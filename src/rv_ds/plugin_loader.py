import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, cast

from .errors import ValidationFailure
from .plugin_api import Exporter, Extractor
from .plugins.exporters import build_default_yolo_exporter
from .plugins.extractors import (
    build_default_both_extractor,
    build_default_detection_extractor,
    build_default_segment_extractor,
)

ExtractorBuilder = Callable[[dict[str, Any]], Extractor]
ExporterBuilder = Callable[[dict[str, Any]], Exporter]


@dataclass(frozen=True)
class LoadedPlugin:
    name: str
    source: str


BUILTIN_EXTRACTORS: dict[str, ExtractorBuilder] = {
    "default-segment": build_default_segment_extractor,
    "default-detection": build_default_detection_extractor,
    "default-both": build_default_both_extractor,
}

BUILTIN_EXPORTERS: dict[str, ExporterBuilder] = {
    "default-yolo": build_default_yolo_exporter,
}


def load_extractor(spec: str, opts: dict[str, Any]) -> tuple[Extractor, LoadedPlugin]:
    if spec in BUILTIN_EXTRACTORS:
        extractor = BUILTIN_EXTRACTORS[spec](opts)
        return extractor, LoadedPlugin(name=spec, source="builtin")

    module = _load_module_from_path(spec)
    builder = _load_symbol(module, "build_extractor")
    extractor = cast(Extractor, builder(opts))
    if not hasattr(extractor, "extract_dataset"):
        raise ValidationFailure(
            f"extractor plugin '{spec}' did not return object with extract_dataset(ctx)"
        )
    return extractor, LoadedPlugin(name=spec, source="path")


def load_exporter(spec: str, opts: dict[str, Any]) -> tuple[Exporter, LoadedPlugin]:
    if spec in BUILTIN_EXPORTERS:
        exporter = BUILTIN_EXPORTERS[spec](opts)
        return exporter, LoadedPlugin(name=spec, source="builtin")

    module = _load_module_from_path(spec)
    builder = _load_symbol(module, "build_exporter")
    exporter = cast(Exporter, builder(opts))
    if not hasattr(exporter, "export_dataset"):
        raise ValidationFailure(
            f"exporter plugin '{spec}' did not return object with export_dataset(ctx)"
        )
    return exporter, LoadedPlugin(name=spec, source="path")


def _load_module_from_path(spec: str) -> ModuleType:
    path = Path(spec)
    if not path.exists() or not path.is_file():
        raise ValidationFailure(f"plugin path does not exist: '{spec}'")
    if path.suffix != ".py":
        raise ValidationFailure(f"plugin path must be a .py file: '{spec}'")

    module_name = f"rv_ds_plugin_{path.stem}_{abs(hash(path.resolve()))}"
    import_spec = importlib.util.spec_from_file_location(module_name, path)
    if import_spec is None or import_spec.loader is None:
        raise ValidationFailure(f"failed to create module spec for plugin '{spec}'")

    module = importlib.util.module_from_spec(import_spec)
    try:
        import_spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise ValidationFailure(f"failed to import plugin '{spec}': {exc}") from exc

    return module


def _load_symbol(module: ModuleType, symbol: str) -> Callable[[dict[str, Any]], Any]:
    if not hasattr(module, symbol):
        raise ValidationFailure(
            f"plugin '{module.__name__}' missing required function '{symbol}(opts: dict)'"
        )
    candidate = getattr(module, symbol)
    if not callable(candidate):
        raise ValidationFailure(
            f"plugin '{module.__name__}' symbol '{symbol}' is not callable"
        )
    return cast(Callable[[dict[str, Any]], Any], candidate)
