import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, TypeAlias, TypeVar, cast

from pydantic import ValidationError

from .errors import ValidationFailure
from .plugin_api import BaseExporter, BaseExtractor, PluginOptions
from .plugins.exporters import (
    DefaultPreviewBBoxExporter,
    DefaultPreviewSegExporter,
    DefaultYoloExporter,
)
from .plugins.extractors import (
    DefaultBothExtractor,
    DefaultDetectionExtractor,
    DefaultSegmentExtractor,
)

ExtractorType: TypeAlias = BaseExtractor[Any]
ExporterType: TypeAlias = BaseExporter[Any]
PluginType: TypeAlias = ExtractorType | ExporterType

TPlugin = TypeVar("TPlugin", bound=PluginType)
PluginKind = Literal["extractor", "exporter"]


@dataclass(frozen=True)
class LoadedPlugin:
    name: str
    source: str


BUILTIN_EXTRACTORS: dict[str, type[ExtractorType]] = {
    "default-segment": DefaultSegmentExtractor,
    "default-detection": DefaultDetectionExtractor,
    "default-both": DefaultBothExtractor,
}

BUILTIN_EXPORTERS: dict[str, type[ExporterType]] = {
    "default-yolo": DefaultYoloExporter,
    "default-preview-bbox": DefaultPreviewBBoxExporter,
    "default-preview-seg": DefaultPreviewSegExporter,
}


def load_extractor(
    spec: str, opts: dict[str, Any]
) -> tuple[ExtractorType, LoadedPlugin]:
    plugin, loaded = _load_plugin(
        kind="extractor",
        spec=spec,
        opts=opts,
        builtin_registry=BUILTIN_EXTRACTORS,
        plugin_symbol="ExtractorPlugin",
        expected_base=cast(type[ExtractorType], BaseExtractor),
    )
    return plugin, loaded


def load_exporter(
    spec: str, opts: dict[str, Any]
) -> tuple[ExporterType, LoadedPlugin]:
    plugin, loaded = _load_plugin(
        kind="exporter",
        spec=spec,
        opts=opts,
        builtin_registry=BUILTIN_EXPORTERS,
        plugin_symbol="ExporterPlugin",
        expected_base=cast(type[ExporterType], BaseExporter),
    )
    return plugin, loaded


def _load_plugin(
    kind: PluginKind,
    spec: str,
    opts: dict[str, Any],
    builtin_registry: dict[str, type[TPlugin]],
    plugin_symbol: str,
    expected_base: type[TPlugin],
) -> tuple[TPlugin, LoadedPlugin]:
    if spec in builtin_registry:
        plugin_class = builtin_registry[spec]
        plugin = _instantiate_plugin(
            plugin_class,
            opts,
            plugin_type=kind,
            plugin_name=spec,
        )
        return plugin, LoadedPlugin(name=spec, source="builtin")

    module = _load_module_from_path(spec)
    plugin_class = _load_plugin_class(module, plugin_symbol, expected_base)
    plugin = _instantiate_plugin(
        plugin_class,
        opts,
        plugin_type=kind,
        plugin_name=spec,
    )
    return plugin, LoadedPlugin(name=spec, source="path")


def _instantiate_plugin(
    plugin_class: type[TPlugin],
    raw_opts: dict[str, Any],
    plugin_type: str,
    plugin_name: str,
) -> TPlugin:
    options_model = plugin_class.OptionsModel
    if not issubclass(options_model, PluginOptions):
        raise ValidationFailure(
            f"{plugin_type} plugin '{plugin_name}' has invalid OptionsModel; "
            "must inherit PluginOptions"
        )

    try:
        validated_opts = options_model.model_validate(raw_opts)
    except ValidationError as exc:
        raise ValidationFailure(
            f"invalid {plugin_type} options for '{plugin_name}': {exc}"
        ) from exc

    plugin = plugin_class(validated_opts)
    if plugin_type == "extractor" and not hasattr(plugin, "extract_dataset"):
        raise ValidationFailure(
            f"extractor plugin '{plugin_name}' did not implement extract_dataset(ctx)"
        )
    if plugin_type == "exporter" and not hasattr(plugin, "export_dataset"):
        raise ValidationFailure(
            f"exporter plugin '{plugin_name}' did not implement export_dataset(ctx)"
        )

    return cast(TPlugin, plugin)


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


def _load_plugin_class(
    module: ModuleType,
    symbol: str,
    expected_base: type[TPlugin],
) -> type[TPlugin]:
    if not hasattr(module, symbol):
        raise ValidationFailure(
            f"plugin '{module.__name__}' missing required class '{symbol}'"
        )

    candidate = getattr(module, symbol)
    if not isinstance(candidate, type):
        raise ValidationFailure(
            f"plugin '{module.__name__}' symbol '{symbol}' must be a class"
        )
    if not issubclass(candidate, expected_base):
        base_name = expected_base.__name__
        raise ValidationFailure(
            f"plugin '{module.__name__}' class '{symbol}' must inherit {base_name}"
        )

    return candidate
