import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, cast

from pydantic import ValidationError


from .errors import ValidationFailure
from .plugin_api import Exporter, Extractor, PluginOptions
from .plugins.exporters import DefaultYoloExporterOptions, build_default_yolo_exporter
from .plugins.extractors import (
    DefaultExtractorOptions,
    build_default_both_extractor,
    build_default_detection_extractor,
    build_default_segment_extractor,
)

ExtractorBuilder = Callable[[PluginOptions], Extractor]
ExporterBuilder = Callable[[PluginOptions], Exporter]


@dataclass(frozen=True)
class LoadedPlugin:
    name: str
    source: str


@dataclass(frozen=True)
class BuiltinExtractorSpec:
    options_model: type[PluginOptions]
    builder: ExtractorBuilder


@dataclass(frozen=True)
class BuiltinExporterSpec:
    options_model: type[PluginOptions]
    builder: ExporterBuilder


BUILTIN_EXTRACTORS: dict[str, BuiltinExtractorSpec] = {
    "default-segment": BuiltinExtractorSpec(
        options_model=DefaultExtractorOptions,
        builder=cast(ExtractorBuilder, build_default_segment_extractor),
    ),
    "default-detection": BuiltinExtractorSpec(
        options_model=DefaultExtractorOptions,
        builder=cast(ExtractorBuilder, build_default_detection_extractor),
    ),
    "default-both": BuiltinExtractorSpec(
        options_model=DefaultExtractorOptions,
        builder=cast(ExtractorBuilder, build_default_both_extractor),
    ),
}

BUILTIN_EXPORTERS: dict[str, BuiltinExporterSpec] = {
    "default-yolo": BuiltinExporterSpec(
        options_model=DefaultYoloExporterOptions,
        builder=cast(ExporterBuilder, build_default_yolo_exporter),
    )
}


def load_extractor(spec: str, opts: dict[str, Any]) -> tuple[Extractor, LoadedPlugin]:
    if spec in BUILTIN_EXTRACTORS:
        builtin = BUILTIN_EXTRACTORS[spec]
        validated_opts = _validate_options(
            opts,
            builtin.options_model,
            plugin_type="extractor",
            plugin_name=spec,
        )
        extractor = builtin.builder(validated_opts)
        return extractor, LoadedPlugin(name=spec, source="builtin")

    module = _load_module_from_path(spec)
    options_model = _load_options_model(module, "ExtractorOptions")
    builder = _load_symbol(module, "build_extractor")
    validated_opts = _validate_options(
        opts,
        options_model,
        plugin_type="extractor",
        plugin_name=spec,
    )
    extractor = cast(Extractor, builder(validated_opts))
    if not hasattr(extractor, "extract_dataset"):
        raise ValidationFailure(
            f"extractor plugin '{spec}' did not return object with extract_dataset(ctx)"
        )
    return extractor, LoadedPlugin(name=spec, source="path")


def load_exporter(spec: str, opts: dict[str, Any]) -> tuple[Exporter, LoadedPlugin]:
    if spec in BUILTIN_EXPORTERS:
        builtin = BUILTIN_EXPORTERS[spec]
        validated_opts = _validate_options(
            opts,
            builtin.options_model,
            plugin_type="exporter",
            plugin_name=spec,
        )
        exporter = builtin.builder(validated_opts)
        return exporter, LoadedPlugin(name=spec, source="builtin")

    module = _load_module_from_path(spec)
    options_model = _load_options_model(module, "ExporterOptions")
    builder = _load_symbol(module, "build_exporter")
    validated_opts = _validate_options(
        opts,
        options_model,
        plugin_type="exporter",
        plugin_name=spec,
    )
    exporter = cast(Exporter, builder(validated_opts))
    if not hasattr(exporter, "export_dataset"):
        raise ValidationFailure(
            f"exporter plugin '{spec}' did not return object with export_dataset(ctx)"
        )
    return exporter, LoadedPlugin(name=spec, source="path")


def _validate_options(
    opts: dict[str, Any],
    options_model: type[PluginOptions],
    plugin_type: str,
    plugin_name: str,
) -> PluginOptions:
    try:
        return options_model.model_validate(opts)
    except ValidationError as exc:
        raise ValidationFailure(
            f"invalid {plugin_type} options for '{plugin_name}': {exc}"
        ) from exc


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


def _load_options_model(module: ModuleType, symbol: str) -> type[PluginOptions]:
    if not hasattr(module, symbol):
        raise ValidationFailure(
            f"plugin '{module.__name__}' missing required options model '{symbol}'"
        )

    candidate = getattr(module, symbol)
    if not isinstance(candidate, type):
        raise ValidationFailure(
            f"plugin '{module.__name__}' symbol '{symbol}' must be a class"
        )
    if not issubclass(candidate, PluginOptions):
        raise ValidationFailure(
            f"plugin '{module.__name__}' options model '{symbol}' must inherit PluginOptions"
        )

    return cast(type[PluginOptions], candidate)


def _load_symbol(module: ModuleType, symbol: str) -> Callable[[PluginOptions], Any]:
    if not hasattr(module, symbol):
        raise ValidationFailure(
            f"plugin '{module.__name__}' missing required function '{symbol}(opts)'"
        )
    candidate = getattr(module, symbol)
    if not callable(candidate):
        raise ValidationFailure(
            f"plugin '{module.__name__}' symbol '{symbol}' is not callable"
        )
    return cast(Callable[[PluginOptions], Any], candidate)
