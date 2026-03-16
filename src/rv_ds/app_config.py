from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .errors import ValidationFailure
from .pipeline import ExportConfig as PipelineExportConfig

DEFAULT_IMAGE_FILE = "Image.png"
DEFAULT_OUTPUT_DIR = "./exports"
DEFAULT_CONFIG_PATH = "rv-ds.yaml"


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    path: Path

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("must be a non-empty path")
            return Path(stripped)
        raise ValueError("must be a path string")


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    output_dir: Path = Path(DEFAULT_OUTPUT_DIR)

    @field_validator("output_dir", mode="before")
    @classmethod
    def _validate_output_dir(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("must be a non-empty path")
            return Path(stripped)
        raise ValueError("must be a path string")


class PluginConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    spec: str
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("spec")
    @classmethod
    def _validate_spec(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be a non-empty string")
        return stripped

    @field_validator("options", mode="before")
    @classmethod
    def _validate_options(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("must be an object")
        return dict(value)


class DebugConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dump_ir: bool = False
    fail_on_warning: bool = False


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset: DatasetConfig
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    extractor: PluginConfig
    exporter: PluginConfig
    debug: DebugConfig = Field(default_factory=DebugConfig)


class ResolvedAppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    config_path: Path
    config_dir: Path
    app: AppConfig
    dataset_dir: Path
    output_dir: Path
    extractor_spec: str
    exporter_spec: str


def load_app_config(path: Path) -> ResolvedAppConfig:
    if not path.exists() or not path.is_file():
        raise ValidationFailure(f"config file does not exist: '{path}'")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationFailure(f"failed to parse config '{path}': {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ValidationFailure(f"config '{path}' must contain a YAML object")

    try:
        app = AppConfig.model_validate(dict(raw))
    except ValidationError as exc:
        raise ValidationFailure(_format_validation_error(exc)) from exc

    config_path = path.resolve()
    config_dir = config_path.parent
    dataset_dir = _resolve_path(app.dataset.path, config_dir)
    output_dir = _resolve_path(app.pipeline.output_dir, config_dir)

    return ResolvedAppConfig(
        config_path=config_path,
        config_dir=config_dir,
        app=app,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        extractor_spec=_resolve_plugin_spec(app.extractor.spec, config_dir),
        exporter_spec=_resolve_plugin_spec(app.exporter.spec, config_dir),
    )


def dump_app_config(app_config: AppConfig) -> str:
    payload = app_config.model_dump(mode="json")
    return str(yaml.safe_dump(payload, sort_keys=False))


def build_pipeline_export_config(resolved: ResolvedAppConfig) -> PipelineExportConfig:
    return PipelineExportConfig(
        dataset_dir=resolved.dataset_dir,
        output_dir=resolved.output_dir,
        image_file=DEFAULT_IMAGE_FILE,
        extractor_spec=resolved.extractor_spec,
        extractor_opts=dict(resolved.app.extractor.options),
        exporter_spec=resolved.exporter_spec,
        exporter_opts=dict(resolved.app.exporter.options),
        fail_on_plugin_warning=resolved.app.debug.fail_on_warning,
        dump_ir=resolved.app.debug.dump_ir,
    )


def default_app_config(dataset_dir: Path) -> AppConfig:
    return AppConfig(
        dataset=DatasetConfig(path=dataset_dir),
        extractor=PluginConfig(
            spec="default-seg",
            options={
                "class_mapping": [{"class": "example", "required_tags": ["example"]}]
            },
        ),
        exporter=PluginConfig(
            spec="default-yolo-seg",
            options={"include_empty": False, "splits": {"train": 0.8, "val": 0.2}},
        ),
    )


def list_builtin_capabilities() -> dict[str, Any]:
    presets = {
        "detection-preview": {
            "extractor": "default-bbox",
            "exporter": "default-preview-bbox",
        },
        "detection-yolo_bbox": {
            "extractor": "default-bbox",
            "exporter": "default-yolo-bbox",
        },
        "segmentation-preview": {
            "extractor": "default-seg",
            "exporter": "default-preview-seg",
        },
        "segmentation-yolo_seg": {
            "extractor": "default-seg",
            "exporter": "default-yolo-seg",
        },
        "both-preview": {
            "extractor": "default-seg-bbox",
            "exporter": "default-preview-seg",
        },
        "both-yolo_bbox": {
            "extractor": "default-seg-bbox",
            "exporter": "default-yolo-bbox",
        },
        "both-yolo_seg": {
            "extractor": "default-seg-bbox",
            "exporter": "default-yolo-seg",
        },
    }
    return {
        "extractors": ["default-seg", "default-bbox", "default-seg-bbox"],
        "exporters": [
            "default-preview-bbox",
            "default-preview-seg",
            "default-yolo-bbox",
            "default-yolo-seg",
        ],
        "presets": presets,
    }


def _resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _resolve_plugin_spec(spec: str, base_dir: Path) -> str:
    path = Path(spec)
    looks_like_path = path.suffix == ".py" or len(path.parts) > 1
    if not looks_like_path:
        return spec
    return str(_resolve_path(path, base_dir))


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["invalid config:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)
