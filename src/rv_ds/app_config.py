from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import ValidationFailure
from .pipeline import ExportConfig as PipelineExportConfig

DEFAULT_IMAGE_FILE = "Image.png"
DEFAULT_OUTPUT_DIR = "./exports"
DEFAULT_CONFIG_PATH = "rv-ds.yaml"
DEFAULT_YOLO_SPLITS = {"train": 0.8, "val": 0.2}
UNMAPPED_CLASS_NAME = "unmapped"

TaskName = Literal["detection", "segmentation", "both"]
OutputFormatName = Literal["preview", "yolo_bbox", "yolo_seg"]


def _normalize_tag_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("must be a list of strings")
    return [item.strip() for item in raw if item.strip()]


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

    task: TaskName
    output_format: OutputFormatName
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


class SceneSelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    require_tags: list[str] = Field(default_factory=list)
    exclude_tags: list[str] = Field(default_factory=list)

    @field_validator("require_tags", "exclude_tags", mode="before")
    @classmethod
    def _validate_tags(cls, value: Any) -> list[str]:
        return _normalize_tag_list(value)


class ObjectSelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    target_tags: list[str] = Field(default_factory=list)
    include_unmapped: bool = False

    @field_validator("target_tags", mode="before")
    @classmethod
    def _validate_tags(cls, value: Any) -> list[str]:
        return _normalize_tag_list(value)


class SelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scene: SceneSelectionConfig = Field(default_factory=SceneSelectionConfig)
    objects: ObjectSelectionConfig = Field(default_factory=ObjectSelectionConfig)


class TagMatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    all_tags: list[str]

    @field_validator("all_tags", mode="before")
    @classmethod
    def _validate_tags(cls, value: Any) -> list[str]:
        tags = _normalize_tag_list(value)
        if not tags:
            raise ValueError("must contain at least one tag")
        return tags


class ClassMappingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    match: TagMatchConfig

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be a non-empty string")
        return stripped


class ClassesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    mapping: list[ClassMappingConfig]

    @field_validator("mapping")
    @classmethod
    def _validate_mapping(cls, value: list[ClassMappingConfig]) -> list[ClassMappingConfig]:
        if not value:
            raise ValueError("must contain at least one class mapping")
        seen: set[str] = set()
        duplicates: set[str] = set()
        for item in value:
            if item.name in seen:
                duplicates.add(item.name)
            seen.add(item.name)
        if duplicates:
            joined = ", ".join(sorted(duplicates))
            raise ValueError(f"contains duplicate class names: {joined}")
        return value


class ExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    include_empty: bool = False
    splits: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_YOLO_SPLITS))

    @field_validator("splits")
    @classmethod
    def _validate_splits(cls, value: dict[str, float]) -> dict[str, float]:
        allowed = {"train", "val", "test"}
        if not value:
            raise ValueError("must not be empty")
        unknown = set(value) - allowed
        if unknown:
            joined = ", ".join(sorted(unknown))
            raise ValueError(f"contains unsupported split names: {joined}")
        if "train" not in value or "val" not in value:
            raise ValueError("must include train and val")
        total = 0.0
        for name, ratio in value.items():
            if ratio <= 0.0:
                raise ValueError(f"{name} must be > 0")
            total += ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError("must sum to 1.0")
        return value


class DebugConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dump_ir: bool = False
    fail_on_warning: bool = False


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset: DatasetConfig
    pipeline: PipelineConfig
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    classes: ClassesConfig
    export: ExportConfig = Field(default_factory=ExportConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

    @model_validator(mode="after")
    def _validate_pipeline_compatibility(self) -> "AppConfig":
        task = self.pipeline.task
        output_format = self.pipeline.output_format

        if task == "detection" and output_format == "yolo_seg":
            raise ValueError(
                "pipeline.output_format 'yolo_seg' requires task 'segmentation' or 'both'"
            )
        if task == "segmentation" and output_format == "yolo_bbox":
            raise ValueError(
                "pipeline.output_format 'yolo_bbox' requires task 'detection' or 'both'"
            )
        return self


class ResolvedAppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    config_path: Path
    config_dir: Path
    app: AppConfig
    dataset_dir: Path
    output_dir: Path


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
    )


def dump_app_config(app_config: AppConfig) -> str:
    payload = app_config.model_dump(mode="json")
    return str(yaml.safe_dump(payload, sort_keys=False))


def build_pipeline_export_config(resolved: ResolvedAppConfig) -> PipelineExportConfig:
    extractor_spec, exporter_spec = _select_pipeline_specs(
        resolved.app.pipeline.task, resolved.app.pipeline.output_format
    )
    extractor_opts = {
        "class_mapping": [
            {
                "class": item.name,
                "required_tags": item.match.all_tags,
            }
            for item in resolved.app.classes.mapping
        ],
        "target_tags": resolved.app.selection.objects.target_tags,
        "require_tags": resolved.app.selection.scene.require_tags,
        "exclude_tags": resolved.app.selection.scene.exclude_tags,
        "include_empty": resolved.app.export.include_empty,
        "include_unmapped": resolved.app.selection.objects.include_unmapped,
        "unmapped_class_name": UNMAPPED_CLASS_NAME,
    }
    exporter_opts: dict[str, Any] = {
        "include_empty": resolved.app.export.include_empty,
    }
    if resolved.app.pipeline.output_format != "preview":
        exporter_opts["splits"] = resolved.app.export.splits

    return PipelineExportConfig(
        dataset_dir=resolved.dataset_dir,
        output_dir=resolved.output_dir,
        image_file=DEFAULT_IMAGE_FILE,
        extractor_spec=extractor_spec,
        extractor_opts=extractor_opts,
        exporter_spec=exporter_spec,
        exporter_opts=exporter_opts,
        fail_on_plugin_warning=resolved.app.debug.fail_on_warning,
        dump_ir=resolved.app.debug.dump_ir,
    )


def default_app_config(dataset_dir: Path) -> AppConfig:
    return AppConfig(
        dataset=DatasetConfig(path=dataset_dir),
        pipeline=PipelineConfig(task="segmentation", output_format="yolo_seg"),
        classes=ClassesConfig(
            mapping=[
                ClassMappingConfig(
                    name="example",
                    match=TagMatchConfig(all_tags=["example"]),
                )
            ]
        ),
        export=ExportConfig(include_empty=False),
    )


def list_builtin_capabilities() -> dict[str, list[str]]:
    return {
        "tasks": ["detection", "segmentation", "both"],
        "output_formats": ["preview", "yolo_bbox", "yolo_seg"],
        "presets": [
            "detection-preview",
            "detection-yolo_bbox",
            "segmentation-preview",
            "segmentation-yolo_seg",
            "both-preview",
            "both-yolo_bbox",
            "both-yolo_seg",
        ],
    }


def _select_pipeline_specs(
    task: TaskName, output_format: OutputFormatName
) -> tuple[str, str]:
    extractor_by_task = {
        "detection": "default-bbox",
        "segmentation": "default-seg",
        "both": "default-seg-bbox",
    }
    extractor_spec = extractor_by_task[task]

    if output_format == "preview":
        exporter_spec = (
            "default-preview-bbox" if task == "detection" else "default-preview-seg"
        )
        return extractor_spec, exporter_spec
    if output_format == "yolo_bbox":
        return extractor_spec, "default-yolo-bbox"
    return extractor_spec, "default-yolo-seg"


def _resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["invalid config:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)
