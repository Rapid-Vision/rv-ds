from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from ..filters import (
    object_passes_target_tags,
    passes_min_counts,
    scene_passes_filters,
)
from ..ir import DatasetIR, InstanceRecord, SampleRecord
from ..mask_ops import object_mask, read_index_map
from ..models import SceneObject, load_scene_meta
from ..plugin_api import ExtractionContext, Extractor, PluginOptions
from ..sdk import extract_bbox, extract_largest_polygon, normalize_bbox

TaskMode = Literal["detect", "segment", "both"]


class ClassMappingRule(PluginOptions):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)

    class_name: str = Field(alias="class")
    required_tags: list[str] = Field(default_factory=list)
    optional_tags: list[str] = Field(default_factory=list)

    @field_validator("class_name")
    @classmethod
    def _validate_class_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("class must be a non-empty string")
        return stripped

    @field_validator("required_tags", "optional_tags")
    @classmethod
    def _normalize_tags(cls, tags: list[str]) -> list[str]:
        return [tag.strip() for tag in tags if tag.strip()]

    @model_validator(mode="after")
    def _validate_tag_schema(self) -> "ClassMappingRule":
        has_required = bool(self.required_tags)
        has_optional = bool(self.optional_tags)
        if not has_required and not has_optional:
            raise ValueError("must set required_tags or optional_tags")
        if has_required and has_optional:
            raise ValueError("must not set both required_tags and optional_tags")
        return self

    def matches(self, object_tags: set[str]) -> bool:
        required = set(self.required_tags)
        optional = set(self.optional_tags)
        required_match = bool(required) and required.issubset(
            object_tags
        )
        optional_match = bool(optional) and bool(
            optional.intersection(object_tags)
        )
        return required_match or optional_match


class DefaultExtractorOptions(PluginOptions):
    class_mapping: list[ClassMappingRule]
    target_tags: list[str] | str = Field(default_factory=list)
    min_count: dict[str, int] = Field(default_factory=dict)
    require_tags: list[str] | str = Field(default_factory=list)
    exclude_tags: list[str] | str = Field(default_factory=list)
    include_empty: bool = False
    epsilon_ratio: float = 0.002

    @field_validator("target_tags", "require_tags", "exclude_tags", mode="before")
    @classmethod
    def _parse_tag_list(cls, raw: Any) -> list[str]:
        if isinstance(raw, str):
            return [item.strip() for item in raw.split(",") if item.strip()]
        if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
            return [item.strip() for item in raw if item.strip()]
        raise ValueError("must be list[str] or CSV string")

    @field_validator("min_count")
    @classmethod
    def _validate_min_count(cls, value: dict[str, int]) -> dict[str, int]:
        validated: dict[str, int] = {}
        for key, count in value.items():
            if not key.strip():
                raise ValueError("min_count keys must be non-empty strings")
            if count < 0:
                raise ValueError(
                    f"min_count[{key}] must be a non-negative integer"
                )
            validated[key.strip()] = count
        return validated

    @field_validator("epsilon_ratio")
    @classmethod
    def _validate_epsilon_ratio(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("epsilon_ratio must be a non-negative number")
        return value

    @model_validator(mode="after")
    def _validate_class_mapping(self) -> "DefaultExtractorOptions":
        if not self.class_mapping:
            raise ValueError("class_mapping must be non-empty")

        seen: set[str] = set()
        duplicates: set[str] = set()
        for item in self.class_mapping:
            if item.class_name in seen:
                duplicates.add(item.class_name)
            seen.add(item.class_name)

        if duplicates:
            joined = ", ".join(sorted(duplicates))
            raise ValueError(f"class_mapping contains duplicate class name(s): {joined}")

        return self


class DefaultExtractor(Extractor):
    def __init__(self, mode: TaskMode, opts: DefaultExtractorOptions) -> None:
        self.mode = mode
        self.opts = opts

    def extract_dataset(self, ctx: ExtractionContext) -> DatasetIR:
        class_names = [rule.class_name for rule in self.opts.class_mapping]
        target_tags = set(self.opts.target_tags)
        required_tags = set(self.opts.require_tags)
        exclude_tags = set(self.opts.exclude_tags)
        sample_records: list[SampleRecord] = []

        for sample in ctx.samples:
            scene = load_scene_meta(sample.meta_path)
            if not scene_passes_filters(scene, required_tags, exclude_tags):
                continue

            selected: list[tuple[SceneObject, int, str]] = []
            for obj in scene.objects:
                if not object_passes_target_tags(obj, target_tags):
                    continue
                class_id, class_name = _resolve_class(obj.tags, self.opts.class_mapping)
                if class_id is None or class_name is None:
                    continue
                selected.append((obj, class_id, class_name))

            if not passes_min_counts(
                [obj for obj, _, _ in selected], self.opts.min_count
            ):
                continue

            index_map = read_index_map(sample.index_path)
            instances: list[InstanceRecord] = []

            for obj, class_id, class_name in selected:
                mask = object_mask(index_map, obj.index)
                area_px = int(mask.sum())

                bbox_xyxy = None
                bbox_norm = None
                polygon_norm = None

                if self.mode in ("detect", "both"):
                    bbox_xyxy = extract_bbox(mask)
                    if bbox_xyxy is not None:
                        bbox_norm = normalize_bbox(
                            bbox_xyxy,
                            width=index_map.shape[1],
                            height=index_map.shape[0],
                        )

                if self.mode in ("segment", "both"):
                    polygon_norm = extract_largest_polygon(
                        mask, epsilon_ratio=self.opts.epsilon_ratio
                    )

                if bbox_xyxy is None and polygon_norm is None:
                    continue

                instances.append(
                    InstanceRecord(
                        sample_id=sample.sample_id,
                        object_index=obj.index,
                        class_name=class_name,
                        class_id=class_id,
                        object_tags=list(obj.tags),
                        bbox_xyxy=bbox_xyxy,
                        bbox_norm_cxcywh=bbox_norm,
                        polygon_norm=polygon_norm,
                        area_px=area_px,
                        extra={},
                    )
                )

            if not instances and not self.opts.include_empty:
                continue

            sample_records.append(
                SampleRecord(
                    sample_id=sample.sample_id,
                    scene_tags=list(scene.tags),
                    image_src_path=sample.image_path,
                    image_out_name=f"{sample.sample_id}.png",
                    width=index_map.shape[1],
                    height=index_map.shape[0],
                    instances=instances,
                    extra={},
                )
            )

        dataset = DatasetIR(
            samples=sample_records,
            class_names=class_names,
            meta={"extractor": "default", "mode": self.mode},
        )
        dataset.validate()
        return dataset


def build_default_segment_extractor(opts: DefaultExtractorOptions) -> Extractor:
    return DefaultExtractor(mode="segment", opts=opts)


def build_default_detection_extractor(opts: DefaultExtractorOptions) -> Extractor:
    return DefaultExtractor(mode="detect", opts=opts)


def build_default_both_extractor(opts: DefaultExtractorOptions) -> Extractor:
    return DefaultExtractor(mode="both", opts=opts)


def _resolve_class(
    object_tags: list[str], class_mapping: list[ClassMappingRule]
) -> tuple[int | None, str | None]:
    tags = set(object_tags)
    for class_id, rule in enumerate(class_mapping):
        if rule.matches(tags):
            return class_id, rule.class_name
    return None, None
