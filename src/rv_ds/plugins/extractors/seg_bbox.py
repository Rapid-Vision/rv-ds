from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from ...contracts import INSTANCE_BBOX, INSTANCE_CLASS, INSTANCE_SEGMENT
from ...filters import (
    object_passes_target_tags,
    passes_min_counts,
    scene_passes_filters,
)
from ...ir import InstanceRecord, SampleRecord
from ...mask_ops import object_mask, read_index_map
from ...models import SceneObject, load_scene_meta
from ...plugin_api import (
    BaseExtractor,
    ExtractionContext,
    ExtractorDatasetInfo,
    PluginOptions,
)
from ...scanner import SamplePaths
from ...sdk import extract_bbox, extract_largest_polygon, normalize_bbox

TaskMode = Literal["detect", "segment"]


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
    max_samples: int | None = Field(default=None, ge=1)
    include_unmapped: bool = False
    unmapped_class_name: str = "unmapped"
    polygon_tolerance: float = 1.0
    max_polygon_points: int | None = Field(default=None, ge=3)
    min_segment_area: float = 0.0

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

    @field_validator("polygon_tolerance")
    @classmethod
    def _validate_polygon_tolerance(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("polygon_tolerance must be a non-negative number")
        return value

    @field_validator("min_segment_area")
    @classmethod
    def _validate_min_segment_area(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("min_segment_area must be between 0.0 and 1.0")
        return value

    @field_validator("unmapped_class_name")
    @classmethod
    def _validate_unmapped_class_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("unmapped_class_name must be a non-empty string")
        return stripped

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

        if self.include_unmapped and self.unmapped_class_name in seen:
            raise ValueError(
                "unmapped_class_name must not duplicate a configured class_mapping name"
            )

        return self


class DefaultExtractor(BaseExtractor[DefaultExtractorOptions]):
    OptionsModel = DefaultExtractorOptions
    MODE: TaskMode = "segment"

    def __init__(self, opts: DefaultExtractorOptions) -> None:
        super().__init__(opts)
        self.mode = self.MODE
        self.opts = opts

    def describe_dataset(self, ctx: ExtractionContext) -> ExtractorDatasetInfo:
        _ = ctx
        class_names = [rule.class_name for rule in self.opts.class_mapping]
        if self.opts.include_unmapped:
            class_names.append(self.opts.unmapped_class_name)
        return ExtractorDatasetInfo(
            class_names=class_names,
            meta={"extractor": "default", "mode": self.mode},
        )

    def extract_sample(
        self, ctx: ExtractionContext, sample: SamplePaths
    ) -> SampleRecord | None:
        target_tags = set(self.opts.target_tags)
        required_tags = set(self.opts.require_tags)
        exclude_tags = set(self.opts.exclude_tags)
        scene = load_scene_meta(sample.meta_path)
        if not scene_passes_filters(scene, required_tags, exclude_tags):
            return None

        selected: list[tuple[SceneObject, int, str]] = []
        for obj in scene.objects:
            if not object_passes_target_tags(obj, target_tags):
                continue
            class_id, class_name = _resolve_class(obj.tags, self.opts.class_mapping)
            if (
                class_id is None
                and class_name is None
                and self.opts.include_unmapped
            ):
                class_id = len(self.opts.class_mapping)
                class_name = self.opts.unmapped_class_name
            if class_id is None or class_name is None:
                continue
            selected.append((obj, class_id, class_name))

        if not passes_min_counts([obj for obj, _, _ in selected], self.opts.min_count):
            return None

        index_map = read_index_map(sample.index_path)
        image_area = index_map.shape[0] * index_map.shape[1]
        instances: list[InstanceRecord] = []

        for obj, class_id, class_name in selected:
            mask = object_mask(index_map, obj.index)
            area_px = int(mask.sum())

            bbox_xyxy = extract_bbox(mask)
            bbox_norm = None
            polygon_norm = None

            if bbox_xyxy is not None:
                bbox_norm = normalize_bbox(
                    bbox_xyxy,
                    width=index_map.shape[1],
                    height=index_map.shape[0],
                )

            if self.mode == "segment":
                area_ratio = float(area_px) / float(image_area)
                if area_ratio >= self.opts.min_segment_area:
                    polygon_norm = extract_largest_polygon(
                        mask,
                        polygon_tolerance=self.opts.polygon_tolerance,
                        max_polygon_points=self.opts.max_polygon_points,
                    )

            if self.mode == "detect" and bbox_xyxy is None:
                continue
            if self.mode == "segment" and polygon_norm is None:
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
            return None

        return SampleRecord(
            sample_id=sample.sample_id,
            scene_tags=list(scene.tags),
            image_src_path=sample.image_path,
            image_out_name=f"{sample.sample_id}.png",
            width=index_map.shape[1],
            height=index_map.shape[0],
            instances=instances,
            extra={},
        )


class DefaultSegmentExtractor(DefaultExtractor):
    MODE: TaskMode = "segment"
    produced_features = frozenset(
        {INSTANCE_CLASS, INSTANCE_SEGMENT, INSTANCE_BBOX}
    )


class DefaultDetectionExtractor(DefaultExtractor):
    MODE: TaskMode = "detect"
    produced_features = frozenset({INSTANCE_CLASS, INSTANCE_BBOX})


def _resolve_class(
    object_tags: list[str], class_mapping: list[ClassMappingRule]
) -> tuple[int | None, str | None]:
    tags = set(object_tags)
    for class_id, rule in enumerate(class_mapping):
        if rule.matches(tags):
            return class_id, rule.class_name
    return None, None
