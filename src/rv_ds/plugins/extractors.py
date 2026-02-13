from dataclasses import dataclass
from typing import Any, Literal

from ..errors import ValidationFailure
from ..filters import (
    object_passes_target_tags,
    passes_min_counts,
    scene_passes_filters,
)
from ..ir import DatasetIR, InstanceRecord, SampleRecord
from ..mask_ops import object_mask, read_index_map
from ..models import SceneObject, load_scene_meta
from ..plugin_api import ExtractionContext, Extractor
from ..sdk import extract_bbox, extract_largest_polygon, normalize_bbox

TaskMode = Literal["detect", "segment", "both"]


@dataclass
class DefaultExtractorOptions:
    class_mapping: list["ClassMappingRule"]
    target_tags: set[str]
    min_count: dict[str, int]
    require_tags: set[str]
    exclude_tags: set[str]
    include_empty: bool
    epsilon_ratio: float


@dataclass(frozen=True)
class ClassMappingRule:
    class_name: str
    required_tags: set[str]
    optional_tags: set[str]

    def matches(self, object_tags: set[str]) -> bool:
        required_match = bool(self.required_tags) and self.required_tags.issubset(
            object_tags
        )
        optional_match = bool(self.optional_tags) and bool(
            self.optional_tags.intersection(object_tags)
        )
        return required_match or optional_match


class DefaultExtractor(Extractor):
    def __init__(self, mode: TaskMode, opts: DefaultExtractorOptions) -> None:
        self.mode = mode
        self.opts = opts

    def extract_dataset(self, ctx: ExtractionContext) -> DatasetIR:
        class_names = [rule.class_name for rule in self.opts.class_mapping]
        sample_records: list[SampleRecord] = []

        for sample in ctx.samples:
            scene = load_scene_meta(sample.meta_path)
            if not scene_passes_filters(
                scene, self.opts.require_tags, self.opts.exclude_tags
            ):
                continue

            selected: list[tuple[SceneObject, int, str]] = []
            for obj in scene.objects:
                if not object_passes_target_tags(obj, self.opts.target_tags):
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


def build_default_segment_extractor(opts: dict[str, Any]) -> Extractor:
    return DefaultExtractor(mode="segment", opts=_parse_default_extractor_options(opts))


def build_default_detection_extractor(opts: dict[str, Any]) -> Extractor:
    return DefaultExtractor(mode="detect", opts=_parse_default_extractor_options(opts))


def build_default_both_extractor(opts: dict[str, Any]) -> Extractor:
    return DefaultExtractor(mode="both", opts=_parse_default_extractor_options(opts))


def _parse_default_extractor_options(opts: dict[str, Any]) -> DefaultExtractorOptions:
    class_mapping = _parse_class_mapping(opts.get("class_mapping"))

    min_count_raw = opts.get("min_count", {})
    if not isinstance(min_count_raw, dict):
        raise ValidationFailure(
            "extractor option 'min_count' must be an object {tag: count}"
        )
    min_count: dict[str, int] = {}
    for key, value in min_count_raw.items():
        if not isinstance(key, str) or not key.strip():
            raise ValidationFailure(
                "extractor option 'min_count' keys must be non-empty strings"
            )
        if not isinstance(value, int) or value < 0:
            raise ValidationFailure(
                f"extractor option 'min_count[{key}]' must be a non-negative integer"
            )
        min_count[key] = value

    epsilon_ratio_raw = opts.get("epsilon_ratio", 0.002)
    if (
        not isinstance(epsilon_ratio_raw, (int, float))
        or float(epsilon_ratio_raw) < 0.0
    ):
        raise ValidationFailure(
            "extractor option 'epsilon_ratio' must be a non-negative number"
        )

    return DefaultExtractorOptions(
        class_mapping=class_mapping,
        target_tags=_read_str_list(opts, "target_tags"),
        min_count=min_count,
        require_tags=_read_str_list(opts, "require_tags"),
        exclude_tags=_read_str_list(opts, "exclude_tags"),
        include_empty=_read_bool(opts, "include_empty", default=False),
        epsilon_ratio=float(epsilon_ratio_raw),
    )


def _parse_class_mapping(raw: Any) -> list[ClassMappingRule]:
    if not isinstance(raw, list) or not raw:
        raise ValidationFailure(
            "default extractor requires non-empty options.class_mapping list"
        )

    class_mapping: list[ClassMappingRule] = []
    seen_classes: set[str] = set()
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValidationFailure(f"class_mapping[{idx}] must be an object")

        class_name_raw = item.get("class")
        if not isinstance(class_name_raw, str) or not class_name_raw.strip():
            raise ValidationFailure(
                f"class_mapping[{idx}].class must be a non-empty string"
            )
        class_name = class_name_raw.strip()

        if class_name in seen_classes:
            raise ValidationFailure(
                f"class_mapping contains duplicate class name '{class_name}'"
            )
        seen_classes.add(class_name)

        required_tags = _read_rule_tags(item, "required_tags", idx)
        optional_tags = _read_rule_tags(item, "optional_tags", idx)
        if not required_tags and not optional_tags:
            raise ValidationFailure(
                f"class_mapping[{idx}] must set required_tags or optional_tags"
            )
        if required_tags and optional_tags:
            raise ValidationFailure(
                f"class_mapping[{idx}] must not set both required_tags and optional_tags"
            )

        class_mapping.append(
            ClassMappingRule(
                class_name=class_name,
                required_tags=required_tags,
                optional_tags=optional_tags,
            )
        )

    return class_mapping


def _read_rule_tags(item: dict[str, Any], key: str, idx: int) -> set[str]:
    raw = item.get(key, [])
    if not isinstance(raw, list) or not all(isinstance(v, str) for v in raw):
        raise ValidationFailure(f"class_mapping[{idx}].{key} must be list[str]")
    return {value.strip() for value in raw if value.strip()}


def _resolve_class(
    object_tags: list[str], class_mapping: list[ClassMappingRule]
) -> tuple[int | None, str | None]:
    tags = set(object_tags)
    for class_id, rule in enumerate(class_mapping):
        if rule.matches(tags):
            return class_id, rule.class_name
    return None, None


def _read_str_list(opts: dict[str, Any], key: str) -> set[str]:
    raw = opts.get(key, [])
    if isinstance(raw, str):
        return {item.strip() for item in raw.split(",") if item.strip()}
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValidationFailure(
            f"extractor option '{key}' must be list[str] or CSV string"
        )
    return {item.strip() for item in raw if item.strip()}


def _read_bool(opts: dict[str, Any], key: str, default: bool) -> bool:
    raw = opts.get(key, default)
    if not isinstance(raw, bool):
        raise ValidationFailure(f"extractor option '{key}' must be boolean")
    return raw
