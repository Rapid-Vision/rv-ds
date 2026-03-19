from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .geometry import bbox_from_mask, largest_polygon_from_mask
from .mask_ops import object_mask, read_index_map
from .models import SceneObject, load_scene_meta
from .errors import ValidationFailure


class NumericSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    min: int
    max: int
    mean: float
    total: int


class SampleMaskStats(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    object_index: int
    object_name: str | None = None
    object_tags: list[str] = Field(default_factory=list)
    area_px: int
    polygon_points: int
    bbox_xyxy: tuple[int, int, int, int] | None = None


class SampleInspectReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_dir: Path
    sample_id: str
    sample_dir: Path
    image_file: str
    image_width: int
    image_height: int
    object_count_in_meta: int
    mask_count: int
    missing_mask_indexes: list[int] = Field(default_factory=list)
    orphan_mask_indexes: list[int] = Field(default_factory=list)
    area_px: NumericSummary | None = None
    polygon_points: NumericSummary | None = None
    masks: list[SampleMaskStats] = Field(default_factory=list)


class SampleIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sample_id: str
    problems: list[str]


class TagCount(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tag: str
    count: int


class TagCombinationCount(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tags: list[str]
    count: int


class SuggestedClassMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    class_name: str
    all_tags: list[str]
    count: int


class InspectReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_dir: Path
    image_file: str
    total_sample_dirs: int
    valid_sample_count: int
    invalid_sample_count: int
    issues: list[SampleIssue] = Field(default_factory=list)
    scene_tags: list[TagCount] = Field(default_factory=list)
    object_tags: list[TagCount] = Field(default_factory=list)
    object_tag_combinations: list[TagCombinationCount] = Field(default_factory=list)
    suggested_class_mappings: list[SuggestedClassMapping] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


def inspect_sample(dataset_dir: Path, image_file: str, sample_id: str) -> SampleInspectReport:
    sample_dir = dataset_dir / sample_id
    if not sample_dir.exists() or not sample_dir.is_dir():
        raise ValidationFailure(f"sample directory does not exist: '{sample_dir}'")

    meta_path = sample_dir / "_meta.json"
    index_path = sample_dir / "IndexOB.png"
    image_path = sample_dir / image_file
    missing = [
        path.name
        for path in (meta_path, index_path, image_path)
        if not path.exists() or not path.is_file()
    ]
    if missing:
        raise ValidationFailure(
            f"sample '{sample_id}' is missing required files: {', '.join(missing)}"
        )

    scene = load_scene_meta(meta_path)
    index_map = read_index_map(index_path)
    image_height, image_width = index_map.shape[:2]

    object_by_index = {obj.index: obj for obj in scene.objects}
    present_indexes = sorted(int(index) for index in np.unique(index_map) if int(index) > 0)

    masks: list[SampleMaskStats] = []
    for object_index in present_indexes:
        mask = object_mask(index_map, object_index)
        area_px = int(np.count_nonzero(mask))
        if area_px == 0:
            continue

        polygon = largest_polygon_from_mask(mask)
        bbox = bbox_from_mask(mask)
        scene_object = object_by_index.get(object_index)
        masks.append(
            SampleMaskStats(
                object_index=object_index,
                object_name=scene_object.name if scene_object is not None else None,
                object_tags=list(scene_object.tags) if scene_object is not None else [],
                area_px=area_px,
                polygon_points=len(polygon) if polygon is not None else 0,
                bbox_xyxy=bbox,
            )
        )

    masks.sort(key=lambda item: item.object_index)
    meta_indexes = sorted(object_by_index)
    mask_indexes = [item.object_index for item in masks]

    return SampleInspectReport(
        dataset_dir=dataset_dir,
        sample_id=sample_id,
        sample_dir=sample_dir,
        image_file=image_file,
        image_width=image_width,
        image_height=image_height,
        object_count_in_meta=len(scene.objects),
        mask_count=len(masks),
        missing_mask_indexes=sorted(index for index in meta_indexes if index not in set(mask_indexes)),
        orphan_mask_indexes=sorted(index for index in mask_indexes if index not in set(meta_indexes)),
        area_px=_summarize_numeric([item.area_px for item in masks]),
        polygon_points=_summarize_numeric([item.polygon_points for item in masks]),
        masks=masks,
    )


def inspect_dataset(dataset_dir: Path, image_file: str) -> InspectReport:
    sample_dirs = sorted(
        [path for path in dataset_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name,
    )
    scene_tags: Counter[str] = Counter()
    object_tags: Counter[str] = Counter()
    tag_combinations: Counter[tuple[str, ...]] = Counter()
    issues: list[SampleIssue] = []
    valid_sample_count = 0

    for sample_dir in sample_dirs:
        meta_path = sample_dir / "_meta.json"
        index_path = sample_dir / "IndexOB.png"
        image_path = sample_dir / image_file

        problems: list[str] = []
        if not meta_path.exists() or not meta_path.is_file():
            problems.append("missing _meta.json")
        if not index_path.exists() or not index_path.is_file():
            problems.append("missing IndexOB.png")
        if not image_path.exists() or not image_path.is_file():
            problems.append(f"missing {image_file}")

        scene = None
        if meta_path.exists() and meta_path.is_file():
            try:
                scene = load_scene_meta(meta_path)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"invalid _meta.json: {exc}")

        if scene is not None:
            scene_tags.update(tag for tag in scene.tags if tag)
            for obj in scene.objects:
                tags = tuple(sorted({tag for tag in obj.tags if tag}))
                object_tags.update(tags)
                if tags:
                    tag_combinations[tags] += 1

        if problems:
            issues.append(SampleIssue(sample_id=sample_dir.name, problems=problems))
        else:
            valid_sample_count += 1

    invalid_sample_count = len(issues)
    recommendations = _build_recommendations(
        total_sample_dirs=len(sample_dirs),
        valid_sample_count=valid_sample_count,
        invalid_sample_count=invalid_sample_count,
        object_tags=object_tags,
    )

    return InspectReport(
        dataset_dir=dataset_dir,
        image_file=image_file,
        total_sample_dirs=len(sample_dirs),
        valid_sample_count=valid_sample_count,
        invalid_sample_count=invalid_sample_count,
        issues=issues,
        scene_tags=_sorted_tag_counts(scene_tags),
        object_tags=_sorted_tag_counts(object_tags),
        object_tag_combinations=[
            TagCombinationCount(tags=list(tags), count=count)
            for tags, count in islice(_sorted_combination_items(tag_combinations), 10)
        ],
        suggested_class_mappings=[
            SuggestedClassMapping(class_name=tag, all_tags=[tag], count=count)
            for tag, count in islice(_sorted_tag_items(object_tags), 10)
        ],
        recommendations=recommendations,
    )


def _sorted_tag_counts(counter: Counter[str]) -> list[TagCount]:
    return [TagCount(tag=tag, count=count) for tag, count in _sorted_tag_items(counter)]


def _sorted_tag_items(counter: Counter[str]) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def _sorted_combination_items(
    counter: Counter[tuple[str, ...]],
) -> list[tuple[tuple[str, ...], int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def _build_recommendations(
    *,
    total_sample_dirs: int,
    valid_sample_count: int,
    invalid_sample_count: int,
    object_tags: Counter[str],
) -> list[str]:
    recommendations: list[str] = []
    if total_sample_dirs == 0:
        recommendations.append("Dataset contains no sample directories.")
    if invalid_sample_count:
        recommendations.append(
            f"{invalid_sample_count} sample directories are missing required files or metadata."
        )
    if valid_sample_count == 0 and total_sample_dirs:
        recommendations.append(
            "No fully valid samples were found; export will fail until the dataset is fixed."
        )
    if not object_tags:
        recommendations.append(
            "No object tags found; create class mappings manually before export."
        )
    return recommendations


def _summarize_numeric(values: list[int]) -> NumericSummary | None:
    if not values:
        return None

    return NumericSummary(
        min=min(values),
        max=max(values),
        mean=float(sum(values)) / float(len(values)),
        total=sum(values),
    )
