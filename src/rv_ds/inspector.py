from collections import Counter
from itertools import islice
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .models import load_scene_meta


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
