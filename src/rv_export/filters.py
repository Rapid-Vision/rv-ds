from collections import Counter

from .models import SceneMeta, SceneObject


def parse_csv_tags(raw: str | None) -> set[str]:
    if raw is None:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def scene_passes_filters(
    scene: SceneMeta,
    required_tags: set[str],
    excluded_tags: set[str],
) -> bool:
    scene_tags = set(scene.tags)
    if required_tags and not required_tags.issubset(scene_tags):
        return False
    if excluded_tags and scene_tags.intersection(excluded_tags):
        return False
    return True


def object_passes_target_tags(obj: SceneObject, target_tags: set[str]) -> bool:
    if not target_tags:
        return True
    return bool(set(obj.tags).intersection(target_tags))


def passes_min_counts(
    selected_objects: list[SceneObject], min_count: dict[str, int]
) -> bool:
    if not min_count:
        return True

    tag_counter = Counter(tag for obj in selected_objects for tag in obj.tags)
    for tag, required_count in min_count.items():
        if tag_counter[tag] < required_count:
            return False
    return True
