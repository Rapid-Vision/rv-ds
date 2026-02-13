from rv_export.filters import (
    object_passes_target_tags,
    parse_csv_tags,
    passes_min_counts,
    scene_passes_filters,
)
from rv_export.models import SceneMeta, SceneObject


def test_parse_csv_tags() -> None:
    assert parse_csv_tags(" a, b ,,c ") == {"a", "b", "c"}


def test_scene_passes_filters() -> None:
    scene = SceneMeta(objects=[], tags=["sunny", "outdoor"])

    assert scene_passes_filters(scene, {"sunny"}, {"night"})
    assert not scene_passes_filters(scene, {"indoor"}, set())
    assert not scene_passes_filters(scene, set(), {"outdoor"})


def test_object_passes_target_tags() -> None:
    obj = SceneObject(index=2, name="Cube", tags=["cube", "red"])

    assert object_passes_target_tags(obj, {"red"})
    assert not object_passes_target_tags(obj, {"sphere"})


def test_passes_min_counts() -> None:
    objs = [
        SceneObject(index=1, name="o1", tags=["a", "b"]),
        SceneObject(index=2, name="o2", tags=["a"]),
    ]

    assert passes_min_counts(objs, {"a": 2})
    assert not passes_min_counts(objs, {"a": 3})
