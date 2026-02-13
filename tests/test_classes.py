from pathlib import Path

import pytest

from rv_export.classes import ClassResolver, parse_classes_file
from rv_export.errors import ValidationFailure
from rv_export.models import SceneObject


def test_parse_classes_file_success(tmp_path: Path) -> None:
    classes = tmp_path / "classes.txt"
    classes.write_text("cube\nsphere\n", encoding="utf-8")

    parsed = parse_classes_file(classes)

    assert parsed == ["cube", "sphere"]


def test_parse_classes_file_duplicate_fails(tmp_path: Path) -> None:
    classes = tmp_path / "classes.txt"
    classes.write_text("cube\nsphere\ncube\n", encoding="utf-8")

    with pytest.raises(ValidationFailure):
        parse_classes_file(classes)


def test_class_resolver_respects_class_order() -> None:
    resolver = ClassResolver(["sphere", "cube"])
    obj = SceneObject(index=3, name="Cube", tags=["cube", "sphere"])

    assert resolver.resolve_object_class(obj) == 0
