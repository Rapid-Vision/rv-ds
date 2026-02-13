import pytest

from rv_ds.errors import ValidationFailure
from rv_ds.plugins.extractors import _parse_class_mapping, _resolve_class


def test_required_tags_matching() -> None:
    mapping = _parse_class_mapping(
        [
            {"class": "sphere", "required_tags": ["sphere", "round"]},
            {"class": "animal", "optional_tags": ["cat", "dog"]},
        ]
    )

    class_id, class_name = _resolve_class(["sphere", "round", "metal"], mapping)

    assert class_id == 0
    assert class_name == "sphere"


def test_optional_tags_matching() -> None:
    mapping = _parse_class_mapping(
        [
            {"class": "sphere", "required_tags": ["sphere", "round"]},
            {"class": "animal", "optional_tags": ["cat", "dog"]},
        ]
    )

    class_id, class_name = _resolve_class(["dog", "pet"], mapping)

    assert class_id == 1
    assert class_name == "animal"


def test_rule_with_both_required_and_optional_tags_fails() -> None:
    with pytest.raises(ValidationFailure):
        _parse_class_mapping(
            [
                {
                    "class": "sphere",
                    "required_tags": ["sphere"],
                    "optional_tags": ["round"],
                }
            ]
        )
