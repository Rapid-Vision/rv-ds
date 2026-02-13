import pytest

from rv_ds.errors import ValidationFailure
from rv_ds.plugin_loader import load_extractor


def test_required_tags_matching() -> None:
    extractor, info = load_extractor(
        "default-seg",
        {
            "class_mapping": [
                {"class": "sphere", "required_tags": ["sphere", "round"]},
                {"class": "animal", "optional_tags": ["cat", "dog"]},
            ]
        },
    )

    assert hasattr(extractor, "extract_dataset")
    assert info.source == "builtin"


def test_optional_tags_matching() -> None:
    extractor, info = load_extractor(
        "default-seg",
        {
            "class_mapping": [
                {"class": "sphere", "required_tags": ["sphere", "round"]},
                {"class": "animal", "optional_tags": ["cat", "dog"]},
            ]
        },
    )

    assert hasattr(extractor, "extract_dataset")
    assert info.source == "builtin"


def test_rule_with_both_required_and_optional_tags_fails() -> None:
    with pytest.raises(ValidationFailure):
        load_extractor(
            "default-seg",
            {
                "class_mapping": [
                    {
                        "class": "sphere",
                        "required_tags": ["sphere"],
                        "optional_tags": ["round"],
                    }
                ]
            },
        )
