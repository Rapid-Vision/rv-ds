import re

from .errors import ValidationFailure

SAMPLE_CLASS = "sample_class"
INSTANCE_CLASS = "instance_class"
INSTANCE_SEGMENT = "instance_segment"
INSTANCE_BBOX = "instance_bbox"

STANDARD_FEATURES = frozenset(
    {
        SAMPLE_CLASS,
        INSTANCE_CLASS,
        INSTANCE_SEGMENT,
        INSTANCE_BBOX,
    }
)

_FEATURE_RE = re.compile(r"^[a-z0-9_:.]+$")


def validate_feature_name(name: str) -> None:
    if not isinstance(name, str):
        raise ValidationFailure(f"feature name must be string, got {type(name)!r}")

    feature = name.strip()
    if not feature:
        raise ValidationFailure("feature name cannot be empty")

    if not _FEATURE_RE.fullmatch(feature):
        raise ValidationFailure(
            f"invalid feature name '{name}': allowed chars are [a-z0-9_:.]"
        )

    if feature in STANDARD_FEATURES:
        return

    if not feature.startswith("custom:"):
        raise ValidationFailure(
            f"non-standard feature '{name}' must use 'custom:' prefix"
        )
