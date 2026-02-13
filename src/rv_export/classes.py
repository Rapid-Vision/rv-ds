from pathlib import Path

from .errors import ValidationFailure
from .models import SceneObject


class ClassResolver:
    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names
        self._class_to_id = {name: idx for idx, name in enumerate(class_names)}

    def resolve_object_class(self, obj: SceneObject) -> int | None:
        for class_name in self.class_names:
            if class_name in obj.tags:
                return self._class_to_id[class_name]
        return None


def parse_classes_file(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationFailure(f"failed to read classes file '{path}': {exc}") from exc

    class_names = [line.strip() for line in lines if line.strip()]
    if not class_names:
        raise ValidationFailure(
            f"classes file '{path}' does not contain any class names"
        )

    if any(not name for name in class_names):
        raise ValidationFailure(f"classes file '{path}' contains empty class names")

    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in class_names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)

    if duplicates:
        joined = ", ".join(sorted(duplicates))
        raise ValidationFailure(
            f"classes file '{path}' contains duplicate class names: {joined}"
        )

    return class_names
