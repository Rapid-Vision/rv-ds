from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import ValidationFailure


class SceneObject(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    index: int = Field(ge=1)
    name: str
    tags: list[str] = Field(default_factory=list)
    custom_meta: dict[str, Any] = Field(default_factory=dict)


class SceneMeta(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    tags: list[str] = Field(default_factory=list)
    objects: list[SceneObject]
    resolution: list[int] | tuple[int, int] | None = None


def load_scene_meta(path: Path) -> SceneMeta:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationFailure(f"failed to read meta file '{path}': {exc}") from exc

    try:
        return SceneMeta.model_validate_json(raw)
    except ValidationError as exc:
        raise ValidationFailure(f"invalid metadata schema at '{path}': {exc}") from exc
