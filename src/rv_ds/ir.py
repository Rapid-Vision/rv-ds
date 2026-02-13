from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import ValidationFailure


@dataclass
class InstanceRecord:
    sample_id: str
    object_index: int
    class_name: str | None
    class_id: int | None
    object_tags: list[str]
    bbox_xyxy: tuple[int, int, int, int] | None
    bbox_norm_cxcywh: tuple[float, float, float, float] | None
    polygon_norm: list[tuple[float, float]] | None
    area_px: int
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.object_index < 1:
            raise ValidationFailure(
                f"instance '{self.sample_id}' has invalid object_index={self.object_index}"
            )

        if self.class_id is not None and self.class_id < 0:
            raise ValidationFailure(
                f"instance '{self.sample_id}' has invalid class_id={self.class_id}"
            )

        if (self.class_id is None) != (self.class_name is None):
            raise ValidationFailure(
                f"instance '{self.sample_id}' class_id/class_name must either both be set "
                "or both be null"
            )

        if self.area_px < 0:
            raise ValidationFailure(f"instance '{self.sample_id}' has negative area_px")

        if self.polygon_norm is not None and len(self.polygon_norm) < 3:
            raise ValidationFailure(
                f"instance '{self.sample_id}' polygon must contain at least 3 points"
            )


@dataclass
class SampleRecord:
    sample_id: str
    scene_tags: list[str]
    image_src_path: Path
    image_out_name: str
    width: int
    height: int
    instances: list[InstanceRecord]
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValidationFailure(
                f"sample '{self.sample_id}' has invalid dimensions {self.width}x{self.height}"
            )
        if not self.image_out_name:
            raise ValidationFailure(
                f"sample '{self.sample_id}' has empty image_out_name"
            )

        for instance in self.instances:
            if instance.sample_id != self.sample_id:
                raise ValidationFailure(
                    f"sample '{self.sample_id}' contains instance with mismatched "
                    f"sample_id='{instance.sample_id}'"
                )
            instance.validate()


@dataclass
class DatasetIR:
    samples: list[SampleRecord]
    class_names: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if len(set(self.class_names)) != len(self.class_names):
            raise ValidationFailure("dataset IR class_names must be unique")

        for sample in self.samples:
            sample.validate()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for sample in payload["samples"]:
            sample["image_src_path"] = str(sample["image_src_path"])
        return payload
