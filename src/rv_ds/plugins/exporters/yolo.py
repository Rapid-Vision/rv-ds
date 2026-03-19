from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, model_validator

from ...contracts import INSTANCE_BBOX, INSTANCE_CLASS, INSTANCE_SEGMENT
from ...errors import ValidationFailure
from ...plugin_api import BaseExporter, ExportContext, ExporterRunResult, PluginOptions
from ...yolo import format_detect_line, format_segment_line, write_data_yaml

DEFAULT_SPLIT_RATIOS = {"train": 0.8, "val": 0.2}
ALLOWED_SPLITS = frozenset({"train", "val", "test"})


class DefaultYoloExporterOptions(PluginOptions):
    model_config = ConfigDict(extra="forbid", strict=True)

    include_empty: bool = False
    splits: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_SPLIT_RATIOS))

    @model_validator(mode="after")
    def _validate_splits(self) -> "DefaultYoloExporterOptions":
        if not self.splits:
            raise ValueError("splits must be non-empty")

        names = set(self.splits)
        unknown = names - ALLOWED_SPLITS
        if unknown:
            joined = ", ".join(sorted(unknown))
            raise ValueError(f"unsupported split name(s): {joined}")
        if "train" not in names or "val" not in names:
            raise ValueError("splits must include train and val")

        total = 0.0
        for name, ratio in self.splits.items():
            if ratio <= 0.0:
                raise ValueError(f"splits[{name}] must be > 0")
            total += ratio

        if abs(total - 1.0) > 1e-6:
            raise ValueError("split ratios must sum to 1.0")

        return self


class _BaseYoloExporter(BaseExporter[DefaultYoloExporterOptions]):
    OptionsModel = DefaultYoloExporterOptions
    exporter_name = "default-yolo"

    def __init__(self, opts: DefaultYoloExporterOptions) -> None:
        super().__init__(opts)
        self.opts = opts

    def _format_instance_line(self, class_id: int, inst: Any) -> str | None:
        raise NotImplementedError

    def export_dataset(self, ctx: ExportContext) -> ExporterRunResult:
        samples = list(ctx.iter_samples(order="random"))
        split_to_samples = _assign_splits(samples, self.opts.splits)

        exported_samples = 0
        empty_label_files = 0

        for split_name, split_samples in split_to_samples.items():
            split_dir = Path(split_name)
            images_dir = ctx.mkdir(split_dir / "images")
            labels_dir = ctx.mkdir(split_dir / "labels")

            for sample in split_samples:
                ctx.copy_image(sample.image_src_path, images_dir / sample.image_out_name)

                lines: list[str] = []
                for inst in sample.instances:
                    if inst.class_id is None:
                        continue

                    line = self._format_instance_line(inst.class_id, inst)
                    if line is not None:
                        lines.append(line)

                if lines or self.opts.include_empty:
                    text = "\n".join(lines) + ("\n" if lines else "")
                    ctx.write_text(labels_dir / f"{sample.sample_id}.txt", text)
                    if not lines:
                        empty_label_files += 1

                exported_samples += 1

        data_yaml_path = ctx.safe_path(Path("data.yaml"))
        write_data_yaml(
            data_yaml_path,
            ctx.dataset_info.class_names,
            splits={name: f"./{name}/" for name in split_to_samples},
        )
        ctx.add_output(data_yaml_path)

        return ExporterRunResult(
            stats={
                "exported_samples": exported_samples,
                "empty_label_files": empty_label_files,
            },
            outputs=[str(path) for path in ctx.outputs],
            meta={"exporter": self.exporter_name},
        )


class DefaultYoloSegExporter(_BaseYoloExporter):
    required_features = frozenset({INSTANCE_CLASS, INSTANCE_SEGMENT})
    exporter_name = "default-yolo-seg"

    def _format_instance_line(self, class_id: int, inst: Any) -> str | None:
        polygon = inst.polygon_norm
        if polygon is None:
            return None
        return format_segment_line(class_id, polygon)


class DefaultYoloBBoxExporter(_BaseYoloExporter):
    required_features = frozenset({INSTANCE_CLASS, INSTANCE_BBOX})
    exporter_name = "default-yolo-bbox"

    def _format_instance_line(self, class_id: int, inst: Any) -> str | None:
        bbox = inst.bbox_norm_cxcywh
        if bbox is None:
            return None
        return format_detect_line(class_id, bbox)


def _assign_splits(
    samples: list[Any],
    split_ratios: dict[str, float],
) -> dict[str, list[Any]]:
    ordered_splits = list(split_ratios.items())
    split_to_samples: dict[str, list[Any]] = {name: [] for name, _ in ordered_splits}
    total = len(samples)
    if total == 0:
        return split_to_samples

    raw_counts = {name: ratio * total for name, ratio in ordered_splits}
    counts = {name: int(value) for name, value in raw_counts.items()}
    assigned = sum(counts.values())
    leftovers = total - assigned

    remainders = sorted(
        (
            (raw_counts[name] - counts[name], idx, name)
            for idx, (name, _) in enumerate(ordered_splits)
        ),
        reverse=True,
    )
    for _, _, name in remainders[:leftovers]:
        counts[name] += 1

    cursor = 0
    for name, _ in ordered_splits:
        next_cursor = cursor + counts[name]
        split_to_samples[name] = samples[cursor:next_cursor]
        cursor = next_cursor

    if cursor != total:
        raise ValidationFailure("failed to assign all samples to splits")

    return split_to_samples
