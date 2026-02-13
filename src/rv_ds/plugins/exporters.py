from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ValidationFailure
from ..plugin_api import ExportContext, Exporter, ExporterRunResult
from ..yolo import format_detect_line, format_segment_line, write_data_yaml


@dataclass
class DefaultYoloExporterOptions:
    include_empty: bool


class DefaultYoloExporter(Exporter):
    def __init__(self, opts: DefaultYoloExporterOptions) -> None:
        self.opts = opts

    def export_dataset(self, ctx: ExportContext) -> ExporterRunResult:
        images_dir = ctx.mkdir(Path("images"))
        labels_dir = ctx.mkdir(Path("labels"))

        exported_samples = 0
        empty_label_files = 0

        for sample in ctx.dataset.samples:
            ctx.copy_image(sample.image_src_path, images_dir / sample.image_out_name)

            lines: list[str] = []
            for inst in sample.instances:
                if inst.class_id is None:
                    continue

                if inst.bbox_norm_cxcywh is not None:
                    lines.append(
                        format_detect_line(inst.class_id, inst.bbox_norm_cxcywh)
                    )

                if inst.polygon_norm is not None:
                    lines.append(format_segment_line(inst.class_id, inst.polygon_norm))

            if lines or self.opts.include_empty:
                text = "\n".join(lines) + ("\n" if lines else "")
                ctx.write_text(labels_dir / f"{sample.sample_id}.txt", text)
                if not lines:
                    empty_label_files += 1

            exported_samples += 1

        data_yaml_path = ctx.safe_path(Path("data.yaml"))
        write_data_yaml(data_yaml_path, ctx.dataset.class_names)
        ctx._outputs.add(data_yaml_path)

        return ExporterRunResult(
            stats={
                "exported_samples": exported_samples,
                "empty_label_files": empty_label_files,
            },
            outputs=[str(path) for path in sorted(ctx._outputs)],
            meta={"exporter": "default-yolo"},
        )


def build_default_yolo_exporter(opts: dict[str, Any]) -> Exporter:
    include_empty = opts.get("include_empty", False)
    if not isinstance(include_empty, bool):
        raise ValidationFailure("exporter option 'include_empty' must be boolean")
    return DefaultYoloExporter(DefaultYoloExporterOptions(include_empty=include_empty))
