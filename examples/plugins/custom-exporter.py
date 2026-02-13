from pathlib import Path

from rv_ds.plugin_api import BaseExporter, ExporterRunResult, PluginOptions
from rv_ds.yolo import format_detect_line, format_segment_line, write_data_yaml


class ExporterOptions(PluginOptions):
    include_empty: bool = False


class ExporterPlugin(BaseExporter):
    OptionsModel = ExporterOptions

    def __init__(self, opts: ExporterOptions) -> None:
        super().__init__(opts)
        self.opts = opts

    def export_dataset(self, ctx):
        include_empty = self.opts.include_empty

        images_dir = ctx.mkdir(Path("images"))
        labels_dir = ctx.mkdir(Path("labels"))

        for sample in ctx.dataset.samples:
            ctx.copy_image(sample.image_src_path, images_dir / sample.image_out_name)
            lines = []
            for inst in sample.instances:
                if inst.class_id is None:
                    continue
                if inst.bbox_norm_cxcywh is not None:
                    lines.append(format_detect_line(inst.class_id, inst.bbox_norm_cxcywh))
                if inst.polygon_norm:
                    lines.append(format_segment_line(inst.class_id, inst.polygon_norm))

            if lines or include_empty:
                ctx.write_text(
                    labels_dir / f"{sample.sample_id}.txt",
                    "\n".join(lines) + ("\n" if lines else ""),
                )

        data_yaml = ctx.safe_path(Path("data.yaml"))
        write_data_yaml(data_yaml, ctx.dataset.class_names)
        return ExporterRunResult(
            outputs=[str(data_yaml)],
            stats={"samples": len(ctx.dataset.samples)},
        )
