import json
from pathlib import Path

from rv_ds.plugin_api import BaseExporter, ExporterRunResult, PluginOptions


class ExporterOptions(PluginOptions):
    pass


class ExporterPlugin(BaseExporter[ExporterOptions]):
    OptionsModel = ExporterOptions

    def __init__(self, opts: ExporterOptions) -> None:
        super().__init__(opts)
        self.opts = opts

    def export_dataset(self, ctx):
        images_dir = ctx.mkdir(Path("images"))
        labels = []

        for sample in ctx.dataset.samples:
            ctx.copy_image(sample.image_src_path, images_dir / sample.image_out_name)

            for inst in sample.instances:
                if inst.class_id is None or inst.bbox_norm_cxcywh is None:
                    continue

                labels.append(
                    {
                        "sample_id": sample.sample_id,
                        "image": sample.image_out_name,
                        "class_id": inst.class_id,
                        "class_name": inst.class_name,
                        "bbox": list(inst.bbox_norm_cxcywh),
                    }
                )

        labels_path = ctx.write_text(
            Path("labels.json"),
            json.dumps(labels, indent=2),
        )

        return ExporterRunResult(
            outputs=[str(labels_path)],
            stats={"samples": len(ctx.dataset.samples), "labels": len(labels)},
            meta={"format": "json-bbox"},
        )
