import json
from pathlib import Path

import cv2

from ..contracts import INSTANCE_BBOX, INSTANCE_CLASS, INSTANCE_SEGMENT
from ..plugin_api import BaseExporter, ExportContext, ExporterRunResult, PluginOptions
from .utils import class_color, class_label, draw_bbox, draw_polygon


class _PreviewBaseOptions(PluginOptions):
    include_empty: bool = True


class DefaultPreviewBBoxExporterOptions(_PreviewBaseOptions):
    pass


class DefaultPreviewSegExporterOptions(_PreviewBaseOptions):
    pass


class _PreviewOverlayBase(BaseExporter[_PreviewBaseOptions]):
    mode: str = "bbox"
    fallback_to_bbox: bool = False

    def _build_preview(
        self, ctx: ExportContext, include_empty: bool
    ) -> ExporterRunResult:
        images_dir = ctx.mkdir(Path("images"))
        overlays_dir = ctx.mkdir(Path("overlays"))

        exported_samples = 0
        drawn_samples = 0
        skipped_samples = 0

        for sample in ctx.dataset.samples:
            source = sample.image_src_path
            overlay = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if overlay is None:
                ctx.warn(f"failed to read image for preview: '{source}'")
                skipped_samples += 1
                continue

            drawable = False
            for inst in sample.instances:
                if inst.class_id is None:
                    continue

                color = class_color(inst.class_id)
                label = class_label(inst.class_name, inst.class_id)

                if self.mode == "bbox":
                    if draw_bbox(overlay, inst.bbox_xyxy, color, label):
                        drawable = True
                else:
                    drawn_seg = draw_polygon(
                        overlay,
                        inst.polygon_norm,
                        sample.width,
                        sample.height,
                        color,
                        label,
                    )
                    if drawn_seg:
                        drawable = True
                    elif self.fallback_to_bbox and draw_bbox(
                        overlay, inst.bbox_xyxy, color, label
                    ):
                        drawable = True

            if not drawable and not include_empty:
                skipped_samples += 1
                continue

            ctx.copy_image(source, images_dir / sample.image_out_name)
            overlay_path = ctx.safe_path(overlays_dir / sample.image_out_name)
            cv2.imwrite(str(overlay_path), overlay)
            ctx.add_output(overlay_path)

            exported_samples += 1
            if drawable:
                drawn_samples += 1

        preview_meta = {
            "exporter": f"default-preview-{self.mode}",
            "include_empty": include_empty,
            "class_names": ctx.dataset.class_names,
            "stats": {
                "exported_samples": exported_samples,
                "drawn_samples": drawn_samples,
                "skipped_samples": skipped_samples,
            },
        }
        meta_path = ctx.write_text(
            Path("preview_meta.json"),
            json.dumps(preview_meta, indent=2),
        )

        return ExporterRunResult(
            stats={
                "exported_samples": exported_samples,
                "drawn_samples": drawn_samples,
                "skipped_samples": skipped_samples,
            },
            outputs=[str(path) for path in ctx.outputs],
            meta={
                "exporter": f"default-preview-{self.mode}",
                "include_empty": include_empty,
                "class_names": ctx.dataset.class_names,
                "preview_meta": str(meta_path),
            },
        )


class DefaultPreviewBBoxExporter(
    _PreviewOverlayBase, BaseExporter[DefaultPreviewBBoxExporterOptions]
):
    OptionsModel = DefaultPreviewBBoxExporterOptions
    required_features = frozenset({INSTANCE_CLASS, INSTANCE_BBOX})
    mode = "bbox"

    def __init__(self, opts: DefaultPreviewBBoxExporterOptions) -> None:
        super().__init__(opts)
        self.opts = opts

    def export_dataset(self, ctx: ExportContext) -> ExporterRunResult:
        return self._build_preview(ctx, include_empty=self.opts.include_empty)


class DefaultPreviewSegExporter(
    _PreviewOverlayBase, BaseExporter[DefaultPreviewSegExporterOptions]
):
    OptionsModel = DefaultPreviewSegExporterOptions
    required_features = frozenset({INSTANCE_CLASS, INSTANCE_SEGMENT})
    mode = "seg"
    fallback_to_bbox = True

    def __init__(self, opts: DefaultPreviewSegExporterOptions) -> None:
        super().__init__(opts)
        self.opts = opts

    def export_dataset(self, ctx: ExportContext) -> ExporterRunResult:
        return self._build_preview(ctx, include_empty=self.opts.include_empty)
