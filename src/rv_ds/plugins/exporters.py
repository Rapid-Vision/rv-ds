import json
from pathlib import Path

import cv2
import numpy as np

from ..plugin_api import BaseExporter, ExportContext, ExporterRunResult, PluginOptions
from ..yolo import format_detect_line, format_segment_line, write_data_yaml


class DefaultYoloExporterOptions(PluginOptions):
    include_empty: bool = False


class DefaultYoloExporter(BaseExporter[DefaultYoloExporterOptions]):
    OptionsModel = DefaultYoloExporterOptions

    def __init__(self, opts: DefaultYoloExporterOptions) -> None:
        super().__init__(opts)
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
        ctx.add_output(data_yaml_path)

        return ExporterRunResult(
            stats={
                "exported_samples": exported_samples,
                "empty_label_files": empty_label_files,
            },
            outputs=[str(path) for path in ctx.outputs],
            meta={"exporter": "default-yolo-seg"},
        )


class _PreviewBaseOptions(PluginOptions):
    include_empty: bool = True


class DefaultPreviewBBoxExporterOptions(_PreviewBaseOptions):
    pass


class DefaultPreviewSegExporterOptions(_PreviewBaseOptions):
    pass


_PALETTE: list[tuple[int, int, int]] = [
    (52, 152, 219),
    (231, 76, 60),
    (46, 204, 113),
    (241, 196, 15),
    (155, 89, 182),
    (26, 188, 156),
    (230, 126, 34),
    (149, 165, 166),
]


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

                color = _class_color(inst.class_id)
                label = _class_label(inst.class_name, inst.class_id)

                if self.mode == "bbox":
                    if _draw_bbox(overlay, inst.bbox_xyxy, color, label):
                        drawable = True
                else:
                    drawn_seg = _draw_polygon(
                        overlay,
                        inst.polygon_norm,
                        sample.width,
                        sample.height,
                        color,
                        label,
                    )
                    if drawn_seg:
                        drawable = True
                    elif self.fallback_to_bbox and _draw_bbox(
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
    mode = "seg"
    fallback_to_bbox = True

    def __init__(self, opts: DefaultPreviewSegExporterOptions) -> None:
        super().__init__(opts)
        self.opts = opts

    def export_dataset(self, ctx: ExportContext) -> ExporterRunResult:
        return self._build_preview(ctx, include_empty=self.opts.include_empty)


def _class_color(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def _class_label(class_name: str | None, class_id: int) -> str:
    if class_name is not None:
        return class_name
    return str(class_id)


def _draw_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    color: tuple[int, int, int],
    label: str,
) -> bool:
    if bbox is None:
        return False

    x0, y0, x1, y1 = bbox
    h, w = image.shape[:2]
    x0 = int(max(0, min(x0, w - 1)))
    y0 = int(max(0, min(y0, h - 1)))
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))

    if x1 <= x0 or y1 <= y0:
        return False

    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        image,
        label,
        (x0, max(0, y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    return True


def _draw_polygon(
    image: np.ndarray,
    polygon_norm: list[tuple[float, float]] | None,
    width: int,
    height: int,
    color: tuple[int, int, int],
    label: str,
) -> bool:
    if polygon_norm is None or len(polygon_norm) < 3:
        return False

    points: list[tuple[int, int]] = []
    for x_norm, y_norm in polygon_norm:
        x = int(round(x_norm * width))
        y = int(round(y_norm * height))
        x = int(max(0, min(x, width - 1)))
        y = int(max(0, min(y, height - 1)))
        points.append((x, y))

    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

    x0, y0 = points[0]
    cv2.putText(
        image,
        label,
        (x0, max(0, y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    return True
