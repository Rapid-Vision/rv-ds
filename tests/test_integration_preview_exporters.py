import json
from pathlib import Path

import cv2
import numpy as np

from rv_ds.pipeline import ExportConfig, run_export
from rv_ds.plugin_loader import load_exporter


def _write_sample(
    root: Path,
    sample_id: str,
    object_tags: dict[int, list[str]],
    *,
    draw_mask_for: set[int] | None = None,
) -> None:
    sample_dir = root / sample_id
    sample_dir.mkdir(parents=True)

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(sample_dir / "Image.png"), image)

    index_map = np.zeros((16, 16), dtype=np.uint16)
    draw = draw_mask_for if draw_mask_for is not None else set(object_tags.keys())
    if 2 in draw:
        index_map[2:8, 2:8] = 2
    if 3 in draw:
        index_map[8:14, 8:14] = 3
    cv2.imwrite(str(sample_dir / "IndexOB.png"), index_map)

    objects = [
        {"index": idx, "name": f"o{idx}", "tags": tags, "custom_meta": {}}
        for idx, tags in object_tags.items()
    ]

    (sample_dir / "_meta.json").write_text(
        json.dumps({"tags": ["scene"], "objects": objects, "resolution": [16, 16]}),
        encoding="utf-8",
    )


def _default_detection_opts() -> dict:
    return {"class_mapping": [{"class": "sphere", "required_tags": ["sphere"]}]}


def _default_segment_opts() -> dict:
    return {"class_mapping": [{"class": "sphere", "required_tags": ["sphere"]}]}


def test_preview_bbox_exports_images_overlays_and_meta(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", {2: ["sphere"]})

    config = ExportConfig(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "exports",
        image_file="Image.png",
        extractor_spec="default-bbox",
        extractor_opts=_default_detection_opts(),
        exporter_spec="default-preview-bbox",
        exporter_opts={},
        fail_on_plugin_warning=False,
        dump_ir=False,
    )

    result = run_export(config)

    assert (result.export_dir / "images" / "s1.png").exists()
    assert (result.export_dir / "overlays" / "s1.png").exists()

    meta = json.loads((result.export_dir / "preview_meta.json").read_text(encoding="utf-8"))
    assert meta["stats"]["exported_samples"] == 1
    assert meta["stats"]["drawn_samples"] == 1
    assert meta["stats"]["skipped_samples"] == 0


def test_preview_seg_exports_polygon_overlay(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", {2: ["sphere"]})

    config = ExportConfig(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "exports",
        image_file="Image.png",
        extractor_spec="default-seg",
        extractor_opts=_default_segment_opts(),
        exporter_spec="default-preview-seg",
        exporter_opts={},
        fail_on_plugin_warning=False,
        dump_ir=False,
    )

    result = run_export(config)

    assert (result.export_dir / "images" / "s1.png").exists()
    assert (result.export_dir / "overlays" / "s1.png").exists()

    meta = json.loads((result.export_dir / "preview_meta.json").read_text(encoding="utf-8"))
    assert meta["stats"]["drawn_samples"] == 1


def test_preview_bbox_include_empty_false_skips_samples_without_boxes(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", {2: ["sphere"]}, draw_mask_for=set())

    config = ExportConfig(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "exports",
        image_file="Image.png",
        extractor_spec="default-bbox",
        extractor_opts={**_default_detection_opts(), "include_empty": True},
        exporter_spec="default-preview-bbox",
        exporter_opts={"include_empty": False},
        fail_on_plugin_warning=False,
        dump_ir=False,
    )

    result = run_export(config)

    assert not (result.export_dir / "images" / "s1.png").exists()
    assert not (result.export_dir / "overlays" / "s1.png").exists()

    meta = json.loads((result.export_dir / "preview_meta.json").read_text(encoding="utf-8"))
    assert meta["stats"]["exported_samples"] == 0
    assert meta["stats"]["skipped_samples"] == 1


def test_preview_bbox_include_empty_true_keeps_empty_samples(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", {2: ["sphere"]}, draw_mask_for=set())

    config = ExportConfig(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "exports",
        image_file="Image.png",
        extractor_spec="default-bbox",
        extractor_opts={**_default_detection_opts(), "include_empty": True},
        exporter_spec="default-preview-bbox",
        exporter_opts={"include_empty": True},
        fail_on_plugin_warning=False,
        dump_ir=False,
    )

    result = run_export(config)

    assert (result.export_dir / "images" / "s1.png").exists()
    assert (result.export_dir / "overlays" / "s1.png").exists()

    meta = json.loads((result.export_dir / "preview_meta.json").read_text(encoding="utf-8"))
    assert meta["stats"]["exported_samples"] == 1
    assert meta["stats"]["drawn_samples"] == 0
    assert meta["stats"]["skipped_samples"] == 0


def test_preview_exporters_are_registered() -> None:
    bbox_exporter, _ = load_exporter("default-preview-bbox", {})
    seg_exporter, _ = load_exporter("default-preview-seg", {})

    assert hasattr(bbox_exporter, "export_dataset")
    assert hasattr(seg_exporter, "export_dataset")
