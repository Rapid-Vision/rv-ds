import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from rv_ds.pipeline import ExportConfig, run_export


def _write_sample(
    root: Path, sample_id: str, object_tags: dict[int, list[str]]
) -> None:
    sample_dir = root / sample_id
    sample_dir.mkdir(parents=True)

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(sample_dir / "Image.png"), image)

    index_map = np.zeros((16, 16), dtype=np.uint16)
    index_map[2:8, 2:8] = 2
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


def _write_custom_extractor(path: Path) -> None:
    path.write_text(
        """
from rv_ds.ir import DatasetIR, InstanceRecord, SampleRecord
from rv_ds.plugin_api import BaseExtractor
from rv_ds.plugin_api import ExtractorDatasetInfo
from rv_ds.sdk import (
    extract_bbox,
    load_scene_meta,
    normalize_bbox,
    object_mask,
    read_index_map,
)
from rv_ds.plugin_api import PluginOptions


class ExtractorOptions(PluginOptions):
    include_empty_samples: bool = True


class ExtractorPlugin(BaseExtractor):
    OptionsModel = ExtractorOptions
    produced_features = frozenset({"instance_class", "instance_bbox"})

    def __init__(self, opts: ExtractorOptions):
        super().__init__(opts)
        self.opts = opts

    def describe_dataset(self, ctx):
        return ExtractorDatasetInfo(class_names=["sphere"], meta={})

    def extract_sample(self, ctx, sample):
        scene = load_scene_meta(sample.meta_path)
        idx = read_index_map(sample.index_path)
        instances = []
        for obj in scene.objects:
            if "sphere" not in obj.tags:
                continue
            mask = object_mask(idx, obj.index)
            bbox = extract_bbox(mask)
            norm = normalize_bbox(bbox, idx.shape[1], idx.shape[0]) if bbox else None
            instances.append(
                InstanceRecord(
                    sample_id=sample.sample_id,
                    object_index=obj.index,
                    class_name="sphere",
                    class_id=0,
                    object_tags=list(obj.tags),
                    bbox_xyxy=bbox,
                    bbox_norm_cxcywh=norm,
                    polygon_norm=None,
                    area_px=int(mask.sum()),
                    extra={},
                )
            )

        if not instances and not self.opts.include_empty_samples:
            return None

        return SampleRecord(
            sample_id=sample.sample_id,
            scene_tags=list(scene.tags),
            image_src_path=sample.image_path,
            image_out_name=f"{sample.sample_id}.png",
            width=idx.shape[1],
            height=idx.shape[0],
            instances=instances,
            extra={},
        )
""",
        encoding="utf-8",
    )


def _write_custom_exporter(path: Path) -> None:
    path.write_text(
        """
from pathlib import Path

from rv_ds.plugin_api import BaseExporter
from rv_ds.plugin_api import PluginOptions
from rv_ds.plugin_api import ExporterRunResult


class ExporterOptions(PluginOptions):
    write_summary: bool = True


class ExporterPlugin(BaseExporter):
    OptionsModel = ExporterOptions
    required_features = frozenset({"instance_class"})

    def __init__(self, opts: ExporterOptions):
        super().__init__(opts)
        self.opts = opts

    def export_dataset(self, ctx):
        ctx.mkdir(Path("images"))
        outputs = []
        sample_count = sum(1 for _ in ctx.iter_samples(order="sequential"))
        if self.opts.write_summary:
            summary = ctx.write_text(
                Path("summary.txt"),
                str(sample_count),
            )
            outputs.append(str(summary))
        return ExporterRunResult(
            stats={"samples": sample_count},
            outputs=outputs,
            meta={},
        )
""",
        encoding="utf-8",
    )


def test_builtin_extractor_and_exporter(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", {2: ["sphere"], 3: ["cube"]})

    config = ExportConfig(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "exports",
        image_file="Image.png",
        extractor_spec="default-seg-bbox",
        extractor_opts={
            "class_mapping": [{"class": "sphere", "required_tags": ["sphere"]}],
            "target_tags": ["sphere"],
        },
        exporter_spec="default-yolo-seg",
        exporter_opts={},
        fail_on_plugin_warning=False,
        dump_ir=False,
    )

    result = run_export(config)

    label = result.export_dir / "labels" / "s1.txt"
    assert label.exists()
    data_yaml = yaml.safe_load(
        (result.export_dir / "data.yaml").read_text(encoding="utf-8")
    )
    assert data_yaml["names"] == ["sphere"]


def test_custom_extractor_builtin_exporter(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", {2: ["sphere"]})

    extractor = tmp_path / "extractor.py"
    _write_custom_extractor(extractor)

    config = ExportConfig(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "exports",
        image_file="Image.png",
        extractor_spec=str(extractor),
        extractor_opts={},
        exporter_spec="default-yolo-seg",
        exporter_opts={},
        fail_on_plugin_warning=False,
        dump_ir=False,
    )

    result = run_export(config)
    assert (result.export_dir / "labels" / "s1.txt").exists()


def test_builtin_extractor_custom_exporter(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", {2: ["sphere"]})

    exporter = tmp_path / "exporter.py"
    _write_custom_exporter(exporter)

    config = ExportConfig(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "exports",
        image_file="Image.png",
        extractor_spec="default-bbox",
        extractor_opts={
            "class_mapping": [{"class": "sphere", "required_tags": ["sphere"]}]
        },
        exporter_spec=str(exporter),
        exporter_opts={},
        fail_on_plugin_warning=False,
        dump_ir=False,
    )

    result = run_export(config)
    assert (result.export_dir / "summary.txt").exists()


def test_custom_extractor_custom_exporter(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", {2: ["sphere"]})

    extractor = tmp_path / "extractor.py"
    exporter = tmp_path / "exporter.py"
    _write_custom_extractor(extractor)
    _write_custom_exporter(exporter)

    config = ExportConfig(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "exports",
        image_file="Image.png",
        extractor_spec=str(extractor),
        extractor_opts={},
        exporter_spec=str(exporter),
        exporter_opts={},
        fail_on_plugin_warning=False,
        dump_ir=True,
    )

    result = run_export(config)
    assert (result.export_dir / "summary.txt").exists()
    assert (result.export_dir / "ir_dump.json").exists()
