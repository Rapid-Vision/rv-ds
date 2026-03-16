import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from rv_ds.errors import ValidationFailure
from rv_ds.pipeline import ExportConfig, run_export
from rv_ds.plugin_loader import load_exporter, load_extractor


def _write_sample(root: Path, sample_id: str = "s1") -> None:
    sample_dir = root / sample_id
    sample_dir.mkdir(parents=True)

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(sample_dir / "Image.png"), image)

    index_map = np.zeros((16, 16), dtype=np.uint16)
    index_map[2:8, 2:8] = 2
    cv2.imwrite(str(sample_dir / "IndexOB.png"), index_map)

    objects = [{"index": 2, "name": "o2", "tags": ["sphere"], "custom_meta": {}}]
    (sample_dir / "_meta.json").write_text(
        json.dumps({"tags": ["scene"], "objects": objects, "resolution": [16, 16]}),
        encoding="utf-8",
    )


def _bbox_opts() -> dict:
    return {"class_mapping": [{"class": "sphere", "required_tags": ["sphere"]}]}


def _seg_opts() -> dict:
    return {"class_mapping": [{"class": "sphere", "required_tags": ["sphere"]}]}


def test_builtin_feature_contract_compatible_pairs(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir)

    configs = [
        ExportConfig(
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "out1",
            image_file="Image.png",
            extractor_spec="default-seg-bbox",
            extractor_opts=_bbox_opts(),
            exporter_spec="default-yolo-seg",
            exporter_opts={},
            fail_on_plugin_warning=False,
            dump_ir=False,
        ),
        ExportConfig(
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "out1_bbox",
            image_file="Image.png",
            extractor_spec="default-bbox",
            extractor_opts=_bbox_opts(),
            exporter_spec="default-yolo-bbox",
            exporter_opts={},
            fail_on_plugin_warning=False,
            dump_ir=False,
        ),
        ExportConfig(
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "out2",
            image_file="Image.png",
            extractor_spec="default-bbox",
            extractor_opts=_bbox_opts(),
            exporter_spec="default-preview-bbox",
            exporter_opts={},
            fail_on_plugin_warning=False,
            dump_ir=False,
        ),
        ExportConfig(
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "out3",
            image_file="Image.png",
            extractor_spec="default-seg",
            extractor_opts=_seg_opts(),
            exporter_spec="default-preview-seg",
            exporter_opts={},
            fail_on_plugin_warning=False,
            dump_ir=False,
        ),
    ]

    for cfg in configs:
        result = run_export(cfg)
        assert result.export_dir.exists()


def test_builtin_feature_contract_mismatch_fails(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir)

    cfg = ExportConfig(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "out",
        image_file="Image.png",
        extractor_spec="default-bbox",
        extractor_opts=_bbox_opts(),
        exporter_spec="default-preview-seg",
        exporter_opts={},
        fail_on_plugin_warning=False,
        dump_ir=False,
    )

    with pytest.raises(ValidationFailure, match="instance_segment"):
        run_export(cfg)

    cfg = ExportConfig(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "out_bbox",
        image_file="Image.png",
        extractor_spec="default-seg",
        extractor_opts=_seg_opts(),
        exporter_spec="default-yolo-bbox",
        exporter_opts={},
        fail_on_plugin_warning=False,
        dump_ir=False,
    )

    with pytest.raises(ValidationFailure, match="instance_bbox"):
        run_export(cfg)


def test_missing_contract_declaration_fails_load(tmp_path: Path) -> None:
    extractor = tmp_path / "extractor.py"
    extractor.write_text(
        """
from rv_ds.plugin_api import BaseExtractor, ExtractorDatasetInfo, PluginOptions


class ExtractorOptions(PluginOptions):
    pass


class ExtractorPlugin(BaseExtractor):
    OptionsModel = ExtractorOptions

    def describe_dataset(self, ctx):
        return ExtractorDatasetInfo(class_names=[], meta={})

    def extract_sample(self, ctx, sample):
        return None
""",
        encoding="utf-8",
    )

    exporter = tmp_path / "exporter.py"
    exporter.write_text(
        """
from rv_ds.plugin_api import BaseExporter, ExporterRunResult, PluginOptions


class ExporterOptions(PluginOptions):
    pass


class ExporterPlugin(BaseExporter):
    OptionsModel = ExporterOptions

    def export_dataset(self, ctx):
        return ExporterRunResult(stats={}, outputs=[], meta={})
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationFailure, match="produced_features"):
        load_extractor(str(extractor), {})
    with pytest.raises(ValidationFailure, match="required_features"):
        load_exporter(str(exporter), {})


def test_custom_feature_contract_pass_and_fail(tmp_path: Path) -> None:
    extractor_ok = tmp_path / "extractor_ok.py"
    extractor_ok.write_text(
        """
from rv_ds.plugin_api import BaseExtractor, ExtractorDatasetInfo, PluginOptions


class ExtractorOptions(PluginOptions):
    pass


class ExtractorPlugin(BaseExtractor):
    OptionsModel = ExtractorOptions
    produced_features = frozenset({"custom:instance_bbox_6d"})

    def describe_dataset(self, ctx):
        return ExtractorDatasetInfo(class_names=[], meta={})

    def extract_sample(self, ctx, sample):
        return None
""",
        encoding="utf-8",
    )

    extractor_bad = tmp_path / "extractor_bad.py"
    extractor_bad.write_text(
        """
from rv_ds.plugin_api import BaseExtractor, ExtractorDatasetInfo, PluginOptions


class ExtractorOptions(PluginOptions):
    pass


class ExtractorPlugin(BaseExtractor):
    OptionsModel = ExtractorOptions
    produced_features = frozenset({"instance_bbox"})

    def describe_dataset(self, ctx):
        return ExtractorDatasetInfo(class_names=[], meta={})

    def extract_sample(self, ctx, sample):
        return None
""",
        encoding="utf-8",
    )

    exporter = tmp_path / "exporter.py"
    exporter.write_text(
        """
from rv_ds.plugin_api import BaseExporter, ExporterRunResult, PluginOptions


class ExporterOptions(PluginOptions):
    pass


class ExporterPlugin(BaseExporter):
    OptionsModel = ExporterOptions
    required_features = frozenset({"custom:instance_bbox_6d"})

    def export_dataset(self, ctx):
        return ExporterRunResult(stats={}, outputs=[], meta={})
""",
        encoding="utf-8",
    )

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir)

    ok_cfg = ExportConfig(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "ok_out",
        image_file="Image.png",
        extractor_spec=str(extractor_ok),
        extractor_opts={},
        exporter_spec=str(exporter),
        exporter_opts={},
        fail_on_plugin_warning=False,
        dump_ir=False,
    )
    run_export(ok_cfg)

    bad_cfg = ExportConfig(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "bad_out",
        image_file="Image.png",
        extractor_spec=str(extractor_bad),
        extractor_opts={},
        exporter_spec=str(exporter),
        exporter_opts={},
        fail_on_plugin_warning=False,
        dump_ir=False,
    )
    with pytest.raises(ValidationFailure, match="custom:instance_bbox_6d"):
        run_export(bad_cfg)


def test_invalid_feature_name_rejected(tmp_path: Path) -> None:
    extractor = tmp_path / "extractor.py"
    extractor.write_text(
        """
from rv_ds.plugin_api import BaseExtractor, ExtractorDatasetInfo, PluginOptions


class ExtractorOptions(PluginOptions):
    pass


class ExtractorPlugin(BaseExtractor):
    OptionsModel = ExtractorOptions
    produced_features = frozenset({"BAD FEATURE"})

    def describe_dataset(self, ctx):
        return ExtractorDatasetInfo(class_names=[], meta={})

    def extract_sample(self, ctx, sample):
        return None
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationFailure, match="invalid feature name"):
        load_extractor(str(extractor), {})


def test_legacy_extract_dataset_api_is_rejected(tmp_path: Path) -> None:
    extractor = tmp_path / "extractor.py"
    extractor.write_text(
        """
from rv_ds.ir import DatasetIR
from rv_ds.plugin_api import BaseExtractor, PluginOptions


class ExtractorOptions(PluginOptions):
    pass


class ExtractorPlugin(BaseExtractor):
    OptionsModel = ExtractorOptions
    produced_features = frozenset({"instance_bbox"})

    def extract_dataset(self, ctx):
        return DatasetIR(samples=[], class_names=[], meta={})
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationFailure, match="describe_dataset"):
        load_extractor(str(extractor), {})


def test_fail_on_warning_stops_before_exporter_runs(tmp_path: Path) -> None:
    extractor = tmp_path / "extractor.py"
    extractor.write_text(
        """
from rv_ds.plugin_api import BaseExtractor, ExtractorDatasetInfo, PluginOptions


class ExtractorOptions(PluginOptions):
    pass


class ExtractorPlugin(BaseExtractor):
    OptionsModel = ExtractorOptions
    produced_features = frozenset({"instance_class"})

    def describe_dataset(self, ctx):
        ctx.warn("describe warning")
        return ExtractorDatasetInfo(class_names=[], meta={})

    def extract_sample(self, ctx, sample):
        return None
""",
        encoding="utf-8",
    )

    exporter = tmp_path / "exporter.py"
    exporter.write_text(
        """
from rv_ds.plugin_api import BaseExporter, ExporterRunResult, PluginOptions


class ExporterOptions(PluginOptions):
    pass


class ExporterPlugin(BaseExporter):
    OptionsModel = ExporterOptions
    required_features = frozenset({"instance_class"})

    def export_dataset(self, ctx):
        raise RuntimeError("exporter-should-not-run")
""",
        encoding="utf-8",
    )

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir)

    cfg = ExportConfig(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "out",
        image_file="Image.png",
        extractor_spec=str(extractor),
        extractor_opts={},
        exporter_spec=str(exporter),
        exporter_opts={},
        fail_on_plugin_warning=True,
        dump_ir=False,
    )

    with pytest.raises(
        ValidationFailure,
        match="extractor produced warnings and fail-on-warning is set",
    ):
        run_export(cfg)
