import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from rv_ds.app_config import load_app_config
from rv_ds.cli import main


def _write_sample(
    root: Path,
    sample_id: str,
    *,
    scene_tags: list[str] | None = None,
    object_tags: dict[int, list[str]] | None = None,
    include_image: bool = True,
) -> None:
    sample_dir = root / sample_id
    sample_dir.mkdir(parents=True)

    if include_image:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        cv2.imwrite(str(sample_dir / "Image.png"), image)

    index_map = np.zeros((16, 16), dtype=np.uint16)
    index_map[2:8, 2:8] = 2
    index_map[8:14, 8:14] = 3
    cv2.imwrite(str(sample_dir / "IndexOB.png"), index_map)

    objects = [
        {"index": index, "name": f"o{index}", "tags": tags, "custom_meta": {}}
        for index, tags in (object_tags or {}).items()
    ]
    (sample_dir / "_meta.json").write_text(
        json.dumps(
            {
                "tags": scene_tags or [],
                "objects": objects,
                "resolution": [16, 16],
            }
        ),
        encoding="utf-8",
    )


def test_inspect_json_reports_tags_and_invalid_samples(tmp_path: Path, capsys) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(
        dataset_dir,
        "valid",
        scene_tags=["indoor"],
        object_tags={2: ["sphere"], 3: ["cube", "blue"]},
    )
    _write_sample(
        dataset_dir,
        "invalid",
        scene_tags=["outdoor"],
        object_tags={2: ["sphere"]},
        include_image=False,
    )

    exit_code = main(["inspect", str(dataset_dir), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_sample_dirs"] == 2
    assert payload["valid_sample_count"] == 1
    assert payload["invalid_sample_count"] == 1
    assert payload["issues"][0]["sample_id"] == "invalid"
    assert payload["object_tags"][0]["tag"] == "sphere"


def test_inspect_sample_json_reports_mask_statistics(
    tmp_path: Path, capsys
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(
        dataset_dir,
        "s1",
        object_tags={2: ["sphere"], 3: ["cube"], 4: ["missing"]},
    )

    exit_code = main(["inspect", str(dataset_dir), "--sample", "s1", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sample_id"] == "s1"
    assert payload["mask_count"] == 2
    assert payload["object_count_in_meta"] == 3
    assert payload["missing_mask_indexes"] == [4]
    assert payload["orphan_mask_indexes"] == []
    assert payload["area_px"] == {"min": 36, "max": 36, "mean": 36.0, "total": 72}
    assert payload["polygon_points"]["min"] >= 4
    assert payload["masks"][0]["object_index"] == 2
    assert payload["masks"][0]["object_name"] == "o2"


def test_inspect_sample_text_reports_orphan_masks(tmp_path: Path, capsys) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", object_tags={2: ["sphere"]})

    index_map = np.zeros((16, 16), dtype=np.uint16)
    index_map[1:5, 1:5] = 2
    index_map[8:12, 8:12] = 5
    cv2.imwrite(str(dataset_dir / "s1" / "IndexOB.png"), index_map)

    exit_code = main(["inspect", str(dataset_dir), "--sample", "s1"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Masks: total=2 meta_objects=1" in output
    assert "Masks missing metadata: 5" in output
    assert "index=5 name=unknown" in output


def test_inspect_sample_accepts_lexicographic_index(
    tmp_path: Path, capsys
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "b-sample", object_tags={2: ["sphere"]})
    _write_sample(dataset_dir, "a-sample", object_tags={2: ["cube"]})

    exit_code = main(["inspect", str(dataset_dir), "--sample", "1", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sample_id"] == "a-sample"


def test_inspect_sample_random_selects_sample(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "a-sample", object_tags={2: ["sphere"]})
    _write_sample(dataset_dir, "b-sample", object_tags={2: ["cube"]})

    monkeypatch.setattr("rv_ds.cli.random.choice", lambda items: items[-1])

    exit_code = main(["inspect", str(dataset_dir), "--sample", "random", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sample_id"] == "b-sample"


def test_inspect_sample_rejects_out_of_range_index(tmp_path: Path, capsys) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "a-sample", object_tags={2: ["sphere"]})

    exit_code = main(["inspect", str(dataset_dir), "--sample", "2"])

    assert exit_code == 2
    assert "sample index out of range: 2. Expected 1..1." in capsys.readouterr().err


def test_init_writes_yaml_config(monkeypatch, tmp_path: Path, capsys) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", object_tags={2: ["sphere"], 3: ["cube"]})
    config_path = tmp_path / "rv-ds.yaml"

    answers = iter(["", "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    exit_code = main(
        ["init", str(dataset_dir), "--output-config", str(config_path)]
    )

    assert exit_code == 0
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert payload["pipeline"]["output_dir"] == "exports"
    assert payload["extractor"]["spec"] == "default-seg"
    assert payload["exporter"]["spec"] == "default-yolo-seg"
    assert [
        item["class"] for item in payload["extractor"]["options"]["class_mapping"]
    ] == ["cube", "sphere"]
    assert payload["extractor"]["options"]["max_polygon_points"] == 100
    assert payload["exporter"]["options"]["splits"] == {"train": 0.8, "val": 0.2}
    assert f"rv-ds export {config_path.resolve()}" in capsys.readouterr().out


def test_init_preview_config_sets_extractor_max_samples(
    monkeypatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", object_tags={2: ["sphere"], 3: ["cube"]})
    config_path = tmp_path / "rv-ds.yaml"

    answers = iter(["", "1", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    exit_code = main(
        ["init", str(dataset_dir), "--output-config", str(config_path)]
    )

    assert exit_code == 0
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert payload["exporter"]["spec"] == "default-preview-seg"
    assert payload["extractor"]["options"]["max_samples"] == 100
    assert payload["extractor"]["options"]["max_polygon_points"] == 100


def test_init_fails_for_invalid_dataset(tmp_path: Path, capsys) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "broken", object_tags={2: ["sphere"]}, include_image=False)

    exit_code = main(["init", str(dataset_dir), "--output-config", str(tmp_path)])

    assert exit_code == 2
    assert "no fully valid samples" in capsys.readouterr().err


def test_init_accepts_output_config_directory(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", object_tags={2: ["sphere"]})
    output_dir = tmp_path / "configs"

    answers = iter(["", "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    exit_code = main(["init", str(dataset_dir), "--output-config", str(output_dir)])

    assert exit_code == 0
    assert (output_dir / "rv-ds.yaml").exists()


def test_validate_resolves_paths_relative_to_config(tmp_path: Path, capsys) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    dataset_dir = project_dir / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", object_tags={2: ["sphere"]})

    config_path = project_dir / "rv-ds.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"path": "./dataset"},
                "pipeline": {"output_dir": "./exports"},
                "extractor": {
                    "spec": "default-bbox",
                    "options": {
                        "class_mapping": [
                            {"class": "sphere", "required_tags": ["sphere"]}
                        ]
                    },
                },
                "exporter": {
                    "spec": "default-yolo-bbox",
                    "options": {
                        "include_empty": False,
                        "splits": {"train": 0.8, "val": 0.2},
                    },
                },
                "debug": {"dump_ir": False, "fail_on_warning": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate", "--config", str(config_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "extractor=default-bbox" in output
    assert "exporter=default-yolo-bbox" in output


def test_validate_resolves_relative_custom_plugin_paths(tmp_path: Path, capsys) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    dataset_dir = project_dir / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", object_tags={2: ["sphere"]})

    extractor_path = project_dir / "plugins" / "extractor.py"
    extractor_path.parent.mkdir()
    extractor_path.write_text(
        """
from rv_ds.plugin_api import BaseExtractor, ExtractorDatasetInfo, PluginOptions


class ExtractorOptions(PluginOptions):
    pass


class ExtractorPlugin(BaseExtractor):
    OptionsModel = ExtractorOptions
    produced_features = frozenset({"instance_class", "instance_bbox"})

    def describe_dataset(self, ctx):
        return ExtractorDatasetInfo(class_names=["sphere"], meta={})

    def extract_sample(self, ctx, sample):
        return None
""",
        encoding="utf-8",
    )
    exporter_path = project_dir / "plugins" / "exporter.py"
    exporter_path.write_text(
        """
from rv_ds.plugin_api import BaseExporter, ExporterRunResult, PluginOptions


class ExporterOptions(PluginOptions):
    pass


class ExporterPlugin(BaseExporter):
    OptionsModel = ExporterOptions
    required_features = frozenset({"instance_class"})

    def export_dataset(self, ctx):
        return ExporterRunResult(stats={}, outputs=[], meta={})
""",
        encoding="utf-8",
    )

    config_path = project_dir / "rv-ds.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"path": "./dataset"},
                "pipeline": {"output_dir": "./exports"},
                "extractor": {"spec": "./plugins/extractor.py", "options": {}},
                "exporter": {"spec": "./plugins/exporter.py", "options": {}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    resolved = load_app_config(config_path)

    assert resolved.extractor_spec == str(extractor_path.resolve())
    assert resolved.exporter_spec == str(exporter_path.resolve())

    exit_code = main(["validate", "--config", str(config_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"extractor={extractor_path.resolve()}" in output
    assert f"exporter={exporter_path.resolve()}" in output


def test_export_uses_yaml_config(tmp_path: Path, capsys) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    dataset_dir = project_dir / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", object_tags={2: ["sphere"]})

    config_path = project_dir / "rv-ds.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"path": "./dataset"},
                "pipeline": {"output_dir": "./exports"},
                "extractor": {
                    "spec": "default-seg",
                    "options": {
                        "class_mapping": [
                            {"class": "sphere", "required_tags": ["sphere"]}
                        ]
                    },
                },
                "exporter": {
                    "spec": "default-preview-seg",
                    "options": {"include_empty": True},
                },
                "debug": {"dump_ir": False, "fail_on_warning": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(["export", str(config_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "progress: start:" in output
    assert "progress: samples:" in output
    assert "progress: done:" in output
    export_dir_line = next(line for line in output.splitlines() if line.startswith("export_dir="))
    export_dir = Path(export_dir_line.split("=", 1)[1])
    assert not (export_dir / "images" / "s1.png").exists()
    assert (export_dir / "overlays" / "s1.png").exists()


def test_export_dry_run_prints_preflight_and_writes_nothing(
    tmp_path: Path, capsys
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    dataset_dir = project_dir / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", object_tags={2: ["sphere"]})

    config_path = project_dir / "rv-ds.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"path": "./dataset"},
                "pipeline": {"output_dir": "./exports"},
                "extractor": {
                    "spec": "default-seg",
                    "options": {
                        "class_mapping": [
                            {"class": "sphere", "required_tags": ["sphere"]}
                        ]
                    },
                },
                "exporter": {
                    "spec": "default-preview-seg",
                    "options": {"include_empty": True},
                },
                "debug": {"dump_ir": False, "fail_on_warning": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(["export", str(config_path), "--dry-run"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"config_ok: {config_path.resolve()}" in output
    assert f"dataset: {dataset_dir.resolve()}" in output
    assert f"output_dir: {(project_dir / 'exports').resolve()}" in output
    assert "extractor: default-seg" in output
    assert "exporter: default-preview-seg" in output
    assert "summary: valid_samples=1 class_names=1" in output
    assert "dry_run: no files written" in output
    assert not (project_dir / "exports").exists()


def test_validate_rejects_old_unified_schema(tmp_path: Path, capsys) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", object_tags={2: ["sphere"]})

    config_path = tmp_path / "rv-ds.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"path": str(dataset_dir)},
                "pipeline": {
                    "output_dir": "./exports",
                },
                "selection": {},
                "classes": {"mapping": []},
                "export": {"include_empty": False},
                "debug": {"dump_ir": False, "fail_on_warning": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate", "--config", str(config_path)])

    assert exit_code == 2
    error = capsys.readouterr().err
    assert "invalid config:" in error
    assert "pipeline.task" in error or "extractor" in error
