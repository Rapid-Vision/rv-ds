import json
from pathlib import Path

import cv2
import numpy as np
import yaml

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
    assert payload["pipeline"]["task"] == "segmentation"
    assert payload["pipeline"]["output_format"] == "yolo_seg"
    assert [item["name"] for item in payload["classes"]["mapping"]] == [
        "cube",
        "sphere",
    ]
    assert f"rv-ds export --config {config_path.resolve()}" in capsys.readouterr().out


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
                "pipeline": {
                    "task": "detection",
                    "output_format": "yolo_bbox",
                    "output_dir": "./exports",
                },
                "selection": {
                    "scene": {"require_tags": [], "exclude_tags": []},
                    "objects": {"target_tags": [], "include_unmapped": False},
                },
                "classes": {
                    "mapping": [{"name": "sphere", "match": {"all_tags": ["sphere"]}}]
                },
                "export": {"include_empty": False, "splits": {"train": 0.8, "val": 0.2}},
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
                "pipeline": {
                    "task": "segmentation",
                    "output_format": "preview",
                    "output_dir": "./exports",
                },
                "selection": {
                    "scene": {"require_tags": [], "exclude_tags": []},
                    "objects": {"target_tags": [], "include_unmapped": False},
                },
                "classes": {
                    "mapping": [{"name": "sphere", "match": {"all_tags": ["sphere"]}}]
                },
                "export": {"include_empty": True, "splits": {"train": 0.8, "val": 0.2}},
                "debug": {"dump_ir": False, "fail_on_warning": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(["export", "--config", str(config_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    export_dir_line = next(line for line in output.splitlines() if line.startswith("export_dir="))
    export_dir = Path(export_dir_line.split("=", 1)[1])
    assert (export_dir / "images" / "s1.png").exists()
    assert (export_dir / "overlays" / "s1.png").exists()


def test_validate_rejects_invalid_task_output_combo(tmp_path: Path, capsys) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_sample(dataset_dir, "s1", object_tags={2: ["sphere"]})

    config_path = tmp_path / "rv-ds.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"path": str(dataset_dir)},
                "pipeline": {
                    "task": "detection",
                    "output_format": "yolo_seg",
                    "output_dir": "./exports",
                },
                "selection": {
                    "scene": {"require_tags": [], "exclude_tags": []},
                    "objects": {"target_tags": [], "include_unmapped": False},
                },
                "classes": {
                    "mapping": [{"name": "sphere", "match": {"all_tags": ["sphere"]}}]
                },
                "export": {"include_empty": False, "splits": {"train": 0.8, "val": 0.2}},
                "debug": {"dump_ir": False, "fail_on_warning": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate", "--config", str(config_path)])

    assert exit_code == 2
    assert (
        "pipeline.output_format 'yolo_seg' requires task 'segmentation' or 'both'"
        in capsys.readouterr().err
    )
