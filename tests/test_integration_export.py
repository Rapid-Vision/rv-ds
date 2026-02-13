import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from rv_export.pipeline import build_config, run_export


def _write_sample(
    root: Path,
    sample_id: str,
    scene_tags: list[str],
    object_tags: dict[int, list[str]],
    image_file: str = "Image.png",
) -> None:
    sample_dir = root / sample_id
    sample_dir.mkdir(parents=True)

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    cv2.imwrite(str(sample_dir / image_file), image)

    index_map = np.zeros((16, 16), dtype=np.uint16)
    index_map[2:8, 2:8] = 2
    index_map[8:14, 9:15] = 3
    cv2.imwrite(str(sample_dir / "IndexOB.png"), index_map)

    objects = []
    for idx, tags in object_tags.items():
        objects.append(
            {"index": idx, "name": f"obj-{idx}", "tags": tags, "custom_meta": {}}
        )

    meta = {"tags": scene_tags, "objects": objects, "resolution": [16, 16]}
    (sample_dir / "_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_export_detect_and_segment_with_filters(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    _write_sample(dataset_dir, "s1", ["sunny"], {2: ["cube", "red"], 3: ["sphere"]})
    _write_sample(dataset_dir, "s2", ["night"], {2: ["cube"], 3: ["sphere"]})

    classes_path = tmp_path / "classes.txt"
    classes_path.write_text("cube\nsphere\n", encoding="utf-8")

    config = build_config(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "exports",
        classes_path=classes_path,
        image_file="Image.png",
        task="both",
        target_tags_csv="cube,sphere",
        min_count={"cube": 1},
        require_tags_csv="sunny",
        exclude_tags_csv="night",
        include_empty=False,
        format_name="yolo",
    )

    result = run_export(config)

    assert result.stats.processed == 2
    assert result.stats.exported == 1
    assert result.stats.skipped == 1

    export_dir = result.export_dir
    labels_file = export_dir / "labels" / "s1.txt"
    labels = labels_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(labels) == 4

    data_yaml = yaml.safe_load((export_dir / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["names"] == ["cube", "sphere"]

    meta = json.loads((export_dir / "rv_export_meta.json").read_text(encoding="utf-8"))
    assert meta["stats"]["exported"] == 1


def test_include_empty_keeps_scene(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    _write_sample(dataset_dir, "s1", ["ok"], {2: ["cube"]})

    classes_path = tmp_path / "classes.txt"
    classes_path.write_text("sphere\n", encoding="utf-8")

    config = build_config(
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "exports",
        classes_path=classes_path,
        image_file="Image.png",
        task="detect",
        target_tags_csv=None,
        min_count={},
        require_tags_csv=None,
        exclude_tags_csv=None,
        include_empty=True,
        format_name="yolo",
    )

    result = run_export(config)

    assert result.stats.exported == 1
    label_text = (result.export_dir / "labels" / "s1.txt").read_text(encoding="utf-8")
    assert label_text == ""
