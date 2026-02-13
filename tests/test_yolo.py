from pathlib import Path

import yaml

from rv_ds.yolo import format_detect_line, format_segment_line, write_data_yaml


def test_format_detect_line() -> None:
    line = format_detect_line(1, (0.1, 0.2, 0.3, 0.4))
    assert line == "1 0.100000 0.200000 0.300000 0.400000"


def test_format_segment_line() -> None:
    line = format_segment_line(0, [(0.0, 0.0), (1.0, 0.5), (0.5, 1.0)])
    assert line == "0 0.000000 0.000000 1.000000 0.500000 0.500000 1.000000"


def test_write_data_yaml(tmp_path: Path) -> None:
    path = tmp_path / "data.yaml"

    write_data_yaml(path, ["cube", "sphere"])

    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert parsed == {"path": ".", "train": "images", "names": ["cube", "sphere"]}
