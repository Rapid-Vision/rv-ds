import json
from pathlib import Path

import pytest

from rv_ds.errors import ValidationFailure
from rv_ds.ir import DatasetIR, InstanceRecord, SampleRecord
from rv_ds.pipeline import load_json_opts


def test_load_json_opts(tmp_path: Path) -> None:
    path = tmp_path / "opts.json"
    path.write_text('{"a": 1}', encoding="utf-8")

    parsed = load_json_opts(path)

    assert parsed == {"a": 1}


def test_load_json_opts_invalid(tmp_path: Path) -> None:
    path = tmp_path / "opts.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValidationFailure):
        load_json_opts(path)


def test_ir_validation() -> None:
    inst = InstanceRecord(
        sample_id="s1",
        object_index=2,
        class_name="sphere",
        class_id=0,
        object_tags=["sphere"],
        bbox_xyxy=(0, 0, 2, 2),
        bbox_norm_cxcywh=(0.1, 0.1, 0.2, 0.2),
        polygon_norm=[(0.0, 0.0), (0.2, 0.0), (0.1, 0.3)],
        area_px=9,
        extra={},
    )

    sample = SampleRecord(
        sample_id="s1",
        scene_tags=[],
        image_src_path=Path("/tmp/img.png"),
        image_out_name="s1.png",
        width=10,
        height=10,
        instances=[inst],
        extra={},
    )

    dataset = DatasetIR(samples=[sample], class_names=["sphere"], meta={})
    dataset.validate()

    payload = dataset.to_dict()
    assert isinstance(payload, dict)
    assert json.loads(json.dumps(payload))["class_names"] == ["sphere"]
