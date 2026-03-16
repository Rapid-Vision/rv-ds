from pathlib import Path

import yaml  # type: ignore[import-untyped]


def format_detect_line(class_id: int, bbox: tuple[float, float, float, float]) -> str:
    cx, cy, w, h = bbox
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def format_segment_line(class_id: int, polygon: list[tuple[float, float]]) -> str:
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
    return f"{class_id} {coords}"


def write_data_yaml(
    path: Path,
    class_names: list[str],
    splits: dict[str, str] | None = None,
) -> None:
    payload = {
        "path": ".",
        "names": class_names,
    }
    if splits is None:
        payload["train"] = "images"
    else:
        payload.update(splits)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
