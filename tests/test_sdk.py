import numpy as np

from rv_ds.sdk import (
    extract_bbox,
    extract_largest_polygon,
    normalize_bbox,
    object_mask,
    resolve_class_from_tags,
)


def test_resolve_class_from_tags() -> None:
    class_name, class_id = resolve_class_from_tags(
        ["cube", "sphere"], ["sphere", "cube"]
    )
    assert class_name == "sphere"
    assert class_id == 0


def test_bbox_and_polygon_helpers() -> None:
    index_map = np.zeros((20, 20), dtype=np.uint16)
    index_map[5:10, 7:13] = 2

    mask = object_mask(index_map, 2)
    bbox = extract_bbox(mask)
    assert bbox == (7, 5, 12, 9)

    norm = normalize_bbox(bbox, width=20, height=20) if bbox is not None else None
    assert norm is not None

    polygon = extract_largest_polygon(mask)
    assert polygon is not None
    assert len(polygon) >= 3
