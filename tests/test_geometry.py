import numpy as np

from rv_ds.geometry import bbox_from_mask, largest_polygon_from_mask, normalize_bbox


def test_bbox_normalization() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:7, 3:8] = 1

    bbox = bbox_from_mask(mask)
    assert bbox is not None

    cx, cy, w, h = normalize_bbox(bbox, width=10, height=10)

    assert (cx, cy, w, h) == (0.55, 0.45, 0.5, 0.5)


def test_polygon_extraction() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 6:16] = 1

    polygon = largest_polygon_from_mask(mask)

    assert polygon is not None
    assert len(polygon) >= 4
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in polygon)
