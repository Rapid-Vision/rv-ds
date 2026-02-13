from pathlib import Path

import cv2
import numpy as np

from rv_export.mask_ops import object_mask, read_index_map


def test_read_index_map_u16(tmp_path: Path) -> None:
    path = tmp_path / "IndexOB.png"
    arr = np.array([[0, 1], [2, 655]], dtype=np.uint16)
    cv2.imwrite(str(path), arr)

    read = read_index_map(path)

    assert read.dtype == np.uint16
    assert np.array_equal(read, arr)


def test_read_index_map_from_8bit_bgra(tmp_path: Path) -> None:
    path = tmp_path / "IndexOB.png"
    lo = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    hi = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    arr = np.dstack([lo, hi, np.zeros_like(lo), np.zeros_like(lo)])
    cv2.imwrite(str(path), arr)

    read = read_index_map(path)

    assert read[0, 0] == 1
    assert read[0, 1] == 258
    assert read[1, 1] == 260


def test_object_mask() -> None:
    index_map = np.array([[0, 1], [1, 2]], dtype=np.uint16)

    mask = object_mask(index_map, 1)

    assert mask.dtype == np.uint8
    assert mask.tolist() == [[0, 1], [1, 0]]
