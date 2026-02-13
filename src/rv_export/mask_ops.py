from pathlib import Path

import cv2
import numpy as np

from .errors import ValidationFailure


def read_index_map(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValidationFailure(f"failed to read index map image '{path}'")

    if image.ndim == 2:
        if image.dtype == np.uint16:
            return image
        if image.dtype == np.uint8:
            return image.astype(np.uint16)
        raise ValidationFailure(
            f"unsupported index map dtype for '{path}': {image.dtype}"
        )

    if image.ndim == 3:
        if image.dtype == np.uint8 and image.shape[2] >= 2:
            lo = image[:, :, 0].astype(np.uint16)
            hi = image[:, :, 1].astype(np.uint16)
            return lo | (hi << 8)

        if image.dtype == np.uint16 and image.shape[2] >= 1:
            return image[:, :, 0]

    raise ValidationFailure(
        f"unsupported IndexOB format for '{path}': shape={image.shape}, dtype={image.dtype}"
    )


def object_mask(index_map: np.ndarray, object_index: int) -> np.ndarray:
    mask = (index_map == object_index).astype(np.uint8)
    return mask
