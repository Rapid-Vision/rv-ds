from typing import Final

import cv2
import numpy as np

MIN_CONTOUR_POINTS: Final[int] = 3


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    return x_min, y_min, x_max, y_max


def normalize_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = bbox

    box_w = x_max - x_min + 1
    box_h = y_max - y_min + 1
    cx = x_min + box_w / 2
    cy = y_min + box_h / 2

    return cx / width, cy / height, box_w / width, box_h / height


def largest_polygon_from_mask(
    mask: np.ndarray, epsilon_ratio: float = 0.002
) -> list[tuple[float, float]] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
    points = approx.reshape(-1, 2)
    if len(points) < MIN_CONTOUR_POINTS:
        return None

    h, w = mask.shape[:2]
    polygon: list[tuple[float, float]] = []
    for x, y in points:
        polygon.append((float(x) / float(w), float(y) / float(h)))

    return polygon
