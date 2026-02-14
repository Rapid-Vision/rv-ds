import cv2
import numpy as np

_PALETTE: list[tuple[int, int, int]] = [
    (52, 152, 219),
    (231, 76, 60),
    (46, 204, 113),
    (241, 196, 15),
    (155, 89, 182),
    (26, 188, 156),
    (230, 126, 34),
    (149, 165, 166),
]


def class_color(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def class_label(class_name: str | None, class_id: int) -> str:
    if class_name is not None:
        return class_name
    return str(class_id)


def draw_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    color: tuple[int, int, int],
    label: str,
    *,
    fill: bool = False,
    fill_alpha: float = 0.5,
) -> bool:
    if bbox is None:
        return False

    x0, y0, x1, y1 = bbox
    h, w = image.shape[:2]
    x0 = int(max(0, min(x0, w - 1)))
    y0 = int(max(0, min(y0, h - 1)))
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))

    if x1 <= x0 or y1 <= y0:
        return False

    if fill:
        overlay = image.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness=-1)
        alpha = float(max(0.0, min(fill_alpha, 1.0)))
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)

    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        image,
        label,
        (x0, max(0, y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    return True


def draw_polygon(
    image: np.ndarray,
    polygon_norm: list[tuple[float, float]] | None,
    width: int,
    height: int,
    color: tuple[int, int, int],
    label: str,
    *,
    fill: bool = False,
    fill_alpha: float = 0.5,
) -> bool:
    if polygon_norm is None or len(polygon_norm) < 3:
        return False

    points: list[tuple[int, int]] = []
    for x_norm, y_norm in polygon_norm:
        x = int(round(x_norm * width))
        y = int(round(y_norm * height))
        x = int(max(0, min(x, width - 1)))
        y = int(max(0, min(y, height - 1)))
        points.append((x, y))

    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    if fill:
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], color)
        alpha = float(max(0.0, min(fill_alpha, 1.0)))
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

    x0, y0 = points[0]
    cv2.putText(
        image,
        label,
        (x0, max(0, y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    return True
