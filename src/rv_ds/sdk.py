import numpy as np

from .classes import parse_classes_file
from .geometry import bbox_from_mask, largest_polygon_from_mask, normalize_bbox
from .mask_ops import object_mask, read_index_map
from .models import SceneMeta, load_scene_meta


def extract_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    return bbox_from_mask(mask)


def extract_largest_polygon(
    mask: np.ndarray,
    epsilon_ratio: float = 0.002,
) -> list[tuple[float, float]] | None:
    return largest_polygon_from_mask(mask, epsilon_ratio=epsilon_ratio)


def resolve_class_from_tags(
    object_tags: list[str], class_names: list[str]
) -> tuple[str | None, int | None]:
    for idx, class_name in enumerate(class_names):
        if class_name in object_tags:
            return class_name, idx
    return None, None


__all__ = [
    "SceneMeta",
    "extract_bbox",
    "extract_largest_polygon",
    "load_scene_meta",
    "normalize_bbox",
    "object_mask",
    "parse_classes_file",
    "read_index_map",
    "resolve_class_from_tags",
]
