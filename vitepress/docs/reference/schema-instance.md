# Instance Schema Reference

`InstanceRecord` fields:

| Field | Type | Description |
|---|---|---|
| `sample_id` | `str` | Parent sample id. |
| `object_index` | `int` | Source object index, must be >= 1. |
| `class_name` | `str | null` | Class label. |
| `class_id` | `int | null` | Class index, must be >= 0 when set. |
| `object_tags` | `list[str]` | Original object tags. |
| `bbox_xyxy` | `(x0,y0,x1,y1) | null` | Pixel bounding box. |
| `bbox_norm_cxcywh` | `(cx,cy,w,h) | null` | Normalized bbox for YOLO-style outputs. |
| `polygon_norm` | `list[(x,y)] | null` | Normalized polygon points. |
| `area_px` | `int` | Mask area in pixels, non-negative. |
| `extra` | `dict[str, Any]` | Instance-level extension metadata. |

## Validation rules

- `class_id` and `class_name` must both be set or both be null
- `polygon_norm` must contain at least 3 points when present
- negative values for `class_id` or `area_px` are invalid
