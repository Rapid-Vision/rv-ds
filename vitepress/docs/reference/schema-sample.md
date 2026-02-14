# Sample Schema Reference

`SampleRecord` fields:

| Field | Type | Description |
|---|---|---|
| `sample_id` | `str` | Sample identifier, usually folder name. |
| `scene_tags` | `list[str]` | Scene-level tags from metadata. |
| `image_src_path` | `Path` | Source image location in dataset. |
| `image_out_name` | `str` | Output image filename. |
| `width` | `int` | Image width in pixels, must be > 0. |
| `height` | `int` | Image height in pixels, must be > 0. |
| `instances` | `list[InstanceRecord]` | Object annotation records. |
| `extra` | `dict[str, Any]` | Sample-level extension metadata. |

## Validation rules

- width and height must be positive
- `image_out_name` cannot be empty
- each instance `sample_id` must match sample `sample_id`
