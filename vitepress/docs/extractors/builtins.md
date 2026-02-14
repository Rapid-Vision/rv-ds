# Built-in Extractors

Built-ins are registered in `plugin_loader.BUILTIN_EXTRACTORS`.

## IDs and Features

| ID | Produced features |
|---|---|
| `default-seg` | `instance_class`, `instance_segment` |
| `default-bbox` | `instance_class`, `instance_bbox` |
| `default-seg-bbox` | `instance_class`, `instance_segment`, `instance_bbox` |

All three use the same options schema (`DefaultExtractorOptions`).

## Options

| Field | Type | Default | Notes |
|---|---|---|---|
| `class_mapping` | `list[ClassMappingRule]` | required | Non-empty; class names must be unique. |
| `target_tags` | `list[str]` or CSV string | `[]` | Object-level tag filter. |
| `min_count` | `dict[str, int]` | `{}` | Per-tag minimum object counts; non-negative values. |
| `require_tags` | `list[str]` or CSV string | `[]` | Scene-level required tags. |
| `exclude_tags` | `list[str]` or CSV string | `[]` | Scene-level excluded tags. |
| `include_empty` | `bool` | `false` | Include samples without valid instances. |
| `epsilon_ratio` | `float` | `0.002` | Polygon simplification ratio (segment modes). |

### `class_mapping` item

| Field | Type | Required | Notes |
|---|---|---|---|
| `class` | `str` | yes | Non-empty class name. |
| `required_tags` | `list[str]` | conditional | Set exactly one of `required_tags` or `optional_tags`. |
| `optional_tags` | `list[str]` | conditional | Set exactly one of `required_tags` or `optional_tags`. |
