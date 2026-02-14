# SDK Helpers

Use helpers from `rv_ds.sdk` for stable plugin logic.

## Functions

- `read_index_map(path)`
- `object_mask(index_map, object_index)`
- `extract_bbox(mask)`
- `normalize_bbox(bbox, width, height)`
- `extract_largest_polygon(mask, epsilon_ratio=0.002)`
- `load_scene_meta(path)`
- `parse_classes_file(path)`
- `resolve_class_from_tags(object_tags, class_names)`

## Why use SDK layer

- avoids coupling plugin code to internal module layout
- centralizes normalization and geometry behavior
- improves forward compatibility for custom plugins
