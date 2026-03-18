# Built-In Extractors

Use an extractor to turn RV samples into class, bbox, and segment records for the exporter.

## Available Extractors

| Extractor | Produces | Use for |
| --- | --- | --- |
| `default-bbox` | `INSTANCE_CLASS`, `INSTANCE_BBOX` | detection |
| `default-seg` | `INSTANCE_CLASS`, `INSTANCE_SEGMENT`, `INSTANCE_BBOX` | segmentation and segmentation-derived bbox exports |

## Common Config

All built-in extractors use the same options shape.

```yaml
extractor:
  spec: default-seg
  options:
    class_mapping:
      - class: cube
        required_tags: [cube]
      - class: sphere
        required_tags: [sphere]
    include_empty: false
```

## Options You Will Actually Use

- `class_mapping`: map RV object tags to output class names
- `include_empty`: keep samples with no matched objects
- `target_tags`: only keep objects with these tags
- `require_tags`: require scene tags
- `exclude_tags`: skip scene tags
- `max_samples`: cap how many samples the extractor yields
- `polygon_tolerance`: polygon simplification tolerance in pixels
- `max_polygon_points`: cap the number of points in produced polygons
- `min_segment_area`: skip very small segments

## `class_mapping`

Each item must set a class name and one matching rule:

```yaml
class_mapping:
  - class: sphere
    required_tags: [sphere]
  - class: cube
    optional_tags: [cube, blue]
```

Rules:

- `required_tags`: all tags must be present
- `optional_tags`: any listed tag may match
- class names must be unique

## Which One Should I Use?

- Pick `default-bbox` for detection-only exports
- Pick `default-seg` for segmentation exports and any export that can use segmentation-derived boxes
