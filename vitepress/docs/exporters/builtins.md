# Built-in Exporters

Built-ins are registered in `plugin_loader.BUILTIN_EXPORTERS`.

## IDs and Required Features

| ID | Required features |
|---|---|
| `default-yolo-seg` | `instance_class`, `instance_bbox` |
| `default-preview-bbox` | `instance_class`, `instance_bbox` |
| `default-preview-seg` | `instance_class`, `instance_segment` |

## `default-yolo-seg`

Options (`DefaultYoloExporterOptions`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `include_empty` | `bool` | `false` | Write empty label files for samples without lines. |

Outputs:

- `images/*.png`
- `labels/*.txt`
- `data.yaml`

## Preview Exporters

Shared options (`_PreviewBaseOptions`):

| Field | Type | Default | Applies to |
|---|---|---|---|
| `include_empty` | `bool` | `true` | bbox + seg |
| `max_samples` | `int | null` | `null` | bbox + seg |
| `fill_bbox` | `bool` | `false` | bbox + seg fallback boxes |

Segment-only option (`DefaultPreviewSegExporterOptions`):

| Field | Type | Default | Applies to |
|---|---|---|---|
| `fill_segment` | `bool` | `true` | seg exporter |

Behavior:

- `fill_bbox=true`: half-transparent bbox fill plus border.
- `fill_segment=true`: half-transparent polygon fill plus border.
- `max_samples`: cap exported sample count.
- order: random sampling via `iter_samples(order="random")`.

### `_framework.random_seed`

Set `_framework.random_seed` in extractor and/or exporter options to make random ordering deterministic.

- must be integer
- if both set, values must match
