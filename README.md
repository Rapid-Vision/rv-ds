# rv-ds

`rv-ds` is a two-stage export framework for RV intermediate datasets.

1. **Extractor**: builds a standardized intermediate representation (IR) from RV samples.
2. **Exporter**: writes IR to a target format (YOLO in v1 builtins).

## Install

```bash
uv sync --dev
```

## CLI

```bash
uv run rv-ds export <dataset_dir> \
  --extractor default-seg-bbox \
  --extractor-opts ./examples/opts/extractor.default-seg.json \
  --exporter default-yolo-seg \
  --exporter-opts ./examples/opts/exporter.default-yolo-seg.json
```

### Command

`rv-ds export <dataset_dir>`

Required flags:
- `--extractor <builtin-or-plugin.py>`
- `--exporter <builtin-or-plugin.py>`

Optional flags:
- `--extractor-opts <path.json>`
- `--exporter-opts <path.json>`
- `--output <dir>` default `./exports`
- `--image-file <name>` default `Image.png`
- `--fail-on-plugin-warning`
- `--dump-ir`
- `--debug`

Builtin extractors:
- `default-seg`
- `default-bbox`
- `default-seg-bbox`

Builtin exporters:
- `default-yolo-seg`
- `default-preview-bbox`
- `default-preview-seg`

### Built-in extractor options (`class_mapping`)

`default-seg`, `default-bbox`, and `default-seg-bbox` expect:

```json
{
  "class_mapping": [
    {
      "class": "sphere",
      "required_tags": ["sphere"]
    },
    {
      "class": "animal",
      "optional_tags": ["cat", "dog"]
    }
  ],
  "min_count": {},
  "require_tags": [],
  "exclude_tags": [],
  "include_empty": false,
  "epsilon_ratio": 0.002
}
```

Matching rules:
- `required_tags`: object must contain all tags.
- `optional_tags`: object must contain at least one tag.
- Class IDs are the order in `class_mapping`.

## Input dataset shape

```text
<dataset_dir>/<sample_uuid>/_meta.json
<dataset_dir>/<sample_uuid>/IndexOB.png
<dataset_dir>/<sample_uuid>/<image-file>
```

## Data Model Terminology

- `Dataset`: the whole export run payload.
- `Sample`: one scene/image item in a dataset.
- `Instance`: one object annotation in a sample.

## Output shape (default YOLO)

```text
<output>/<timestamp>/images/<sample_uuid>.png
<output>/<timestamp>/labels/<sample_uuid>.txt
<output>/<timestamp>/data.yaml
<output>/<timestamp>/rv_ds_meta.json
```

## Output shape (preview exporters)

`default-preview-bbox` and `default-preview-seg` write:

```text
<output>/<timestamp>/images/<sample_uuid>.png
<output>/<timestamp>/overlays/<sample_uuid>.png
<output>/<timestamp>/preview_meta.json
<output>/<timestamp>/rv_ds_meta.json
```

## Plugin API

Extractor plugin file must expose:

```python
from rv_ds.plugin_api import (
    INSTANCE_BBOX,
    INSTANCE_CLASS,
    BaseExtractor,
    PluginOptions,
)


class ExtractorOptions(PluginOptions):
    ...


class ExtractorPlugin(BaseExtractor):
    OptionsModel = ExtractorOptions
    produced_features = frozenset({INSTANCE_CLASS, INSTANCE_BBOX})

    def __init__(self, opts: ExtractorOptions):
        super().__init__(opts)
        self.opts = opts

    def extract_dataset(self, ctx):
        ...
```

Exporter plugin file must expose:

```python
from rv_ds.plugin_api import (
    INSTANCE_BBOX,
    INSTANCE_CLASS,
    BaseExporter,
    PluginOptions,
)


class ExporterOptions(PluginOptions):
    ...


class ExporterPlugin(BaseExporter):
    OptionsModel = ExporterOptions
    required_features = frozenset({INSTANCE_CLASS, INSTANCE_BBOX})

    def __init__(self, opts: ExporterOptions):
        super().__init__(opts)
        self.opts = opts

    def export_dataset(self, ctx):
        ...
```

## Feature Contracts

Extractor and exporter compatibility is checked before extraction starts:

`exporter.required_features` must be a subset of `extractor.produced_features`.

If something is missing, `rv-ds` fails fast with an error listing missing features.

### Standard Features

- `sample_class`
- `instance_class`
- `instance_segment`
- `instance_bbox`

### Custom Features

Custom features must use the `custom:` prefix (for example, `custom:instance_bbox_6d`).

Use existing IR extension fields for payload:
- dataset-level: `dataset.meta["custom:<feature>"]`
- sample-level: `sample.extra["custom:<feature>"]`
- instance-level: `instance.extra["custom:<feature>"]`

See examples:
- `/Users/mishapankin/Work/RapidVision/rv-ds/examples/plugins/custom-extractor.py`
- `/Users/mishapankin/Work/RapidVision/rv-ds/examples/plugins/custom-exporter.py`

## Stable SDK helpers for custom plugins

Import from `rv_ds.sdk`:
- `read_index_map(path)`
- `object_mask(index_map, object_index)`
- `extract_bbox(mask)`
- `normalize_bbox(bbox, width, height)`
- `extract_largest_polygon(mask, epsilon_ratio=0.002)`
- `load_scene_meta(path)`
- `parse_classes_file(path)`
- `resolve_class_from_tags(object_tags, class_names)`

## Example custom combinations

Custom extractor + built-in YOLO exporter:

```bash
uv run rv-ds export ./examples/dataset \
  --extractor ./examples/plugins/custom-extractor.py \
  --extractor-opts ./examples/opts/extractor.custom-simple.json \
  --exporter default-yolo-seg \
  --exporter-opts ./examples/opts/exporter.default-yolo-seg.json
```

Built-in extractor + custom exporter:

```bash
uv run rv-ds export ./examples/dataset \
  --extractor default-bbox \
  --extractor-opts ./examples/opts/extractor.default-seg.json \
  --exporter ./examples/plugins/custom-exporter.py \
  --exporter-opts ./examples/opts/exporter.custom-simple.json
```

Built-in extractor + preview bbox exporter:

```bash
uv run rv-ds export ./examples/dataset \
  --extractor default-bbox \
  --extractor-opts ./examples/opts/extractor.default-seg.json \
  --exporter default-preview-bbox \
  --exporter-opts ./examples/opts/exporter.default-preview-bbox.json
```

Built-in extractor + preview seg exporter:

```bash
uv run rv-ds export ./examples/dataset \
  --extractor default-seg \
  --extractor-opts ./examples/opts/extractor.default-seg.json \
  --exporter default-preview-seg \
  --exporter-opts ./examples/opts/exporter.default-preview-seg.json
```

Custom extractor + custom exporter:
```bash
uv run rv-ds export ./examples/dataset \
  --extractor ./examples/plugins/custom-extractor.py \
  --extractor-opts ./examples/opts/extractor.custom-simple.json \
  --exporter ./examples/plugins/custom-exporter.py \
  --exporter-opts ./examples/opts/exporter.custom-simple.json
```


## Development checks

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```
