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
  --extractor default-segment \
  --extractor-opts ./examples/opts/extractor.default-segment.json \
  --exporter default-yolo \
  --exporter-opts ./examples/opts/exporter.default-yolo.json
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
- `default-segment`
- `default-detection`
- `default-both`

Builtin exporters:
- `default-yolo`
- `default-preview-bbox`
- `default-preview-seg`

### Built-in extractor options (`class_mapping`)

`default-segment`, `default-detection`, and `default-both` expect:

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
from rv_ds.plugin_api import BaseExtractor, PluginOptions


class ExtractorOptions(PluginOptions):
    ...


class ExtractorPlugin(BaseExtractor):
    OptionsModel = ExtractorOptions

    def __init__(self, opts: ExtractorOptions):
        super().__init__(opts)
        self.opts = opts

    def extract_dataset(self, ctx):
        ...
```

Exporter plugin file must expose:

```python
from rv_ds.plugin_api import BaseExporter, PluginOptions


class ExporterOptions(PluginOptions):
    ...


class ExporterPlugin(BaseExporter):
    OptionsModel = ExporterOptions

    def __init__(self, opts: ExporterOptions):
        super().__init__(opts)
        self.opts = opts

    def export_dataset(self, ctx):
        ...
```

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
  --exporter default-yolo \
  --exporter-opts ./examples/opts/exporter.default-yolo.json
```

Built-in extractor + custom exporter:

```bash
uv run rv-ds export ./examples/dataset \
  --extractor default-detection \
  --extractor-opts ./examples/opts/extractor.default-segment.json \
  --exporter ./examples/plugins/custom-exporter.py \
  --exporter-opts ./examples/opts/exporter.custom-simple.json
```

Built-in extractor + preview bbox exporter:

```bash
uv run rv-ds export ./examples/dataset \
  --extractor default-detection \
  --extractor-opts ./examples/opts/extractor.default-segment.json \
  --exporter default-preview-bbox \
  --exporter-opts '{}'
```

Built-in extractor + preview seg exporter:

```bash
uv run rv-ds export ./examples/dataset \
  --extractor default-segment \
  --extractor-opts ./examples/opts/extractor.default-segment.json \
  --exporter default-preview-seg \
  --exporter-opts '{}'
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
