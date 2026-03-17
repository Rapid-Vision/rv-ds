# Getting Started

`rv-ds` reads an RV dataset, turns it into an internal sample stream with an extractor, and writes the target format with an exporter.

Use this flow:

```bash
uv sync --dev
uv run rv-ds inspect ./examples/dataset
uv run rv-ds init ./examples/dataset
uv run rv-ds validate --config ./rv-ds.yaml
uv run rv-ds export --config ./rv-ds.yaml
```

## Dataset Layout

Each sample must be a directory under the dataset root:

```text
<dataset_dir>/<sample_id>/_meta.json
<dataset_dir>/<sample_id>/IndexOB.png
<dataset_dir>/<sample_id>/Image.png
```

## Minimal Config

```yaml
dataset:
  path: ./examples/dataset

pipeline:
  output_dir: ./exports

extractor:
  spec: default-seg
  options:
    class_mapping:
      - class: sphere
        required_tags: [sphere]

exporter:
  spec: default-yolo-seg
  options:
    include_empty: false
    splits:
      train: 0.8
      val: 0.2
```

Relative paths are resolved from the config file location.

## Pick A Pair

- `default-bbox` + `default-yolo-bbox`: detection dataset
- `default-seg` + `default-yolo-seg`: segmentation dataset
- `default-bbox` + `default-preview-bbox`: bbox preview images
- `default-seg` or `default-seg-bbox` + `default-preview-seg`: segmentation preview images

## CLI

- `rv-ds inspect <dataset_dir>`: show dataset health and tags
- `rv-ds init [dataset_dir]`: generate a starter config
- `rv-ds validate --config <path>`: validate config, dataset, and plugin compatibility
- `rv-ds export --config <path>`: run export
- `rv-ds export --config <path> --dry-run`: preflight only, no files written
- `rv-ds list`: list built-in extractors, exporters, and presets

## Next

- [Built-in extractors](/extractors/builtins)
- [Writing a custom extractor](/extractors/custom)
- [Built-in exporters](/exporters/builtins)
- [Writing a custom exporter](/exporters/custom)
