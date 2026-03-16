# rv-ds

`rv-ds` is a CLI for inspecting RV datasets, generating export configs, and exporting them into built-in target formats.

## Install

```bash
uv sync --dev
```

## Quickstart

Inspect the dataset first:

```bash
uv run rv-ds inspect ./examples/dataset
```

Generate a starter config with the terminal wizard:

```bash
uv run rv-ds init ./examples/dataset
```

Validate a config without exporting:

```bash
uv run rv-ds validate --config ./rv-ds.yaml
```

Run the export:

```bash
uv run rv-ds export --config ./rv-ds.yaml
```

## Commands

- `rv-ds inspect <dataset_dir>`: summarize sample health, tags, and suggested class mappings
- `rv-ds init [dataset_dir]`: generate a YAML config interactively
- `rv-ds validate --config <path>`: validate config and dataset compatibility
- `rv-ds export --config <path>`: execute an export from YAML config
- `rv-ds list`: list supported tasks, output formats, and presets

## Config Example

```yaml
dataset:
  path: ./examples/dataset

pipeline:
  task: segmentation
  output_format: yolo_seg
  output_dir: ./exports

selection:
  scene:
    require_tags: []
    exclude_tags: []
  objects:
    target_tags: []
    include_unmapped: false

classes:
  mapping:
    - name: sphere
      match:
        all_tags: [sphere]
    - name: cube
      match:
        all_tags: [cube]

export:
  include_empty: false
  splits:
    train: 0.8
    val: 0.2

debug:
  dump_ir: false
  fail_on_warning: false
```

## Built-in Output Formats

- `preview`: image copies plus overlay previews
- `yolo_bbox`: YOLO detection labels
- `yolo_seg`: YOLO segmentation labels

## Compatibility

- Python `>=3.12`
