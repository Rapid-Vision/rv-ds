# rv-ds

`rv-ds` is a CLI for inspecting RV datasets, generating export configs, and exporting them through built-in or custom plugins.

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
- `rv-ds validate --config <path>`: validate config and plugin compatibility
- `rv-ds export --config <path>`: execute an export from YAML config
- `rv-ds list`: list built-in extractors, exporters, and presets

## Config Example

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
      - class: cube
        required_tags: [cube]

exporter:
  spec: default-yolo-seg
  options:
    include_empty: false
    splits:
      train: 0.8
      val: 0.2

debug:
  dump_ir: false
  fail_on_warning: false
```

`spec` can be a built-in plugin name such as `default-seg` or a relative/absolute path to a custom `.py` plugin file. Relative plugin paths are resolved relative to the config file.

## Built-in Plugins

- Extractors: `default-bbox`, `default-seg`, `default-seg-bbox`
- Exporters: `default-preview-bbox`, `default-preview-seg`, `default-yolo-bbox`, `default-yolo-seg`

## Compatibility

- Python `>=3.12`
