# Quickstart

## Minimal Dataset Requirements

Dataset directory must contain sample subdirectories:

```text
<dataset_dir>/<sample_id>/_meta.json
<dataset_dir>/<sample_id>/IndexOB.png
<dataset_dir>/<sample_id>/Image.png
```

`Image.png` can be replaced with another filename via `--image-file`.

## Run a Built-in Pipeline

```bash
uv run rv-ds export ./examples/dataset \
  --extractor default-bbox \
  --extractor-opts ./examples/opts/extractor.default-seg.json \
  --exporter default-preview-bbox \
  --exporter-opts ./examples/opts/exporter.default-preview-bbox.json
```

## Check Results

- Inspect overlays in `<export_dir>/overlays`
- Inspect run metadata in `<export_dir>/rv_ds_meta.json`
- For preview exporter details, inspect `<export_dir>/preview_meta.json`
