# Getting Started

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- Dataset with sample folders containing `_meta.json`, `IndexOB.png`, and image file (default `Image.png`)

## Install

```bash
uv sync --dev
```

## First Export

```bash
uv run rv-ds export ./examples/dataset \
  --extractor default-seg-bbox \
  --extractor-opts ./examples/opts/extractor.default-seg.json \
  --exporter default-yolo-seg \
  --exporter-opts ./examples/opts/exporter.default-yolo-seg.json
```

On success:

- `export_dir=<path>` is printed
- summary counters are printed

## Output Anatomy

Typical output directory:

```text
<output>/<timestamp>/
  images/
  labels/                  # YOLO exporter
  overlays/                # preview exporters
  data.yaml                # YOLO exporter
  preview_meta.json        # preview exporters
  rv_ds_meta.json          # always
  ir_dump.json             # only with --dump-ir
```

Continue with:

- `/installation`
- `/quickstart`
- `/plugins/overview`
