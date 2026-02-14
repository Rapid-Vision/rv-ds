# rv-ds

`rv-ds` is a two-stage export framework for RV intermediate datasets.

- **Extractor**: describes dataset metadata and yields sample records on demand.
- **Exporter**: pulls samples from extractor stream and writes target format outputs.

## Quick Install

```bash
uv sync --dev
```

## Minimal Quickstart

```bash
uv run rv-ds export ./examples/dataset \
  --extractor default-seg-bbox \
  --extractor-opts ./examples/opts/extractor.default-seg.json \
  --exporter default-yolo-seg \
  --exporter-opts ./examples/opts/exporter.default-yolo-seg.json
```

## Full Documentation (Canonical)

VitePress docs live in `/Users/mishapankin/Work/RapidVision/rv-export/vitepress`.

Run locally:

```bash
cd vitepress
bun run docs:dev
```

Build static docs:

```bash
cd vitepress
bun run docs:build
```

## Compatibility

- Python `>=3.12`
- Current docs describe the latest codebase (single latest version, not versioned docs)
