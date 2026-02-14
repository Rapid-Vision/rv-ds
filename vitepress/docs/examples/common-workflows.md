# Common Workflows

## Built-in extractor + built-in YOLO exporter

```bash
uv run rv-ds export ./examples/dataset \
  --extractor default-seg-bbox \
  --extractor-opts ./examples/opts/extractor.default-seg.json \
  --exporter default-yolo-seg \
  --exporter-opts ./examples/opts/exporter.default-yolo-seg.json
```

## Built-in extractor + preview bbox exporter

```bash
uv run rv-ds export ./examples/dataset \
  --extractor default-bbox \
  --extractor-opts ./examples/opts/extractor.default-seg.json \
  --exporter default-preview-bbox \
  --exporter-opts ./examples/opts/exporter.default-preview-bbox.json
```

## Deterministic random previews

Set matching `_framework.random_seed` in opts JSON.
