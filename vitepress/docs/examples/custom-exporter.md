# Example: Custom Exporter

Reference implementation:

- `/Users/mishapankin/Work/RapidVision/rv-export/examples/plugins/custom-exporter.py`

## Run with built-in extractor

```bash
uv run rv-ds export ./examples/dataset \
  --extractor default-bbox \
  --extractor-opts ./examples/opts/extractor.default-seg.json \
  --exporter ./examples/plugins/custom-exporter.py \
  --exporter-opts ./examples/opts/exporter.custom-simple.json
```

## Exporter checklist

- declare `required_features`
- stream data with `ctx.iter_samples(...)`
- write only within output directory via context helpers
- return `ExporterRunResult` with stats/outputs/meta
