# Example: Custom Extractor

Reference implementation:

- `/Users/mishapankin/Work/RapidVision/rv-export/examples/plugins/custom-extractor.py`

## Run with built-in exporter

```bash
uv run rv-ds export ./examples/dataset \
  --extractor ./examples/plugins/custom-extractor.py \
  --extractor-opts ./examples/opts/extractor.custom-simple.json \
  --exporter default-yolo-seg \
  --exporter-opts ./examples/opts/exporter.default-yolo-seg.json
```

## Implementation checklist

- declare `produced_features`
- return `ExtractorDatasetInfo` in `describe_dataset`
- return `SampleRecord | None` in `extract_sample`
- use `ctx.warn(...)` for non-fatal warnings
