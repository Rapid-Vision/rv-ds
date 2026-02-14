# Example: Preview Exporters

Preview exporters generate image copies and overlay visualizations.

## BBox preview

```bash
uv run rv-ds export ./examples/dataset \
  --extractor default-bbox \
  --extractor-opts ./examples/opts/extractor.default-seg.json \
  --exporter default-preview-bbox \
  --exporter-opts ./examples/opts/exporter.default-preview-bbox.json
```

## Segmentation preview

```bash
uv run rv-ds export ./examples/dataset \
  --extractor default-seg \
  --extractor-opts ./examples/opts/extractor.default-seg.json \
  --exporter default-preview-seg \
  --exporter-opts ./examples/opts/exporter.default-preview-seg.json
```

## Key options

- `include_empty`
- `max_samples`
- `fill_bbox` (default `false`)
- `fill_segment` (segment exporter, default `true`)
