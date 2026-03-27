# Built-In Exporters

Use an exporter to write the extracted sample stream to disk.

## Available Exporters

| Exporter | Requires | Writes |
| --- | --- | --- |
| `default-yolo-bbox` | `INSTANCE_CLASS`, `INSTANCE_BBOX` | YOLO detection dataset |
| `default-yolo-seg` | `INSTANCE_CLASS`, `INSTANCE_SEGMENT` | YOLO segmentation dataset |
| `default-preview-bbox` | `INSTANCE_CLASS`, `INSTANCE_BBOX` | preview overlay images |
| `default-preview-seg` | `INSTANCE_CLASS`, `INSTANCE_SEGMENT` | segmentation preview overlay images |

## YOLO Exporters

```yaml
exporter:
  spec: default-yolo-seg
  options:
    include_empty: false
    splits:
      train: 0.8
      val: 0.2
```

Options:

- `include_empty`: write empty label files when a sample has no annotations
- `splits`: split ratios, must include `train` and `val`, and must sum to `1.0`

Output:

- `<split>/images/*.png`
- `<split>/labels/*.txt`
- `data.yaml`

## Preview Exporters

```yaml
exporter:
  spec: default-preview-seg
  options:
    include_empty: true
    compare_original: true
    max_samples: 50
```

Common options:

- `include_empty`: export samples even when nothing is drawn
- `compare_original`: write original and overlay side by side
- `max_samples`: limit how many samples are exported
- `fill_bbox`: draw filled boxes instead of outline only

Segmentation preview also supports:

- `fill_segment`: draw filled polygons

Output:

- `overlays/*.png`
- `preview_meta.json`

## Which One Should I Use?

- Pick `default-yolo-bbox` for training a detection model
- Pick `default-yolo-seg` for training a segmentation model
- Pick a preview exporter when you want a visual sanity check before training
