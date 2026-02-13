# rv-export

`rv-export` converts RV intermediate output folders into YOLO-formatted datasets without re-rendering.

## Install

```bash
uv sync --dev
```

## CLI

```bash
uv run rv-export export <dataset_dir> \
  --format yolo \
  --classes ./classes.txt
```

### Supported flags

- `--format yolo` (required, only v1 format)
- `--task detect|segment|both` (default: `both`)
- `--classes <path>` (required, one class label per line)
- `--output <dir>` (default: `./exports`)
- `--image-file <name>` (default: `Image.png`)
- `--target-tags <csv>`
- `--min-count <tag=N>` (repeatable)
- `--require-tags <csv>`
- `--exclude-tags <csv>`
- `--include-empty`

## Input dataset shape

```text
<dataset_dir>/<sample_uuid>/_meta.json
<dataset_dir>/<sample_uuid>/IndexOB.png
<dataset_dir>/<sample_uuid>/<image-file>
```

## Output shape

Each run creates a timestamped folder inside `--output`:

```text
<output>/<timestamp>/images/<sample_uuid>.png
<output>/<timestamp>/labels/<sample_uuid>.txt
<output>/<timestamp>/data.yaml
<output>/<timestamp>/rv_export_meta.json
```

`data.yaml` contains:

- `path: .`
- `train: images`
- `names: [...]`

## Behavior notes

- Class IDs are determined by the order in `--classes`.
- Object class resolution uses first matching tag by class-file order.
- Filtering order:
  1. Scene `require-tags` / `exclude-tags`
  2. Object selection (`target-tags` + class match)
  3. `min-count` on selected objects
  4. Empty-scene handling via `--include-empty`
- `--task both` emits both detection and segmentation lines.

## Development checks

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```
