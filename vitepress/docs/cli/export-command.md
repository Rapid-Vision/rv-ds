# `rv-ds export`

## Syntax

```bash
rv-ds export <dataset_dir> \
  --extractor <spec> \
  --exporter <spec> \
  [--extractor-opts <file.json>] \
  [--exporter-opts <file.json>] \
  [--output <dir>] \
  [--image-file <name>] \
  [--fail-on-plugin-warning] \
  [--dump-ir] \
  [--debug]
```

## Execution Lifecycle

1. Discover sample folders in `<dataset_dir>`.
2. Load and validate extractor and exporter plugins.
3. Validate feature contracts.
4. Run extractor `describe_dataset(ctx)`.
5. Run exporter `export_dataset(ctx)` with streaming `ctx.iter_samples(...)`.
6. Validate outputs and write `rv_ds_meta.json`.

## Exit Behavior

- Success: exit code `0`
- User-facing validation/runtime failures: exit code `2`
- With `--debug`, exceptions are re-raised with traceback
