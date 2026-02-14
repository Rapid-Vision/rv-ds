# CLI Overview

`rv-ds` currently exposes one subcommand:

- `export`: discover samples, run extractor, run exporter, write outputs and metadata

## Command Form

```bash
rv-ds export <dataset_dir> [flags]
```

Required flags:

- `--extractor <builtin-or-plugin.py>`
- `--exporter <builtin-or-plugin.py>`

Built-in extractors:

- `default-seg`
- `default-bbox`
- `default-seg-bbox`

Built-in exporters:

- `default-yolo-seg`
- `default-preview-bbox`
- `default-preview-seg`

See `/cli/export-command` and `/cli/options-reference` for full details.
