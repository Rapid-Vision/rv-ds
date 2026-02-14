# CLI Options Reference

| Option | Required | Default | Description |
|---|---|---|---|
| `dataset_dir` | yes | n/a | Dataset root containing sample folders. |
| `--extractor` | yes | n/a | Built-in extractor id or plugin `.py` path. |
| `--extractor-opts` | no | none | JSON file loaded as extractor options object. |
| `--exporter` | yes | n/a | Built-in exporter id or plugin `.py` path. |
| `--exporter-opts` | no | none | JSON file loaded as exporter options object. |
| `--output`, `-o` | no | `./exports` | Base output directory for timestamped run folders. |
| `--image-file` | no | `Image.png` | Expected image filename inside each sample directory. |
| `--fail-on-plugin-warning` | no | `false` | Fail run if extractor/exporter calls `ctx.warn(...)`. |
| `--dump-ir` | no | `false` | Write `ir_dump.json` from streamed samples. |
| `--debug` | no | `false` | Re-raise exceptions after printing user-facing error. |

## Options Files

`--extractor-opts` and `--exporter-opts` must point to JSON files containing a top-level object.

Validation failures include:

- file missing or unreadable
- invalid JSON
- top-level value not an object
- plugin option schema mismatch (`strict=True`, `extra="forbid"`)
