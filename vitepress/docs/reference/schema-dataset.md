# Dataset Schema Reference

`DatasetIR` fields:

| Field | Type | Description |
|---|---|---|
| `samples` | `list[SampleRecord]` | Exported sample records. |
| `class_names` | `list[str]` | Ordered class catalog used for class ids. Must be unique. |
| `meta` | `dict[str, Any]` | Dataset-level extension metadata. |

## Validation rules

- `class_names` must be unique
- each sample must pass `SampleRecord.validate()`

When `--dump-ir` is enabled, framework writes serialized `DatasetIR` to `ir_dump.json`.
