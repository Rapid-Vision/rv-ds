# Error Reference

Framework raises `ValidationFailure` for user-facing validation problems.

## Options and CLI Loading

| Trigger | Typical message | Fix |
|---|---|---|
| Missing options file | `options file does not exist` | Pass existing JSON path or omit flag. |
| Invalid JSON | `failed to parse options JSON` | Fix JSON syntax. |
| Non-object options JSON | `must be an object` | Use top-level object. |
| Unknown/typed wrong plugin option | `invalid extractor/exporter options` | Match `OptionsModel` exactly. |

## Scanner and Dataset Shape

| Trigger | Typical message | Fix |
|---|---|---|
| Missing dataset dir | `dataset directory does not exist` | Fix path. |
| No sample folders | `has no sample folders` | Add sample directories. |
| Missing sample files | `missing required files` | Add `_meta.json`, `IndexOB.png`, and image file. |

## Plugin Loading and Contracts

| Trigger | Typical message | Fix |
|---|---|---|
| Missing contract declaration | `must declare non-empty produced_features/required_features` | Declare non-empty frozenset. |
| Invalid feature name | `invalid feature name` / `must use 'custom:' prefix` | Use valid naming rules. |
| Contract mismatch | `extractor/exporter feature contract mismatch` | Align required and produced features. |
| Legacy extractor API | `did not implement describe_dataset` | Implement streaming extractor methods. |

## Streaming and Export

| Trigger | Typical message | Fix |
|---|---|---|
| Multiple stream calls | `only one pass is supported` | Iterate once in exporter. |
| Invalid `max_samples` | `iter_samples max_samples must be positive` | Use `None` or positive integer. |
| Output path escape | `outside output directory` | Use context safe paths under export root. |
| Random seed mismatch | `_framework.random_seed mismatch` | Set same seed or only one side. |
