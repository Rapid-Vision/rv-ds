# Troubleshooting

## Plugin migration from legacy API

If your extractor still implements `extract_dataset(ctx)`, migrate to:

- `describe_dataset(ctx) -> ExtractorDatasetInfo`
- `extract_sample(ctx, sample) -> SampleRecord | None`

## Common issues

1. `extractor/exporter feature contract mismatch`
- Cause: required feature absent from extractor output declaration.
- Fix: update `produced_features` or `required_features`.

2. `invalid extractor options`
- Cause: strict option validation failed.
- Fix: remove unknown keys, correct types, verify required fields.

3. `exporter requested sample stream multiple times`
- Cause: called `ctx.iter_samples(...)` more than once.
- Fix: process in one pass and accumulate results in memory if needed.

4. `extractor produced warnings and fail-on-warning is set`
- Cause: extractor emitted `ctx.warn` and strict warning mode enabled.
- Fix: disable flag or eliminate warning condition.

## Dev commands

```bash
uv run pytest -q
uv run mypy src
uv run ruff check
```
