# Streaming Model

`rv-ds` uses exporter-driven streaming.

## Lifecycle

1. Scanner discovers all candidate sample paths.
2. Extractor describes dataset once via `describe_dataset(ctx)`.
3. Exporter requests samples via `ctx.iter_samples(...)`.
4. Framework invokes `extract_sample(ctx, sample)` lazily for each candidate.

## Single-Pass Guarantee

`ExportContext.iter_samples(...)` can be used only once per export run. Re-requesting the stream raises:

- `ValidationFailure: exporter requested sample stream multiple times; only one pass is supported`

## Ordering

`iter_samples(order=...)` supports:

- `sequential` (scanner order)
- `random` (optionally deterministic via `_framework.random_seed`)

## Sampling

`max_samples` limits yielded records, not candidate scans. Exporters may still scan candidates when filtering empty records.
