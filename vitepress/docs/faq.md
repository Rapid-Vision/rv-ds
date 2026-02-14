# FAQ

## Is documentation versioned?

Not yet. Current docs describe the latest codebase state.

## Should I use README or VitePress as source of truth?

VitePress is canonical. README is intentionally concise and links here.

## Can exporter call `iter_samples` twice for two passes?

No. Stream is single-pass by design. Buffer what you need during the first pass.

## How do I make random preview sampling deterministic?

Set `_framework.random_seed` in extractor and/or exporter options. If both are set, values must match.

## Why do unknown plugin options fail?

`PluginOptions` enforces strict schema with `extra="forbid"` and strict typing.

## How do I include custom data fields?

Use `meta`/`extra` dictionaries with `custom:<feature>` keys and declare matching custom feature contracts.
