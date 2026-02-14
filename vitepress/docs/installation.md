# Installation

## Python Environment

```bash
uv sync --dev
```

This installs runtime and development dependencies from `pyproject.toml` and `uv.lock`.

## Verify CLI

```bash
uv run rv-ds --help
uv run rv-ds export --help
```

## Local Docs

From `/Users/mishapankin/Work/RapidVision/rv-export/vitepress`:

```bash
bun run docs:dev
```

Build static docs:

```bash
bun run docs:build
```
