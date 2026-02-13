# Repository Guidelines

## Project Structure & Module Organization
- Source code lives in `src/rv_ds/`.
- CLI entrypoint is `src/rv_ds/cli.py` (`main()`), re-exported by `src/rv_ds/__init__.py`.
- Project metadata and dependencies are in `pyproject.toml`; lockfile is `uv.lock`.
- Top-level docs are in `README.md`.
- There is currently no `tests/` directory; add it at the repository root when introducing tests.

## Build, Test, and Development Commands
- `uv sync`: create/update the local environment from `pyproject.toml` and `uv.lock`.
- `uv run rv-ds`: run the packaged CLI script defined in `[project.scripts]`.
- `uv run python -m rv_ds.cli`: run the CLI module directly during development.
- `uv build`: build source/wheel distributions using `uv_build` backend.

## Coding Style & Naming Conventions
- Target Python `>=3.12`; keep compatibility with this baseline. Do not required by older versions like `from __future__ import annotations`
- Follow PEP 8: 4-space indentation, snake_case for functions/modules, CapWords for classes, UPPER_CASE for constants.
- Keep CLI-related code in `cli.py`; move reusable logic into additional modules under `src/rv_ds/` as complexity grows.
- Prefer small, single-purpose functions with explicit argument names.

## Testing Guidelines
- Use `pytest` for new tests.
- Place tests in `tests/` and name files `test_*.py` (for example, `tests/test_cli.py`).
- Run tests with `uv run pytest`.
- For CLI changes, include at least one behavior test covering argument handling or output.

## Commit & Pull Request Guidelines
- Current history uses short, imperative messages (`init`). Continue with concise imperative subjects, e.g., `add cli argument parsing`.
- Keep commits focused and atomic.
- PRs should include:
  - clear problem/solution summary,
  - testing notes (commands run and results),
  - linked issue (if applicable),
  - output examples for CLI behavior changes.

## Security & Configuration Tips
- Do not commit secrets, tokens, or machine-specific paths.
- Keep dependencies pinned via `uv.lock`; update lockfile alongside dependency changes.
