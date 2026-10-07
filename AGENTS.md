# Agent instructions

## Commands

Use `uv` to manage the project. Add runtime dependencies with `uv add <package>` and development dependencies with `uv add --dev <package>`.

## Project structure

- Application code lives in `src/distributed_learning/`.
- Curated papers and figures live in `docs/`.
- Analysis notebooks live in `notebooks/`.
- Archived experiment and model outputs live under `artifacts/*/archive/`.
- New generated runs belong under `artifacts/*/runs/` and stay out of version control.

Run the CLI from the repository root with `uv run python scripts/train.py`.
