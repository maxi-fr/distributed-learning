# Distributed Learning

Experiments in centralized and distributed deep learning for the DAAI 2024/25 project. The code compares training strategies and optimizer configurations on CIFAR-100.

## Project layout

- `src/distributed_learning/`: training, model, optimizer, and experiment code.
- `docs/references/`: papers and project materials.
- `docs/figures/`: curated result figures.
- `notebooks/`: exploratory analysis notebooks.

## Setup

Install [uv](https://docs.astral.sh/uv/) and sync the project dependencies:

```powershell
uv sync
```

The project targets Python 3.13 or later. Dataset downloads are stored under `data/`.

## Run

Run a training mode from the repository root. The CLI parser displays all
options with `--help`; boolean options use `--use-cuda` or `--no-use-cuda`:

```powershell
uv run python scripts/train.py --mode train --n_epochs 150 --n_workers 4 --n_local_steps 10 --local_batch_size 16 --local_optimizer_class SGD --local_lr 0.01 --local_weight_decay 0.0001 --local_momentum 0.9 --scheduler_class WarmupCosineAnnealing --per_warmup_epochs 0.55 --global_optimizer_class DoNothing --global_optimizer_lr 0.001 --global_optimizer_momentum 0.9
```

Run a hyperparameter experiment with a predefined configuration:

```powershell
uv run python scripts/train.py --mode experiment --dict_name mini_batch_sgd --num_samples 20
```

Run the `local_ada_scale` tuning configuration with the same argparse-based
interface:

```powershell
uv run python scripts/train.py --mode experiment --dict_name local_ada_scale --num_samples 10
```

## Development

Install the repository hooks after initializing or cloning the repository:

```powershell
uv run pre-commit install
```

The template configures Ruff for linting and formatting and `ty` for type checking. Pytest is available as a development dependency, but the current repository has no test suite, so it is not run automatically by the commit hooks.
