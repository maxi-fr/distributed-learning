@echo off
cd /d "%~dp0"
uv run python -m distributed_learning.hyperparameter local_ada_scale
