from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR
from torch.utils.data import DataLoader, Dataset

from .model import Instantiator, LeNet5, Trainer, average_model_params, evaluate_model, set_model_params
from .optimizers import AverageOptimizers, DoNothing, WarmupCosineAnnealing

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sized
    from pathlib import Path


def distributed_learning(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917
    train_dataset: Dataset,
    n_epochs: int,
    n_workers: int,
    n_local_steps: int,
    local_batch_size: int,
    local_optimizer_I: Instantiator[Any],
    *,
    optimizer_manager_I: Instantiator[Any] | None = None,
    global_optimizer_I: Instantiator[Any] | None = None,
    scheduler_I: Instantiator[Any] | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
    val_dataset: Dataset | None = None,
) -> tuple[LeNet5, pd.DataFrame]:
    """Train local models and aggregate their parameters across workers.

    Parameters
    ----------
    train_dataset : Dataset
        Dataset shared by the workers.
    n_epochs : int
        Base epoch budget.
    n_workers : int
        Number of local models to train.
    n_local_steps : int
        Optimizer steps performed by each worker per synchronization.
    local_batch_size : int
        Number of samples in each worker batch.
    local_optimizer_I : Instantiator
        Factory for each worker's local optimizer.
    optimizer_manager_I : Instantiator, optional
        Factory for the optimizer-state synchronization manager.
    global_optimizer_I : Instantiator, optional
        Factory for the optimizer applied to the aggregated model.
    scheduler_I : Instantiator, optional
        Factory for each worker's learning-rate scheduler.
    device : torch.device, optional
        Device used for training. Defaults to CUDA when available.
    verbose : bool, default=False
        Whether to log training progress.
    val_dataset : Dataset, optional
        Dataset evaluated after each epoch.

    Returns
    -------
    tuple of LeNet5 and pandas.DataFrame
        Aggregated model and per-epoch training metrics, with validation
        metrics when ``val_dataset`` is provided.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_steps = n_epochs * len(cast("Sized", train_dataset)) // (n_workers * local_batch_size)
    if scheduler_I is not None:
        if issubclass(scheduler_I.var_class, CosineAnnealingLR):
            scheduler_I.kwargs["T_max"] = total_steps
        elif issubclass(scheduler_I.var_class, PolynomialLR):
            scheduler_I.kwargs["total_iters"] = total_steps
        elif issubclass(scheduler_I.var_class, WarmupCosineAnnealing):
            scheduler_I.kwargs["total_epochs"] = total_steps

    trainers = get_workers(n_workers, local_optimizer_I, scheduler_I, device)

    if optimizer_manager_I is None:
        optimizer_manager_I = Instantiator(AverageOptimizers, {})

    optimizer_manager = optimizer_manager_I.instantiate(
        [t.optimizer for t in trainers], step_invariant_epochs=n_epochs, n_local_steps=n_local_steps
    )

    global_model = LeNet5()
    global_model.to(device)

    if global_optimizer_I is None:
        global_optimizer_I = Instantiator(DoNothing, {})

    global_optimizer = global_optimizer_I.instantiate(global_model.parameters(), local_opt=trainers[0].optimizer)

    local_models = [tr.model for tr in trainers]

    if len(cast("Sized", train_dataset)) <= n_workers * n_local_steps * local_batch_size:
        msg = "The dataset must exceed workers x local steps x local batch size."
        raise ValueError(msg)

    # number of iterations until the workers together have seen the whole dataset
    steps_per_epoch = len(cast("Sized", train_dataset)) // (n_workers * n_local_steps * local_batch_size)

    if verbose:
        logger.info("Starting training on device: %s", device)
        start_time = time.monotonic()

    # dataset doesn't have to be split, since both iid and none overlapping data are given
    # works exactly as using DistributedSampler for distributed systems
    data_loader = DataLoader(
        train_dataset,
        local_batch_size,
        shuffle=True,
        num_workers=6,
        prefetch_factor=12,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    train_metrics = []
    val_metrics = []

    epoch = 0
    try:
        while optimizer_manager.epoch_budget - epoch > 0:
            data_loader_i = iter(data_loader)

            train_metrics_w = []
            for _ in range(steps_per_epoch):
                set_model_params(local_models, global_model)
                optimizer_manager.step()

                train_metrics_w.extend(trainer.train_model(data_loader_i, n_local_steps) for trainer in trainers)

                average_model_params(global_model, local_models)

                global_optimizer.step()

            train_metrics.append(np.mean(train_metrics_w, 0))

            if verbose:
                logger.info(
                    "Training progress: [%s/%s], %.2fs per epoch",
                    epoch + 1,
                    optimizer_manager.epoch_budget,
                    (time.monotonic() - start_time) / (epoch + 1),
                )
                logger.info("Current training loss/acc: %.3f/%.2f%%", train_metrics[-1][0], train_metrics[-1][1] * 100)

            if val_dataset is not None:
                val_metrics.append(evaluate_model(global_model, val_dataset, device, verbose=True))

            optimizer_manager.update_epoch()
            epoch += 1

    except ValueError as e:
        logger.warning("Training stopped: %s", e)

    performance = pd.DataFrame(train_metrics, columns=["train_loss", "train_acc"])

    if val_dataset is not None:
        performance[["val_loss", "val_acc"]] = val_metrics

    return global_model, performance


def get_workers(
    n_workers: int,
    local_optimizer_I: Instantiator[Any],
    scheduler_I: Instantiator[Any] | None = None,
    device: torch.device | None = None,
) -> list[Trainer]:
    """Create initialized trainers for the requested number of workers.

    Parameters
    ----------
    n_workers : int
        Number of trainers to create.
    local_optimizer_I : Instantiator
        Factory used to create each trainer's optimizer.
    scheduler_I : Instantiator, optional
        Factory used to create each trainer's scheduler.
    device : torch.device, optional
        Device assigned to each trainer. Defaults to CUDA when available.

    Returns
    -------
    list of Trainer
        Initialized trainers with independent models.
    """
    trainers = []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _i in range(n_workers):
        model = LeNet5()
        model.to(device)

        trainers.append(Trainer(model, local_optimizer_I, device, scheduler_I, verbose=False))

    return trainers


def plot_metric(metric: pd.Series, ax: plt.Axes, label: str) -> None:
    """Draw one metric series on an axis.

    Parameters
    ----------
    metric : pandas.Series
        Metric values indexed by training step or epoch.
    ax : matplotlib.axes.Axes
        Axis on which to draw the series.
    label : str
        Legend label for the series.
    """
    ax.plot(range(len(metric)), metric, label=label)
    ax.set_xlabel("Epochs")
    ax.grid(visible=True)


def plot_metrics(
    df: pd.DataFrame,
    fname: str | Path | None = None,
) -> tuple[Any, tuple[plt.Axes, plt.Axes]]:
    """Plot loss and accuracy metrics in separate panels.

    Parameters
    ----------
    df : pandas.DataFrame
        Metrics with column names identifying loss or accuracy values.
    fname : str or pathlib.Path, optional
        Destination path for saving the figure.

    Returns
    -------
    tuple
        Figure and the loss and accuracy axes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for col in df.columns:
        if "loss" in col:
            plot_metric(df[col], ax1, col)
            ax1.set_ylabel(col.split("_")[1].capitalize())
        else:
            plot_metric(df[col], ax2, col)
            ax2.set_ylabel(col.split("_")[1].capitalize())
    if fname is not None:
        fig.savefig(fname)
    return fig, (ax1, ax2)
