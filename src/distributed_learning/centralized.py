from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR
from torch.utils.data import DataLoader, Dataset

from .experiments_config import OPT_SGD_LR, OPT_SGD_W_DECAY
from .model import Instantiator, LeNet5, Trainer, evaluate_model, load_data
from .optimizers import WarmupCosineAnnealing
from .plotting_metrics import plot_metrics

if TYPE_CHECKING:
    from collections.abc import Sized

logger = logging.getLogger(__name__)


def centralized_learning(  # noqa: PLR0913
    train_dataset: Dataset,
    n_epochs: int,
    batch_size: int,
    optimizer_I: Instantiator[Any],
    *,
    scheduler_I: Instantiator[Any] | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
    val_dataset: Dataset | None = None,
) -> tuple[LeNet5, pd.DataFrame]:
    """Train a LeNet model on one dataset using a centralized optimizer.

    Parameters
    ----------
    train_dataset : Dataset
        Dataset used for optimization.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Number of samples per training batch.
    optimizer_I : Instantiator
        Factory for the local optimizer.
    scheduler_I : Instantiator, optional
        Factory for the learning-rate scheduler.
    device : torch.device, optional
        Device used for training. Defaults to CUDA when available.
    verbose : bool, default=False
        Whether to log epoch progress.
    val_dataset : Dataset, optional
        Dataset evaluated after each epoch.

    Returns
    -------
    tuple of LeNet5 and pandas.DataFrame
        Trained model and per-epoch training metrics, with validation metrics
        when ``val_dataset`` is provided.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeNet5()
    model.to(device)

    total_steps = n_epochs * len(cast("Sized", train_dataset)) // batch_size
    if scheduler_I is not None:
        if issubclass(scheduler_I.var_class, CosineAnnealingLR):
            scheduler_I.kwargs["T_max"] = total_steps
        elif issubclass(scheduler_I.var_class, PolynomialLR):
            scheduler_I.kwargs["total_iters"] = total_steps
        elif issubclass(scheduler_I.var_class, WarmupCosineAnnealing):
            scheduler_I.kwargs["total_epochs"] = total_steps

    trainer = Trainer(model, optimizer_I, device, scheduler_I, verbose=False)

    d_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=6,
        prefetch_factor=20,
        persistent_workers=True,
    )

    train_metrics = []
    val_metrics = []
    logger.info("Starting training on device: %s", device)
    start_time = time.monotonic()
    for epoch in range(n_epochs):
        d_iter = iter(d_loader)

        train_metrics.append(trainer.train_model(d_iter, len(d_loader)))

        if verbose:
            logger.info(
                "Training progress: [%s/%s], %.2fs per epoch",
                epoch + 1,
                n_epochs,
                (time.monotonic() - start_time) / (epoch + 1),
            )
            logger.info("Current training loss/acc: %.3f/%.2f%%", train_metrics[-1][0], train_metrics[-1][1] * 100)

        if val_dataset is not None:
            val_metrics.append(evaluate_model(model, val_dataset, device, verbose=verbose))

    performance = pd.DataFrame(train_metrics, columns=["train_loss", "train_acc"])
    if val_dataset is not None:
        performance[["val_loss", "val_acc"]] = val_metrics

    return model, performance


if __name__ == "__main__":
    train_dataset, val_dataset = load_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_epochs = 150
    batch_size = 64
    opt_class = SGD

    optimizer_i = Instantiator(opt_class, {"lr": OPT_SGD_LR, "momentum": 0.9, "weight_decay": OPT_SGD_W_DECAY})
    scheduler_i = Instantiator(CosineAnnealingLR, {})

    model, performance = centralized_learning(
        train_dataset,
        n_epochs,
        batch_size,
        optimizer_i,
        scheduler_I=scheduler_i,
        device=device,
        verbose=True,
        val_dataset=val_dataset,
    )

    test_dataset = load_data(test_data=True)

    test_acc = evaluate_model(model, test_dataset, device)

    output_dir = Path("artifacts/models/runs/centralized")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir / f"{opt_class.__name__}.lenet"), {"test_acc": test_acc})

    performance.to_csv(output_dir / f"{opt_class.__name__}performance.csv")
    plot_metrics(performance, output_dir / f"{opt_class.__name__}performance.png")
