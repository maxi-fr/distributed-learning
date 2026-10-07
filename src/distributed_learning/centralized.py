import os

import pandas as pd
from .model import Trainer, LeNet5, evaluate_model, load_data
import torch
from torch.optim import SGD
from .experiments_config import OPT_SGD_LR, OPT_SGD_W_DECAY
from torch.utils.data import DataLoader, random_split, Dataset
import time
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, PolynomialLR
from typing import Type
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from .plotting_metrics import plot_metrics
from .model import Instantiator
from .optimizers import WarmupCosineAnnealing


def centralized_learning(train_dataset: Dataset, n_epochs: int, batch_size: int,
                         optimizer_I: Instantiator[Optimizer],
                         scheduler_I: Instantiator[LRScheduler] = None, 
                         device=None, verbose=False, val_dataset=None) -> tuple[LeNet5, pd.DataFrame]:

    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if issubclass(scheduler_I.var_class, CosineAnnealingLR):
        scheduler_I.kwargs["T_max"] = n_epochs * len(train_dataset) // batch_size 
    elif issubclass(scheduler_I.var_class, PolynomialLR):
        scheduler_I.kwargs["total_iters"] = n_epochs * len(train_dataset) // batch_size 

    model = LeNet5()
    model.to(device)

    total_steps = n_epochs * len(train_dataset) // batch_size
    if issubclass(scheduler_I.var_class, CosineAnnealingLR):
        scheduler_I.kwargs["T_max"] = total_steps 
    elif issubclass(scheduler_I.var_class, PolynomialLR):
        scheduler_I.kwargs["total_iters"] = total_steps 
    elif issubclass(scheduler_I.var_class, WarmupCosineAnnealing):
        scheduler_I.kwargs["total_epochs"] = total_steps 


    trainer = Trainer(model, optimizer_I, device, scheduler_I, verbose=False)

    d_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, pin_memory=True,
                          num_workers=6, prefetch_factor=20, persistent_workers=True)

    train_metrics = []
    val_metrics = []
    print("Starting training on device:", device)
    start_time = time.monotonic()
    for epoch in range(n_epochs):
        d_iter = iter(d_loader)

        train_metrics.append(trainer.train_model(d_iter, len(d_loader)))

        if verbose:
            print(f"Training progress: [{(epoch+1)}/{n_epochs}], {(time.monotonic()-start_time)/((epoch+1)):.2f}s per epoch")
            print(f"Current training loss/acc: {train_metrics[-1][0]:.3f}/{train_metrics[-1][1]*100:.2f}%")
        
        if val_dataset:
            val_metrics.append(evaluate_model(model, val_dataset, device, verbose=verbose))

    performance = pd.DataFrame(train_metrics, columns=["train_loss", "train_acc"])
    if val_dataset:
        performance[["val_loss", "val_acc"]] = val_metrics

    return model, performance


if __name__ == "__main__":
    train_dataset, val_dataset = load_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_epochs = 150
    b_sizie = 64
    opt_class = SGD

    optimizer_I = Instantiator(opt_class, {"lr": OPT_SGD_LR, "momentum": 0.9, "weight_decay": OPT_SGD_W_DECAY})
    scheduler_I = Instantiator(CosineAnnealingLR)

    model, performance = centralized_learning(train_dataset, n_epochs, b_sizie, optimizer_I,
                                              scheduler_I, device, True, val_dataset)

    test_dataset = load_data(test_data=True)

    test_acc = evaluate_model(model, test_dataset, device)

    output_dir = os.path.join("artifacts", "models", "runs", "centralized")
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, opt_class.__name__ + ".lenet"), {"test_acc": test_acc})

    performance.to_csv(os.path.join(output_dir, opt_class.__name__ + "performance.csv"))
    plot_metrics(performance, os.path.join(output_dir, opt_class.__name__ + "performance.png"))
