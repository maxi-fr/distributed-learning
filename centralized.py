import os

import pandas as pd
from model import Trainer, LeNet5, evaluate_model, load_data
import torch
from torch.optim import SGD
from Results.experiments_config import OPT_SGD_LR, OPT_SGD_W_DECAY
from torch.utils.data import DataLoader, random_split, Dataset
import time
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, PolynomialLR
from typing import Type
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from models.plotting_metrics import plot_metrics


def centralized_learning(train_dataset: Dataset, n_epochs: int, batch_size: int,
                         optimizer_class: Type[Optimizer], optimizer_params: dict,
                         scheduler_class: Type[LRScheduler] = None, scheduler_params: dict = None, 
                         device=None, verbose=False, val_dataset=None) -> tuple[LeNet5, pd.DataFrame]:

    trainer = Trainer(model, optimizer_class, optimizer_params,
                      device, scheduler_class, scheduler_params, verbose=False)

    d_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, pin_memory=True,
                          num_workers=8, prefetch_factor=8, persistent_workers=True)

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

    model = LeNet5()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    n_epochs = 150
    b_sizie = 64
    opt_class = SGD

    model, performance = centralized_learning(train_dataset, n_epochs, b_sizie, opt_class, 
                                              {"lr": OPT_SGD_LR, "momentum": 0.9, "weight_decay": OPT_SGD_W_DECAY},
                                              CosineAnnealingLR, {"T_max": n_epochs*len(train_dataset)//b_sizie}, 
                                              device, True, val_dataset)

    test_dataset = load_data(test_data=True)

    test_acc = evaluate_model(model, test_dataset, device)

    add_on = "_blabla_"
    model.save(os.path.join("models", opt_class.__name__ + add_on + ".lenet"), {"test_acc": test_acc})

    performance.to_csv(os.path.join("models", opt_class.__name__ + add_on + "performance.csv"))
    plot_metrics(performance, os.path.join("models", opt_class.__name__ + add_on + "performance.png"))
