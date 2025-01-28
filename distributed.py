
import math
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, PolynomialLR
from typing import Generic, Iterator, Type, TypeVar
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from model import Instantiator, load_data
from model import LeNet5, Trainer, average_model_params, evaluate_model, set_model_params
from optimizers import DoNothing, SlowMo, WarmupCosineAnnealing, LARS, LAMB, OptimizerManager


def distributed_learning(train_dataset: Dataset, n_epochs: int, n_workers: int, n_local_steps: int, local_batch_size: int, 
                         local_optimizer_I: Instantiator[Optimizer],
                         optimizer_manager_I: Instantiator[OptimizerManager],
                         global_optimizer_I: Instantiator[Optimizer]=None,
                         scheduler_I: Instantiator[LRScheduler]=None, 
                         device=None, verbose=False, val_dataset=None):
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_steps = n_epochs * len(train_dataset) // (n_workers * local_batch_size)
    if issubclass(scheduler_I.var_class, CosineAnnealingLR):
        scheduler_I.kwargs["T_max"] = total_steps 
    elif issubclass(scheduler_I.var_class, PolynomialLR):
        scheduler_I.kwargs["total_iters"] = total_steps 
    elif issubclass(scheduler_I.var_class, WarmupCosineAnnealing):
        scheduler_I.kwargs["total_epochs"] = total_steps 

    trainers = get_workers(n_workers, local_optimizer_I, scheduler_I, device)
    
    optimizer_manager = optimizer_manager_I.instantiate([t.optimizer for t in trainers]) 

    global_model = LeNet5()
    global_model.to(device)

    if global_optimizer_I is None:
        global_optimizer_I = Instantiator(DoNothing, {})

    global_optimizer = global_optimizer_I.instantiate(global_model.parameters(), local_opt=trainers[0].optimizer)

    local_models = [tr.model for tr in trainers]
    
    assert len(train_dataset) > (n_workers * n_local_steps * local_batch_size), "Combination of n_workers, n_local_steps and local_batch_size is bigger than the dataset!"
    
    # number of iterations until the workers together have seen the whole dataset
    steps_per_epoch = len(train_dataset) // (n_workers * n_local_steps * local_batch_size) 

    if verbose:
        print("Starting training on device:", device)
        start_time = time.monotonic()


    # dataset doesn't have to be split, since both iid and none overlapping data are given 
    # works exactly as using DistributedSampler for distributed systems 
    data_loader = DataLoader(train_dataset, local_batch_size, shuffle=True, num_workers=6, prefetch_factor=12,
                            pin_memory=True, drop_last=True, persistent_workers=True)
    
    
    train_metrics = []
    val_metrics = []
    try:
        for epoch in range(n_epochs):
            data_loader_i = iter(data_loader)

            train_metrics_w = []
            for _ in range(steps_per_epoch):

                set_model_params(local_models, global_model)
                optimizer_manager.step()

                for trainer in trainers:
                    train_metrics_w.append(trainer.train_model(data_loader_i, n_local_steps))

                average_model_params(global_model, local_models)

                global_optimizer.step()

            train_metrics.append(np.mean(train_metrics_w, 0))

            if verbose:
                print(f"Training progress: [{(epoch+1)}/{n_epochs}], {(time.monotonic()-start_time)/((epoch+1)):.2f}s per epoch")
                print(f"Current training loss/acc: {train_metrics[-1][0]:.3f}/{train_metrics[-1][1]*100:.2f}%")

            if val_dataset is not None:
                val_metrics.append(evaluate_model(global_model, val_dataset, device, verbose=True))
                
    except ValueError as e:
        print(e)

    performance = pd.DataFrame(train_metrics, columns=["train_loss", "train_acc"])

    if val_dataset is not None:
        performance[["val_loss", "val_acc"]] = val_metrics
        
    return global_model, performance


def get_workers(n_workers: int, local_optimizer_I: Instantiator[Optimizer],
                scheduler_I: Instantiator[LRScheduler]=None, device=None) -> list[Trainer]:
    trainers = []

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(n_workers):
        model = LeNet5()
        model.to(device)

        trainers.append(Trainer(model, local_optimizer_I, device, scheduler_I, verbose=False))
        
    return trainers

def plot_metric(metric, ax: plt.Axes, **kwargs):
    ax.plot(range(len(metric)), metric, **kwargs)
    ax.set_xlabel("Epochs")
    ax.grid(True)
    
def plot_metrics(df, fname=None):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for col in df.columns:
        if "loss" in col:
            plot_metric(df[col], ax1, label=col)
            ax1.set_ylabel(col)
        else:
            plot_metric(df[col], ax2, label=col)
            ax2.set_ylabel(col)
    if fname is not None:
        fig.savefig(fname)
    return fig, (ax1, ax2)

if __name__ == "__main__":
    from Results2.experiments_config import OPT_SGD_LR, OPT_SGD_W_DECAY, OPT_ADAMW_LR, OPT_ADAMW_W_DECAY
    train_dataset, val_dataset = load_data()

    name = torch.optim.SGD
    model, performance = distributed_learning(train_dataset, n_epochs=150, n_workers=1, n_local_steps=100, local_batch_size=64, 
                                 global_optimizer_class=DoNothing, global_optimizer_params={},
                                 local_optimizer_class=name, local_optimizer_params={"lr": OPT_SGD_LR, "momentum": 0.9, "weight_decay": OPT_SGD_W_DECAY},
                                 scheduler_I.var_class=CosineAnnealingLR, scheduler_I.kwargs={"eta_min": 1e-7},
                                 verbose=True, val_dataset=val_dataset)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tran = transforms.Compose((transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))))
    test_dataset = datasets.CIFAR100(root="data", train=False, download=True, transform=tran)

    test_acc = evaluate_model(model, test_dataset, torch.nn.CrossEntropyLoss(), device)
    model.save(os.path.join("models", name.__name__ + ".lenet"), {"test_acc": test_acc})

    performance.to_csv(os.path.join("models", name.__name__ + "performance.csv"))
    plot_metrics(performance, os.path.join("models", name.__name__ + "performance.png"))