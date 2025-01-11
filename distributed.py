
import math
import os
import time
from matplotlib import pyplot as plt
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, PolynomialLR
from typing import Iterator, Type
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split, Dataset, DistributedSampler
from torchvision import datasets, transforms
from model import load_data
from model import LeNet5, Trainer, average_model_params, evaluate_model, set_model_params
from optimizers import DoNothing, SlowMo, average_optimizers, LARS, LAMB




def distributed_learning(train_dataset: Dataset, n_epochs: int, n_workers: int, n_local_steps: int, local_batch_size: int, 
                         global_optimizer_class: Type[Optimizer], global_optimizer_params: dict,
                         local_optimizer_class: Type[Optimizer], local_optimizer_params: dict,
                         scheduler_class: Type[LRScheduler]=None, scheduler_params: dict=None, device=None, verbose=False, val_dataset=None):
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if issubclass(scheduler_class, CosineAnnealingLR):
        scheduler_params["T_max"] = n_epochs * len(train_dataset) // (n_workers * local_batch_size) 
    elif issubclass(scheduler_class, PolynomialLR):
        scheduler_params["total_iters"] = n_epochs * len(train_dataset) // (n_workers * local_batch_size) 

    trainers = get_workers(n_workers, local_optimizer_class, local_optimizer_params, 
                           scheduler_class, scheduler_params, device)
    
    global_model = LeNet5()
    global_model.to(device)
    global_optimizer = global_optimizer_class(global_model.parameters(), 
                                              local_lr=local_optimizer_params["lr"], **global_optimizer_params)

    local_models = [tr.model for tr in trainers]
    local_optimizers = [tr.optimizer for tr in trainers]

    
    assert len(train_dataset) > (n_workers * n_local_steps * local_batch_size), "Combination of n_workers, n_local_steps and local_batch_size is bigger than the dataset!"
    
    # number of iterations until the workers together have seen the whole dataset
    steps_per_epoch = len(train_dataset) // (n_workers * n_local_steps * local_batch_size) 

    if verbose:
        print("Starting training on device:", device)
        start_time = time.monotonic()
    
    train_metrics = []
    val_metrics = []
    try:
        for epoch in range(n_epochs):
            split_data = shuffle_and_split(train_dataset, n_workers, local_batch_size, device=device, pre_fetch=n_local_steps)

            for _ in range(steps_per_epoch):

                set_model_params(local_models, global_model)
                average_optimizers(local_optimizers)

                train_metrics_w = []
                for i, trainer in enumerate(trainers):
                    data_loader = split_data[i]
                    train_metrics_w.append(trainer.train_model(data_loader, n_local_steps))
                print("Len dataloader:", len(data_loader))
                print("n_local_steps:", n_local_steps)
                print("steps_per_epoch * n_local_steps: ", steps_per_epoch * n_local_steps)

                average_model_params(global_model, local_models)

                global_optimizer.step()

                train_metrics.append(torch.mean(train_metrics_w, 0))

            if verbose:
                print(f"Training progress: [{(epoch+1)}/{n_epochs}], {(time.monotonic()-start_time)/((epoch+1)):.2f}s per epoch")
                print(f"Current training loss/acc: {train_metrics[-1][0]:.3f}/{train_metrics[-1][1]*100:.2f}%")

            if val_dataset is not None:
                val_metrics.append(evaluate_model(global_model, val_dataset, torch.nn.CrossEntropyLoss(), device, verbose=True))
                
    except ValueError as e:
        print(e)

    if val_dataset is not None:
        df = pd.DataFrame(train_metrics, columns=["train_loss", "train_acc"])
        df[["val_loss", "val_acc"]] = val_metrics
        return global_model, df
    
    return global_model


def get_workers(n_workers: int, local_optimizer_class: Type[Optimizer], local_optimizer_params: dict,
                scheduler_class: Type[LRScheduler]=None, scheduler_params: dict=None, device=None) -> list[Trainer]:
    trainers = []

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(n_workers):
        model = LeNet5()
        model.to(device)

        trainers.append(Trainer(model, local_optimizer_class, local_optimizer_params,
                                device, scheduler_class, scheduler_params, verbose=False))
        
    return trainers


def shuffle_and_split(train_data, N, batch_size, random_seed=None, device= "", pre_fetch=1) -> list[Iterator[DataLoader]]:
    if random_seed is not None:
        random_seed = torch.Generator().manual_seed(random_seed)

    lengths = [len(train_data) // N] * N

    ret = [iter(DataLoader(subset, batch_size=batch_size, shuffle=False, 
                               pin_memory=True, drop_last=True)) for subset in random_split(train_data, lengths, random_seed)]
    
    return ret

def plot_metric(metric, ax: plt.Axes, **kwargs):
    ax.plot(range(len(metric)), metric, **kwargs)
    ax.set_xlabel("Epochs")
    ax.grid(True)
    
def plot_metrics(df, fname):
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
    from experiments_config import OPT_SGD_LR, OPT_SGD_W_DECAY, OPT_ADAMW_LR, OPT_ADAMW_W_DECAY
    train_dataset, val_dataset = load_data()

    name = torch.optim.SGD
    model, performance, val_metrics = distributed_learning(train_dataset, n_epochs=150, n_workers=1, n_local_steps=100, local_batch_size=64, 
                                 global_optimizer_class=DoNothing, global_optimizer_params={},
                                 local_optimizer_class=name, local_optimizer_params={"lr": OPT_SGD_LR, "weight_decay": OPT_SGD_W_DECAY},
                                 scheduler_class=CosineAnnealingLR, scheduler_params={"eta_min": 1e-7},
                                 verbose=True, val_dataset=val_dataset)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tran = transforms.Compose((transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))))
    test_dataset = datasets.CIFAR100(root="data", train=False, download=True, transform=tran)

    test_acc = evaluate_model(model, test_dataset, torch.nn.CrossEntropyLoss(), device)
    model.save(os.path.join("models", name.__name__ + ".lenet"), {"test_acc": test_acc})

    performance.to_csv(os.path.join("models", name.__name__ + "performance.csv"))
    plot_metrics(performance, os.path.join("models", name.__name__ + "performance.png"))