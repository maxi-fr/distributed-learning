
import math
import time
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from typing import Iterator, Type
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split, Dataset, DistributedSampler
from torchvision import datasets, transforms
from model import load_data
from model import LeNet5, Trainer, average_model_params, evaluate_model, set_model_params
from optimizers import DoNothing, SlowMo, average_optimizers




def distributed_learning(train_dataset: Dataset, n_epochs: int, n_workers: int, n_local_steps: int, local_batch_size: int, 
                         global_optimizer_class: Type[Optimizer], global_optimizer_params: dict,
                         local_optimizer_class: Type[Optimizer], local_optimizer_params: dict,
                         scheduler_class: Type[LRScheduler]=None, scheduler_params: dict=None, device=None, verbose=False):
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainers = get_workers(n_workers, local_optimizer_class, local_optimizer_params, 
                           scheduler_class, scheduler_params, device)
    
    # stop_early = EarlyStopping()

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
        print("Starting training...")
        start_time = time.monotonic()

    try:
        for epoch in range(n_epochs):
            split_data = shuffle_and_split(train_dataset, n_workers, local_batch_size, device=device, pre_fetch=n_local_steps)

            for _ in range(steps_per_epoch):

                set_model_params(local_models, global_model)
                average_optimizers(local_optimizers)

                for i, trainer in enumerate(trainers):
                    data_loader = split_data[i]
                    train_loss, train_acc = trainer.train_model(data_loader, n_local_steps)

                average_model_params(global_model, local_models)

                global_optimizer.step()

            if verbose:
                print(f"Training progress: [{(epoch+1)}/{n_epochs}], {(time.monotonic()-start_time)/((epoch+1)):.2f}s per epoch")
                print(f"Current training loss/acc: {train_loss.max().item():.3f}/{train_acc.max().item()*100:.2f}%")

            # if stop_early(train_loss):
            #     break
    except ValueError as e:
        print(e)

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
    # TODO: training data can't be evenly split into subsets
    return [iter(DataLoader(subset, batch_size=batch_size, shuffle=False, 
                            pin_memory=True, drop_last=True, pin_memory_device=str(device),
                            prefetch_factor=math.ceil(pre_fetch/2))) for subset in random_split(train_data, lengths, random_seed)]


class EarlyStopping:

    def __init__(self, min_improvement = 0.001, patience = 5):
        self.min_improv = min_improvement
        self.patience = patience
        self.best = 0.0
        self.steps_without_improvement = 0

    def __call__(self, value: torch.Tensor):
        value = value.mean().item()
        if value > self.best + self.min_improv:
            self.best = value
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        return self.steps_without_improvement >= self.patience

if __name__ == "__main__":
    train_dataset, val_dataset = load_data()

    model = distributed_learning(train_dataset, n_epochs=150, n_workers=8, n_local_steps=1, local_batch_size=64, 
                                 global_optimizer_class=DoNothing, global_optimizer_params={},
                                 local_optimizer_class=torch.optim.AdamW, local_optimizer_params={"lr": 0.01},
                                 scheduler_class=CosineAnnealingLR, scheduler_params={"T_max": 150, "eta_min": 1e-5},
                                 verbose=True)