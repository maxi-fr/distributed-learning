
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from typing import Type
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from model import LeNet5, Trainer, average_model_params, set_model_params
from optimizers import SlowMo


def distributed_learning(n_global_steps, n_workers, local_optimizer_class: Type[Optimizer], local_optimizer_params: dict,
                scheduler_class: Type[LRScheduler], scheduler_params: dict, full_train_dataset: Dataset = None):
    
    trainers = get_workers(n_workers, local_optimizer_class, local_optimizer_params, 
                           scheduler_class, scheduler_params, full_train_dataset)

    global_model = LeNet5()
    global_optimizer = SlowMo(global_model.parameters())

    local_models = [tr.model for tr in trainers]

    for t in range(n_global_steps):
        set_model_params(local_models, global_model)
        # TODO: manage local optimizers
        for trainer in trainers:
            losss = trainer.train_model()

        average_model_params(global_model, local_models)

        global_optimizer.step()

    return global_model


def get_workers(n_workers: int, local_optimizer_class: Type[Optimizer], local_optimizer_params: dict,
                scheduler_class: Type[LRScheduler], scheduler_params: dict, full_train_dataset: Dataset = None) -> list[Trainer]:
    trainers = []

    if train_dataset is None:
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())

    train_size = 0.8  # 80% for training
    val_size = 1 - train_size  # 20% for validation

    train_dataset, val_dataset = random_split(full_train_dataset, (train_size, val_size))
    split_train_datasets = split_N_ways(train_dataset, n_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, t_data in zip(range(n_workers), split_train_datasets):
        model = LeNet5()

        trainers.append(Trainer(t_data, val_dataset, model, local_optimizer_class, local_optimizer_params,
                                device, scheduler_class, scheduler_params, verbose=False))
        
    return trainers


def split_N_ways(train_data, N, random_seed=None):
    if random_seed is not None:
        random_seed = torch.Generator().manual_seed(random_seed)

    lengths = [len(train_data) / N] * N

    return random_split(train_data, lengths, random_seed)

