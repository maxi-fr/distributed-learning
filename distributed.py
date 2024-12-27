
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from typing import Iterator, Type
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split, Dataset, DistributedSampler
from torchvision import datasets, transforms
from model import LeNet5, Trainer, average_model_params, evaluate_model, set_model_params
from optimizers import SlowMo




def distributed_learning(train_dataset: Dataset, n_epochs: int, n_workers: int, n_local_steps: int, local_batch_size: int, 
                         global_optimizer_class: Type[Optimizer], global_optimizer_params: dict,
                         local_optimizer_class: Type[Optimizer], local_optimizer_params: dict,
                         scheduler_class: Type[LRScheduler]=None, scheduler_params: dict=None, device=None):
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainers = get_workers(n_workers, local_optimizer_class, local_optimizer_params, 
                           scheduler_class, scheduler_params, device)
    

    global_model = LeNet5()
    global_optimizer = global_optimizer_class(global_model.parameters(), **global_optimizer_params)

    local_models = [tr.model for tr in trainers]

    b_per_epoch = len(train_dataset) // (n_workers * n_local_steps * local_batch_size)

    assert b_per_epoch > 1, "Combination of n_workers, n_local_steps and local_batch_size is bigger than the dataset!"

    for epoch in range(n_epochs):
        split_data = shuffle_and_split(train_dataset, n_workers, local_batch_size)
        for _ in range(b_per_epoch):

            set_model_params(local_models, global_model)
            # TODO: manage local optimizers

            for i, trainer in enumerate(trainers):
                data_loader = split_data[i]
                losss = trainer.train_model(data_loader, n_local_steps)

            average_model_params(global_model, local_models)

            global_optimizer.step()

        
    return global_model


def get_workers(n_workers: int, local_optimizer_class: Type[Optimizer], local_optimizer_params: dict,
                scheduler_class: Type[LRScheduler]=None, scheduler_params: dict=None, device=None) -> list[Trainer]:
    trainers = []

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(n_workers):
        model = LeNet5()

        trainers.append(Trainer(model, local_optimizer_class, local_optimizer_params,
                                device, scheduler_class, scheduler_params, verbose=False))
        
    return trainers


def shuffle_and_split(train_data, N, batch_size, random_seed=None) -> list[Iterator[DataLoader]]:
    if random_seed is not None:
        random_seed = torch.Generator().manual_seed(random_seed)

    lengths = [len(train_data) // N] * N
    # TODO: training data can't be evenly split into subsets

    return [iter(DataLoader(subset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)) for subset in random_split(train_data, lengths, random_seed)]


if __name__ == "__main__":
    
    # search_space = {}
    
    # tune.Tuner(tune_distributed_learning, )

    tran = transforms.Compose((transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))))
    full_train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=tran)

    train_size = int(0.8 * len(full_train_dataset))  # 80% for training
    val_size = len(full_train_dataset) - train_size  # 20% for validation

    train_dataset, val_dataset = random_split(full_train_dataset, (train_size, val_size))

    model = distributed_learning(train_dataset, n_epochs=2, n_workers=2, n_local_steps=4, local_batch_size=2000, 
                                 global_optimizer_class=SlowMo, global_optimizer_params={},
                                 local_optimizer_class=torch.optim.Adam, local_optimizer_params={"lr": 0.01})