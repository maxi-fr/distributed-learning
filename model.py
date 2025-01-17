from itertools import cycle
import os
import torch.nn as nn
import time
from typing import Iterator, Type
import torch
from torch import nn
import copy
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import copy
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms.v2 as transforms


"""
For CIFAR-10 and CIFAR-100 experiments, we use a CNN similar to LeNet-
5 which has two 5×5, 64-channel convolution layers, each precedes a 2×2
max-pooling layer, followed by two fully-connected layers with 384 and 192
channels respectively and finally a softmax linear classifier. 

This model is not
the state-of-the-art on the CIFAR datasets, but is sufficient to show the relative
performance for our investigation. Weight decay is set to 4 × 10−4.
Unless otherwise stated, the client learning rate is 0.01 and momentum
β = 0.9 is used for FedAvgM. The learning rate is kept constant without decay
for simplicity. The client batch size is 32 for Landmarks-User-160k and 64 for
others.
"""

CONV1_CH = 64
CONV2_CH = 64

LIN1_CH = 384
LIN2_CH = 192

IMG_WH = 28

class LeNet5(nn.Module):
    def __init__(self, num_classes=100): 
        super(LeNet5, self).__init__()

        self._feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=CONV1_CH, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=CONV1_CH, out_channels=CONV2_CH, kernel_size=5, stride=1, padding=0), 
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )

        self._classifier = nn.Sequential(
            nn.Linear(in_features=CONV2_CH * (IMG_WH//(2*2) - 2)**2, out_features=LIN1_CH),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(in_features=LIN1_CH, out_features=LIN2_CH),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(in_features=LIN2_CH, out_features=num_classes)
        )

        self._softmax = nn.Softmax(dim=1)

    def forward(self, x, apply_softmax=False):
        x = self._feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self._classifier(x)

        if apply_softmax:
            x = self._softmax(x)

        return x
    
    def save(self, path: str, other: dict):
        """
        Saves the model architecture and parameters to the specified path.

        Args:
            path (str): Path to save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({'model_state_dict': self.state_dict(),
                    'model_class': self.__class__.__name__, **other}, path)

    @classmethod
    def load(cls, path: str):
        """
        Loads the model architecture and parameters from the specified path.

        Args:
            path (str): Path to the saved model.

        Returns:
            LeNet5: An instance of the LeNet5 class with loaded parameters.
        """
        checkpoint = torch.load(path)
        
        model = cls()
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model


def average_model_params(out: nn.Module, inp: list[nn.Module]) -> None:
    """
    Averages the parameters of a list of models and stores the result in the output model.

    Args:
        out (nn.Module): The model where the averaged parameters will be stored.
        inp (list[nn.Module]): A list of models whose parameters will be averaged.
    """
    if not inp:
        raise ValueError("The list of input models is empty.")

    for i in range(1, len(inp)):
        if len(list(inp[0].parameters())) != len(list(inp[i].parameters())):
            raise ValueError("All models must have the same structure.")

    for out_param, *inp_params in zip(out.parameters(), *[m.parameters() for m in inp]):

        if not all(out_param.shape == inp_param.shape for inp_param in inp_params):
            raise ValueError("Mismatch in parameter shapes among models.")

        avg_param = torch.mean(torch.stack([inp_param.data for inp_param in inp_params]), dim=0)

        out_param.data.copy_(avg_param)

def set_model_params(out: list[nn.Module], inp: nn.Module) -> None:
    """
    Sets the parameters of all models in the list to match the parameters of the input model.

    Args:
        out (list[nn.Module]): A list of models whose parameters will be updated.
        inp (nn.Module): The input model whose parameters will be copied to the list of models.
    """
    if not out:
        raise ValueError("The list of models is empty.")
    
    for model in out:
        if len(list(model.parameters())) != len(list(inp.parameters())):
            raise ValueError("Mismatch in the structure of models and the input model.")
    
    inp_params = list(inp.parameters())

    for model in out:
        for model_param, inp_param in zip(model.parameters(), inp_params):
            if model_param.shape != inp_param.shape:
                raise ValueError("Mismatch in parameter shapes between models and the input model.")
            
            model_param.data.copy_(inp_param.data)


def evaluate_model(model, eval_data: Dataset, device, verbose=True):
    """
    Evaluates the model on the given dataset.

    Args:
        eval_data (torch.utils.data.Dataset): Evaluation dataset.

    Returns:
        tuple: Contains:
            - av_loss (float): Average loss over the evaluation dataset.
            - accuracy (float): Accuracy over the evaluation dataset.
    """
    eval_data_loader = DataLoader(eval_data, batch_size=2048, shuffle=False, pin_memory=True)

    model.eval()
    correct = 0
    running_loss = 0

    with torch.no_grad():
        for images, labels in eval_data_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels, reduction="sum")

            # Accumulate loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    av_loss = running_loss / len(eval_data)
    accuracy = correct / len(eval_data)

    if verbose:
        print(f"Validation loss/acc: {av_loss:.3f}/{accuracy*100:.2f}%\n")

    return av_loss, accuracy

class Trainer:

    def __init__(self, model: nn.Module, optimizer_class: Type[Optimizer],
                 optimizer_params: dict, device, scheduler_class: Type[LRScheduler] = None, scheduler_params: dict = None, verbose=True):
        """
        Initializes the Trainer class for managing the training and evaluation process of a neural network.

        Args:
            training_data (torch.utils.data.Dataset): The dataset to be used for training the model.
            validation_data (torch.utils.data.Dataset): The dataset to be used for validation during training.
            model (torch.nn.Module): The neural network model to train and evaluate.
            optimizer_class (Type[torch.optim.Optimizer]): The class of the optimizer to use (e.g., `torch.optim.Adam`).
            optimizer_params (dict): A dictionary of parameters to initialize the optimizer.
            device (torch.device): The device on which to perform computations (`torch.device('cuda')` or `torch.device('cpu')`).
            scheduler_class (Type[torch.optim.lr_scheduler._LRScheduler], optional): 
                The class of the learning rate scheduler to use (e.g., `torch.optim.lr_scheduler.CosineAnnealingLR`). 
                Default is None. Scheduler gets updated every batch
            scheduler_params (dict, optional): A dictionary of parameters to initialize the scheduler. Default is None.
            verbose (bool, optional): If True, prints progress and logs during training and evaluation. Default is True.

        Attributes:
            training_data (torch.utils.data.Dataset): Stores the training dataset.
            validation_data (torch.utils.data.Dataset): Stores the validation dataset.
            model (torch.nn.Module): The neural network model being trained.
            _model_copy (torch.nn.Module): A deep copy of the initial model for resetting purposes.
            optimizer (torch.optim.Optimizer): The optimizer instance initialized with the given parameters.
            scheduler (torch.optim.lr_scheduler._LRScheduler or None): The learning rate scheduler instance.
            _optimizer_class (Type[torch.optim.Optimizer]): Stores the optimizer class for reset purposes.
            _optimizer_params (dict): Stores the optimizer parameters for reset purposes.
            _scheduler_class (Type[torch.optim.lr_scheduler._LRScheduler] or None): 
                Stores the scheduler class for reset purposes.
            _scheduler_params (dict or None): Stores the scheduler parameters for reset purposes.
            device (torch.device): The device on which computations are performed.
            criterion (torch.nn.CrossEntropyLoss): The loss function used during training.
            verbose (bool): Indicates whether to print progress logs during training and evaluation.
        """
        self.model = model.to(device)

        self.optimizer: Optimizer = optimizer_class(model.parameters(), **optimizer_params)

        if scheduler_class is not None:
            self.scheduler: LRScheduler = scheduler_class(self.optimizer, **scheduler_params)

        self._optimizer_class = optimizer_class

        self._scheduler_class = scheduler_class

        self.device = device
        self.verbose = verbose

    def train_model(self, train_loader: Iterator[DataLoader], n_steps: int, eval_data: Dataset = None):
        """
        Trains the model on the given training dataset.

        Args:
            train_loader (works like: DataLoader): Returns a batch of training data when next is called on it.
            n_epochs (int): Number of training epochs.
            eval_data (torch.utils.data.Dataset, optional): Evaluation dataset. Default is None.

        Returns:
            tuple: Contains two or four tensors:
                - train_loss (torch.Tensor): Training loss per epoch.
                - train_acc (torch.Tensor): Training accuracy per epoch.
                - eval_loss (torch.Tensor): Evaluation loss per epoch (if `eval_data` is provided).
                - eval_acc (torch.Tensor): Evaluation accuracy per epoch (if `eval_data` is provided).
        """

        self.model.train()

        train_loss = torch.empty(n_steps)
        train_acc = torch.empty(n_steps)
        eval_loss = torch.empty(n_steps)
        eval_acc = torch.empty(n_steps)

        if self.verbose:
            start_time = time.monotonic()
            print(f"Training progress: [0/{n_steps}]")

        for step in range(n_steps):
            try:
                images, labels = next(train_loader)
            except StopIteration:
                if eval_data: 
                    return train_loss.mean().item(), train_acc.mean().item(), eval_loss.mean().item(), eval_acc.mean().item()
                return train_loss.mean().item(), train_acc.mean().item() 
            
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            batch_size = labels.size(0)

            # Forward pass
            outputs = self.model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels, reduction="mean")

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()

            if torch.isnan(loss):
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item()
                print(f"Gradient norm: {total_norm}")

                raise ValueError("Loss is NaN. Stopping...")

            self.optimizer.step()

            if self._scheduler_class is not None:
                self.scheduler.step()
                
                if self.verbose:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Training step {step + 1}: Learning rate {current_lr:.6f}")

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()

            train_loss[step] = loss.item()
            train_acc[step] = correct / batch_size

            if eval_data is not None:
                e_loss, e_acc = self.evaluate(eval_data)
                eval_loss[step] = e_loss
                eval_acc[step] = e_acc
                self.model.train()

            if self.verbose:
                print(f"Training progress: [{(step+1)}/{n_steps}], {(time.monotonic()-start_time)/((step+1)):.2f}s per step (batch size: {batch_size})")

        if eval_data: 
            return train_loss.mean().item(), train_acc.mean().item(), eval_loss.mean().item(), eval_acc.mean().item()
        
        return train_loss.mean().item(), train_acc.mean().item()

    def evaluate(self, eval_data: Dataset = None):
        """
        Evaluates the model on the given dataset.

        Args:
            eval_data (torch.utils.data.Dataset): Evaluation dataset.

        Returns:
            tuple: Contains:
                - av_loss (float): Average loss over the evaluation dataset.
                - accuracy (float): Accuracy over the evaluation dataset.
        """
        if eval_data is None:
            eval_data = self.validation_data

        return evaluate_model(self.model, eval_data, self.device, self.verbose)


def load_data(data_dir=None, random_seed=None, test_data=False):
    if data_dir is None:
        data_dir = os.path.abspath("./data")

    if random_seed is not None:
        random_seed = torch.Generator().manual_seed(random_seed)
    
    train_transforms = transforms.Compose([
        transforms.RandomCrop((IMG_WH, IMG_WH)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), #
        # transforms.ToImage(),
        # transforms.ToDtype(torch.float32, scale=True), # to tensor is faster 
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)) 
    ])

    val_transforms = transforms.Compose([
        transforms.CenterCrop((IMG_WH, IMG_WH)),
        transforms.ToTensor(),
        # transforms.ToImage(),
        # transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
    ])

    dataset = datasets.CIFAR100(root="data", train=(not test_data), download=True)


    if test_data:
        dataset.transform = train_transforms
        return dataset


    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataset = DatasetFromSubset(train_dataset, train_transforms)
    val_dataset = DatasetFromSubset(val_dataset, val_transforms)

    return train_dataset, val_dataset


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
    
    import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupPolyScheduler(_LRScheduler):
    """
    Custom Learning Rate Scheduler with Warmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epochs (int): Number of warmup epochs.
        total_epochs (int): Total number of training epochs.
        power (float): Power for polynomial decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, power=2.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.power = power
        super(WarmupPolynomialDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Polynomial decay phase
            decay_factor = (1 - (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)) ** self.power
            return [base_lr * decay_factor for base_lr in self.base_lrs]
        
from torch.optim.lr_scheduler import _LRScheduler

class WarmupSqRootScheduler(_LRScheduler):
    """
    Custom Learning Rate Scheduler with Warmup and Square Root Decay.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epochs (int): Number of warmup epochs.
        total_epochs (int): Total number of training epochs.
        last_epoch (int): The index of the last epoch. Default: -1.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super(WarmupSquareRootLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Square root decay phase
            decay_factor = (1 - (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)) ** 0.5
            return [base_lr * decay_factor for base_lr in self.base_lrs]
