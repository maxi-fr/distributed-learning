import torch.nn as nn
import time
from typing import Type
import torch
from torch import nn
import copy
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import copy
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from checkpoint import Checkpoint
from optimizers import LARS


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

class LeNet5(nn.Module):
    def __init__(self, num_classes=100): 
        super(LeNet5, self).__init__()

        self._feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=CONV1_CH, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=CONV1_CH, out_channels=CONV2_CH, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self._classifier = nn.Sequential(
            nn.Linear(in_features=CONV2_CH * 6 * 6, out_features=LIN1_CH),
            nn.ReLU(),
            nn.Linear(in_features=LIN1_CH, out_features=LIN2_CH),
            nn.ReLU(),
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


class Trainer:

    def __init__(self, training_data: Dataset, validation_data: Dataset, model: nn.Module, optimizer_class: Type[Optimizer],
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
                Default is None. Scheduler gets updated every EPOCH
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

        self.training_data = training_data
        self.validation_data = validation_data

        self.model = model
        self._model_copy = copy.deepcopy(model)

        self.optimizer: Optimizer = optimizer_class(model.parameters(), **optimizer_params)

        if scheduler_class is not None:
            self.scheduler: LRScheduler = scheduler_class(self.optimizer, **scheduler_params)

        self._optimizer_class = optimizer_class
        self._optimizer_params = copy.deepcopy(optimizer_params)

        self._scheduler_class = scheduler_class
        self._scheduler_params = copy.deepcopy(scheduler_params)

        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.verbose = verbose

    def train_model(self, batch_size: int, n_epochs: int, eval_data: Dataset = None,
                    cp: Checkpoint = None, save_every: int = None):
        """
        Trains the model on the given training dataset.

        Args:
            batch_size (int): Number of batches into which the dataset is split.
            n_epochs (int): Number of training epochs.
            eval_data (torch.utils.data.Dataset, optional): Evaluation dataset. Default is None.
            cp (Checkpoint, optional): If training is supposed to start from a saved Checkpoint it can be passed through cp. 
            save_every (int, optional): Number of epochs between saving checkpoints
        Returns:
            tuple: Contains four lists:
                - train_loss (list): Training loss per epoch.
                - train_acc (list): Training accuracy per epoch.
                - eval_loss (list): Evaluation loss per epoch (if `eval_data` is provided otherwise empty list).
                - eval_acc (list): Evaluation accuracy per epoch (if `eval_data` is provided, otherwise empty list).
        """

        train_loader = DataLoader(self.training_data, batch_size, shuffle=True, pin_memory=True)

        patience = 5  # parameters for early stopping
        delta = 1e-4

        self.model.train()

        if cp is None:
            cp = Checkpoint()

        start_epoch = cp.epoch
        if self.verbose:
            start_time = time.monotonic()
            print(f"Training progress: [0/{n_epochs-start_epoch}]")

        for epoch in range(start_epoch, n_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                batch_size = labels.size(0)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate loss and accuracy
                running_loss += loss.item() * batch_size
                total += batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

            cp.train_loss.append(running_loss / total)
            cp.train_acc.append(correct / total)

            if eval_data is not None:
                e_loss, e_acc = self.evaluate(eval_data)
                cp.eval_loss.append(e_loss)
                cp.eval_acc.append(e_acc)
                self.model.train()

                # TODO: early stopping?
                # if len(cp.eval_loss) > patience:
                #     recent_losses = cp.eval_loss[-patience:]
                #     if all(recent_losses[i] >= recent_losses[i + 1] - delta for i in range(len(recent_losses) - 1)):
                #         print(f"Early stopping at epoch {epoch + 1}")
                #         break

            if self._scheduler_class is not None:
                self.scheduler.step()
                if self.verbose:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch + 1}: Learning rate {current_lr:.6f}")

            if save_every is not None:
                if (epoch + 1) % save_every == 0:
                    cp.epoch = epoch
                    cp.save(self.model, self.optimizer)

            if self.verbose:
                print(f"Training progress: [{(epoch+1)-start_epoch}/{n_epochs - start_epoch}], {(time.monotonic()-start_time)/((epoch+1)-start_epoch):.2f}s per epoch")

        return cp.train_loss, cp.train_acc, cp.eval_loss, cp.eval_acc

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

        eval_data_loader = DataLoader(eval_data, batch_size=16384, shuffle=False, pin_memory=True)

        self.model.eval()
        correct = 0
        running_loss = 0

        with torch.no_grad():
            for images, labels in eval_data_loader:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Accumulate loss and accuracy
                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        av_loss = running_loss / len(eval_data)
        accuracy = correct / len(eval_data)

        if self.verbose:
            print(f"Evaluation loss:     {av_loss:.4f}")
            print(f"Evaluation accuracy: {100 * accuracy:.2f}%")

        return av_loss, accuracy

    def reset_model(self, optimizer_params: dict = None, scheduler_params: dict = None):
        self.model: nn.Module = copy.deepcopy(self._model_copy)

        if optimizer_params is None:
            optimizer_params = self._optimizer_params

        self.optimizer = self._optimizer_class(self.model.parameters(), **optimizer_params)

        if scheduler_params is None:
            scheduler_params = self._scheduler_params

        if self._scheduler_class is not None:
            self.scheduler = self._scheduler_class(self.optimizer, **scheduler_params)

    @classmethod
    def standard_init(cls, optimizer_class=torch.optim.Adam, optimizer_params={"lr": 0.001},
                      scheduler_class=None, scheduler_params=None):
        """
        A standard initialization method for the Trainer class using the CIFAR-100 dataset 
        and the LeNet5 model. Configures the optimizer, scheduler, and splits the dataset 
        into training and validation subsets.

        Args:
            optimizer_class (Type[torch.optim.Optimizer], optional): 
                The optimizer class to use (e.g., `torch.optim.Adam`). Default is `torch.optim.Adam`.
            optimizer_params (dict, optional): 
                A dictionary of parameters to initialize the optimizer. Default is `{"lr": 0.001}`.
            scheduler_class (Type[torch.optim.lr_scheduler._LRScheduler], optional): 
                The scheduler class to use for learning rate adjustments (e.g., `torch.optim.lr_scheduler.CosineAnnealingLR`). 
                Default is None.
            scheduler_params (dict, optional): 
                A dictionary of parameters to initialize the scheduler. Default is None.

        Returns:
            Trainer: An instance of the `Trainer` class configured with the LeNet5 model, 
            CIFAR-100 dataset, the specified optimizer, and optional learning rate scheduler.

        Example:
            trainer = Trainer.standard_init(
                optimizer_class=torch.optim.SGD,
                optimizer_params={"lr": 0.01, "momentum": 0.9},
                scheduler_class=torch.optim.lr_scheduler.StepLR,
                scheduler_params={"step_size": 10, "gamma": 0.1}
            )
        """
        full_train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())

        train_size = 0.8  # 80% for training
        val_size = 1 - train_size  # 20% for validation

        train_dataset, val_dataset = random_split(full_train_dataset, (train_size, val_size))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LeNet5(num_classes=100).to(device)

        return cls(
            training_data=train_dataset,
            validation_data=val_dataset,
            model=model,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
            device=device,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params
        )


if __name__ == "__main__":
    model = LeNet5(num_classes=100)
    input_tensor = torch.rand(1, 3, 32, 32)  # Batch size: 1, Channels: 1, Height: 32, Width: 32
    output = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
