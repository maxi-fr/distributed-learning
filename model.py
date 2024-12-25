import time
import torch
from torch import nn
import copy
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
import copy
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from checkpoint import Checkpoint


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


class LeNet5(nn.Module):
    def __init__(self, num_classes=100):  # Default is 10 classes (e.g., for MNIST or CIFAR-10)
        super(LeNet5, self).__init__()

        self._feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self._classifier = nn.Sequential(
            nn.Linear(in_features=16 * 6 * 6, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self._feature_extractor(x)

        x = torch.flatten(x, start_dim=1)

        out = self._classifier(x)

        return out


class Trainer:

    def __init__(self, training_data: Dataset, validation_data: Dataset, model: nn.Module, optimizer_class: Optimizer, 
                 optimizer_params: dict, device, verbose=True):
        self.training_data = training_data
        self.validation_data = validation_data

        self.model = model
        self._model_copy = copy.deepcopy(model)
        
        self.optimizer: Optimizer = optimizer_class(model.parameters(), **optimizer_params)

        self._optimizer_class = optimizer_class
        self._optimizer_params = copy.deepcopy(optimizer_params)

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

        train_loader = DataLoader(self.training_data, batch_size, shuffle=True)

        if cp is None:
            cp = Checkpoint()

        start_epoch = cp.epoch
        if self.verbose:
            start_time = time.monotonic()
            print(f"Training progress: [0/{n_epochs-start_epoch}]")

        for epoch in range(start_epoch, n_epochs):
            self.model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device, copy=False), labels.to(self.device, copy=False)
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

            if save_every is not None:
                if (epoch + 1) % save_every == 0:
                    cp.epoch = epoch
                    cp.save(self.model, self.optimizer)

            if self.verbose:
                print(f"Training progress: [{(epoch+1)-start_epoch}/{n_epochs-start_epoch}], {(time.monotonic()-start_time)/((epoch+1)-start_epoch):.2f}s per epoch")

        return cp.train_loss, cp.train_acc, cp.eval_loss, cp.eval_acc


    def evaluate(self, eval_data: Dataset=None):
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

        eval_data_loader = DataLoader(eval_data, batch_size=8192, shuffle=False)

        self.model.eval()
        correct = 0
        running_loss = 0

        with torch.no_grad():
            for images, labels in eval_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)

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
            print(f"Evaluation loss:     {av_loss}")
            print(f"Evaluation accuracy: {100 * accuracy:.2f}%")

        return av_loss, accuracy

    def reset_model(self, optimizer_params: dict=None):
        self.model: nn.Module = copy.deepcopy(self._model_copy) 

        if optimizer_params is None:
            optimizer_params = self._optimizer_params

        self.optimizer = self._optimizer_class(self.model.parameters(), **optimizer_params)

    @classmethod
    def standard_init(cls, optimizer_class= torch.optim.Adam, optimizer_params = {"lr": 0.001}):

        full_train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
        
        train_size = 0.8  # 80% for training
        val_size = 1 - train_size  # 20% for validation

        train_dataset, val_dataset = random_split(full_train_dataset, (train_size, val_size))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LeNet5(num_classes=100).to(device)

        return cls(train_dataset, val_dataset, model, optimizer_class, optimizer_params, device)

if __name__ == "__main__":
    model = LeNet5(num_classes=100)
    input_tensor = torch.rand(1, 3, 32, 32)  # Batch size: 1, Channels: 1, Height: 32, Width: 32
    output = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
