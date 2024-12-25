import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module
from torch.optim import Optimizer
import torch.nn as nn
from tqdm import tqdm

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


    def fit(self, train_loader: DataLoader, n_epochs: int, 
            criterion: Module, optimizer: Optimizer, device, eval_data=None, 
            cp: Checkpoint= None, save_every: int|None =None):
        """
        Trains the model on the given training dataset.

        Args:
            train_loader (DataLoader): Dataloader for the training dataset.
            n_epochs (int): Number of training epochs.
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            device (torch.device): Device to train on ('cuda' or 'cpu').
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

        if cp is None:
            cp = Checkpoint()

        for epoch in tqdm(range(cp.epoch, n_epochs), desc="Training progress"):
            self.train() 

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                batch_size = labels.size(0)

                # Forward pass
                outputs = self(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss and accuracy
                running_loss += loss.item() * batch_size
                total += batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()


            cp.train_loss.append(running_loss / total) 
            cp.train_acc.append(correct / total) 

            if eval_data is not None:
                e_loss, e_acc = self.evaluate(eval_data, criterion, device)
                cp.eval_loss.append(e_loss)
                cp.eval_acc.append(e_acc)

            if save_every is not None:
                if (epoch + 1) % save_every == 0:
                    cp.epoch = epoch
                    cp.save(self, optimizer) 
            

        return cp.train_loss, cp.train_acc, cp.eval_loss, cp.eval_acc


    def evaluate(self, eval_data: Dataset, criterion, device):
        """
        Evaluates the model on the given dataset.

        Args:
            eval_data (torch.utils.data.Dataset): Evaluation dataset.
            criterion (torch.nn.Module): Loss function.
            device (torch.device): Device to evaluate on ('cuda' or 'cpu').

        Returns:
            tuple: Contains:
                - av_loss (float): Average loss over the evaluation dataset.
                - accuracy (float): Accuracy over the evaluation dataset.
        """
        eval_data_loader = DataLoader(eval_data, batch_size=256, shuffle=False)

        self.eval()
        correct = 0
        running_loss = 0

        with torch.no_grad():
            for images, labels in eval_data_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = self(images)
                loss = criterion(outputs, labels)

                # Accumulate loss and accuracy
                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        av_loss = running_loss / len(eval_data)
        accuracy = correct / len(eval_data)

        print(f"Evaluation loss:     {av_loss}")
        print(f"Evaluation accuracy: {100 * accuracy:.2f}%")

        return av_loss, accuracy


if __name__ == "__main__":
    model = LeNet5(num_classes=100)
    input_tensor = torch.rand(1, 3, 32, 32)  # Batch size: 1, Channels: 1, Height: 32, Width: 32
    output = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
