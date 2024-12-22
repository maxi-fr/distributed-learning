from torch.nn import Module
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


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

if __name__ == "__main__":
    model = LeNet5(num_classes=10)
    input_tensor = torch.rand(1, 1, 32, 32)  # Batch size: 1, Channels: 1, Height: 32, Width: 32
    output = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
