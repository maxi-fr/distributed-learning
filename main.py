

# Move script from test_notebook into here
# google colab notebook should just call this script
# but can also be called from command line


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import LeNet5


def main(epochs, device=None, criterion=None, optimizer=None):

    full_train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())

    train_size = 0.8  # 80% for training
    val_size = 1 - train_size  # 20% for validation

    train_dataset, val_dataset = random_split(full_train_dataset, (train_size, val_size))

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5(num_classes=100).to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss, train_acc, eval_loss, eval_acc = model.fit(train_loader, epochs, criterion, optimizer, device, val_dataset, save_every=2)

    plot_metric(train_loss, marker="x")
    plot_metric(eval_loss)
    plt.ylabel("Loss")
    plt.show()
    plot_metric(train_acc, marker="x")
    plot_metric(eval_acc)
    plt.ylabel("Accuracy")
    plt.show()
    

def plot_metric(metric, **kwargs):
    plt.plot(range(len(metric)), metric, **kwargs)
    plt.xlabel("Epochs")
    plt.grid(True)


if __name__ == "__main__":

    # parse command line args

    main(...)


"""
Aufgaben:
- done: split into training and validation sets
- done: Metriken, usw. entwickeln
- done: implement experiment logging (maybe better experiment logging)
- done: implement when checkpoints get saved and who calls them
- done: test checkpoints
- test setup on GPU



- implement centralized baseline
- simulate parallel training of local methods
- Optimizers implementieren: LARS, LAMB, AdamW, SGDM

"""
