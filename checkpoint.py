import os
import torch
from datetime import datetime


class Checkpoint:

    def __init__(self, folder: str ="Checkpoints"):
        self.reset_checkpoint()

        if not os.path.isdir(folder):
            os.makedirs(folder)
        self.folder = folder

    def reset_checkpoint(self):
        self.epoch = 0
        self.train_loss = []
        self.train_acc = []
        self.eval_loss = []
        self.eval_acc = []

    @classmethod
    def from_saved(cls, model: torch.nn.Module, optimizer: torch.optim.Optimizer, folder="Checkpoints"):
        """
        Loads the model, optimizer, and training metrics from the newest checkpoint file in the folder.

        Args:
            model (torch.nn.Module): Model instance.
            optimizer (torch.optim.Optimizer): Optimizer instance.
            folder (str): Path to the folder where checkpointfiles are stored.

        Returns:
            tuple: Contains:
                - int: Resumed epoch.
                - list: Training losses.
                - list: Training accuracies.
        """
        path = max(file for file in os.listdir(folder) if file.endswith(".pth"))

        path = os.path.join(folder, path)
        if os.path.exists(path):
            state = torch.load(path)
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            print(f"Checkpoint loaded from {path}")

            self = cls(folder)
            self.epoch = state['epoch']
            self.train_loss = state['train_loss']
            self.train_acc = state['train_acc']
            self.eval_loss = state["eval_loss"]
            self.eval_acc = state["eval_acc"]

            return self
        
        raise Exception(f"No checkpoint found at {path}")


    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """
        Saves the model, optimizer, and training metrics to a checkpoint file.

        Args:
            epoch (int): Current epoch.
            model (torch.nn.Module): Model instance.
            optimizer (torch.optim.Optimizer): Optimizer instance.
            train_loss (list): List of training losses.
            train_acc (list): List of training accuracies.
            folder (str): Path to the folder where the checkpoint file will be stored.
        """
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

        state = {
            'epoch': self.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'eval_loss': self.eval_loss,
            'eval_acc': self.eval_acc
        }
        path = os.path.join(self.folder, datetime.now().strftime("%Y_%m_%d__%H-%M-%S") + ".pth")
        torch.save(state, path)
        print(f"Checkpoint saved at {path}")

    def clear_folder(self):
        for f in os.listdir(self.folder):
            if f.endswith(".pth"):
                os.remove(os.path.join(self.folder, f))


