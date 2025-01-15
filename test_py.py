import os

import pandas as pd
from model import Trainer, LeNet5, evaluate_model, load_data
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from Results.experiments_config import OPT_SGD_LR, OPT_SGD_W_DECAY
from torch.utils.data import DataLoader, random_split, Dataset
import time
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import multiprocessing


if __name__ == "__main__":
    def plot_metric(metric, ax: plt.Axes, **kwargs):
        ax.plot(range(len(metric)), metric, **kwargs)
        ax.set_xlabel("Epochs")
        ax.grid(True)
        ax.legend()
        
    def plot_metrics(df, fname):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for col in df.columns:
            if "loss" in col:
                plot_metric(df[col], ax1, label=col)
                ax1.set_ylabel("loss")
            else:
                plot_metric(df[col], ax2, label=col)
                ax2.set_ylabel("accuracy")

        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname)
        return fig, (ax1, ax2)

    train_dataset, val_dataset = load_data()


    model = LeNet5()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    n_epochs = 150
    b_sizie = 64
    opt_class = SGD

    trainer = Trainer(model, opt_class, {"lr": OPT_SGD_LR, "momentum": 0.9, "weight_decay": OPT_SGD_W_DECAY},
                    device, CosineAnnealingLR, {"T_max": n_epochs*len(train_dataset)//b_sizie}, verbose=False)

    d_loader = DataLoader(train_dataset, b_sizie, shuffle=True, drop_last=True, pin_memory=True, num_workers=8, prefetch_factor=16, persistent_workers=True)


    train_metrics = []
    val_metrics = []
    print("Starting training on device:", device)
    start_time = time.monotonic()
    for epoch in range(n_epochs):
        d_iter = iter(d_loader)

        train_metrics.append(trainer.train_model(d_iter, len(d_loader)))

        print(f"Training progress: [{(epoch+1)}/{n_epochs}], {(time.monotonic()-start_time)/((epoch+1)):.2f}s per epoch")
        print(f"Current training loss/acc: {train_metrics[-1][0]:.3f}/{train_metrics[-1][1]*100:.2f}%")
        val_metrics.append(evaluate_model(model, val_dataset, torch.nn.CrossEntropyLoss(), device, verbose=True))
        print("")

    performance = pd.DataFrame(train_metrics, columns=["train_loss", "train_acc"])
    performance[["val_loss", "val_acc"]] = val_metrics


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = load_data(test_data=True)

    test_acc = evaluate_model(model, test_dataset, torch.nn.CrossEntropyLoss(), device)

    add_on = "_more_DO_"
    model.save(os.path.join("models", opt_class.__name__ + add_on + ".lenet"), {"test_acc": test_acc})

    performance.to_csv(os.path.join("models", opt_class.__name__ + add_on + "performance.csv"))
    plot_metrics(performance, os.path.join("models", opt_class.__name__ + add_on + "performance.png"))