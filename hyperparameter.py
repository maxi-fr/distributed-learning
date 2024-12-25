import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd 
import torch.optim as optim
from tqdm import tqdm

from checkpoint import Checkpoint
from model import LeNet5


def sample_uniform(inp):
    if np.squeeze(inp).ndim == 1:
        out = random.uniform(*inp)
    else:
        out = [random.uniform(*x) for x in inp]

    return out

def random_search(max_iter: int, search_space: dict, folder: str, 
                  model_class, train_loader, n_epochs, criterion, 
                  optimizer_class, device, eval_data, save_every=None):
    """
    Performs grid search for a general optimizer with logging for each parameter combination.

    Args:
        model_class (torch.nn.Module): The model with `fit` and `evaluate` methods implemented, i.e LeNet5.
        train_loader (DataLoader): Training data loader.
        eval_data (Dataset, optional): Validation or test data.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to train on (e.g., 'cuda' or 'cpu').
        n_epochs (int): Number of epochs to train for each configuration.
        optimizer_class (Optimizer): Optimizer class (e.g., torch.optim.AdamW).
        search_space (dict): Hyperparameter ranges to search.
        folder (str): Folder for saving checkpoints into 
        save_every (int, optional): Save a checkpoint every `save_every` epochs. Default is None.

    Returns:
        pd.DataFrame: Dataframe containing all parameter combinations and corresponding metrics.
    """
    cp_folder = os.path.join(folder, "Checkpoints")
    if not os.path.isdir(cp_folder):
        os.makedirs(cp_folder)

    save_path = os.path.join(folder, "random_search_results.csv")

    results = []
    for it in tqdm(range(max_iter), f"Hyperparameter search {optimizer_class}"):

        params = {k: sample_uniform(v) for k, v in search_space.items()}
        print(f"\nTesting parameters: {params}")

        model: LeNet5 = model_class()
        optimizer = optimizer_class(model.parameters(), **params)

        cp = Checkpoint(cp_folder)
        model.fit(train_loader, n_epochs, criterion, optimizer, device, cp=cp, save_every=save_every)
        cp.clear_folder()
        
        
        final_eval_loss, final_eval_accuracy = model.evaluate(eval_data, criterion, device)

    
        results.append((*params.values(), final_eval_loss, final_eval_accuracy))

        pd.DataFrame(results, columns=(*params.keys(), "eval_loss", "eval_acc")).to_csv(save_path)

    results = pd.DataFrame(results, columns=(*params.keys(), "eval_loss", "eval_acc"))

    print("The best hyperparameter set is:")
    print(results.sort_values("eval_loss").iloc[0])
    print("and for accuracy:")
    print(results.sort_values("eval_acc").iloc[0])
    
    return results
    

def plot_2d_results(results_df, metric="eval_acc"):
    params = results_df.columns[:-2]

    fig, axss = plt.subplots(len(params), len(params), sharex="col", sharey="row")

    # fig.suptitle(metric)
    for i, (axs, x_param) in enumerate(zip(axss, params)):
        for j in range(len(params)): 
            ax = axs[j]
            if j >= i: 
                ax.set_visible(False)
            else: 
                y_param = params[j]
                heatmap_data = results_df.pivot_table(index=x_param, columns=y_param, values=metric)
                im = ax.imshow(heatmap_data, aspect="auto", cmap="viridis")
                ax.set_xlabel(y_param)
                ax.set_ylabel(x_param)
                ax.set_xticks(range(len(heatmap_data.columns)))
                ax.set_xticklabels(heatmap_data.columns, rotation=90)
                ax.set_yticks(range(len(heatmap_data.index)))

                ax.set_yticklabels(heatmap_data.index)
    return fig, ax

def plot_1d_results(results_df: pd.DataFrame, metric="eval_loss"):
    params = results_df.iloc[:, :-2]

    fig, axs = plt.subplots(len(params.columns)) 

    for ax, col in zip(axs, params):
        ax.scatter(params[col], results_df[metric])
        ax.set_xlabel(col)
        ax.grid()
    fig.supylabel(metric)

    return fig, ax




if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms
    
    full_train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())

    train_size = 0.8  # 80% for training
    val_size = 1 - train_size  # 20% for validation

    train_dataset, val_dataset = random_split(full_train_dataset, (train_size, val_size))

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5(num_classes=100).to(device)

    criterion = nn.CrossEntropyLoss()
    
    optimizer_class = optim.SGD

    search_space = {"lr": (0.0001, 0.01), "momentum": (0, 0.9)}


    results = random_search(10, search_space, "hyper", LeNet5, train_loader, 3, criterion, optimizer_class, device, val_dataset)

    print(results)