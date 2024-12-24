import os
import random
import numpy as np
import pandas as pd
import itertools
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

def random_search(max_iter, search_space, folder, 
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
    for it in tqdm(range(max_iter), f"Hyperparameter search <{optimizer_class}>"):

        params = {k: sample_uniform(v) for k, v in search_space}
        print(f"\nTesting parameters: {params}")

        model: LeNet5 = model_class()
        optimizer = optimizer_class(model.parameters(), **params)

        cp = Checkpoint(cp_folder)
        model.fit(train_loader, n_epochs, criterion, optimizer, device, cp=cp, save_every=save_every)
        cp.clear_folder()
        
        
        final_eval_loss, final_eval_accuracy = model.evaluate(eval_data, criterion, device)

    
        results.append((*params.values(), final_eval_loss, final_eval_accuracy))

        pd.DataFrame(results, columns=(*params.keys(), "eval_loss", "eval_acc")).to_csv(save_path)

    return pd.DataFrame(results, columns=(*params.keys(), "eval_loss", "eval_acc"))
    
