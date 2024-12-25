import copy
import os
import random
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import copy

from checkpoint import Checkpoint
from model import LeNet5, Trainer


VAL_LOSS = "validation_loss"
VAL_ACC = "validation_acc"

def sample_uniform(inp):
    if np.squeeze(inp).ndim == 1:
        out = random.uniform(*inp)
    else:
        out = [random.uniform(*x) for x in inp]

    return out

def random_search(trainer: Trainer, max_iter: int, n_epochs: int, batch_size: int,
                  search_space: dict, folder: str, save_every=None):
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
    trainer.verbose = False

    cp_folder = os.path.join(folder, "Checkpoints")
    if not os.path.isdir(cp_folder):
        os.makedirs(cp_folder)

    cp = Checkpoint(cp_folder)


    save_path = os.path.join(folder, "random_search_results.csv")

    start_time = time.monotonic()
    print(f"Random search it: [0/{max_iter}]")

    results = []
    for it in range(1, max_iter+1):

        params = {k: sample_uniform(v) for k, v in search_space.items()}
        print(f"Testing parameters: {params} for {trainer._optimizer_class}")

        trainer.reset_model(params)
        cp.reset_checkpoint()
        cp.clear_folder()

        trainer.train_model(batch_size, n_epochs, cp=cp, save_every=save_every)

        final_eval_loss, final_eval_accuracy = trainer.evaluate()

        results.append((*params.values(), final_eval_loss, final_eval_accuracy))
        pd.DataFrame(results, columns=(*params.keys(), VAL_LOSS, VAL_ACC)).to_csv(save_path)

        print(f"Random search it: [{it}/{max_iter}], {(time.monotonic()-start_time)/it:.2f}s per iteration")


    results = pd.DataFrame(results, columns=(*params.keys(), VAL_LOSS, VAL_ACC))

    print("The best hyperparameter set is:")
    print(results.sort_values(VAL_LOSS).iloc[0])
    print("and for accuracy:")
    print(results.sort_values(VAL_ACC).iloc[0])

    return results


def plot_2d_results(results_df, metric=VAL_LOSS):
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


def plot_1d_results(results_df: pd.DataFrame, metric=VAL_LOSS):
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

    trainer = Trainer.standard_init()

    results = random_search(trainer, max_iter=10, n_epochs=3, batch_size=256, 
                            search_space={"lr": (0.0001, 0.01)}, folder="hyper", save_every=5)

    print(results)
