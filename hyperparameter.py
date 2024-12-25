import copy
import math
import os
import random
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import copy

import torch

from checkpoint import Checkpoint
from model import LeNet5, Trainer


VAL_LOSS = "validation_loss"
VAL_ACC = "validation_acc"

class SearchSpace:

    def __init__(self, lower_bound, upper_bound, log_scale=False):
        self.interval = np.array((lower_bound, upper_bound))
        self.log_scale = log_scale
        if log_scale:
            self.interval = np.log(self.interval)

    def sample(self):
        if self.log_scale:
            return math.exp(random.uniform(*self.interval))
        
        return random.uniform(*self.interval)


def sample_float_or_list(inp: SearchSpace|list[SearchSpace]) -> float|list[float]:
    if isinstance(inp, SearchSpace):
        out = inp.sample()
    else:
        out = [x.sample() for x in inp]

    return out
def random_search(trainer: Trainer, max_iter: int, n_epochs: int, batch_size: int | list[int], 
                  search_space_opt: dict[str, SearchSpace|list[SearchSpace]], folder: str, save_every=None) -> pd.DataFrame:
    """
    Perform random search over a defined hyperparameter space for training a model.

    Args:
        trainer (Trainer): 
            An instance of the Trainer class used to manage model training and evaluation.
        max_iter (int): 
            The maximum number of hyperparameter combinations to sample and evaluate.
        n_epochs (int): 
            The number of training epochs for each hyperparameter configuration.
        batch_size (int | list[int]): 
            The batch size(s) to test. If a list is provided, one value is sampled per iteration.
        search_space_opt (dict): 
            A dictionary defining the hyperparameter space for optimization. 
            Keys are hyperparameter names, and values are tuples defining the intervall of the possible parameters of the optimizer.
        folder (str): 
            Directory path to save training checkpoints and logs for each experiment.
        save_every (int, optional): 
            Frequency (in epochs) to save model checkpoints. Defaults to None, meaning no saves.

    Returns:
        pd.DataFrame: 
            A DataFrame containing the results of the random search. Each row corresponds to a trial, and columns
            include hyperparameters, training/validation loss, accuracy, and additional metrics for comparison.

    Example:
        ```python
        trainer = Trainer(training_data, validation_data, model, optimizer_class, optimizer_params, device)

        search_space = {
            "lr": [0.0001, 0.1],
            "momentum": [0, 0.9]
        }

        results = random_search(
            trainer=trainer,
            max_iter=20,
            n_epochs=50,
            batch_size=[32, 64, 128],
            search_space_opt=search_space,
            folder="./checkpoints",
            save_every=10
        )

        print(results)
        ```
    """
    trainer.verbose = False

    hyp_names = ["batch_size", *search_space_opt.keys()]

    cp_folder = os.path.join(folder, "Checkpoints")
    if not os.path.isdir(cp_folder):
        os.makedirs(cp_folder)

    cp = Checkpoint(cp_folder)


    save_path = os.path.join(folder, "random_search_results.csv")

    start_time = time.monotonic()
    print(f"Random search it: [0/{max_iter}]")

    results = []
    for it in range(1, max_iter+1):

        params_opt = {k: sample_float_or_list(v) for k, v in search_space_opt.items()}

        if hasattr(batch_size, "__len__"):
            bs = random.choice(batch_size)
        else:
            bs = batch_size

        print(f"Testing parameters: {params_opt} for {trainer._optimizer_class}")

        trainer.reset_model(params_opt)
        cp.reset_checkpoint()
        cp.clear_folder()

        trainer.train_model(bs, n_epochs, cp=cp, save_every=save_every)

        final_eval_loss, final_eval_accuracy = trainer.evaluate()

        results.append((bs, *params_opt.values(), final_eval_loss, final_eval_accuracy))
        pd.DataFrame(results, columns=(*hyp_names, VAL_LOSS, VAL_ACC)).to_csv(save_path)

        print(f"Random search it: [{it}/{max_iter}], {(time.monotonic()-start_time)/it:.2f}s per iteration")


    results = pd.DataFrame(results, columns=(*hyp_names, VAL_LOSS, VAL_ACC))
    if "betas" in hyp_names:
        results[["beta1", "beta2"]] = pd.DataFrame(results["betas"].to_list())
        results.drop("betas", axis=1, inplace=True)
        hyp_names.remove("betas")
        hyp_names.extend(["beta1", "beta2"])

    results.set_index(hyp_names, inplace=True)

    print("The best hyperparameter set is:")
    print(results.sort_values(VAL_LOSS).iloc[0])
    print("\nand for accuracy:")
    print(results.sort_values(VAL_ACC).iloc[0])

    return results


def plot_2d_results(results_df: pd.DataFrame, metric=VAL_LOSS) -> tuple[plt.Figure, plt.Axes]:
    params = results_df.index.names

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
                vmin = results_df[metric].min()
                vmax = results_df[metric].max()
                im = ax.imshow(heatmap_data, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)

                ax.set_xlabel(y_param)
                ax.set_ylabel(x_param)
                ax.set_xticks(range(len(heatmap_data.columns)))
                ax.set_xticklabels(heatmap_data.columns, rotation=90)
                ax.set_yticks(range(len(heatmap_data.index)))
                ax.set_yticklabels(heatmap_data.index)
    return fig, ax


def plot_1d_results(results_df: pd.DataFrame, metric=VAL_LOSS) -> tuple[plt.Figure, plt.Axes]:
    params = results_df.index.names

    results_df = results_df.reset_index()

    fig, axs = plt.subplots(len(params))

    for ax, col in zip(axs, params):
        results_df.sort_values(col, inplace=True)
        ax.plot(results_df[col], results_df[metric], marker="x")
        ax.set_xlabel(col)
        ax.grid()
    fig.supylabel(metric)
    fig.tight_layout()

    return fig, ax


if __name__ == "__main__":

    trainer = Trainer.standard_init(torch.optim.AdamW, 
                                    scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR, scheduler_params={"T_max": 150})

    search_space = {"lr": SearchSpace(1e-5, 1e-2, log_scale=True), 
                    "weight_decay": SearchSpace(1e-6, 1e-2, True),
                    "betas": [SearchSpace(0.8, 0.95), SearchSpace(0.95, 0.999)]}

    results = random_search(trainer, max_iter=2, n_epochs=1, batch_size=256, 
                            search_space_opt=search_space, folder="hyper", save_every=5)

    print(results)
