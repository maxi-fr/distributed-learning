from functools import partial
import os
import time
import ray
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
from ray import tune
from ray.tune import CLIReporter
# from ray.tune.search.variant_generator import BasicVariantGenerator

from centralized import centralized_learning
from distributed import distributed_learning
from model import Instantiator, evaluate_model, load_data
import Results.experiments_config as experiments_config
import pickle

from optimizers import DoNothing
from optimizers import AverageOptimizers, DoNothing


# TODO: test hyperparameter search for this branch with all the new stuff, then merge

def tune_distributed_learning(config: dict[str, float | int], train_data_obj_ref):
    """
    Wrapper for distributed learning to enable Ray Tune hyperparameter tuning.

    Args:
        config (dict): Configuration containing hyperparameters to test.

    Returns:
        None
    """

    n_epochs = config.pop("n_epochs", 150)
    n_workers = config.pop("n_workers", 1)
    local_batch_size = config.pop("local_batch_size")
    n_local_steps = config.pop("n_local_steps", 1)

    is_distributed = n_workers > 1  # or n_local_steps > 1 <- only if optimizer buffers get averaged,
    # problem for when merging with local_ada_scale_implementaition branch TODO

    global_optimizer_class = config.pop("global_optimizer_class", DoNothing)
    local_optimizer_class = config.pop("local_optimizer_class")
    local_optimizer_manager_class = config.pop("local_optimizer_manager_class", AverageOptimizers)
    scheduler_class = config.pop("scheduler_class")

    local_optimizer_params = {}
    global_optimizer_params = {}
    optimizer_manager_params = {}
    scheduler_params = {}

    for key, val in config.items():
        (scope, param) = key.split(".")
        if scope == "local_opt":
            local_optimizer_params[param] = val

        elif scope == "global_opt":
            global_optimizer_params[param] = val

        elif scope == "opt_manager":
            optimizer_manager_params[param] = val

        elif scope == "scheduler":
            scheduler_params[param] = val
        else:
            raise KeyError("Wrong parameter in 'config' dict: ", scope, param)
        

    optimizer_I = Instantiator(local_optimizer_class, local_optimizer_params)
    scheduler_I = Instantiator(scheduler_class, scheduler_params)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, val_dataset = ray.get(train_data_obj_ref)

    if is_distributed:
        optimizer_manager_I = Instantiator(local_optimizer_manager_class, optimizer_manager_params)
        global_optimizer_I = Instantiator(global_optimizer_class, global_optimizer_params)

        global_model, _ = distributed_learning(train_dataset, n_epochs, n_workers, n_local_steps, 
                                               local_batch_size, optimizer_I, optimizer_manager_I, 
                                               global_optimizer_I, scheduler_I, device)
    else:
        global_model, _ = centralized_learning(train_dataset, n_epochs, local_batch_size,
                                               optimizer_I, scheduler_I, device)

    val_loss, val_acc = evaluate_model(global_model, val_dataset, device, verbose=False)

    return {"val_loss": val_loss, "val_acc": val_acc}


def custom_trial_name(trial):
    return f"trial_{trial.trial_id}"


if __name__ == "__main__":
    experiment_name = "mini_batch_adamw"
    search_space = getattr(experiments_config, experiment_name)

    experment_folder = os.path.join(os.path.abspath("Results2"))

    train_dataset, val_dataset = load_data()

    ray.init()
    train_data_obj_ref = ray.put((train_dataset, val_dataset))

    reporter = CLIReporter(metric_columns=["val_loss", "val_acc"])

    print("Cuda available:", torch.cuda.is_available())

    analysis = tune.run(
        partial(tune_distributed_learning, train_data_obj_ref=train_data_obj_ref),
        config=search_space,
        num_samples=-1,
        time_budget_s= 12 * 60 * 60,
        progress_reporter=reporter,
        storage_path=experment_folder,
        max_concurrent_trials=1,
        trial_name_creator=custom_trial_name,
        trial_dirname_creator=custom_trial_name,
        metric="val_acc",
        mode="max"
    )
    analysis.dataframe().to_pickle(os.path.join(experment_folder, f"{experiment_name}.pkl"))

    print("Best hyperparameters found: ", analysis.best_config)
    print("Best validation accuracy: ", analysis.best_result)
