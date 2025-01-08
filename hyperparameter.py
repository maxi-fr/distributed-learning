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

from distributed import distributed_learning
from model import evaluate_model, load_data
import experiments_config
import pickle


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

    global_optimizer_class = config.pop("global_optimizer_class")
    local_optimizer_class = config.pop("local_optimizer_class")
    local_optimizer_manager_class = config.pop("local_optimizer_manager_class")
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, val_dataset = ray.get(train_data_obj_ref)

    global_model = distributed_learning(train_dataset, n_epochs, n_workers, n_local_steps, local_batch_size,
                                        global_optimizer_class, global_optimizer_params,
                                        local_optimizer_manager_class, optimizer_manager_params,
                                        local_optimizer_class, local_optimizer_params,
                                        scheduler_class, scheduler_params, device)

    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate_model(global_model, val_dataset, criterion, device, verbose=False)

    return {"val_loss": val_loss, "val_acc": val_acc}


def custom_trial_name(trial):
    return f"trial_{trial.trial_id}"


if __name__ == "__main__":
    experiment_name = "large_batch_lars"
    search_space = getattr(experiments_config, experiment_name)

    experment_folder = os.path.join(os.path.abspath("ray_results"))

    train_dataset, val_dataset = load_data()

    ray.init()
    train_data_obj_ref = ray.put((train_dataset, val_dataset))

    reporter = CLIReporter(metric_columns=["val_loss", "val_acc"])

    print("Cuda available:", torch.cuda.is_available())

    analysis = tune.run(
        partial(tune_distributed_learning, train_data_obj_ref=train_data_obj_ref),
        config=search_space,
        num_samples=25,
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

