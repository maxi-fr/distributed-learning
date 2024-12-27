from functools import partial
import os
import ray
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split, DataLoader

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from typing import Type

from distributed import distributed_learning
from model import evaluate_model
from optimizers import SlowMo


def load_data(data_dir=None, random_seed=69):
    if data_dir is None:
        data_dir = os.path.abspath("./data")

    if random_seed is not None:
        random_seed = torch.Generator().manual_seed(random_seed)


    tran = transforms.Compose((transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))))
    full_train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=tran)

    train_size = int(0.8 * len(full_train_dataset))  # 80% for training
    val_size = len(full_train_dataset) - train_size  # 20% for validation


    train_dataset, val_dataset = random_split(full_train_dataset, (train_size, val_size), random_seed)

    return train_dataset, val_dataset


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
    scheduler_class = config.pop("scheduler_class")

    local_optimizer_params = {}
    global_optimizer_params = {}
    scheduler_params = {}

    for key, val in config.items():
        (scope, param) = key.split(".")
        if scope == "local_opt":
            local_optimizer_params[param] = val

        elif scope == "global_opt":
            global_optimizer_params[param] = val

        elif scope == "scheduler":
            scheduler_params[param] = val
        else:
            raise KeyError("Wrong parameter in 'config' dict: ", scope, param)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, val_dataset = ray.get(train_data_obj_ref)

    global_model = distributed_learning(train_dataset, n_epochs, n_workers, n_local_steps, local_batch_size,
                                        global_optimizer_class, global_optimizer_params,
                                        local_optimizer_class, local_optimizer_params,
                                        scheduler_class, scheduler_params, device)

    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate_model(global_model, val_dataset, criterion, device, verbose=False)

    return {"val_loss": val_loss, "val_acc": val_acc}



# Define the hyperparameter search space
def get_hyperparameter_search_space():
    """
    Returns the hyperparameter search space for Ray Tune.
    """
    return {
        "n_workers": tune.grid_search([2, 4, 8]),
        "n_epochs": 1,
        "n_local_steps": tune.choice([1, 5, 10]),
        "local_batch_size": tune.choice([32, 64, 128]),

        "local_optimizer_class": torch.optim.SGD,
        "local_opt.lr": tune.loguniform(1e-4, 1e-1),
        "local_opt.momentum": tune.uniform(0.5, 0.9),

        "global_optimizer_class": SlowMo,
        "global_opt.lr": tune.loguniform(1e-4, 1e-1),
        "global_opt.momentum": tune.uniform(0.8, 0.95),

        "scheduler_class": CosineAnnealingLR,
        "scheduler.T_max": 150
        }

def custom_trial_name(trial):
    return f"trial_{trial.trial_id}"

# Main entry point
if __name__ == "__main__":

    search_space = get_hyperparameter_search_space()

    train_dataset, val_dataset = load_data()

    ray.init()
    train_data_obj_ref = ray.put((train_dataset, val_dataset))

    reporter = CLIReporter(
        metric_columns=["val_loss", "val_acc", "training_iteration"]
    )

    analysis = tune.run(
        partial(tune_distributed_learning, train_data_obj_ref=train_data_obj_ref),
        config=search_space,
        num_samples=2, 
        progress_reporter=reporter,
        storage_path=os.path.abspath("ray_results"),  
        max_concurrent_trials=1,
        trial_dirname_creator=custom_trial_name
    )

    # Print the best hyperparameters
    print("Best hyperparameters found: ", analysis.get_best_config("val_acc"))
    print("Best validation accuracy: ", analysis.best_result["val_acc"])