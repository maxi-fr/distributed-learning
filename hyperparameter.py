from functools import partial
import math
import os
import sys
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
from model import evaluate_model, load_data
import experiments_config as experiments_config
import pickle

from optimizers import DoNothing


def tune_distributed_learning(config: dict[str, float | int], train_data_obj_ref, exp_folder):
    """
    Wrapper for distributed learning to enable Ray Tune hyperparameter tuning.

    Args:
        config (dict): Configuration containing hyperparameters to test.

    Returns:
        None
    """
    test_mode = config.pop("test_mode", True)
    n_epochs = config.pop("n_epochs", 150)
    n_workers = config.pop("n_workers", 1)
    local_batch_size = config.pop("local_batch_size")
    n_local_steps = config.pop("n_local_steps", 1)

    is_distributed = n_workers > 1 and n_local_steps > 1 
    # only if optimizer buffers get averaged, 
    # problem for when merging with local_ada_scale_implementaition branch TODO
    # is_distributed = n_workers > 1 or (n_local_steps > 1 and issubclass(optimizer_manager, AverageOptimizers))

    global_optimizer_class = config.pop("global_optimizer_class", DoNothing)
    local_optimizer_class = config.pop("local_optimizer_class")
    scheduler_class = config.pop("scheduler_class")

    local_optimizer_params = {}
    global_optimizer_params = {}
    scheduler_params = {}

    base_lr = config.pop("local_opt.base_lr", None)
    if base_lr is not None:
        lr_scaling = config.pop("local_opt.lr_scaling")

        if lr_scaling == "linear":
            scale = n_workers
        elif lr_scaling == "sqrt":
            scale = math.sqrt(n_workers)

            # n_epochs *= n_workers / scale # make n_epoch "scale invariant"
            # FIXME: mit oder ohne scale inveriant epochs??

        else:
            raise ValueError("Scaling rule should be one of 'linear' and 'sqrt'")

        config["local_opt.lr"] = base_lr * scale


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

    train_dataset, val_dataset, test_dataset = ray.get(train_data_obj_ref)
    val_or_test_set = test_dataset if test_mode else val_dataset 

    if is_distributed:
        model, train_performance = distributed_learning(train_dataset, n_epochs, n_workers, n_local_steps, local_batch_size,
                                            global_optimizer_class, global_optimizer_params,
                                            local_optimizer_class, local_optimizer_params,
                                            scheduler_class, scheduler_params, device, False, val_or_test_set)
    else:
        model, train_performance = centralized_learning(train_dataset, n_epochs, n_workers * local_batch_size, 
                                               local_optimizer_class, local_optimizer_params,
                                               scheduler_class, scheduler_params, device, False, val_or_test_set)
    
    exp_sub_folder = max([f for f in os.listdir(exp_folder) if os.path.isdir(os.path.join(exp_folder, f))])
    exp_folder = os.path.join(exp_folder, exp_sub_folder)
    curr_trial_folder = max([f for f in os.listdir(exp_folder) if os.path.isdir(os.path.join(exp_folder, f))])
    train_performance.to_csv(os.path.join(exp_folder, curr_trial_folder, "performance.csv"))
    model.save(os.path.join(exp_folder, curr_trial_folder, "model.pkl"))

    if test_mode:
        test_loss, test_acc = evaluate_model(model, test_dataset, device, verbose=False)
        return {"test_loss": test_loss, "test_acc": test_acc}
    
    val_loss, val_acc = evaluate_model(model, val_dataset, device, verbose=False)
    return {"val_loss": val_loss, "val_acc": val_acc}


def custom_trial_name(trial):
    return f"trial_{trial.trial_id.split('_')[-1]}"


if __name__ == "__main__":
    if len(sys.argv) == 2:
        experiment_name = sys.argv[1]
    else:
        raise Exception("set experiment name through CLI")

    print("Starting experiment: ", experiment_name)
    search_space: dict = getattr(experiments_config, experiment_name)

    experment_folder = os.path.join(os.path.abspath("Results2"), experiment_name)

    train_dataset, val_dataset = load_data()
    test_dataset = load_data(test_data=True)

    ray.init()
    train_data_obj_ref = ray.put((train_dataset, val_dataset, test_dataset))

    reporter = CLIReporter(metric_columns=["test_acc" if search_space.get("test_mode", True) else "val_acc"])

    print("Cuda available:", torch.cuda.is_available())

    analysis = tune.run(
        partial(tune_distributed_learning, train_data_obj_ref=train_data_obj_ref, exp_folder=experment_folder),
        config=search_space,
        num_samples=20,
        # time_budget_s= 10 * 60 * 60,
        progress_reporter=reporter,
        storage_path=experment_folder,
        max_concurrent_trials=1,
        trial_name_creator=custom_trial_name,
        trial_dirname_creator=custom_trial_name,
        metric="test_acc" if search_space.get("test_mode", True) else "val_acc",
        mode="max"
    ) 
    #analysis.dataframe().to_pickle(os.path.join(experment_folder, f"{experiment_name}.pkl"))

    print("Best hyperparameters found: ", analysis.best_config)
    print("Best validation accuracy: ", analysis.best_result)

