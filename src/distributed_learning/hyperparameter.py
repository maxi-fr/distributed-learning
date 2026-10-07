import math
from pathlib import Path
from typing import Any

import ray
import torch
from ray.tune.experiment.trial import Trial

# from ray.tune.search.variant_generator import BasicVariantGenerator  # noqa: ERA001
from .centralized import centralized_learning
from .distributed import distributed_learning
from .model import Instantiator, evaluate_model
from .optimizers import AverageOptimizers, DoNothing


def tune_distributed_learning(  # noqa: C901, PLR0912, PLR0915
    config: dict[str, Any],
    train_data_obj_ref: ray.ObjectRef,
    exp_folder: str | Path,
) -> dict[str, float]:
    """
    Run distributed learning for one Ray Tune configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Mutable Ray Tune configuration containing the hyperparameters to test.
    train_data_obj_ref : ray.ObjectRef
        Reference to the training, validation, and test datasets.
    exp_folder : str or pathlib.Path
        Ray Tune experiment directory.

    Returns
    -------
    dict[str, float]
        Loss and accuracy metrics for the selected evaluation dataset.
    """
    test_mode = config.pop("test_mode", True)
    n_epochs = config.pop("n_epochs", 150)
    n_workers = config.pop("n_workers", 1)
    local_batch_size = config.pop("local_batch_size")
    n_local_steps = config.pop("n_local_steps", 1)

    global_optimizer_class = config.pop("global_optimizer_class", DoNothing)
    local_optimizer_class = config.pop("local_optimizer_class")
    local_optimizer_manager_class = config.pop("optimizer_manager_class", AverageOptimizers)
    scheduler_class = config.pop("scheduler_class")

    local_optimizer_params = {}
    global_optimizer_params = {}
    optimizer_manager_params = {}
    scheduler_params = {}

    base_lr = config.pop("local_opt.base_lr", None)
    if base_lr is not None:
        lr_scaling = config.pop("local_opt.lr_scaling")

        # only if optimizer buffers get averaged,
        # problem for when merging with local_ada_scale_implementaition branch TODO
        # is_distributed = n_workers > 1 or (n_local_steps > 1 and issubclass(optimizer_manager, AverageOptimizers))  # noqa: ERA001

        if lr_scaling == "linear":
            scale = n_workers
        elif lr_scaling == "sqrt":
            scale = math.sqrt(n_workers)

        else:
            msg = "Scaling rule should be one of 'linear' and 'sqrt'"
            raise ValueError(msg)

        config["local_opt.lr"] = base_lr * scale

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
            msg = "Wrong parameter in 'config' dict: "
            raise KeyError(msg, scope, param)

    optimizer_I = Instantiator(local_optimizer_class, local_optimizer_params)
    scheduler_I = Instantiator(scheduler_class, scheduler_params)
    optimizer_manager_I = Instantiator(local_optimizer_manager_class, optimizer_manager_params)
    global_optimizer_I = Instantiator(global_optimizer_class, global_optimizer_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, test_dataset = ray.get(train_data_obj_ref)
    val_or_test_set = test_dataset if test_mode else val_dataset

    is_distributed = not (
        n_workers == 1 or (n_local_steps == 1 and issubclass(optimizer_manager_I.var_class, AverageOptimizers))
    )

    if is_distributed:
        model, train_performance = distributed_learning(
            train_dataset,
            n_epochs,
            n_workers,
            n_local_steps,
            local_batch_size,
            optimizer_I,
            optimizer_manager_I=optimizer_manager_I,
            global_optimizer_I=global_optimizer_I,
            scheduler_I=scheduler_I,
            device=device,
            verbose=False,
            val_dataset=val_or_test_set,
        )
    else:
        model, train_performance = centralized_learning(
            train_dataset,
            n_epochs,
            n_workers * n_local_steps * local_batch_size,
            optimizer_I,
            scheduler_I=scheduler_I,
            device=device,
            verbose=False,
            val_dataset=val_or_test_set,
        )

    experiment_path = Path(exp_folder)
    experiment_runs = [path for path in experiment_path.iterdir() if path.is_dir()]
    if not experiment_runs:
        msg = f"No experiment run directories found under {experiment_path}."
        raise FileNotFoundError(msg)
    current_run = max(experiment_runs, key=lambda path: path.name)
    trial_runs = [path for path in current_run.iterdir() if path.is_dir()]
    if not trial_runs:
        msg = f"No trial directories found under {current_run}."
        raise FileNotFoundError(msg)
    trial_path = max(trial_runs, key=lambda path: path.name)
    train_performance.to_csv(trial_path / "performance.csv")
    model.save(str(trial_path / "model.pkl"))

    if test_mode:
        test_loss, test_acc = evaluate_model(model, test_dataset, device, verbose=False)
        return {"test_loss": test_loss, "test_acc": test_acc}

    val_loss, val_acc = evaluate_model(model, val_dataset, device, verbose=False)
    return {"val_loss": val_loss, "val_acc": val_acc}


def custom_trial_name(trial: Trial) -> str:
    """Return a concise directory name for a Ray Tune trial.

    Parameters
    ----------
    trial : ray.tune.experiment.trial.Trial
        Trial whose identifier determines the directory name.

    Returns
    -------
    str
        Directory name using the trial's numeric identifier.
    """
    return f"trial_{trial.trial_id.split('_')[-1]}"
