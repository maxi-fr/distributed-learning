import argparse
import logging
from functools import partial
from pathlib import Path

import ray
import torch
from ray import tune
from ray.tune import CLIReporter

from . import experiments_config, optimizers
from .centralized import centralized_learning
from .distributed import distributed_learning
from .hyperparameter import custom_trial_name, tune_distributed_learning
from .model import Instantiator, evaluate_model, load_data
from .optimizers import AverageOptimizers
from .plotting_metrics import plot_metrics

logger = logging.getLogger(__name__)


def run_hyperparam(args: argparse.Namespace) -> None:
    """Run the selected Ray Tune experiment.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line options, including the experiment name and sample
        count.
    """
    experiment_name = args.dict_name
    logger.info("Starting experiment: %s", experiment_name)
    logger.info("Cuda available: %s", torch.cuda.is_available())

    search_space: dict = getattr(experiments_config, experiment_name)
    num_samples = args.num_samples
    experiment_folder = Path("artifacts/results/runs/experiments").resolve() / experiment_name

    train_dataset, val_dataset = load_data()
    test_dataset = load_data(test_data=True)
    ray.init()
    train_data_obj_ref = ray.put((train_dataset, val_dataset, test_dataset))

    reporter = CLIReporter(metric_columns=["test_acc" if search_space.get("test_mode", True) else "val_acc"])

    tune.run(
        partial(tune_distributed_learning, train_data_obj_ref=train_data_obj_ref, exp_folder=experiment_folder),
        config=search_space,
        num_samples=num_samples,
        # time_budget_s= 10 * 60 * 60,  # noqa: ERA001
        progress_reporter=reporter,
        storage_path=str(experiment_folder),
        max_concurrent_trials=1,
        trial_name_creator=custom_trial_name,
        trial_dirname_creator=custom_trial_name,
        metric="test_acc" if search_space.get("test_mode", True) else "val_acc",
        mode="max",
    )


def run_centralized(args: argparse.Namespace) -> None:
    """Train and save a model using centralized optimization.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line training options.
    """
    logger.info("Running centralized mode with: %s", args)
    local_optimizer_params = {
        "lr": args.local_lr,
        "weight_decay": args.local_weight_decay,
        "momentum": args.local_momentum,
    }
    # Scheduler params
    scheduler_params = {"per_warmup_epochs": args.per_warmup_epochs}

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    scheduler_class = getattr(torch.optim.lr_scheduler, args.scheduler_class) if args.scheduler_class else None
    optimizer_class = getattr(torch.optim, args.local_optimizer_class, None) if args.local_optimizer_class else None
    if optimizer_class is None and args.local_optimizer_class:
        optimizer_class = getattr(optimizers, args.local_optimizer_class, None)
    if optimizer_class is None:
        msg = "A local optimizer class is required for centralized training."
        raise ValueError(msg)
    train_dataset, val_dataset = load_data()

    optimizer_i = Instantiator(optimizer_class, local_optimizer_params)
    scheduler_i = Instantiator(scheduler_class, scheduler_params) if scheduler_class is not None else None

    model, performance = centralized_learning(
        train_dataset,
        args.n_epochs,
        args.local_batch_size,
        optimizer_i,
        scheduler_I=scheduler_i,
        device=device,
        verbose=True,
        val_dataset=val_dataset,
    )
    test_dataset = load_data(test_data=True)

    test_acc = evaluate_model(model, test_dataset, device)

    add_on = "_crop28_no_DO_w_CJ_"
    output_dir = Path("artifacts/models/runs/centralized")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir / f"{optimizer_class.__name__}{add_on}.lenet"), {"test_acc": test_acc})

    performance.to_csv(output_dir / f"{optimizer_class.__name__}{add_on}performance.csv")
    plot_metrics(performance, output_dir / f"{optimizer_class.__name__}{add_on}performance.png")
    logger.info("Training completed.")


def run_train(args: argparse.Namespace) -> None:
    """Train a distributed model and save its metrics and parameters.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line training options.
    """
    logger.info("Running parallel training with: %s", args)
    # Local optimizer params
    local_optimizer_params = {
        "lr": args.local_lr,
        "weight_decay": args.local_weight_decay,
        "momentum": args.local_momentum,
    }

    # Scheduler params
    scheduler_params = {"per_warmup_epochs": args.per_warmup_epochs}

    # Global optimizer params
    global_optimizer_params = {"lr": args.global_optimizer_lr, "momentum": args.global_optimizer_momentum}

    local_optimizer_class = getattr(torch.optim, args.local_optimizer_class, None)
    if local_optimizer_class is None and args.local_optimizer_class:
        local_optimizer_class = getattr(optimizers, args.local_optimizer_class, None)
    if local_optimizer_class is None:
        msg = f"Unknown local optimizer: {args.local_optimizer_class}"
        raise ValueError(msg)
    local_optimizer_i = Instantiator(local_optimizer_class, local_optimizer_params)

    if args.scheduler_class is not None:
        scheduler_i = Instantiator(getattr(torch.optim.lr_scheduler, args.scheduler_class), scheduler_params)
    else:
        scheduler_i = None

    if args.global_optimizer_class is None:
        global_optimizer_I = None
    else:
        global_optimizer_class = getattr(torch.optim, args.global_optimizer_class, None)
        if global_optimizer_class is None:
            global_optimizer_class = getattr(optimizers, args.global_optimizer_class)
        global_optimizer_I = Instantiator(global_optimizer_class, global_optimizer_params)

    train_dataset, val_dataset = load_data()
    model, performance = distributed_learning(
        train_dataset,
        n_epochs=args.n_epochs,
        n_workers=args.n_workers,
        n_local_steps=args.n_local_steps,
        local_batch_size=args.local_batch_size,
        local_optimizer_I=local_optimizer_i,
        optimizer_manager_I=Instantiator(AverageOptimizers, {}),
        scheduler_I=scheduler_i,
        global_optimizer_I=global_optimizer_I,
        verbose=args.verbose,
        val_dataset=val_dataset,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = load_data(test_data=True)

    name = local_optimizer_class

    test_acc = evaluate_model(model, test_dataset, device)
    output_dir = Path("artifacts/models/runs/distributed")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir / f"{name.__name__}.lenet"), {"test_acc": test_acc})

    performance.to_csv(output_dir / f"{name.__name__}performance.csv")
    plot_metrics(performance, output_dir / f"{name.__name__}performance.png")


def parse_args_with_dict() -> argparse.Namespace:
    """Parse experiment selection and training options from the command line.

    Returns
    -------
    argparse.Namespace
        Parsed command-line options.
    """
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Script to run different modes")

    # Collect the available dicts from experiments_config
    dict_options = {name: value for name, value in experiments_config.__dict__.items() if isinstance(value, dict)}

    # Add argument to specify if dict is used
    parser.add_argument("--dict_name", type=str, help="Name of the dictionary to use", choices=dict_options.keys())
    parser.add_argument("--mode", choices=("centralized", "train", "experiment"), help="Training mode")

    parser.add_argument("--n_epochs", type=int, help="Number of epochs", default=150)
    parser.add_argument(
        "--use_cuda", action=argparse.BooleanOptionalAction, help="Use CUDA for training", default=False
    )
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=None)
    parser.add_argument("--n_workers", type=int, help="Number of workers", default=None)
    parser.add_argument("--n_local_steps", type=int, help="Number of local steps", default=None)
    parser.add_argument("--local_batch_size", type=int, help="Local batch size", default=None)
    parser.add_argument("--local_optimizer_class", type=str, help="Local optimizer class", default=None)
    parser.add_argument("--local_lr", type=float, help="local lr", default=None)
    parser.add_argument("--local_weight_decay", type=float, help="local weight decay", default=None)
    parser.add_argument("--local_momentum", type=float, help="local momentum", default=None)
    parser.add_argument("--scheduler_class", type=str, help="Scheduler class", default=None)
    parser.add_argument("--per_warmup_epochs", type=float, help="Scheduler warumup", default=None)
    parser.add_argument("--global_optimizer_class", type=str, help="Global optimizer class", default=None)
    parser.add_argument("--global_optimizer_lr", type=float, help="Global optimizer lr", default=0.0)
    parser.add_argument("--global_optimizer_momentum", type=float, help="Global optimizer momentum", default=0.0)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, help="Verbose", default=True)
    parser.add_argument("--num_samples", type=int, help="Number of Samples", default=20)

    return parser.parse_args()


def main() -> None:
    """Dispatch to the selected training or tuning workflow."""
    # First Step: check whether to use dict or command-line args
    args = parse_args_with_dict()
    # Second Step: run the function according to the mode
    if args.mode == "centralized":
        run_centralized(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "experiment":
        run_hyperparam(args)
    else:
        logger.warning("Use 'centralized' or 'train' or 'experiment'")
        msg = "--mode must be one of 'centralized', 'train', or 'experiment'."
        raise ValueError(msg)


if __name__ == "__main__":
    main()
