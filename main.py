import os
import sys
import json
import argparse
from functools import partial

import ray
from ray import tune
from ray.tune import CLIReporter
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split
from torchvision import datasets, transforms

from experiments_config import (
    LOCAL_BATCH_SIZE, 
    OPT_SGD_LR, 
    OPT_SGD_W_DECAY, 
    OPT_SGD_NO_MOMENTUM_LR, 
    OPT_SGD_NO_MOMENTUM_W_DECAY, 
    OPT_ADAMW_LR, 
    OPT_ADAMW_W_DECAY, 
    OPT_LARS_LR, 
    OPT_LAMB_LR, 
    OPT_SLOW_MOMENTUM
)
import experiments_config

from model import load_data, evaluate_model
from hyperparameter import tune_distributed_learning, custom_trial_name
from optimizers import LAMB, LARS, DoNothing, SlowMo, WarmupCosineAnnealing
from centralized import centralized_learning
from distributed import distributed_learning
from models.plotting_metrics import plot_metrics


def run_hyperparam(args):
    experiment_name = args.dict_name
    print("Starting experiment: ", experiment_name)

    search_space: dict = getattr(experiments_config, experiment_name)
    num_samples=args.num_samples
    experment_folder = os.path.join(os.path.abspath("Results"), experiment_name)

    train_dataset, val_dataset = load_data()
    test_dataset = load_data(test_data=True)
    ray.init()
    train_data_obj_ref = ray.put((train_dataset, val_dataset, test_dataset))

    reporter = CLIReporter(metric_columns=["test_acc" if search_space.get("test_mode", True) else "val_acc"])

    print("Cuda available:", torch.cuda.is_available())

    analysis = tune.run(
        partial(tune_distributed_learning, train_data_obj_ref=train_data_obj_ref, exp_folder=experment_folder),
        config=search_space,
        num_samples=num_samples,
        # time_budget_s= 10 * 60 * 60,
        progress_reporter=reporter,
        storage_path=experment_folder,
        max_concurrent_trials=1,
        trial_name_creator=custom_trial_name,
        trial_dirname_creator=custom_trial_name,
        metric="test_acc" if search_space.get("test_mode", True) else "val_acc",
        mode="max"
    )
    print("Best hyperparameters found: ", analysis.best_config)
    print("Best validation accuracy: ", analysis.best_result)
    
def run_centralized(args):
    print("Running centralized mode with:", args)
    local_optimizer_params = {
        'lr': args.local_lr,
        'weight_decay': args.local_weight_decay,
        'momentum': args.local_momentum
    }
    # Scheduler params
    scheduler_params = {
        'per_warmup_epochs': args.per_warmup_epochs
    }
    
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    scheduler_class = eval(args.scheduler_class) if args.scheduler_class else None
    optimizer_class = eval(args.local_optimizer_class) if args.local_optimizer_class else None
    train_dataset, val_dataset = load_data()

    model, performance = centralized_learning(train_dataset, args.n_epochs, args.local_batch_size, optimizer_class, 
                                             local_optimizer_params, scheduler_class, scheduler_params, 
                                              device, True, val_dataset)
    print("Training completed.")
    test_dataset = load_data(test_data=True)

    test_acc = evaluate_model(model, test_dataset, device)

    add_on = "_crop28_no_DO_w_CJ_"
    model.save(os.path.join("models2", optimizer_class.__name__ + add_on + ".lenet"), {"test_acc": test_acc})

    performance.to_csv(os.path.join("models2", optimizer_class.__name__ + add_on + "performance.csv"))
    plot_metrics(performance, os.path.join("models2", optimizer_class.__name__ + add_on + "performance.png"))

def run_train(args):
    print("Running parallel training with:", args)
    # Local optimizer params
    local_optimizer_params = {
        'lr': args.local_lr,
        'weight_decay': args.local_weight_decay,
        'momentum': args.local_momentum
    }

    # Scheduler params
    scheduler_params = {
        'per_warmup_epochs': args.per_warmup_epochs
    }

    # Global optimizer params
    global_optimizer_params = {
        'lr': args.global_optimizer_lr,
        'momentum': args.global_optimizer_momentum
    }
    train_dataset, val_dataset = load_data()
    model, performance = distributed_learning(
        train_dataset,
        n_epochs=args.n_epochs,
        n_workers=args.n_workers,
        n_local_steps=args.n_local_steps,
        local_batch_size=args.local_batch_size,
        local_optimizer_class=eval(args.local_optimizer_class) if args.local_optimizer_class else None,
        local_optimizer_params=local_optimizer_params if local_optimizer_params else {},
        scheduler_class=eval(args.scheduler_class) if args.scheduler_class else None,
        scheduler_params=scheduler_params if scheduler_params else {},
        global_optimizer_class=eval(args.global_optimizer_class) if args.global_optimizer_class else None,
        global_optimizer_params=global_optimizer_params if global_optimizer_params else {},
        verbose=args.verbose,
        val_dataset=val_dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = load_data(test_data=True)

    name = eval(args.local_optimizer_class)
   
    test_acc = evaluate_model(model, test_dataset, device)
    model.save(os.path.join("models", name.__name__ + ".lenet"), {"test_acc": test_acc})

    performance.to_csv(os.path.join("models", name.__name__ + "performance.csv"))
    plot_metrics(performance, os.path.join("models", name.__name__ + "performance.png"))

def parse_args_with_dict():
    """Initial step: Check if a dict is provided or manually input args"""
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Script to run different modes")

    # Collect the available dicts from experiments_config
    dict_options = {name: value for name, value in experiments_config.__dict__.items() if isinstance(value, dict)}

    # Add argument to specify if dict is used
    parser.add_argument('--dict_name', type=str, help='Name of the dictionary to use', choices=dict_options.keys())
    parser.add_argument('--mode', type=str, help='Mode')

    parser.add_argument('--n_epochs', type=int, help='Number of epochs', default=150)
    parser.add_argument('--use_cuda', type=bool, help='Use CUDA for training', default=False)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=None)
    parser.add_argument('--n_workers', type=int, help='Number of workers', default=None)
    parser.add_argument('--n_local_steps', type=int, help='Number of local steps', default=None)
    parser.add_argument('--local_batch_size', type=int, help='Local batch size', default=None)
    parser.add_argument('--local_optimizer_class', type=str, help='Local optimizer class', default=None)
    parser.add_argument('--local_lr', type=float, help='local lr', default=None)
    parser.add_argument('--local_weight_decay', type=float, help='local weight decay', default=None)
    parser.add_argument('--local_momentum', type=float, help='local momentum', default=None)
    parser.add_argument('--scheduler_class', type=str, help='Scheduler class', default=None)
    parser.add_argument('--per_warmup_epochs', type=float, help='Scheduler warumup', default=0.0)
    parser.add_argument('--global_optimizer_class', type=str, help='Global optimizer class', default=None)
    parser.add_argument('--global_optimizer_lr', type=float, help='Global optimizer lr', default=0.0)
    parser.add_argument('--global_optimizer_momentum', type=float, help='Global optimizer momentum', default=0.0)
    parser.add_argument('--verbose', type=bool, help='Verbose', default=True)
    parser.add_argument('--num_samples', type=int, help='Number of Samples', default=20)
    
    args = parser.parse_args()

    
    return args
    
def main():
    # First Step: check whether to use dict or command-line args
    args = parse_args_with_dict()
    # Second Step: run the function according to the mode
    if args.mode == 'centralized':
        run_centralized(args)
    elif args.mode == 'train':
        run_train(args)
    elif args.mode == 'experiment':
        run_hyperparam(args)
    else:
        print("Use 'centralized' or 'train' or 'experiment'")


if __name__ == "__main__":
    main()
