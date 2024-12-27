from ray import tune
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from optimizers import LAMB, LARS, DoNothing, SlowMo



play_araound = {
    "n_workers": tune.choice([2, 4, 8]),
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

mini_batch_sdg = {
    "n_workers": 1,
    "n_epochs": 150,
    "n_local_steps": 1000,
    "local_batch_size": 64, # tune.choice([32, 64, 128]), in paper [11] batch size was 64

    "local_optimizer_class": torch.optim.SGD,
    "local_opt.lr": tune.loguniform(1e-5, 1e-1),
    "local_opt.momentum": 0,

    "global_optimizer_class": DoNothing,

    "scheduler_class": CosineAnnealingLR,
    "scheduler.T_max": 150,
    "scheduler.eta_min": 1e-6
    }

local_sgdw = {
    "n_workers": tune.grid_search([2, 4, 8, 16]),
    "n_epochs": 150,
    "n_local_steps": tune.choice(range(3, 10)),
    "local_batch_size": 64, # tune.choice([32, 64, 128]), in paper [11] batch size was 64

    "local_optimizer_class": torch.optim.SGD,
    "local_opt.lr": tune.loguniform(1e-5, 1e-1),
    "local_opt.momentum": 0.9,
    "local_opt.weight_decay": tune.loguniform(1e-5, 1e-2),

    "global_optimizer_class": DoNothing,

    "scheduler_class": CosineAnnealingLR,
    "scheduler.T_max": 150,
    "scheduler.eta_min": 1e-6
    }

local_adamw = {
    "n_workers": tune.grid_search([2, 4, 8, 16]),
    "n_epochs": 150,
    "n_local_steps": tune.choice(range(3, 10)),
    "local_batch_size": 64, # tune.choice([32, 64, 128]), in paper [11] batch size was 64

    "local_optimizer_class": torch.optim.AdamW,
    "local_opt.lr": tune.loguniform(1e-5, 1e-1),
    "local_opt.momentum": 0.9,
    "local_opt.weight_decay": tune.loguniform(1e-5, 1e-2),

    "global_optimizer_class": DoNothing,

    "scheduler_class": CosineAnnealingLR,
    "scheduler.T_max": 150,
    "scheduler.eta_min": 1e-6
    }

large_batch_lars = {
    "n_workers": tune.grid_search([2, 4, 8, 16]),
    "n_epochs": 150,
    "n_local_steps": tune.choice(range(3, 10)),
    "local_batch_size": 64, # tune.choice([32, 64, 128]), in paper [11] batch size was 64

    "local_optimizer_class": LARS,
    "local_opt.lr": tune.loguniform(1e-5, 1e-1),
    "local_opt.momentum": 0.9,
    "local_opt.weight_decay": tune.loguniform(1e-5, 1e-2),

    "global_optimizer_class": DoNothing,

    "scheduler_class": CosineAnnealingLR,
    "scheduler.T_max": 150,
    "scheduler.eta_min": 1e-6
    }

large_batch_lamb = {
    "n_workers": tune.grid_search([2, 4, 8, 16]),
    "n_epochs": 150,
    "n_local_steps": tune.choice(range(3, 10)),
    "local_batch_size": 64, # tune.choice([32, 64, 128]), in paper [11] batch size was 64

    "local_optimizer_class": LAMB,
    "local_opt.lr": tune.loguniform(1e-5, 1e-1),
    "local_opt.momentum": 0.9,
    "local_opt.weight_decay": tune.loguniform(1e-5, 1e-2),

    "global_optimizer_class": DoNothing,

    "scheduler_class": CosineAnnealingLR,
    "scheduler.T_max": 150,
    "scheduler.eta_min": 1e-6
    }


