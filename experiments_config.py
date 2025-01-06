from ray import tune
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR
from optimizers import LAMB, LARS, DoNothing, SlowMo

N_WORKERS = 8
N_LOCAL_STEPS = 8
LOCAL_BATCH_SIZE = 64
OPT_SGD_LR = 0.039576
OPT_SGD_W_DECAY = 0.002376
OPT_ADAMW_LR = 1.881256e-05
OPT_ADAMW_W_DECAY = 0.000186

play_araound = {
    "n_workers": 1,
    "n_epochs": 1,
    "n_local_steps": 1,
    "local_batch_size": 50,

    "local_optimizer_class": torch.optim.SGD,
    "local_opt.lr": tune.loguniform(1e-4, 1e-1),
    "local_opt.momentum": tune.uniform(0.5, 0.9),

    "global_optimizer_class": SlowMo,
    "global_opt.lr": tune.loguniform(1e-4, 1e-1),
    "global_opt.momentum": tune.uniform(0.8, 0.95),

    "scheduler_class": CosineAnnealingLR,
    "scheduler.T_max": 150
    }

mini_batch_sgd = {
    "n_workers": 1,
    "n_epochs": 150,
    "n_local_steps": N_WORKERS * N_LOCAL_STEPS, # for n_workers = 1 doesn't change anything apart from less calls to averaging functions
    "local_batch_size": LOCAL_BATCH_SIZE, # tune.choice([32, 64, 128]), in paper [11] batch size was 64

    "local_optimizer_class": torch.optim.SGD,
    "local_opt.lr": tune.loguniform(1e-6, 1e-1),
    "local_opt.momentum": 0.9,
    "local_opt.weight_decay": tune.loguniform(1e-6, 1e-1),

    "global_optimizer_class": DoNothing,

    "scheduler_class": CosineAnnealingLR,
    "scheduler.T_max": 150,
    "scheduler.eta_min": 1e-7
    }

mini_batch_adamw = {
    "n_workers": 1,
    "n_epochs": 150,
    "n_local_steps": N_WORKERS * N_LOCAL_STEPS, # for n_workers = 1 doesn't change anything apart from less calls to averaging functions
    "local_batch_size": LOCAL_BATCH_SIZE, # tune.choice([32, 64, 128]), in paper [11] batch size was 64

    "local_optimizer_class": torch.optim.AdamW,
    "local_opt.lr": tune.loguniform(1e-7, 1e-1),
    "local_opt.weight_decay": tune.loguniform(1e-6, 1e-1),

    "global_optimizer_class": DoNothing,

    "scheduler_class": CosineAnnealingLR,
    "scheduler.eta_min": 1e-7
    }

local_sgd = {
    "n_workers": tune.grid_search([2, 4, 8]),
    "n_epochs": 150,
    "n_local_steps": tune.grid_search([4, 8, 16, 32, 64]),
    "local_batch_size": LOCAL_BATCH_SIZE, # tune.choice([32, 64, 128]), in paper [11] batch size was 64

    "local_optimizer_class": torch.optim.SGD,
    "local_opt.lr": OPT_SGD_LR,
    "local_opt.momentum": 0.9,
    "local_opt.weight_decay": OPT_SGD_W_DECAY,

    "global_optimizer_class": DoNothing,

    "scheduler_class": CosineAnnealingLR,
    "scheduler.eta_min": 1e-6
    }

local_adamw = {
    "n_workers": tune.grid_search([2, 4, 8]),
    "n_epochs": 150,
    "n_local_steps": tune.grid_search([4, 8, 16, 32, 64]),
    "local_batch_size": LOCAL_BATCH_SIZE, # tune.choice([32, 64, 128]), in paper [11] batch size was 64

    "local_optimizer_class": torch.optim.AdamW,
    "local_opt.lr": OPT_ADAMW_LR,
    "local_opt.weight_decay": OPT_ADAMW_W_DECAY,

    "global_optimizer_class": DoNothing,

    "scheduler_class": CosineAnnealingLR,
    "scheduler.eta_min": 1e-7
    }

large_batch_lars = {
    "n_workers": tune.choice([1, 2, 4, 8, 16, 32]),
    "n_epochs": 150,
    "n_local_steps": 1,
    "local_batch_size": tune.choice([64, 128, 256]), 

    "local_optimizer_class": LARS,
    "local_opt.lr": tune.loguniform(1e-5, 1e-1),

    "global_optimizer_class": DoNothing,

    "scheduler_class": PolynomialLR,
    "scheduler.power": 2
    }

large_batch_lamb = {
    "n_workers": 1,
    "n_epochs": 150,
    "n_local_steps": 1,
    "local_batch_size": tune.grid_search([512, 1024, 2048, 4096]), 

    "local_optimizer_class": LARS,
    "local_opt.lr": tune.loguniform(1e-5, 1e-1),

    "global_optimizer_class": DoNothing,

    "scheduler_class": PolynomialLR,
    "scheduler.power": 2
    }


