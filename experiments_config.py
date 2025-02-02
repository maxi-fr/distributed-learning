import numpy as np
from ray import tune
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR
from optimizers import LAMB, LARS, DoNothing, SlowMo, WarmupCosineAnnealing, LocalAdaScale_Manager, LocalAdaScale_Optimizer

LOCAL_BATCH_SIZE = 64

OPT_SGD_LR = 0.031484	
OPT_SGD_W_DECAY = 0.002244	

OPT_SGD_NO_MOMENTUM_LR = 0.281679	
OPT_SGD_NO_MOMENTUM_W_DECAY = 0.000360

OPT_ADAMW_LR = 0.000495	
OPT_ADAMW_W_DECAY = 0.069738		

OPT_LARS_LR = 0.000526	
OPT_LAMB_LR = 0.000908	

OPT_SLOW_MOMENTUM = -1



"""
TODO: Run experiments:
- local ada scale (with and without momentum)

"""



mini_batch_sgd = {
    "test_mode": False,
    "n_epochs": 150,
    "local_batch_size": LOCAL_BATCH_SIZE,  

    "local_optimizer_class": torch.optim.SGD,
    "local_opt.lr": tune.loguniform(1e-5, 1e-1),
    "local_opt.momentum": 0.9,
    "local_opt.weight_decay": tune.loguniform(1e-5, 1e-1),

    "scheduler_class": CosineAnnealingLR,
    }

mini_batch_sgd_no_momentum = {
    "test_mode": False,
    "n_epochs": 150,
    "local_batch_size": LOCAL_BATCH_SIZE,  

    "local_optimizer_class": torch.optim.SGD,
    "local_opt.lr": tune.loguniform(1e-2, 0.9),
    "local_opt.momentum": 0.0,
    "local_opt.weight_decay": tune.loguniform(1e-4, 1e-2),

    "scheduler_class": CosineAnnealingLR,
    }

mini_batch_adamw = {
    "test_mode": False,
    "n_epochs": 150,
    "local_batch_size": LOCAL_BATCH_SIZE, 
    
    "local_optimizer_class": torch.optim.AdamW,
    "local_opt.lr": tune.loguniform(1e-4, 1e-3),
    "local_opt.weight_decay": tune.loguniform(1e-2, 0.9),

    "scheduler_class": CosineAnnealingLR,
    }


synchronous_sgd = {
    "test_mode": True,
    "n_workers": tune.grid_search([1, 2, 4, 8, 16, 32, 64]), 
    "n_epochs": 150,
    "n_local_steps": 1,
    "local_batch_size": LOCAL_BATCH_SIZE, 

    "local_optimizer_class": torch.optim.SGD,
    "local_opt.base_lr": OPT_SGD_LR,
    "local_opt.lr_scaling": tune.grid_search(("linear", "sqrt")),
    "local_opt.momentum": 0.9,
    "local_opt.weight_decay": OPT_SGD_W_DECAY,

    "global_optimizer_class": DoNothing,

    "scheduler_class": WarmupCosineAnnealing,
    "scheduler.per_warmup_epochs": 5/90,
    }

synchronous_adamw = {
    "test_mode": True,
    "n_workers": tune.grid_search([1, 2, 4, 8, 16, 32, 64]),
    "n_epochs": 150,
    "n_local_steps": 1,
    "local_batch_size": LOCAL_BATCH_SIZE, 

    "local_optimizer_class": torch.optim.AdamW,
    "local_opt.base_lr": OPT_ADAMW_LR,
    "local_opt.lr_scaling": tune.grid_search(("linear", "sqrt")),
    "local_opt.weight_decay": OPT_ADAMW_W_DECAY,

    "global_optimizer_class": DoNothing,

    "scheduler_class": WarmupCosineAnnealing,
    "scheduler.per_warmup_epochs": 5/90,
    }

lars_lr = {
    "test_mode": False,
    "n_workers": 1,
    "n_epochs": 150,
    "n_local_steps": 1,
    "local_batch_size": 64, 

    "local_optimizer_class": LARS,
    "local_opt.lr": tune.loguniform(1e-6, 1e-1),

    "global_optimizer_class": DoNothing,

    "scheduler_class": WarmupCosineAnnealing,
    "scheduler.per_warmup_epochs": 5/90,
    }

lamb_lr = {
    "test_mode": False,
    "n_workers": 1,
    "n_epochs": 150,
    "n_local_steps": 1,
    "local_batch_size": 64, 

    "local_optimizer_class": LAMB,
    "local_opt.lr": tune.loguniform(1e-6, 1e-1),

    "global_optimizer_class": DoNothing,

    "scheduler_class": WarmupCosineAnnealing,
    "scheduler.per_warmup_epochs": 5/90,
    }

lars_bat_size = {
    "test_mode": True,
    "n_workers": tune.grid_search([1, 2, 4, 8, 16]),
    "n_epochs": 150,
    "n_local_steps": 1,
    "local_batch_size": 64, 

    "local_optimizer_class": LARS,
    "local_opt.base_lr": OPT_LARS_LR, 
    "local_opt.lr_scaling": tune.grid_search(("linear", "sqrt")),

    "global_optimizer_class": DoNothing,

    "scheduler_class": WarmupCosineAnnealing,
    "scheduler.per_warmup_epochs": 5/90,
    }


lamb_bat_size = {
    "test_mode": True,
    "n_workers": tune.grid_search([1, 2, 4, 8, 16]),
    "n_epochs": 150,
    "n_local_steps": 1,
    "local_batch_size": 64, 

    "local_optimizer_class": LARS,
    "local_opt.base_lr": OPT_LAMB_LR, # OPT_LARS_BASE_LR 
    "local_opt.lr_scaling": tune.grid_search(("linear", "sqrt")),

    "global_optimizer_class": DoNothing,

    "scheduler_class": WarmupCosineAnnealing,
    "scheduler.per_warmup_epochs": 5/90,
    }


local_sgd = {
    "n_workers": tune.grid_search([1, 2, 4, 8]),
    "n_epochs": 150,
    "n_local_steps": tune.grid_search([1, 2, 4, 8]),
    "local_batch_size": LOCAL_BATCH_SIZE,

    "local_optimizer_class": torch.optim.SGD,
    "local_opt.base_lr": OPT_SGD_LR,
    "local_opt.lr_scaling": "linear",
    "local_opt.momentum": 0.9,
    "local_opt.weight_decay": OPT_SGD_W_DECAY,

    "global_optimizer_class": DoNothing,

    "scheduler_class": CosineAnnealingLR,
    }

local_adamw = {
    "n_workers": tune.grid_search([1, 2, 4, 8]),
    "n_epochs": 150,
    "n_local_steps": tune.grid_search([1, 2, 4, 8]),
    "local_batch_size": LOCAL_BATCH_SIZE, # tune.choice([32, 64, 128]), in paper [11] batch size was 64

    "local_optimizer_class": torch.optim.AdamW,
    "local_opt.base_lr": OPT_ADAMW_LR,
    "local_opt.lr_scaling": "linear",
    "local_opt.weight_decay": OPT_ADAMW_W_DECAY,

    "global_optimizer_class": DoNothing,

    "scheduler_class": CosineAnnealingLR,
    }

slow_mo_momentum = {
    "test_mode": False,
    "n_workers": 8,
    "n_epochs": 150,
    "n_local_steps": 8,
    "local_batch_size": LOCAL_BATCH_SIZE,

    "local_optimizer_class": torch.optim.SGD,
    "local_opt.base_lr": OPT_SGD_LR,
    "local_opt.lr_scaling": "linear",
    "local_opt.momentum": 0.9,
    "local_opt.weight_decay": OPT_SGD_W_DECAY,

    "global_optimizer_class": SlowMo,
    "global_opt.lr": 1.0,
    "global_opt.momentum": tune.grid_search(np.arange(0.3, 1, 0.05)),

    "scheduler_class": CosineAnnealingLR,
    }

slow_mo = {
    "test_mode": True,
    "n_workers": tune.grid_search([1, 2, 4, 8, 16]),
    "n_epochs": 150,
    "n_local_steps": tune.grid_search([1, 2, 4, 8, 16]),
    "local_batch_size": LOCAL_BATCH_SIZE,

    "local_optimizer_class": torch.optim.SGD,
    "local_opt.base_lr": OPT_SGD_LR,
    "local_opt.lr_scaling": "linear",
    "local_opt.momentum": 0.9,
    "local_opt.weight_decay": OPT_SGD_W_DECAY,

    "global_optimizer_class": SlowMo,
    "global_opt.lr": 1, 
    "global_opt.momentum": OPT_SLOW_MOMENTUM, 

    "scheduler_class": CosineAnnealingLR,
    }

local_ada_scale = {
    "n_workers": tune.grid_search([4, 8, 16]),
    "n_epochs": 150,
    "n_local_steps": tune.grid_search([1, 2, 4, 8, 16]),
    "local_batch_size": LOCAL_BATCH_SIZE,

    "local_optimizer_class": LocalAdaScale_Optimizer,
    "local_opt.lr": OPT_SGD_LR,
    "local_opt.momentum": 0.9,
    "local_opt.weight_decay": OPT_SGD_W_DECAY,

    "optimizer_manager_class": LocalAdaScale_Manager,
    
    "global_optimizer_class": DoNothing,

    "scheduler_class": CosineAnnealingLR,
    }