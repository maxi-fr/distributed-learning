# Tutorial on Usage of Project-Code

This repository provides a framework for the project-code for DAAI 2024/25 distributed learning. The code can be run in the different modes utilizing the powershell. The mode is choosen with --mode and "train" "experiment" or "centralized"

## Modes
- **Centralized Training**: Run training where a single machine processes the data.
- **Experimentsg**: Use multiple workers to train the model in parallel using prefedfined configs. The hyperparameters are tuned/ found by ray
- **Training**: Trains a model based on parameters given over the shell

Examples usage for the Shell: your-path>python main.py --mode centralized --n_epochs 150 --use_cuda False --learning_rate 0.001 --n_workers 4 --n_local_steps 10 --local_batch_size 16 --local_optimizer_class 'torch.optim.SGD' --local_lr 0.01 --local_weight_decay 0.0001 --local_momentum 0.9  --scheduler_class 'WarmupCosineAnnealing' --per_warmup_epochs 5 --global_optimizer_class 'DoNothing' --global_optimizer_lr 0.001 --global_optimizer_momentum 0.9 --verbose True

yourpath python main.py --mode experiment --dict_name mini_batch_sgd  --num_samples 20
## Requirements

- Python 3.x
- Ray
- PyTorch
- TorchVision
- CUDA (for GPU acceleration)

