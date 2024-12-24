
import torch
from torch.optim.optimizer import Optimizer

class LARS(Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0, momentum=0.9, eta=0.001):# Maybe change to not set values
        # Initialize the optimizer with learning rate, weight decay, momentum, and LARS scaling factor (eta)
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, eta=eta)
        super().__init__(params, defaults)

    def step(self, closure=None):
        # Iterate over all parameter groups
        for group in self.param_groups:
            lr = group['lr']  # Get the learning rate for this parameter group
            weight_decay = group['weight_decay']  # Get the weight decay (L2 regularization)
            momentum = group['momentum']  # Get the momentum
            eta = group['eta']  # Get the LARS scaling factor

            # Iterate over all parameters in the group
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data # Gradient data
                state = self.state[param]

                # Initialize state variables if they are not already set
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(param.data)

                # Apply weight decay (L2 regularization) to the gradient
                if weight_decay != 0:
                    grad.add_(param.data, alpha=weight_decay)

                momentum_buffer = state['momentum_buffer']
                momentum_buffer.mul_(momentum).add_(grad)

                # Calculate the LARS scaling factor based on the norms of the parameter and gradient
                param_norm = param.data.norm()
                grad_norm = grad.norm()
                lars_factor = eta * param_norm / (grad_norm + 1e-8)  # LARS scaling factor with epsilon to avoid division by zero
                
                # Update the parameter using the momentum buffer and the LARS scaling factor
                param.data.add_(-lr * lars_factor, momentum_buffer)

        # Return loss if closure is provided, otherwise return None
        return loss if closure is not None else None


class LAMB(torch.optim.Optimizer):

    def __init__(self, params, defaults):
        super().__init__(params, defaults)


    def step(self):
        pass
        # update params