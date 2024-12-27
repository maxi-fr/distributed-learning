
import torch
from torch.optim.optimizer import Optimizer

class LARS(Optimizer):
    def __init__(self, params, t,lr=1e-3, weight_decay=0, momentum=0.9, eta=0.001, T=100):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, eta=eta, T=T, t=t)
        super().__init__(params, defaults)

    def step(self, t, closure=None):
        # Iterate over all parameter groups
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum'] 
            eta = group['eta']  # Get the LARS scaling factor
            T = group['T']  

            # Compute the global learning rate at time step t
            gamma_t = lr * (1 - t / T) ** 2

            # Iterate over all parameters in the group
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data  # Gradient data
                state = self.state[param]

                # Initialize state variables if they are not already set
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(param.data)

                # Apply weight decay (L2 regularization) to the gradient
                if weight_decay != 0:
                    grad.add_(param.data, alpha=weight_decay)

                momentum_buffer = state['momentum_buffer']
                momentum_buffer.mul_(momentum).add_(grad)

                # Compute the local LR (lambda^l)
                param_norm = param.data.norm()
                grad_norm = grad.norm()
                lambda_l = param_norm / (grad_norm + weight_decay * param_norm)

                # Update the momentum with the computed local learning rate
                momentum_buffer.mul_(momentum).add_(gamma_t * lambda_l * (grad + weight_decay * param.data))

                # Update the parameter using the momentum buffer
                param.data.add_(-momentum_buffer)

        # Return loss if closure is provided, otherwise return None
        return closure() if closure is not None else None



class LAMB(torch.optim.Optimizer):

    def __init__(self, params, defaults):
        super().__init__(params, defaults)


    def step(self):
        pass
        # update params


class SlowMo(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['prev_param'] = p.data.clone()

                buf = state['momentum_buffer']
                prev_param = state['prev_param']

                # Compute the parameter difference (Δθ_t)
                param_diff = p.data - prev_param

                # Update the momentum buffer: u_t = β * u_{t-1} + Δθ_t
                buf.mul_(momentum).add_(param_diff)

                # Update parameters: θ_t = θ_t - η * v_t
                p.data.add_(buf, alpha=-lr)

                state['prev_param'].copy_(p.data)

        return loss


class DoNothing(Optimizer):

    def __init__(self, params, defaults):
        super().__init__(params, defaults)

    def step(self, bla=None):
        pass


def average_optimizers(opts: list[Optimizer]) -> None:
    #TODO
    pass
