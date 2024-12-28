
import torch
from torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling with Momentum

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Global learning rate.
        momentum (float): Momentum factor (default: 0).
        eta (float, optional): LARS coefficient as used in the paper (default: 1e-3).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0).
        dampening (float, optional): Dampening for momentum (default: 0).
        epsilon (float, optional): Small value to prevent division by zero (default: 0).

    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=required, momentum=0, eta=1e-3, dampening=0,
                 weight_decay=0, epsilon=0):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, eta=eta, dampening=dampening,
                        weight_decay=weight_decay, epsilon=epsilon)
        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            dampening = group['dampening']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Compute weight norm and gradient norm
                w_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # Calculate local learning rate for layer
                if w_norm * g_norm > 0:
                    local_lr = eta * w_norm / (g_norm + weight_decay * w_norm + epsilon)
                else:
                    local_lr = 1

                # Adjust gradient with weight decay
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                # Apply momentum
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(1 - dampening, d_p)

                # Use advanced momentum to adjust gradient further
                d_p = d_p.add(momentum, buf)

                # Update the parameter with the computed step
                p.data.add_(-local_lr * group['lr'], d_p)

        return loss



class LAMB(Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.0, beta1=0.9, beta2=0.999, eps=1e-6):
        """
        LAMB optimizer implementation.
        Args:
            params (iterable): Parameters to optimize.
            lr (float): Learning rate.
            weight_decay (float): Weight decay (L2 penalty).
            beta1 (float): Exponential decay rate for first moment estimates.
            beta2 (float): Exponential decay rate for second moment estimates.
            eps (float): Term added to the denominator to improve numerical stability.
        """
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        Returns:
            Loss value if closure is provided, otherwise None.
        """
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data
                state = self.state[param]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(param.data)  # First moment
                    state['exp_avg_sq'] = torch.zeros_like(param.data)  # Second moment

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                # Apply weight decay (if specified)
                if weight_decay != 0:
                    grad = grad.add(param.data, alpha=weight_decay)

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected moments
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                # Compute LAMB step
                r1 = param.data.norm()
                r2 = corrected_exp_avg.norm() / (corrected_exp_avg_sq.sqrt() + eps)
                trust_ratio = r1 / r2 if r1 > 0 and r2 > 0 else 1.0

                # Update parameters
                step_size = lr * (trust_ratio.item() if isinstance(trust_ratio, torch.Tensor) else trust_ratio)
                param.data.add_(corrected_exp_avg, alpha=-step_size)


        return loss



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
                p.data.add_(-lr, buf)

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
