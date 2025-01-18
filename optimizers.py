
from torch.optim.lr_scheduler import _LRScheduler
import torch
from torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling with Momentum

    Args:
        params (iterable): Iterable of parameters to optimize
        lr (float): Global learning rate
        momentum (float): Momentum factor (default: 0).
        eta (float, optional): LARS coefficient (default: 1e-3).
        weight_decay (float, optional): Weight decay (L2) (default: 0).
        dampening (float, optional): Dampening for momentum (default: 0).
        epsilon (float, optional): Small float to prevent division by zero (default: 0).

    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=0.01, momentum=0.9, eta=1, dampening=0, weight_decay=0.0005, epsilon=1):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, eta=eta, dampening=dampening,
                        weight_decay=weight_decay, epsilon=epsilon)

        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs optimization step.

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

            p: torch.Tensor
            for p in group['params']:
                if p.grad is None:
                    continue

                # Compute weight- and gradient norm
                w_norm = torch.norm(p)
                g_norm = torch.norm(p.grad)

                # Calculate local lr
                if w_norm * g_norm > 0:
                    local_lr = eta * w_norm / (g_norm + weight_decay * w_norm + epsilon)
                else:
                    local_lr = 1

                # Adjust gradient with weight decay
                d_p = p.grad
                if weight_decay != 0:
                    d_p.add_(p, alpha=weight_decay)

                # Apply momentum
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']

                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                # Adjust gradient further with momentum
                d_p = d_p.add(buf, alpha=momentum)

                # Update parameter
                p.add_(d_p, alpha=-local_lr * group['lr'])

        return loss


class LAMB(Optimizer):
    """
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): sets trust ratio to 1, turning it into Adam
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01, adam=False):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super(LAMB, self).__init__(params, defaults)

    @torch.no_grad()
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
            for p in group['params']:
                p: torch.Tensor

                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state: dict[str, torch.Tensor] = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step_size = group['lr']

                weight_norm = torch.norm(p).clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p, alpha=group['weight_decay'])

                adam_norm = torch.norm(adam_step)
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm

                # FIXME: is it necessary to save the folling stuff in the state??
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio

                if self.adam:
                    trust_ratio = 1

                p.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss


class WarmupPolynomialDecayLR(_LRScheduler):
    """
    Custom Learning Rate Scheduler with Warmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epochs (int): Number of warmup epochs.
        total_epochs (int): Total number of training epochs.
        power (float): Power for polynomial decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, power=2.0, last_epoch=-1):
        # TODO: change to take percent value instead of total
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.power = power
        super(WarmupPolynomialDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Polynomial decay phase
            decay_factor = (1 - (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)) ** self.power
            return [base_lr * decay_factor for base_lr in self.base_lrs]


class SlowMo(Optimizer):
    def __init__(self, params, local_lr, lr=0.01, momentum=0.9):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(local_lr=local_lr, lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            local_lr = group["local_lr"]
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                p: torch.Tensor
                state: dict[str, torch.Tensor] = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['prev_param'] = p.clone().detach()

                u = state['momentum_buffer']
                prev_param = state['prev_param']

                # Compute the scaled parameter difference (Δθ_t)
                param_diff = (p - prev_param)/local_lr

                # Update the momentum buffer: u_t = β * u_{t-1} + Δθ_t
                u.mul_(momentum).add_(param_diff)

                # Update parameters: θ_t+1 <- θ_t - lr * local_lr * u_t
                prev_param.add_(u, alpha=-local_lr*lr)
                p.copy_(prev_param)

        return loss


class DoNothing(Optimizer):

    def __init__(self, params, **defaults):
        super().__init__(params, defaults)

    def step(self, bla=None):
        pass


@torch.no_grad()
def average_optimizers(opts: list[Optimizer]) -> None:
    """
    Averages all state values across the given optimizers, inplace.

    Args:
        opts (list[Optimizer]): A list of PyTorch optimizers. All optimizers must have the same state structure.
    """
    if not opts:
        raise ValueError("The list of optimizers is empty.")

    states = [flatten_dict(opt.state) for opt in opts]

    x: list[torch.Tensor]
    for x in zip(*states):
        mean = torch.mean(torch.stack(x), dim=0)

        for state in x:
            state.copy_(mean)


def flatten_dict(d: dict) -> list:
    """
    Flattens an arbitrarily nested dictionary into a list of values.

    Args:
        d (Dict[Any, Any]): The dictionary to flatten.

    Returns:
        List[Any]: A list containing all the values in the dictionary, flattened.
    """
    values = []
    for v in d.values():
        if isinstance(v, dict):
            values.extend(flatten_dict(v))
        else:
            values.append(v)
    return values
