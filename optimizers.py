
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

    def __init__(self, params, lr=0.01, momentum=0, eta=1e-3, dampening=0,
                 weight_decay=0, epsilon=0):
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

            for p in group['params']:
                if p.grad is None:
                    continue

                # Compute weight- and gradient norm
                w_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # Calculate local lr
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

                # Adjust gradient further with momentum
                d_p = d_p.add(momentum, buf)

                # Update parameter
                p.data.add_(-local_lr * group['lr'], d_p)

        return loss



class Lamb(Optimizer):
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

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0, adam=False):
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
        super(Lamb, self).__init__(params, defaults)

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
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step_size = group['lr']

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss



class SlowMo(Optimizer):
    def __init__(self, params, local_lr, lr=0.01, momentum=0.9):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(local_lr=local_lr, lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            local_lr = group["local_lr"]
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['prev_param'] = p.data.clone()

                u = state['momentum_buffer']
                prev_param = state['prev_param']

                # Compute the scaled parameter difference (Δθ_t)
                param_diff = (p.data - prev_param)/local_lr

                # Update the momentum buffer: u_t = β * u_{t-1} + Δθ_t
                u.mul_(momentum).add_(param_diff)

                # Update parameters: θ_t = θ_t - lr * local_lr * u_t
                p.data.add_(u, alpha=-lr*local_lr)

                state['prev_param'].copy_(p.data)

        return loss


class DoNothing(Optimizer):

    def __init__(self, params, **defaults):
        super().__init__(params, defaults)

    def step(self, bla=None):
        pass


def average_optimizers(opts: list[Optimizer]) -> None:
    """
    Averages all state values across the given optimizers, inplace.

    Args:
        opts (list[Optimizer]): A list of PyTorch optimizers. All optimizers must have the same state structure.
    """
    if not opts:
        raise ValueError("The list of optimizers is empty.")
    
    states = [flatten_dict(opt.state) for opt in opts]


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
