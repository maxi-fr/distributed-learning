from collections.abc import Callable, Iterable
from typing import Any, cast, overload

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

LARS_NORM_EPSILON = 1e-8


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling with Momentum.

    Parameters
    ----------
    params : iterable
        Parameters or parameter groups to optimize.
    lr : float, default=0.01
        Global learning rate.
    momentum : float, default=0.9
        Momentum factor.
    eta : float, default=1.0
        LARS trust coefficient.
    dampening : float, default=0.0
        Dampening applied to the momentum update.
    weight_decay : float, default=0.0005
        Weight decay coefficient.
    epsilon : float, default=0.0
        Small value added to the trust-ratio denominator.

    Examples
    --------
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 0.01,
        momentum: float = 0.9,
        eta: float = 1.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0005,
        epsilon: float = 0.0,
    ) -> None:
        if lr < 0.0:
            msg = f"Invalid lr: {lr}"
            raise ValueError(msg)
        if momentum < 0.0:
            msg = f"Invalid momentum: {momentum}"
            raise ValueError(msg)
        if weight_decay < 0.0:
            msg = f"Invalid weight_decay: {weight_decay}"
            raise ValueError(msg)

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "eta": eta,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "epsilon": epsilon,
        }

        super().__init__(params, defaults)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore optimizer state from a serialized checkpoint."""
        super().__setstate__(state)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform one optimization step.

        Parameters
        ----------
        closure : callable, optional
            Function that reevaluates the model and returns its loss.

        Returns
        -------
        float or None
            Closure loss, or ``None`` when no closure is provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eta = group["eta"]
            dampening = group["dampening"]
            epsilon = group["epsilon"]

            p: torch.Tensor
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Compute weight- and gradient norm
                w_norm = torch.norm(p)
                g_norm = torch.norm(p.grad)

                # Calculate local lr
                if abs(w_norm * g_norm) > LARS_NORM_EPSILON:
                    local_lr = eta * w_norm / (g_norm + weight_decay * w_norm + epsilon)
                else:
                    local_lr = 1

                # Adjust gradient with weight decay
                d_p = p.grad
                if weight_decay != 0:
                    d_p.add_(p, alpha=weight_decay)

                # Apply momentum
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                else:
                    buf = param_state["momentum_buffer"]

                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                # Adjust gradient further with momentum
                d_p = d_p.add(buf, alpha=momentum)

                # Update parameter
                p.add_(d_p, alpha=-local_lr * group["lr"])

        return loss


class LAMB(Optimizer):
    """Layer-wise adaptive moments optimizer.

    Parameters
    ----------
    params : iterable
        Parameters or parameter groups to optimize.
    lr : float, default=1e-3
        Learning rate.
    betas : tuple of float, default=(0.9, 0.999)
        Coefficients for the first and second moment estimates.
    eps : float, default=1e-6
        Value added to the denominator for numerical stability.
    weight_decay : float, default=0.01
        Weight decay coefficient.
    adam : bool, default=False
        If true, use a trust ratio of one, equivalent to Adam scaling.
    """

    def __init__(  # noqa: PLR0913
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        *,
        adam: bool = False,
    ) -> None:

        if not lr >= 0.0:
            msg = f"Invalid learning rate: {lr}"
            raise ValueError(msg)
        if not eps >= 0.0:
            msg = f"Invalid epsilon value: {eps}"
            raise ValueError(msg)
        if not 0.0 <= betas[0] < 1.0:
            msg = f"Invalid beta parameter at index 0: {betas[0]}"
            raise ValueError(msg)
        if not 0.0 <= betas[1] < 1.0:
            msg = f"Invalid beta parameter at index 1: {betas[1]}"
            raise ValueError(msg)
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        self.adam = adam
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform one optimization step.

        Parameters
        ----------
        closure : callable, optional
            Function that reevaluates the model and returns its loss.

        Returns
        -------
        float or None
            Closure loss, or ``None`` when no closure is provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                p: torch.Tensor

                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    msg = "Lamb does not support sparse gradients, consider SparseAdam instad."
                    raise RuntimeError(msg)

                state: dict[str, torch.Tensor] = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, device=p.device)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, device=p.device)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step_size = group["lr"]

                weight_norm = torch.norm(p).clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group["eps"])
                if group["weight_decay"] != 0:
                    adam_step.add_(p, alpha=group["weight_decay"])

                adam_norm = torch.norm(adam_step)
                trust_ratio = 1 if weight_norm == 0 or adam_norm == 0 else weight_norm / adam_norm

                # state['weight_norm'] = weight_norm  # noqa: ERA001
                # state['adam_norm'] = adam_norm  # noqa: ERA001
                # state['trust_ratio'] = trust_ratio  # noqa: ERA001

                if self.adam:
                    trust_ratio = 1

                p.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss


class WarmupCosineAnnealing(_LRScheduler):
    """Cosine learning-rate schedule preceded by a linear warmup.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose learning rate is scheduled.
    per_warmup_epochs : float
        Fraction of the total schedule reserved for warmup.
    total_epochs : int
        Total number of scheduler steps.
    last_epoch : int, default=-1
        Index of the last completed scheduler step.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        per_warmup_epochs: float,
        total_epochs: int,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = int(per_warmup_epochs * total_epochs)
        self.total_epochs = total_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - self.warmup_epochs,
        )
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | torch.Tensor]:
        """Return the learning rates for the current scheduler step."""
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [cast("float", base_lr) * warmup_factor for base_lr in self.base_lrs]

        return self.scheduler.get_lr()


class SlowMo(Optimizer):
    """Apply SlowMo updates using a local optimizer's learning rate.

    Parameters
    ----------
    params : iterable
        Parameters or parameter groups to optimize.
    local_opt : Optimizer
        Local optimizer whose learning rates scale the SlowMo update.
    lr : float, default=1.0
        Global SlowMo learning rate.
    momentum : float, default=0.9
        SlowMo momentum factor.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        local_opt: Optimizer,
        lr: float = 1.0,
        momentum: float = 0.9,
    ) -> None:
        if lr <= 0.0:
            msg = f"Invalid learning rate: {lr}"
            raise ValueError(msg)
        if momentum < 0.0 or momentum >= 1.0:
            msg = f"Invalid momentum value: {momentum}"
            raise ValueError(msg)

        defaults = {"lr": lr, "momentum": momentum}
        super().__init__(params, defaults)

        self.local_opt = local_opt

        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["prev_param"] = p.clone().detach()

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Apply one SlowMo update.

        Parameters
        ----------
        closure : callable, optional
            Function that reevaluates the model and returns its loss.

        Returns
        -------
        float or None
            Closure loss, or ``None`` when no closure is provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, group_l in zip(self.param_groups, self.local_opt.param_groups, strict=False):
            local_lr = group_l["lr"]
            lr = group["lr"]
            momentum = group["momentum"]

            for p in group["params"]:
                p: torch.Tensor
                state: dict[str, torch.Tensor] = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["prev_param"] = torch.zeros_like(p)

                u = state["momentum_buffer"]
                prev_param = state["prev_param"]

                param_diff = (prev_param - p) / local_lr

                u.mul_(momentum).add_(param_diff)

                prev_param.add_(u, alpha=-local_lr * lr)
                p.copy_(prev_param)

        return loss


class DoNothing(Optimizer):
    """Optimizer that leaves parameters unchanged."""

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        **defaults: float | bool,
    ) -> None:
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Return the optional closure loss without updating parameters."""
        return closure() if closure is not None else None


class LocalAdaScaleOptimizer(Optimizer):
    """Scale local updates using gradient variance estimates.

    Parameters
    ----------
    params : iterable
        Parameters or parameter groups to optimize.
    lr : float
        Local learning rate.
    momentum : float, default=0.0
        Momentum factor.
    weight_decay : float, default=0.0
        Weight decay coefficient.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            msg = f"Invalid learning rate: {lr}"
            raise ValueError(msg)
        if momentum < 0.0:
            msg = f"Invalid momentum value: {momentum}"
            raise ValueError(msg)
        if weight_decay < 0.0:
            msg = f"Invalid momentum value: {weight_decay}"
            raise ValueError(msg)

        defaults = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        self.state: dict[torch.Tensor, dict[str, torch.Tensor]]

        self.just_synchronized = True
        self.gain_ratio: float | torch.Tensor = 1.0
        self.cache_grad: torch.Tensor | None = None

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Perform one optimization step.

        Parameters
        ----------
        closure : callable, optional
            Function that reevaluates the model and returns its loss.

        Returns
        -------
        float or None
            Closure loss, or ``None`` when no closure is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.just_synchronized:
            self.cache_grad = stacked_grad(self)
        self.just_synchronized = False

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                param: torch.Tensor
                if param.grad is None:
                    continue

                grad = param.grad

                if weight_decay > 0.0:
                    grad = grad.add(param, alpha=weight_decay)

                state = self.state[param]
                if momentum > 0:
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(grad).detach()
                    else:
                        buf = state["momentum_buffer"]

                        buf.mul_(momentum).add_(grad, alpha=-(1 - momentum) * lr)

                    param_update = buf
                else:
                    param_update = -lr * grad

                param.add_(param_update, alpha=self.gain_ratio)

        return loss


def gain_ratio(G: torch.Tensor, sigma_sq: torch.Tensor, H: int, K: int) -> torch.Tensor:
    """Estimate the local update scaling factor from gradient statistics.

    Parameters
    ----------
    G : torch.Tensor
        Estimated squared norm of the mean gradient.
    sigma_sq : torch.Tensor
        Estimated gradient variance.
    H : int
        Number of local steps per synchronization.
    K : int
        Number of workers.

    Returns
    -------
    torch.Tensor
        Estimated gain ratio.
    """
    s_over_k = sigma_sq / K

    numerator = 2 * (G + sigma_sq)
    denominator = G + s_over_k + torch.sqrt((G + s_over_k) ** 2 + torch.abs((3 * (H - 1)) * G * sigma_sq))

    return numerator / denominator


def grad_stats(gradients: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute signal and noise statistics for worker gradients.

    Parameters
    ----------
    gradients : list of torch.Tensor
        Flattened gradients from the workers.

    Returns
    -------
    tuple of torch.Tensor
        Estimated squared mean-gradient norm and gradient variance.
    """
    K = len(gradients)

    gradient_matrix = torch.vstack(gradients)  # shape (K, x)

    sq_av_g_norm = torch.norm(torch.mean(gradient_matrix, dim=0)) ** 2

    sq_g_norms = torch.norm(gradient_matrix, dim=1) ** 2

    sigma_sq = 1 / (K - 1) * (sq_g_norms.sum() - K * sq_av_g_norm)

    G = sq_av_g_norm - sigma_sq / K

    return G, sigma_sq


@torch.no_grad()
def stacked_grad(optimizer: Optimizer) -> torch.Tensor:
    """Flatten and concatenate gradients held by an optimizer.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose parameter gradients are collected.

    Returns
    -------
    torch.Tensor
        One-dimensional concatenation of all available gradients.
    """
    stacked = []
    for group in optimizer.param_groups:
        stacked.extend(param.grad.flatten() for param in group["params"] if param.grad is not None)

    return torch.hstack(stacked)


class OptimizerManager:
    """Coordinate optimizer state across distributed workers."""

    def __init__(
        self,
        optimizers: list[Optimizer],
        step_invariant_epochs: int,
        n_local_steps: int | None = None,
    ) -> None:
        self.optimizers = optimizers
        self.step_invariant_epochs = step_invariant_epochs
        self.epoch_budget = step_invariant_epochs
        self.n_local_steps = n_local_steps

        self.epoch = 0

    def step(self) -> None:
        """Update manager state before each local training round."""
        raise NotImplementedError

    def update_epoch(self) -> None:
        """Advance the manager's epoch counter."""
        self.epoch += 1


class AverageOptimizers(OptimizerManager):
    """Average the state buffers of worker optimizers."""

    @torch.no_grad()
    def step(self) -> None:
        """Average optimizer state across workers."""
        if not self.optimizers:
            msg = "The list of optimizers is empty."
            raise ValueError(msg)

        states = [flatten_dict(opt.state) for opt in self.optimizers]

        for x in zip(*states, strict=False):
            mean = torch.mean(torch.stack(x), dim=0)

            for state in x:
                state.copy_(mean)


class LocalAdaScaleManager(OptimizerManager):
    """Manage gradient statistics and update gains for LocalAdaScale workers."""

    optimizers: list[LocalAdaScaleOptimizer]

    def __init__(
        self,
        optimizers: list[LocalAdaScaleOptimizer],
        step_invariant_epochs: int,
        n_local_steps: int,
    ) -> None:
        self.optimizers = optimizers
        self.H = n_local_steps
        self.step_invariant_epochs = step_invariant_epochs
        self.epoch_budget = step_invariant_epochs

        self.gain_ratios: list[torch.Tensor | float] = []

    @torch.no_grad()
    def step(self) -> None:
        """Estimate a gain ratio and synchronize optimizer buffers."""
        opts = self.optimizers

        if not opts or opts[0].cache_grad is None:
            p = 1.0  # len(opts)
        else:
            gradients = [cast("torch.Tensor", opt.cache_grad) for opt in opts]
            G, sigma_sq = grad_stats(gradients)
            p = gain_ratio(G, sigma_sq, self.H, len(opts))

            # exp moving average
            p_minus_1 = self.gain_ratios[-1]
            p = 0.8 * p_minus_1 + 0.2 * p

        self.gain_ratios.append(p)

        for opt in opts:
            opt.just_synchronized = True
            opt.gain_ratio = p

        # average opt buffers
        states = [flatten_dict(opt.state) for opt in self.optimizers]

        for x in zip(*states, strict=False):
            mean = torch.mean(torch.stack(x), dim=0)

            for state in x:
                state.copy_(mean)

    def update_epoch(self) -> None:
        """Update the epoch budget using the mean observed gain."""
        n_workers = len(self.optimizers)

        self.epoch_budget = self.step_invariant_epochs * (
            n_workers / torch.mean(torch.as_tensor(self.gain_ratios)).item()
        )


def flatten_dict(d: dict[Any, Any]) -> list[Any]:
    """
    Flatten nested dictionary values into a list.

    Parameters
    ----------
    d : dict
        Dictionary whose nested values should be flattened.

    Returns
    -------
    list
        Values from the dictionary in traversal order.
    """
    values = []
    for v in d.values():
        if isinstance(v, dict):
            values.extend(flatten_dict(v))
        else:
            values.append(v)
    return values
