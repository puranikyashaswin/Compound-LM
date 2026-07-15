"""Muon optimizer and Muon/AdamW parameter partitioning.

Muon is intentionally isolated behind a small adapter so each experiment can
record exactly which tensors used which optimizer. PyTorch is optional until a
real training run is launched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    import torch
    from torch.optim import Optimizer
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    Optimizer = object


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for Muon")


def newton_schulz_orthogonalize(matrix, steps: int = 5):
    """Approximate the polar factor used to normalize a matrix update."""
    require_torch()
    if matrix.ndim != 2:
        raise ValueError("Muon expects a 2-D matrix")
    # Coefficients from the common quintic Newton-Schulz iteration.
    x = matrix / (matrix.norm() + 1e-7)
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.transpose(0, 1)
        transposed = True
    for _ in range(steps):
        a = x @ x.transpose(0, 1)
        x = 1.5 * x - 0.5 * (a @ x)
    return x.transpose(0, 1) if transposed else x


if torch is not None:
    class Muon(Optimizer):
        def __init__(self, params: Iterable, lr: float = 0.02, momentum: float = 0.95,
                     weight_decay: float = 0.0, ns_steps: int = 5):
            defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            loss = closure() if closure is not None else None
            for group in self.param_groups:
                lr = group["lr"]
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    if parameter.ndim != 2:
                        raise ValueError("Muon parameter groups may contain only 2-D tensors")
                    grad = parameter.grad
                    state = self.state[parameter]
                    velocity = state.setdefault("momentum_buffer", torch.zeros_like(parameter))
                    velocity.mul_(group["momentum"]).add_(grad)
                    update = newton_schulz_orthogonalize(velocity, group["ns_steps"])
                    if group["weight_decay"]:
                        parameter.mul_(1 - lr * group["weight_decay"])
                    parameter.add_(update, alpha=-lr)
            return loss


@dataclass(frozen=True)
class ParameterPartition:
    muon: tuple[str, ...]
    adamw: tuple[str, ...]


def partition_named_parameters(named_parameters: Iterable[tuple[str, object]]) -> ParameterPartition:
    """Classify only hidden 2-D matrices for Muon; all other tensors use AdamW."""
    muon, adamw = [], []
    for name, parameter in named_parameters:
        # Biases, norms, embeddings, and output heads stay on AdamW.
        is_hidden_matrix = getattr(parameter, "ndim", 0) == 2 and all(
            excluded not in name.lower() for excluded in ("embedding", "embed", "lm_head", "norm", "head")
        )
        (muon if is_hidden_matrix else adamw).append(name)
    return ParameterPartition(tuple(muon), tuple(adamw))
