"""Depth growth: train shallow, then deepen mid-run.

The saving is direct -- a 6-layer model costs half the transformer FLOPs of a
12-layer one -- but only the *transformer* half. At `d_model=256` with a 50257
vocabulary the output head is the larger term, so growth is a weak lever there
and a much stronger one after the vocabulary is right-sized. That interaction
is the reason `scripts/cost_reduction_plan.py` reports growth's gain twice.

Two modes, and the difference is a scientific one:

``zero_init`` inserts new blocks whose residual output projections are exactly
zero. A zero-output block contributes nothing to the residual stream, so the
grown model computes *precisely* what the shallow model computed -- it passes
`assert_logit_equivalence`, which the build plan makes a hard pre-training gate
for growth. Training then moves those projections off zero.

``stack`` duplicates blocks verbatim (the GStack recipe). It is often faster to
recover, but a duplicated block is not the identity, so the grown model does
*not* reproduce the donor's outputs. It cannot pass the equivalence gate and is
refused unless explicitly acknowledged -- otherwise a growth event would
silently discard the capability already paid for.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


@dataclass(frozen=True)
class DepthGrowthReport:
    mode: str
    from_layers: int
    to_layers: int
    inserted_at: tuple[int, ...]
    function_preserving: bool

    def as_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "from_layers": self.from_layers,
                "to_layers": self.to_layers, "inserted_at": list(self.inserted_at),
                "function_preserving": self.function_preserving}


# The residual output projections. Zeroing these makes a block a no-op, because
# every block is applied as `x = x + block(norm(x))`.
_RESIDUAL_OUTPUTS = ("attn.out.weight", "mlp.2.weight", "mlp.down.weight")


def _zero_residual_outputs(block) -> None:
    zeroed = 0
    for name, parameter in block.named_parameters():
        if any(name.endswith(suffix) for suffix in _RESIDUAL_OUTPUTS):
            torch.nn.init.zeros_(parameter)
            zeroed += 1
        elif name.endswith("attn.out.bias") or name.endswith("mlp.2.bias"):
            torch.nn.init.zeros_(parameter)
    if zeroed == 0:
        raise ValueError(
            "no residual output projection found in this block; zero-init growth "
            f"cannot be proven function-preserving for it (looked for {_RESIDUAL_OUTPUTS})"
        )


def grow_depth(model, *, to_layers: int, mode: str = "zero_init"):
    """Return a deeper model built from ``model``'s trained blocks.

    New blocks are interleaved rather than appended, so the added capacity is
    distributed through the stack instead of piled on the output end.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for depth growth")
    if mode not in ("zero_init", "stack"):
        raise ValueError(f"unknown growth mode {mode!r}")

    from_layers = len(model.blocks)
    if to_layers <= from_layers:
        raise ValueError(f"to_layers ({to_layers}) must exceed current depth ({from_layers})")
    if to_layers % from_layers:
        raise ValueError(
            f"to_layers ({to_layers}) must be a multiple of current depth ({from_layers}); "
            "uneven growth has no canonical placement for the new blocks"
        )

    grown = copy.deepcopy(model)
    repeats = to_layers // from_layers
    new_blocks = []
    inserted = []
    for index, block in enumerate(grown.blocks):
        new_blocks.append(block)
        for _ in range(repeats - 1):
            clone = copy.deepcopy(block)
            if mode == "zero_init":
                _zero_residual_outputs(clone)
            inserted.append(len(new_blocks))
            new_blocks.append(clone)

    grown.blocks = nn.ModuleList(new_blocks)
    grown.config = dict(grown.config, n_layers=to_layers)
    return grown, DepthGrowthReport(mode=mode, from_layers=from_layers, to_layers=to_layers,
                                    inserted_at=tuple(inserted),
                                    function_preserving=(mode == "zero_init"))


def growth_savings(*, from_layers: int, to_layers: int, growth_fraction: float,
                   transformer_flop_share: float) -> float:
    """Cost multiplier from spending ``growth_fraction`` of training shallow.

    ``transformer_flop_share`` is the fraction of forward FLOPs in the layer
    stack -- the only part depth affects. Passing 1.0 here is the classic way
    to overstate this lever: it assumes the embedding and output head are free.
    """
    if not 0.0 <= growth_fraction <= 1.0:
        raise ValueError("growth_fraction must be in [0, 1]")
    if not 0.0 <= transformer_flop_share <= 1.0:
        raise ValueError("transformer_flop_share must be in [0, 1]")
    if from_layers >= to_layers or from_layers < 1:
        raise ValueError("from_layers must be positive and less than to_layers")

    depth_ratio = from_layers / to_layers
    # Cost relative to training at full depth throughout.
    shallow_phase = growth_fraction * (
        transformer_flop_share * depth_ratio + (1.0 - transformer_flop_share))
    full_phase = 1.0 - growth_fraction
    return 1.0 / (shallow_phase + full_phase)
