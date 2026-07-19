"""Where a run's compute and dollars actually go, before it is launched.

The `6 * N * tokens` rule the ledger uses for training FLOPs is fine for
comparing two runs of the *same* shape, but it hides which part of the model is
spending the budget -- and at small scale the answer is surprising: with a
50257-entry vocabulary and d_model=256, the tied output head is ~58% of forward
FLOPs. Optimizing the transformer stack while ignoring that is optimizing the
minority of the cost.

This module separates the two, so a cost-reduction plan can be argued from
arithmetic rather than intuition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FlopBreakdown:
    params_total: int
    params_embedding: int
    params_transformer: int
    fwd_flops_per_token_transformer: int
    fwd_flops_per_token_head: int
    fwd_flops_per_token_attention: int
    head_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fwd_flops_per_token(self) -> int:
        return (self.fwd_flops_per_token_transformer + self.fwd_flops_per_token_head
                + self.fwd_flops_per_token_attention)

    def train_flops(self, tokens: int) -> float:
        """Forward + backward ~= 3x forward."""
        return 3.0 * self.fwd_flops_per_token * tokens


def analyze_model(*, vocab_size: int, d_model: int, n_layers: int,
                  sequence_length: int, mlp_ratio: float = 4.0) -> FlopBreakdown:
    """Split a decoder's cost into transformer, attention, and output head.

    ``mlp_ratio`` is 4 for the reference GELU MLP; SwiGLU's three matrices at
    8/3 width come to the same 8*d^2, so the default covers both architectures.
    """
    if min(vocab_size, d_model, n_layers, sequence_length) < 1:
        raise ValueError("model dimensions must be positive")

    attn_params = 4 * d_model * d_model
    mlp_params = int(2 * mlp_ratio * d_model * d_model)
    per_layer = attn_params + mlp_params
    params_transformer = per_layer * n_layers
    params_embedding = vocab_size * d_model

    fwd_transformer = 2 * params_transformer
    fwd_head = 2 * d_model * vocab_size
    # Attention score/value matmuls scale with sequence length, not parameters,
    # and are the term the 6ND rule omits entirely.
    fwd_attention = 4 * n_layers * sequence_length * d_model

    total_fwd = fwd_transformer + fwd_head + fwd_attention
    return FlopBreakdown(
        params_total=params_transformer + params_embedding,
        params_embedding=params_embedding,
        params_transformer=params_transformer,
        fwd_flops_per_token_transformer=fwd_transformer,
        fwd_flops_per_token_head=fwd_head,
        fwd_flops_per_token_attention=fwd_attention,
        head_fraction=fwd_head / total_fwd,
    )


def wall_clock_multiplier(*, flop_multiplier: float, step_cost_ratio: float) -> float:
    """Convert a FLOP-based multiplier into a wall-clock one.

    The ledger prices a run at ``6 * N * tokens``, which counts the model's
    arithmetic and nothing else. An optimizer that does extra work *per step*
    is therefore invisible to it: Muon's Newton-Schulz orthogonalization was
    measured at 1.39x the step cost of AdamW (``scripts/verify_levers.py``),
    so a 1.82x FLOP win is only a 1.31x wall-clock win.

    ``step_cost_ratio`` is the recipe's seconds-per-step divided by the
    baseline's. Above 1.0 means the recipe's steps are more expensive, and the
    FLOP multiplier overstates the real saving.
    """
    if flop_multiplier <= 0:
        raise ValueError("flop_multiplier must be positive")
    if step_cost_ratio <= 0:
        raise ValueError("step_cost_ratio must be positive")
    return flop_multiplier / step_cost_ratio


def vocab_resize_multiplier(*, baseline: FlopBreakdown, new_vocab: int,
                            d_model: int, tokenization_penalty: float) -> float:
    """Compute saving from a smaller vocabulary, net of worse compression.

    A smaller vocabulary shrinks the output head but splits text into more
    tokens, so the same document costs more tokens to read. ``tokenization_penalty``
    is that ratio (e.g. 1.10 = 10% more tokens for the same text); ignoring it
    is the standard way this lever gets overstated.
    """
    if tokenization_penalty < 1.0:
        raise ValueError(
            "tokenization_penalty must be >= 1.0: a smaller vocabulary cannot "
            "compress text better than a larger one built on the same corpus"
        )
    new_fwd = (baseline.fwd_flops_per_token_transformer + 2 * d_model * new_vocab
               + baseline.fwd_flops_per_token_attention)
    per_token_gain = baseline.fwd_flops_per_token / new_fwd
    return per_token_gain / tokenization_penalty
