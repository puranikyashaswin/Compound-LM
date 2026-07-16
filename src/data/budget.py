"""Token-budget sanity gate for a planned run.

A run can be perfectly healthy -- finite losses, no gradient spikes, a clean
ledger -- and still be scientifically void because it looped a small corpus
hundreds of times. That failure is invisible to every other gate in this
framework, and it is expensive: it is only discovered after the GPU hours are
spent. This computes the planned epoch count and Chinchilla-relative token
budget up front so an absurd plan can be refused before training starts.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

# Chinchilla-optimal is ~20 tokens per parameter. Real runs deviate for good
# reasons, so this is the reference point for reporting, not a hard rule.
CHINCHILLA_TOKENS_PER_PARAM = 20.0


@dataclass(frozen=True)
class BudgetReport:
    status: str
    epochs: float
    unique_tokens: int
    consumed_tokens: int
    tokens_per_param: float
    chinchilla_ratio: float
    failures: list[str]
    warnings: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_token_budget(*, unique_tokens: int, steps: int, batch_size: int,
                       sequence_length: int, n_params: int,
                       max_epochs: float = 4.0,
                       min_tokens_per_param: float = 1.0) -> BudgetReport:
    """Report whether a planned run has enough distinct data to be meaningful.

    ``max_epochs`` defaults to 4: repeating a corpus a handful of times costs
    little against fresh data, but beyond roughly that point returns collapse
    and the run measures memorization instead of learning.
    """
    if unique_tokens < 1:
        raise ValueError("unique_tokens must be positive")
    if steps < 1 or batch_size < 1 or sequence_length < 1:
        raise ValueError("steps, batch_size and sequence_length must be positive")
    if n_params < 1:
        raise ValueError("n_params must be positive")

    consumed = steps * batch_size * sequence_length
    epochs = consumed / unique_tokens
    tokens_per_param = unique_tokens / n_params
    chinchilla_ratio = tokens_per_param / CHINCHILLA_TOKENS_PER_PARAM

    failures: list[str] = []
    warnings: list[str] = []
    if epochs > max_epochs:
        failures.append(
            f"corpus_repeated_{epochs:.1f}x: {consumed:,} token-positions over only "
            f"{unique_tokens:,} unique tokens (limit {max_epochs}x). The run would measure "
            f"memorization, not learning. Enlarge the corpus to about "
            f"{int(consumed / max_epochs):,} tokens or reduce steps."
        )
    if tokens_per_param < min_tokens_per_param:
        failures.append(
            f"corpus_too_small_for_model: {tokens_per_param:.2f} unique tokens per parameter "
            f"(limit {min_tokens_per_param}). A {n_params:,}-parameter model needs roughly "
            f"{int(n_params * CHINCHILLA_TOKENS_PER_PARAM):,} tokens to be Chinchilla-optimal."
        )
    if not failures and chinchilla_ratio < 0.5:
        warnings.append(
            f"under_chinchilla: {tokens_per_param:.1f} tokens/param is "
            f"{chinchilla_ratio:.2f}x the ~{CHINCHILLA_TOKENS_PER_PARAM:.0f} reference; "
            "the model is undertrained for its size"
        )
    if not failures and epochs > 1.5:
        warnings.append(f"corpus_repeated_{epochs:.1f}x")

    status = "red" if failures else ("amber" if warnings else "green")
    return BudgetReport(status=status, epochs=epochs, unique_tokens=unique_tokens,
                        consumed_tokens=consumed, tokens_per_param=tokens_per_param,
                        chinchilla_ratio=chinchilla_ratio,
                        failures=failures, warnings=warnings)


def tokens_needed(*, n_params: int, tokens_per_param: float = CHINCHILLA_TOKENS_PER_PARAM) -> int:
    """Unique tokens a model of this size wants at the given ratio."""
    if n_params < 1:
        raise ValueError("n_params must be positive")
    return int(n_params * tokens_per_param)
