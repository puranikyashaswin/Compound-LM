"""Architecture registry: config-selected model construction.

The build plan's model-switch procedure requires that a new architecture is a
declared lineage, never a silent code edit. This registry is the single point
where an architecture name resolves to a model class, so the trainer, the
evaluator, and every checkpoint agree on what was trained.

Checkpoints written before the registry existed carry no ``architecture`` key
in their config; they are ReferenceLM by construction and load as
``reference-v1``, preserving their lineage and hashes unchanged.
"""
from __future__ import annotations

from typing import Any

from src.model.reference import require_torch

DEFAULT_ARCHITECTURE = "reference-v1"


def available_architectures() -> tuple[str, ...]:
    return ("reference-v1", "reex-v2")


def build_model(architecture: str, *, vocab_size: int, d_model: int,
                n_layers: int, n_heads: int, max_seq_len: int):
    require_torch()
    if architecture == "reference-v1":
        from src.model.reference import ReferenceLM
        return ReferenceLM(vocab_size, d_model, n_layers, n_heads, max_seq_len)
    if architecture == "reex-v2":
        from src.model.reex import ReexLM
        return ReexLM(vocab_size, d_model, n_layers, n_heads, max_seq_len)
    raise ValueError(
        f"unknown architecture {architecture!r}; available: {available_architectures()}"
    )


def model_from_config(config: dict[str, Any]):
    """Rebuild the model a checkpoint's config describes."""
    return build_model(config.get("architecture", DEFAULT_ARCHITECTURE),
                       vocab_size=config["vocab_size"], d_model=config["d_model"],
                       n_layers=config["n_layers"], n_heads=config["n_heads"],
                       max_seq_len=config["max_seq_len"])
