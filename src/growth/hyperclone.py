"""Width-growth utilities with a hard donor-equivalence check."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class GrowthReport:
    status: str
    max_abs_logit_diff: float | None
    tolerance: float
    reason: str | None = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for function-preserving growth")


def assert_logit_equivalence(donor, clone, input_ids, *, tolerance: float = 1e-4) -> GrowthReport:
    """Compare donor/clone outputs without allowing training to proceed on failure."""
    require_torch()
    donor.eval(); clone.eval()
    with torch.no_grad():
        donor_logits = donor(input_ids)
        clone_logits = clone(input_ids)
    difference = float((donor_logits.float() - clone_logits.float()).abs().max().cpu())
    if difference > tolerance:
        raise ValueError(f"growth equivalence failed: max_abs_logit_diff={difference} > {tolerance}")
    return GrowthReport("pass", difference, tolerance)


def expand_config(config: dict[str, Any], *, width_multiplier: float = 2.0) -> dict[str, Any]:
    """Return a declared growth config; weight transformation is model-specific."""
    if width_multiplier <= 1:
        raise ValueError("width_multiplier must be greater than 1")
    model = dict(config.get("model", {}))
    old_width = int(model.get("d_model", 0))
    if old_width <= 0:
        raise ValueError("growth requires model.d_model")
    model["d_model"] = int(round(old_width * width_multiplier))
    result = dict(config)
    result["model"] = model
    result["growth"] = {"source_d_model": old_width, "width_multiplier": width_multiplier,
                         "equivalence_tolerance": 1e-4}
    return result
