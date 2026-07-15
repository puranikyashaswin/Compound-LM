"""Small, dependency-free checkpoint health policy."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any


class RollingMedian:
    def __init__(self, window: int = 5):
        self.window = window
        self.history: list[float] = []

    def add(self, value: float) -> None:
        self.history.append(value)
        if len(self.history) > self.window:
            self.history.pop(0)

    def median(self) -> float:
        if not self.history:
            return 0.0
        sorted_history = sorted(self.history)
        n = len(sorted_history)
        if n % 2 == 1:
            return sorted_history[n // 2]
        else:
            return (sorted_history[n // 2 - 1] + sorted_history[n // 2]) / 2.0

    def is_spike(self, value: float, multiplier: float = 3.0) -> bool:
        m = self.median()
        if m <= 0:
            return False
        return value > multiplier * m and value > 0.01


@dataclass
class HealthReport:
    status: str
    failures: list[str]
    warnings: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_checkpoint(
    *, loss: float, grad_norm: float, median_grad_norm: float,
    finite: bool = True, checkpoint_hash: str | None = None,
    provenance_ok: bool = True, throughput: float | None = None,
    reference_throughput: float | None = None,
) -> HealthReport:
    failures: list[str] = []
    warnings: list[str] = []
    if not finite or not math.isfinite(loss) or not math.isfinite(grad_norm):
        failures.append("non_finite_tensor_or_metric")
    if median_grad_norm > 0 and grad_norm > 3.0 * median_grad_norm and grad_norm > 0.01:
        failures.append("gradient_norm_spike")
    if checkpoint_hash is None:
        failures.append("missing_checkpoint_hash")
    if not provenance_ok:
        failures.append("invalid_provenance")
    if throughput and reference_throughput and throughput < 0.9 * reference_throughput:
        warnings.append("throughput_drop")
    status = "red" if failures else ("amber" if warnings else "green")
    return HealthReport(status, failures, warnings)
