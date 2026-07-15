"""Compute capability-at-cost and lever interaction metrics."""
from __future__ import annotations

from typing import Any
from collections import defaultdict


def cost_to_score(checkpoints: list[dict[str, float]], target: float) -> float | None:
    """Linearly interpolate cost at a target score; never extrapolate."""
    points = sorted((float(row["cost"]), float(row["score"])) for row in checkpoints)
    if not points or target < min(score for _, score in points) or target > max(score for _, score in points):
        return None
    for (cost_a, score_a), (cost_b, score_b) in zip(points, points[1:]):
        if score_a == target:
            return cost_a
        if min(score_a, score_b) <= target <= max(score_a, score_b):
            if score_b == score_a:
                return min(cost_a, cost_b)
            return cost_a + (target - score_a) * (cost_b - cost_a) / (score_b - score_a)
    return points[-1][0] if points[-1][1] == target else None


def compounding_report(rows: list[dict[str, Any]], *, target_score: float) -> dict[str, Any]:
    """Compare isolated multipliers with compound multipliers.

    Each row must provide ``name``, ``levers``, ``baseline_cost``, and
    ``recipe_cost`` at the common target score.
    """
    # Enforce replication: must have at least two distinct seeds for every configuration of levers
    # only if seed information is present in the inputs.
    if any("seed" in row for row in rows):
        by_levers = defaultdict(set)
        for row in rows:
            levers_key = tuple(sorted(row.get("levers", [])))
            by_levers[levers_key].add(row.get("seed"))

        for levers_key, seeds in by_levers.items():
            if len(seeds) < 2:
                levers_str = ", ".join(levers_key) or "baseline"
                raise ValueError(
                    f"lever configuration '{levers_str}' must have at least two distinct seeds, but has {len(seeds)}"
                )

    baseline = next((row for row in rows if not row.get("levers")), None)
    if baseline is None:
        raise ValueError("compounding report requires a baseline row")
    baseline_cost = float(baseline["recipe_cost"])
    isolated = {}
    for row in rows:
        levers = row.get("levers", [])
        if len(levers) == 1:
            isolated[levers[0]] = baseline_cost / float(row["recipe_cost"])
    output_rows = []
    for row in rows:
        cost = float(row["recipe_cost"])
        observed = baseline_cost / cost
        product = 1.0
        for lever in row.get("levers", []):
            product *= isolated.get(lever, 1.0)
        output_rows.append({**row, "observed_multiplier": observed,
                            "independent_product": product,
                            "overlap_coefficient": observed / product if product else None})
    return {"target_score": target_score, "baseline_cost": baseline_cost,
            "isolated_multipliers": isolated, "rows": output_rows}
