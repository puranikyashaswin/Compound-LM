"""Compute capability-at-cost and lever interaction metrics."""
from __future__ import annotations

from typing import Any
from collections import defaultdict


def cost_to_score_detail(checkpoints: list[dict[str, float]], target: float) -> dict[str, float | str | None]:
    """Return capability-at-cost with an explicit early-target status.

    If the first checkpoint already exceeds ``target``, the returned cost is a
    lower bound (the first recorded cost), never an extrapolated value.
    """
    points = sorted((float(row["cost"]), float(row["score"])) for row in checkpoints)
    if not points or target > max(score for _, score in points):
        return {"cost": None, "status": "not_reached"}
    if target <= points[0][1]:
        return {"cost": points[0][0], "status": "lower_bound", "first_cost": points[0][0]}
    for (cost_a, score_a), (cost_b, score_b) in zip(points, points[1:]):
        if min(score_a, score_b) <= target <= max(score_a, score_b):
            if score_b == score_a:
                return {"cost": min(cost_a, cost_b), "status": "reached"}
            return {"cost": cost_a + (target - score_a) * (cost_b - cost_a) / (score_b - score_a), "status": "interpolated"}
    return {"cost": points[-1][0], "status": "reached"} if points[-1][1] == target else {"cost": None, "status": "not_reached"}


def cost_to_score(checkpoints: list[dict[str, float]], target: float) -> float | None:
    """Compatibility wrapper returning the cost, including early lower bounds."""
    return cost_to_score_detail(checkpoints, target)["cost"]


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

    # A non-positive target is not a low bar, it is an absent one: every run
    # clears a target of zero at its first checkpoint and ties at 1.000x, which
    # reads as "no lever compounded" when the truth is "nothing was measured".
    if target_score <= 0:
        raise ValueError(
            f"target_score must be positive to compare capability at cost, got {target_score}; "
            "a zero target means no run demonstrated any capability"
        )
    baseline = next((row for row in rows if not row.get("levers")), None)
    if baseline is None:
        raise ValueError("compounding report requires a baseline row")
    if baseline.get("recipe_cost") is None:
        raise ValueError(
            "baseline never reached the target score, so no multiplier is defined against it"
        )
    baseline_cost = float(baseline["recipe_cost"])

    # A None recipe_cost means the run never reached the target score. That is a
    # real outcome, not an error: such a run yields no multiplier and is excluded
    # from every isolated multiplier rather than being scored as if it had won.
    isolated = {}
    for row in rows:
        levers = row.get("levers", [])
        if len(levers) == 1 and row.get("recipe_cost") is not None:
            isolated[levers[0]] = baseline_cost / float(row["recipe_cost"])
    output_rows = []
    for row in rows:
        if row.get("recipe_cost") is None:
            output_rows.append({**row, "observed_multiplier": None,
                                "independent_product": None,
                                "overlap_coefficient": None,
                                "status": "not_reached"})
            continue
        observed = baseline_cost / float(row["recipe_cost"])
        product = 1.0
        for lever in row.get("levers", []):
            product *= isolated.get(lever, 1.0)
        output_rows.append({**row, "observed_multiplier": observed,
                            "independent_product": product,
                            "overlap_coefficient": observed / product if product else None})
    return {"target_score": target_score, "baseline_cost": baseline_cost,
            "isolated_multipliers": isolated, "rows": output_rows}
