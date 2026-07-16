import pytest

from src.ledger.compounding import compounding_report, cost_to_score, cost_to_score_detail


def test_lever_that_never_reaches_target_reports_instead_of_crashing():
    """A None cost is a real outcome ("never got there"), not an error."""
    rows = [
        {"name": "baseline-17", "seed": 17, "levers": [], "recipe_cost": 100},
        {"name": "baseline-23", "seed": 23, "levers": [], "recipe_cost": 100},
        {"name": "muon-17", "seed": 17, "levers": ["muon"], "recipe_cost": None},
        {"name": "muon-23", "seed": 23, "levers": ["muon"], "recipe_cost": None},
    ]
    report = compounding_report(rows, target_score=0.9)
    muon = [row for row in report["rows"] if row["levers"] == ["muon"]]
    assert all(row["observed_multiplier"] is None for row in muon)
    assert all(row["status"] == "not_reached" for row in muon)
    # A run that never reached the target must not earn a multiplier.
    assert "muon" not in report["isolated_multipliers"]


def test_all_lower_bound_costs_are_refused_as_unresolved():
    """Every run crossing the target before its first checkpoint measures nothing."""
    from src.ledger.compounding import assert_costs_resolved
    details = {name: {"cost": 10.0, "status": "lower_bound"}
               for name in ("baseline-s17", "baseline-s23", "muon-s17", "muon-s23")}
    with pytest.raises(ValueError, match="unresolved_capability_cost"):
        assert_costs_resolved(details)


def test_a_resolved_comparison_is_allowed_through():
    from src.ledger.compounding import assert_costs_resolved
    assert_costs_resolved({
        "baseline-s17": {"cost": 10.0, "status": "interpolated"},
        "muon-s17": {"cost": 6.0, "status": "lower_bound"},
    })


def test_compounding_refuses_a_zero_target_that_every_run_trivially_clears():
    """The all-zero-eval trap: a 0 target makes every run tie at 1.000x."""
    rows = [
        {"name": "baseline-17", "seed": 17, "levers": [], "recipe_cost": 100},
        {"name": "baseline-23", "seed": 23, "levers": [], "recipe_cost": 100},
        {"name": "muon-17", "seed": 17, "levers": ["muon"], "recipe_cost": 100},
        {"name": "muon-23", "seed": 23, "levers": ["muon"], "recipe_cost": 100},
    ]
    with pytest.raises(ValueError, match="target_score must be positive"):
        compounding_report(rows, target_score=0.0)


def test_compounding_refuses_a_baseline_that_never_reached_target():
    rows = [
        {"name": "baseline-17", "seed": 17, "levers": [], "recipe_cost": None},
        {"name": "baseline-23", "seed": 23, "levers": [], "recipe_cost": None},
    ]
    with pytest.raises(ValueError, match="baseline never reached"):
        compounding_report(rows, target_score=0.9)


def test_cost_to_score_interpolates_without_extrapolation():
    points = [{"cost": 0, "score": 0}, {"cost": 10, "score": 1}]
    assert cost_to_score(points, 0.5) == 5
    assert cost_to_score(points, 2) is None


def test_cost_to_score_reports_lower_bound_when_target_is_already_exceeded():
    points = [{"cost": 10, "score": 0.6}, {"cost": 20, "score": 0.8}]
    detail = cost_to_score_detail(points, 0.5)
    assert detail["status"] == "lower_bound"
    assert detail["cost"] == 10
    assert cost_to_score(points, 0.5) == 10


def test_compounding_reports_overlap():
    report = compounding_report([
        {"name": "baseline", "levers": [], "recipe_cost": 100},
        {"name": "data", "levers": ["data"], "recipe_cost": 50},
        {"name": "muon", "levers": ["muon"], "recipe_cost": 50},
        {"name": "compound", "levers": ["data", "muon"], "recipe_cost": 30},
    ], target_score=0.5)
    row = report["rows"][-1]
    assert row["observed_multiplier"] == 100 / 30
    assert row["independent_product"] == 4
    assert row["overlap_coefficient"] < 1
