from src.ledger.compounding import compounding_report, cost_to_score, cost_to_score_detail


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
