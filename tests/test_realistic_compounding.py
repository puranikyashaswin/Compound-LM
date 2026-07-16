import json

from src.ledger.compounding import compounding_report, cost_to_score_detail
from src.ledger.writer import append_entry, read_entries


def test_realistic_four_run_compounding_and_lower_bound():
    target = 0.08
    curves = {
        "baseline-s17": [(1.0, .055), (2.0, .110)],
        "baseline-s23": [(1.0, .061), (2.0, .108)],
        "muon-s17": [(1.0, .060), (1.5, .130)],
        "muon-s23": [(1.0, .065), (1.5, .140)],
    }
    rows = []
    for name, points in curves.items():
        kind = "muon" if "muon" in name else "baseline"
        seed = int(name[-2:])
        detail = cost_to_score_detail([{"cost": c, "score": s} for c, s in points], target)
        rows.append({"name": name, "levers": ["optimizer"] if kind == "muon" else [],
                     "seed": seed, "recipe_cost": detail["cost"]})
    report = compounding_report(rows, target_score=target)
    assert report["isolated_multipliers"]["optimizer"] > 1.0
    assert all(row["observed_multiplier"] > 0 for row in report["rows"])
    early = cost_to_score_detail([{"cost": 1.0, "score": .10}], .08)
    assert early["status"] == "lower_bound"


def test_ledger_keeps_four_runs_and_multiple_checkpoints(tmp_path):
    path = tmp_path / "ledger.jsonl"
    base = {"config_hash": "cfg", "commit": "test", "scale": "proxy",
            "levers_on": [], "wall_clock_s": 1.0, "gpu_type": "cpu",
            "est_cost_usd": 1.0, "fully_accounted_cost_usd": 1.0,
            "final_loss": 1.0, "eval_scores": {"val_acc": .1}, "seed": 17,
            "notes": "test"}
    for run_id, seed, lever in [("baseline-s17", 17, []), ("baseline-s23", 23, []),
                                ("muon-s17", 17, ["optimizer"]), ("muon-s23", 23, ["optimizer"])]:
        for tokens in (100, 200):
            entry = {**base, "run_id": run_id, "seed": seed, "levers_on": lever, "tokens": tokens}
            append_entry(path, entry)
    entries = read_entries(path)
    assert len(entries) == 8
    assert {(e["run_id"], e["tokens"]) for e in entries}.__len__() == 8
