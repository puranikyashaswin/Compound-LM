import json

from scripts.run_matrix import plan


def test_matrix_plan_is_deterministic_and_estimates_cost(tmp_path):
    ledger = tmp_path / "runs.jsonl"
    rows = plan([{"id": "A0", "levers": [], "config": "configs/baseline_200m.yaml"}],
                seed=17, tokens=1_000_000, gpu_hours_per_million_tokens=0.001,
                gpu_hour_usd=2.0, ledger_path=str(ledger))
    assert rows[0]["status"] == "pending"
    assert rows[0]["estimated_cost_usd"] == 0.002
    assert len(rows[0]["config_hash"]) == 64
