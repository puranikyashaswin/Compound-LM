import json
from pathlib import Path

import pytest

from src.health.check import check_checkpoint
from src.ledger.writer import append_entry, read_entries
from src.provenance.core import config_hash, load_config
from src.train.dummy import run


def test_config_hash_is_order_independent(tmp_path):
    a = {"schema_version": 1, "x": {"b": 2, "a": 1}}
    b = {"x": {"a": 1, "b": 2}, "schema_version": 1}
    assert config_hash(a) == config_hash(b)


def test_ledger_required_fields_and_duplicate_protection(tmp_path):
    path = tmp_path / "runs.jsonl"
    row = {"run_id": "r", "config_hash": "h", "commit": "c", "scale": "x", "levers_on": [], "tokens": 1, "wall_clock_s": 1, "gpu_type": "cpu", "est_cost_usd": 0, "fully_accounted_cost_usd": 0, "final_loss": 1, "eval_scores": {}, "seed": 1, "notes": ""}
    append_entry(path, row)
    assert read_entries(path)[0]["run_id"] == "r"
    with pytest.raises(ValueError):
        append_entry(path, row)


def test_health_red_on_gradient_spike():
    report = check_checkpoint(loss=1, grad_norm=4, median_grad_norm=1, checkpoint_hash="x")
    assert report.status == "red"
    assert "gradient_norm_spike" in report.failures


def test_dummy_trainer_is_replayable(tmp_path):
    ledger = tmp_path / "runs.jsonl"
    first = run("configs/baseline_200m.yaml", tokens=10_000, ledger_path=str(ledger))
    # A second seed/config identity is required for a separate ledger row.
    text = Path("configs/baseline_200m.yaml").read_text()
    altered = tmp_path / "config.yaml"
    altered.write_text(text.replace("seed: 17", "seed: 18"))
    second = run(str(altered), tokens=10_000, ledger_path=str(ledger))
    assert first["final_loss"] != second["final_loss"]
    assert len(read_entries(ledger)) == 2

