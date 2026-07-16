"""The README table is published evidence, so it must not publish non-evidence."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.ledger.writer import append_entry

# Loaded by path: the repo-root `ledger/` script package shadows `src.ledger`.
_spec = importlib.util.spec_from_file_location(
    "_make_table", Path(__file__).parents[1] / "ledger" / "make_table.py")
_make_table = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_make_table)
make_table = _make_table.make_table

BASE = {"config_hash": "cfg", "commit": "test", "scale": "proxy", "levers_on": [],
        "wall_clock_s": 1.0, "gpu_type": "cpu", "est_cost_usd": 1.0,
        "fully_accounted_cost_usd": 1.0, "final_loss": 1.0, "seed": 17,
        "tokens": 100, "notes": "test"}

README = ("# x\n\n<!-- AUTOGEN:TABLE START -->\nold\n<!-- AUTOGEN:TABLE END -->\n")


def _readme(tmp_path):
    path = tmp_path / "README.md"
    path.write_text(README, encoding="utf-8")
    return path


def test_table_refuses_to_publish_runs_that_all_scored_zero(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    for run_id in ("baseline-s17", "muon-s17"):
        append_entry(ledger, {**BASE, "run_id": run_id, "eval_scores": {"smoke": 0.0}})
    readme = _readme(tmp_path)
    table = make_table(str(ledger), str(readme))
    assert "measured capability signal" in table
    assert "baseline-s17" not in readme.read_text()


def test_table_publishes_runs_with_a_real_signal(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    append_entry(ledger, {**BASE, "run_id": "baseline-s17", "eval_scores": {"smoke": 0.2}})
    append_entry(ledger, {**BASE, "run_id": "muon-s17", "levers_on": ["optimizer"],
                          "fully_accounted_cost_usd": 0.5, "eval_scores": {"smoke": 0.2}})
    readme = _readme(tmp_path)
    table = make_table(str(ledger), str(readme))
    assert "baseline-s17" in table and "muon-s17" in table
    assert "2.00×" in table  # baseline cost 1.0 / muon cost 0.5
    assert "old" not in readme.read_text()
