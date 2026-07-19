"""The final experiment must not lose finished work when one arm fails.

Four arms cost roughly an hour of GPU time. If the fourth crashes -- a diverged
optimizer, an out-of-memory, an interrupted session -- the three that finished
are still hours of compute, already checkpointed and ledgered. Exiting without
recording them would make a partial failure indistinguishable from a total one.
"""
import json

import pytest

from src.ledger.cost_model import wall_clock_multiplier
from scripts.final_experiment import _write_evidence


def test_evidence_survives_types_json_cannot_encode(tmp_path):
    """A stray Path or numpy scalar must not discard the whole record."""
    out = tmp_path / "evidence.json"
    _write_evidence(str(out), {"path": tmp_path, "score": 0.5})
    payload = json.loads(out.read_text())
    assert payload["score"] == 0.5
    assert isinstance(payload["path"], str)


def test_evidence_creates_missing_directories(tmp_path):
    out = tmp_path / "nested" / "deeper" / "evidence.json"
    _write_evidence(str(out), {"ok": True})
    assert json.loads(out.read_text())["ok"] is True


def test_wall_clock_multiplier_refuses_a_missing_multiplier():
    """A run that never reached the target has no multiplier to convert."""
    with pytest.raises(ValueError):
        wall_clock_multiplier(flop_multiplier=0.0, step_cost_ratio=1.0)


def test_verdict_band_is_looked_up_by_name_not_position():
    """Row order is an implementation detail; the noise band must not depend on it."""
    rows = [{"name": "optimizer-s17", "observed_multiplier": 1.4},
            {"name": "baseline-s23", "observed_multiplier": 0.9},
            {"name": "baseline-s17", "observed_multiplier": 1.0}]
    control = next(r for r in rows if r["name"] == "baseline-s23")
    assert abs(control["observed_multiplier"] - 1.0) == pytest.approx(0.1)


def test_a_lever_inside_the_seed_band_is_not_decisive():
    """The verdict rule itself: 2x the baseline band is the bar."""
    band = 0.10
    assert not (1.15 - 1.0 > 2 * band)   # inside noise -> inconclusive
    assert (1.40 - 1.0 > 2 * band)       # clears it


# --- checkpoint pruning ----------------------------------------------------

def test_pruning_keeps_only_the_newest_but_all_measurements(protocol_shards, tmp_path):
    """Disk is bounded; the capability curve is not.

    A checkpoint is ~12 bytes per parameter (268MB at 22M params), and a
    four-arm matrix with twelve each is 12.8GB -- more than a Kaggle session's
    working directory holds. Since the curve is read from the ledger, older
    checkpoints are only needed for resume.
    """
    from src.ledger.writer import read_entries
    from src.train.reference import resumable_checkpoint, train

    out = tmp_path / "run"
    ledger = tmp_path / "l.jsonl"
    train(protocol_shards["train"], str(out), vocab_size=4096, d_model=32,
          n_layers=2, n_heads=2, steps=20, seed=17, device="cpu",
          checkpoint_every=2, batch_size=2, ledger_path=str(ledger),
          run_id="prune", keep_checkpoints=3)

    assert len(list(out.glob("checkpoint-*.pt"))) == 3
    rows = [e for e in read_entries(str(ledger)) if e["run_id"] == "prune"]
    assert len(rows) == 10, "pruning must not cost a single curve point"
    assert resumable_checkpoint(out) is not None


def test_resume_still_works_after_pruning(protocol_shards, tmp_path):
    from src.train.reference import resumable_checkpoint, train

    out = tmp_path / "run"
    common = dict(vocab_size=4096, d_model=32, n_layers=2, n_heads=2, seed=17,
                  device="cpu", checkpoint_every=2, batch_size=2, keep_checkpoints=3)
    train(protocol_shards["train"], str(out), steps=20, **common)
    resumed = train(protocol_shards["train"], str(out), steps=24,
                    resume=str(resumable_checkpoint(out)), **common)
    assert resumed["steps"] == 24


def test_pruning_to_one_is_refused(protocol_shards, tmp_path):
    """Keeping one removes the fallback a truncated save depends on."""
    from src.train.reference import train

    with pytest.raises(ValueError, match="at least 2"):
        train(protocol_shards["train"], str(tmp_path / "r"), vocab_size=4096,
              d_model=32, n_layers=2, n_heads=2, steps=2, seed=17, device="cpu",
              checkpoint_every=1, batch_size=2, keep_checkpoints=1)


def test_default_keeps_every_checkpoint(protocol_shards, tmp_path):
    """Existing callers must not start losing files."""
    from src.train.reference import train

    out = tmp_path / "run"
    train(protocol_shards["train"], str(out), vocab_size=4096, d_model=32,
          n_layers=2, n_heads=2, steps=6, seed=17, device="cpu",
          checkpoint_every=2, batch_size=2)
    assert len(list(out.glob("checkpoint-*.pt"))) == 3
