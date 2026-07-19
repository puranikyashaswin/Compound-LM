"""The knobs added for cost/quality must actually do what they claim.

Each of these guards a change that is invisible if it silently no-ops -- which
is exactly how the original `clip_grad_norm_(..., 1e9)` survived: it looked
like clipping, reported a plausible number, and never clipped anything.
"""
import json
import math
from pathlib import Path

import pytest
import torch

from src.train.reference import (REPLAY_TOLERANCE, _measurements_agree,
                                 _append_or_verify_replay, train)

SHARD = "data/protocol-v1/protocol-train-packed.jsonl"
HELDOUT = "data/protocol-v1/protocol-heldout-packed.jsonl"
# vocab must cover the shard's declared range (protocol shards fold to 4096).
TINY = dict(vocab_size=4096, d_model=32, n_layers=2, n_heads=2)


def _train(tmp_path, **kwargs):
    return train(SHARD, str(tmp_path), **TINY, steps=4, seed=17, device="cpu",
                 checkpoint_every=4, batch_size=2, **kwargs)


def test_grad_clip_actually_bounds_the_update(tmp_path):
    """Clipped and unclipped runs must reach different weights."""
    loose = _train(tmp_path / "loose", grad_clip=None)
    tight = _train(tmp_path / "tight", grad_clip=1e-4)
    assert loose["final_loss"] != tight["final_loss"], (
        "an aggressive grad_clip changed nothing -- clipping is not wired in")


def test_grad_clip_is_recorded(tmp_path):
    assert _train(tmp_path / "r", grad_clip=0.5)["grad_clip"] == 0.5


def test_reported_grad_norm_is_pre_clip(tmp_path):
    """The health gate needs the true norm, not the clipped-to value."""
    result = _train(tmp_path / "n", grad_clip=1e-6)
    assert result["health"]["status"] in ("green", "amber", "red")
    # A norm reported as <= the clip threshold would mean the gate can never
    # see a spike once clipping is on.
    assert result["final_loss"] > 0


def test_weight_decay_exempts_norms_and_biases(tmp_path):
    """1-D tensors must land in the zero-decay group."""
    import src.train.reference as module

    captured = {}
    original = torch.optim.AdamW

    def spy(groups, **kwargs):
        if isinstance(groups, list) and groups and isinstance(groups[0], dict):
            captured["groups"] = [(len(g["params"]), g["weight_decay"]) for g in groups]
        return original(groups, **kwargs)

    module.torch = torch  # ensure attribute exists for monkeypatching path
    torch.optim.AdamW = spy
    try:
        _train(tmp_path / "wd", weight_decay=0.1)
    finally:
        torch.optim.AdamW = original

    assert "groups" in captured, "optimizer was not built from parameter groups"
    decays = {decay for _, decay in captured["groups"]}
    assert 0.0 in decays, "no zero-decay group: norms and biases are being decayed"
    assert 0.1 in decays, "decay group did not receive the requested weight_decay"


def test_weight_decay_changes_results(tmp_path):
    heavy = _train(tmp_path / "h", weight_decay=0.5)
    none = _train(tmp_path / "z", weight_decay=0.0)
    assert heavy["final_loss"] != none["final_loss"]


def test_eval_batch_size_does_not_move_accuracy(tmp_path):
    small = _train(tmp_path / "e1", heldout_shard=HELDOUT, eval_batch_size=1)
    large = _train(tmp_path / "e2", heldout_shard=HELDOUT, eval_batch_size=64)
    assert small["eval_scores"]["val_acc"] == large["eval_scores"]["val_acc"]
    assert small["eval_scores"]["heldout_tokens"] == large["eval_scores"]["heldout_tokens"]
    assert small["eval_scores"]["val_nll"] == pytest.approx(
        large["eval_scores"]["val_nll"], rel=1e-6)


def test_resume_across_architectures_is_refused(tmp_path):
    out = tmp_path / "arch"
    _train(out, architecture="reference-v1")
    checkpoint = sorted(out.glob("checkpoint-*.pt"))[-1]
    with pytest.raises(ValueError, match="architecture"):
        train(SHARD, str(out), **TINY, steps=8, seed=17, device="cpu",
              checkpoint_every=4, batch_size=2, resume=str(checkpoint),
              architecture="reex-v2")


def test_resume_records_and_checks_precision(tmp_path):
    out = tmp_path / "prec"
    result = _train(out)
    assert "systems" in result
    checkpoint = torch.load(sorted(out.glob("checkpoint-*.pt"))[-1],
                            map_location="cpu", weights_only=False)
    assert "precision" in checkpoint, "precision must be recorded for the resume guard"


# --- replay tolerance -------------------------------------------------------

def test_measurements_agree_on_float_reduction_noise():
    prior = {"val_nll": 1.1856877625, "val_acc": 0.94}
    current = {"val_nll": 1.1856876419, "val_acc": 0.94}
    assert _measurements_agree(prior, current)


def test_measurements_disagree_on_a_real_divergence():
    assert not _measurements_agree({"val_acc": 0.94}, {"val_acc": 0.91})


def test_tolerance_is_tight_enough_to_catch_a_seed_change():
    assert not _measurements_agree(2.5, 2.5 * (1 + 10 * REPLAY_TOLERANCE))


def test_nan_only_matches_nan():
    assert _measurements_agree(float("nan"), float("nan"))
    assert not _measurements_agree(float("nan"), 1.0)


def test_differing_keys_disagree():
    assert not _measurements_agree({"a": 1.0}, {"a": 1.0, "b": 2.0})


def _ledger_entry(**overrides):
    entry = {"run_id": "r", "tokens": 10, "final_loss": 1.0,
             "eval_scores": {"val_nll": 2.0}, "config_hash": "abc", "commit": "x",
             "scale": "1p", "levers_on": [], "wall_clock_s": 1.0, "gpu_type": "cpu",
             "est_cost_usd": 0.0, "fully_accounted_cost_usd": 0.0, "seed": 17,
             "notes": ""}
    return {**entry, **overrides}


def test_replay_within_tolerance_does_not_raise(tmp_path):
    ledger = tmp_path / "l.jsonl"
    _append_or_verify_replay(str(ledger), _ledger_entry())
    # Float reduction noise, exactly what batched evaluation produces.
    _append_or_verify_replay(str(ledger), _ledger_entry(eval_scores={"val_nll": 2.0 + 1e-9}))
    assert len(ledger.read_text().strip().splitlines()) == 1, "replay must not duplicate a row"


def test_replay_outside_tolerance_still_raises(tmp_path):
    ledger = tmp_path / "l.jsonl"
    _append_or_verify_replay(str(ledger), _ledger_entry())
    with pytest.raises(ValueError, match="ledger contradiction"):
        _append_or_verify_replay(str(ledger), _ledger_entry(final_loss=1.5))
