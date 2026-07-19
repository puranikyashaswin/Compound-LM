"""A loss-scaling overflow must not destroy a healthy run.

Under fp16, GradScaler deliberately overflows: it starts at a large scale,
halves it whenever gradients exceed the representable range, skips that step,
and probes for a larger scale every `growth_interval` steps -- so overflows
recur throughout training by design. On such a step `clip_grad_norm_` returns
`inf`.

Before this fix that `inf` reached the health gate, which returned `red`, and
`train()` raises on `red`. A routine, correctly-handled fp16 event therefore
killed the entire experiment -- and with checkpoints every 500 steps across
four runs, there were 48 chances for it to happen.
"""
import math

import pytest
import torch

from src.health.check import RollingMedian, check_checkpoint
from src.train.reference import train

TINY = dict(vocab_size=4096, d_model=32, n_layers=2, n_heads=2)


def test_overflowed_step_yields_inf_norm():
    """The precondition: this is what GradScaler leaves behind."""
    model = torch.nn.Linear(8, 8)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler.scale(model(torch.randn(4, 8)).square().mean()).backward()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float("inf"))
    scaler.unscale_(optimizer)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    assert not math.isfinite(norm)


def test_health_gate_would_halt_on_that_norm():
    """Which is why the trainer must never hand it over."""
    report = check_checkpoint(loss=2.0, grad_norm=float("inf"), median_grad_norm=0.5,
                              checkpoint_hash="x", provenance_ok=True)
    assert report.status == "red"


def test_rolling_median_stays_finite_when_overflows_are_excluded():
    history = RollingMedian(window=100)
    for value in [0.5] * 60:
        history.add(value)
    # The trainer skips non-finite values rather than adding them.
    assert math.isfinite(history.median())
    assert all(math.isfinite(v) for v in history.history)


def test_training_survives_an_overflow_at_a_checkpoint_step(protocol_shards, tmp_path,
                                                            monkeypatch):
    """End to end: force an overflow on the exact step a checkpoint lands on."""
    import src.train.reference as module

    real_clip = torch.nn.utils.clip_grad_norm_
    calls = {"n": 0}

    def flaky_clip(parameters, max_norm, *args, **kwargs):
        calls["n"] += 1
        result = real_clip(parameters, max_norm, *args, **kwargs)
        # Step 2 is a checkpoint step below; return inf exactly there.
        if calls["n"] == 2:
            return torch.tensor(float("inf"))
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", flaky_clip)

    result = train(protocol_shards["train"], str(tmp_path / "run"), **TINY, steps=4,
                   seed=17, device="cpu", checkpoint_every=2, batch_size=2)
    assert result["overflow_steps"] >= 1, "the injected overflow was not counted"
    assert result["health"]["status"] in ("green", "amber")


def test_overflow_rate_is_reported(protocol_shards, tmp_path):
    result = train(protocol_shards["train"], str(tmp_path / "run"), **TINY, steps=4,
                   seed=17, device="cpu", checkpoint_every=4, batch_size=2)
    assert result["overflow_steps"] == 0
    assert result["overflow_rate"] == 0.0


def test_a_single_spike_no_longer_halts_training(protocol_shards, tmp_path, monkeypatch):
    """One large step must not end the run."""
    real_clip = torch.nn.utils.clip_grad_norm_
    calls = {"n": 0}

    def spiky_clip(parameters, max_norm, *args, **kwargs):
        calls["n"] += 1
        result = real_clip(parameters, max_norm, *args, **kwargs)
        if calls["n"] == 4:
            return result * 50.0
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spiky_clip)
    result = train(protocol_shards["train"], str(tmp_path / "run"), **TINY, steps=4,
                   seed=17, device="cpu", checkpoint_every=2, batch_size=2)
    assert result["health"]["status"] in ("green", "amber")
