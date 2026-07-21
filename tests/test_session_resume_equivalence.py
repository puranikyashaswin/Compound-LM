"""Session-boundary regression: interrupted and uninterrupted curves agree."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

TRAIN_KWARGS = dict(vocab_size=32768, d_model=16, n_layers=2, n_heads=2,
                    seed=41, device="cpu", checkpoint_every=6, use_muon=True)


def test_resume_restores_curve_optimizer_rng_and_data_position(tmp_path):
    shard = "data/toy-v1/toy-v1-packed.jsonl"
    worker = Path(__file__).with_name("resume_worker.py")
    def run(output, steps, resume=None):
        cmd = [sys.executable, str(worker), "--shard", shard,
               "--output", str(output), "--steps", str(steps)]
        if resume:
            cmd += ["--resume", str(resume)]
        return json.loads(subprocess.check_output(cmd, cwd=Path(__file__).parents[1], text=True))

    uninterrupted = run(tmp_path / "full", 12)
    first_half = run(tmp_path / "split", 6)
    checkpoint = tmp_path / "split" / "checkpoint-00000006.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["step"] == 6
    row_count = sum(1 for line in open(shard) if line.strip())
    assert state["data_position"] == (6 * 2) % row_count
    assert state["batch_size"] == 2
    assert state["muon_optimizer"] is not None
    resumed = run(tmp_path / "split", 12, checkpoint)
    assert len(uninterrupted["losses"]) == len(resumed["losses"]) == 12
    assert torch.allclose(torch.tensor(uninterrupted["losses"]), torch.tensor(resumed["losses"]), atol=1e-6, rtol=1e-6)


def _checkpoint_at_step_6(tmp_path, shard):
    from src.train.reference import train
    train(shard, str(tmp_path / "origin"), steps=6, batch_size=2, **TRAIN_KWARGS)
    return str(tmp_path / "origin" / "checkpoint-00000006.pt")


def test_resume_refuses_a_different_shard(tmp_path):
    """Loader provenance is a gate, not a note: resuming onto other data must fail."""
    from src.train.reference import train

    shard = "data/toy-v1/toy-v1-packed.jsonl"
    checkpoint = _checkpoint_at_step_6(tmp_path, shard)
    impostor = tmp_path / "impostor-packed.jsonl"
    shutil.copy(shard, impostor)
    with pytest.raises(ValueError, match="trained on shard"):
        train(str(impostor), str(tmp_path / "resumed"), steps=12,
              resume=checkpoint, batch_size=2, **TRAIN_KWARGS)


def test_resume_same_name_allows_relocated_shard(tmp_path):
    """Kaggle workdir changes keep the leaf name; absolute paths must not hard-fail."""
    from src.train.reference import train

    shard = Path("data/toy-v1/toy-v1-packed.jsonl")
    checkpoint = _checkpoint_at_step_6(tmp_path, str(shard))
    relocated = tmp_path / "elsewhere" / shard.name
    relocated.parent.mkdir(parents=True)
    shutil.copy(shard, relocated)
    result = train(str(relocated), str(tmp_path / "resumed"), steps=12,
                   resume=checkpoint, batch_size=2,
                   resume_shard_policy="same_name", **TRAIN_KWARGS)
    assert result["reached_step"] == 12


def test_resume_refuses_a_different_batch_size(tmp_path):
    """batch_size changes what data_position means, so it cannot silently differ."""
    from src.train.reference import train

    shard = "data/toy-v1/toy-v1-packed.jsonl"
    checkpoint = _checkpoint_at_step_6(tmp_path, shard)
    with pytest.raises(ValueError, match="batch_size 2, not 4"):
        train(shard, str(tmp_path / "resumed"), steps=12,
              resume=checkpoint, batch_size=4, **TRAIN_KWARGS)
