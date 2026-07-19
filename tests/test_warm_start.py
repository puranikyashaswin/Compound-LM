"""Warm-starting a grown model must borrow weights and nothing else.

Growing Reex-1 into Reex-1.5 is not a resume: after depth growth some
parameters did not exist in the donor, so its optimizer moments do not apply,
its step counter belongs to a different run, and its data cursor to a different
curve. Carrying any of them over would silently present a new run as a
continuation of the old one.
"""
import pytest
import torch

from scripts.grow_and_continue import inspect_donor
from src.growth.depth import grow_depth
from src.growth.hyperclone import assert_logit_equivalence
from src.model.registry import build_model
from src.train.reference import train

DONOR = dict(vocab_size=4096, d_model=64, n_layers=3, n_heads=4)


@pytest.fixture
def donor_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = build_model("reference-v1", **DONOR, max_seq_len=96)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)
    path = tmp_path / "donor.pt"
    torch.save({"model": model.state_dict(), "config": model.config,
                "step": 12345, "optimizer": {}}, path)
    return path, model


def test_donor_param_count_excludes_the_tied_head(donor_checkpoint):
    """state_dict lists a tied lm_head separately; counting both doubles it.

    At 50257x768 that overstates a real model by 38M parameters, which would
    misreport what Reex-1 actually is.
    """
    path, model = donor_checkpoint
    info = inspect_donor(str(path))
    assert info["param_count"] == sum(p.numel() for p in model.parameters())


def test_donor_config_is_read_back(donor_checkpoint):
    path, _ = donor_checkpoint
    info = inspect_donor(str(path))
    assert info["config"]["d_model"] == 64
    assert info["config"]["n_layers"] == 3
    assert info["step"] == 12345


def test_config_is_inferred_from_a_bare_state_dict(tmp_path):
    """A checkpoint from another trainer has no config block."""
    model = build_model("reference-v1", **DONOR, max_seq_len=96)
    path = tmp_path / "bare.pt"
    torch.save(model.state_dict(), path)
    info = inspect_donor(str(path))
    assert info["config"]["d_model"] == 64
    assert info["config"]["vocab_size"] == 4096


def test_growth_preserves_the_donor_function(donor_checkpoint):
    _, model = donor_checkpoint
    grown, report = grow_depth(model.eval(), to_layers=6, mode="zero_init")
    probe = torch.randint(0, 4096, (2, 32))
    assert report.function_preserving
    assert_logit_equivalence(model, grown.eval(), probe, tolerance=1e-5)


def test_warm_start_loads_weights_and_resets_the_step(protocol_shards, tmp_path,
                                                      donor_checkpoint):
    path, _ = donor_checkpoint
    result = train(protocol_shards["train"], str(tmp_path / "run"), **DONOR,
                   steps=2, seed=17, device="cpu", checkpoint_every=2,
                   batch_size=2, init_from=str(path))
    assert result["steps"] == 2, "warm start must not inherit the donor's step count"
    assert result["parent_checkpoint_hash"], "lineage to the donor must be recorded"


def test_warm_start_and_resume_are_mutually_exclusive(protocol_shards, tmp_path,
                                                      donor_checkpoint):
    path, _ = donor_checkpoint
    with pytest.raises(ValueError, match="mutually exclusive"):
        train(protocol_shards["train"], str(tmp_path / "run"), **DONOR, steps=2,
              seed=17, device="cpu", checkpoint_every=2, batch_size=2,
              init_from=str(path), resume=str(path))


def test_a_shape_mismatch_is_refused_not_partially_loaded(protocol_shards, tmp_path,
                                                          donor_checkpoint):
    """Silently ignoring unmatched tensors would train a half-random model."""
    path, _ = donor_checkpoint
    with pytest.raises(ValueError, match="does not fit this architecture"):
        train(protocol_shards["train"], str(tmp_path / "run"), vocab_size=4096,
              d_model=128, n_layers=3, n_heads=4, steps=2, seed=17, device="cpu",
              checkpoint_every=2, batch_size=2, init_from=str(path))


def test_longer_sequence_than_the_donor_trained_on_is_refused(tmp_path):
    """Untrained positions would emit noise beyond the donor's context."""
    model = build_model("reference-v1", **DONOR, max_seq_len=32)
    path = tmp_path / "short.pt"
    torch.save({"model": model.state_dict(), "config": model.config}, path)

    from src.data.packing import pack_shard
    from src.data.pipeline import prepare_documents
    prepare_documents(["alpha beta gamma delta " * 40], source="t", shard_id="s",
                      output_dir=tmp_path, vocab_size=4096)
    packed = tmp_path / "s-packed.jsonl"
    pack_shard(tmp_path / "s.jsonl", packed, sequence_length=64)

    with pytest.raises(ValueError, match="never been trained"):
        train(str(packed), str(tmp_path / "run"), **DONOR, steps=1, seed=17,
              device="cpu", checkpoint_every=1, batch_size=1, init_from=str(path))
