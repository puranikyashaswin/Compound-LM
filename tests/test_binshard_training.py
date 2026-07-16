"""Swapping the shard format must not change the science.

The binary format stores document ids as running integers where the JSONL
format stores truncated hashes. Attention only compares ids for equality, so
identical grouping must yield an identical loss curve -- this pins that.
"""
import json

import pytest
import torch

from src.data.binshard import write_packed_shard
from src.data.packing import pack_shard
from src.train.reference import lr_at_step, train

DOCS = [{"document_id": f"doc-{i}", "tokens": [(i * 7 + j) % 64 for j in range(20)]}
        for i in range(8)]
MODEL = dict(vocab_size=64, d_model=16, n_layers=2, n_heads=2)


def _build_both(tmp_path):
    source = tmp_path / "docs.jsonl"
    source.write_text("\n".join(json.dumps(d) for d in DOCS) + "\n", encoding="utf-8")
    jsonl_packed = tmp_path / "packed.jsonl"
    pack_shard(source, jsonl_packed, sequence_length=8)
    write_packed_shard(DOCS, tmp_path / "bin", sequence_length=8, vocab_size=64,
                       tokenizer_id="test")
    return str(jsonl_packed), str(tmp_path / "bin")


def test_binary_and_jsonl_shards_produce_the_same_loss_curve(tmp_path):
    jsonl_shard, bin_shard = _build_both(tmp_path)
    kwargs = dict(**MODEL, steps=8, seed=17, device="cpu", checkpoint_every=0,
                  batch_size=2)
    from_jsonl = train(jsonl_shard, str(tmp_path / "a"), **kwargs)
    from_bin = train(bin_shard, str(tmp_path / "b"), **kwargs)
    assert torch.allclose(torch.tensor(from_jsonl["losses"]),
                          torch.tensor(from_bin["losses"]), atol=1e-6, rtol=1e-6)


def test_training_reads_a_binary_shard_without_loading_it_all(tmp_path):
    _, bin_shard = _build_both(tmp_path)
    from src.data.loader import BinLoader, open_shard
    loader = open_shard(bin_shard)
    # memmap-backed, not a materialized Python list of every token
    assert isinstance(loader, BinLoader)
    assert loader.shard.tokens.__class__ is __import__("numpy").memmap


def test_muon_lr_is_configurable_not_hardcoded(tmp_path):
    """An LR sweep is impossible while the value is baked into train()."""
    _, bin_shard = _build_both(tmp_path)
    kwargs = dict(**MODEL, steps=6, seed=17, device="cpu", checkpoint_every=0,
                  batch_size=2, use_muon=True)
    slow = train(bin_shard, str(tmp_path / "slow"), muon_lr=1e-6, **kwargs)
    fast = train(bin_shard, str(tmp_path / "fast"), muon_lr=0.05, **kwargs)
    assert slow["losses"] != fast["losses"]


class TestLRSchedule:
    def test_warms_up_then_decays_to_the_floor(self):
        total = 100
        first = lr_at_step(0, total_steps=total, base_lr=1.0, warmup_fraction=0.1)
        peak = lr_at_step(10, total_steps=total, base_lr=1.0, warmup_fraction=0.1)
        last = lr_at_step(99, total_steps=total, base_lr=1.0, warmup_fraction=0.1,
                          min_lr_fraction=0.1)
        assert first < peak
        assert peak == pytest.approx(1.0, abs=1e-6)
        assert last == pytest.approx(0.1, abs=0.01)

    def test_is_monotonic_after_warmup(self):
        values = [lr_at_step(s, total_steps=50, base_lr=1.0, warmup_fraction=0.0)
                  for s in range(50)]
        assert values == sorted(values, reverse=True)

    def test_rejects_nonsense_configuration(self):
        with pytest.raises(ValueError):
            lr_at_step(0, total_steps=0, base_lr=1.0)
        with pytest.raises(ValueError):
            lr_at_step(0, total_steps=10, base_lr=1.0, warmup_fraction=1.0)

    def test_schedule_changes_training_and_stays_off_by_default(self, tmp_path):
        _, bin_shard = _build_both(tmp_path)
        kwargs = dict(**MODEL, steps=10, seed=17, device="cpu", checkpoint_every=0,
                      batch_size=2)
        flat = train(bin_shard, str(tmp_path / "flat"), **kwargs)
        scheduled = train(bin_shard, str(tmp_path / "sched"), lr_schedule=True,
                          warmup_fraction=0.2, **kwargs)
        assert flat["losses"] != scheduled["losses"]
