"""Intrinsic held-out evaluation produces real, bounded capability scores."""
from __future__ import annotations

import importlib.util

import pytest

from src.data.packing import pack_shard
from src.data.pipeline import prepare_documents

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="requires PyTorch runtime"
)

DOCS = [
    "the model predicts the next token in a packed sequence",
    "levers are measured against a fair reproducible baseline",
    "an append only ledger records every audited run",
    "document boundaries prevent cross document attention",
]


def _shard(tmp_path):
    prepare_documents(DOCS, source="test", shard_id="s", output_dir=tmp_path,
                      vocab_size=4096)
    packed = tmp_path / "packed.jsonl"
    pack_shard(tmp_path / "s.jsonl", packed, sequence_length=48)
    return str(packed)


def test_train_records_real_eval_scores(tmp_path):
    from src.train.reference import train

    shard = _shard(tmp_path)
    result = train(shard, str(tmp_path / "run"), vocab_size=4096, d_model=32,
                   n_layers=2, n_heads=4, steps=30, device="cpu",
                   heldout_shard=shard, levers_on=["optimizer"], use_muon=True)
    scores = result["eval_scores"]
    assert 0.0 <= scores["val_acc"] <= 1.0
    assert scores["val_perplexity"] > 0
    assert scores["smoke"] == scores["val_acc"]
    assert result["train_flops"] > 0


def test_evaluate_is_deterministic(tmp_path):
    from src.eval.intrinsic import evaluate
    from src.train.reference import train

    shard = _shard(tmp_path)
    result = train(shard, str(tmp_path / "run"), vocab_size=4096, d_model=32,
                   n_layers=2, n_heads=4, steps=10, device="cpu")
    a = evaluate(result["checkpoint"], shard, device="cpu")
    b = evaluate(result["checkpoint"], shard, device="cpu")
    assert a == b
