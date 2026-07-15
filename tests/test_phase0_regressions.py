"""Trust gates introduced by Phase 0; these must fail on the old instrument."""
import json

import pytest


def test_contamination_gate_rejects_same_text_in_different_shards(tmp_path):
    from src.data.contamination import assert_disjoint_shards

    train = tmp_path / "train.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    row = {"document_id": "different-id", "text": "duplicate semantic content", "text_sha256": "same"}
    train.write_text(json.dumps(row) + "\n")
    heldout.write_text(json.dumps({**row, "document_id": "other-id"}) + "\n")
    with pytest.raises(ValueError, match="contamination"):
        assert_disjoint_shards(train, heldout)


def test_padding_is_excluded_from_training_loss():
    import torch
    from src.train.reference import masked_next_token_loss

    logits = torch.tensor([[[0.0, 12.0, 0.0], [0.0, 0.0, 12.0], [12.0, 0.0, 0.0]]])
    ids = torch.tensor([[1, 2, 0]])
    document_ids = torch.tensor([[4, 4, -1]])
    loss = masked_next_token_loss(logits, ids, document_ids)
    assert loss.item() < 0.01


def test_rolling_median_detects_a_real_spike():
    from src.health.check import RollingMedian

    history = RollingMedian(window=5)
    for value in (1.0, 1.1, 0.9, 1.0, 1.05):
        history.add(value)
    assert history.median() == 1.0
    assert history.is_spike(4.0, multiplier=3.0)


def test_compounding_refuses_single_seed_lever_result():
    from src.ledger.compounding import compounding_report

    rows = [
        {"name": "baseline-17", "seed": 17, "levers": [], "recipe_cost": 100, "ledgered": True},
        {"name": "baseline-23", "seed": 23, "levers": [], "recipe_cost": 100, "ledgered": True},
        {"name": "muon-17", "seed": 17, "levers": ["muon"], "recipe_cost": 60, "ledgered": True},
    ]
    with pytest.raises(ValueError, match="two distinct seeds"):
        compounding_report(rows, target_score=0.9)


def test_custom_state_dict_loader_round_trips_reference_checkpoint(tmp_path):
    import torch
    from src.eval.lm_eval_adapter import load_reference_checkpoint
    from src.model.reference import ReferenceLM

    model = ReferenceLM(vocab_size=32, d_model=8, n_layers=1, n_heads=2, max_seq_len=8)
    path = tmp_path / "reference.pt"
    torch.save({"model": model.state_dict(), "config": model.config}, path)
    loaded = load_reference_checkpoint(path, device="cpu")
    assert loaded.config == model.config
