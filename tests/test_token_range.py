"""A vocabulary mismatch must be an error, never a silent remap.

The trainer used to do ``ids % vocab_size``. That is correct for the hash-based
fallback tokenizer, whose ids are 32-bit hashes -- but it was applied to every
shard, including real tokenized corpora. Loading a 50257-vocab corpus against a
16384-vocab model therefore mapped token 20000 onto token 3616: a *different
valid token*. Training ran, the loss curve looked healthy, and the run measured
nothing.

That is exactly the configuration the vocabulary-reduction lever creates, so
this bug sat directly in the path of the cost plan. The fold now happens once,
explicitly, at data preparation, and is recorded in the datasheet.
"""
import pytest
import torch

from src.data.packing import pack_shard
from src.data.pipeline import prepare_documents, token_ids
from src.train.reference import assert_token_ids_in_range, train

DOCS = ["the model learns to predict the next token in a packed sequence",
        "held out documents measure generalization rather than memorization"]


def test_raw_fallback_tokenizer_emits_out_of_range_ids():
    """The precondition that made the silent fold dangerous."""
    raw = token_ids("hello world")
    assert max(raw) > 2**16, "fallback ids are 32-bit hashes; the fold is load-bearing"


def test_fold_happens_at_preparation_when_vocab_declared():
    folded = token_ids("hello world", vocab_size=4096)
    assert folded and max(folded) < 4096


def test_fold_rejects_a_nonsense_vocab():
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        token_ids("hello", vocab_size=0)


def test_range_check_accepts_valid_ids():
    assert_token_ids_in_range(torch.tensor([[0, 5, 4095]]), 4096)


def test_range_check_rejects_too_large():
    with pytest.raises(ValueError, match="token id out of range"):
        assert_token_ids_in_range(torch.tensor([[0, 5, 4096]]), 4096)


def test_range_check_rejects_negative():
    with pytest.raises(ValueError, match="token id out of range"):
        assert_token_ids_in_range(torch.tensor([[-1, 5]]), 4096)


def test_error_names_both_the_shard_range_and_the_model_vocab():
    """The message must say enough to fix the mismatch without a debugger."""
    with pytest.raises(ValueError) as info:
        assert_token_ids_in_range(torch.tensor([[0, 90000]]), 16384)
    message = str(info.value)
    assert "90000" in message and "16384" in message


def test_training_refuses_a_shard_tokenized_for_a_larger_vocab(tmp_path):
    """The vocab-reduction lever's exact failure mode, end to end."""
    prepare_documents(DOCS, source="test", shard_id="big", output_dir=tmp_path,
                      vocab_size=50257)
    packed = tmp_path / "big-packed.jsonl"
    pack_shard(tmp_path / "big.jsonl", packed, sequence_length=32)

    # Same corpus, model declares the smaller vocabulary the lever would use.
    with pytest.raises(ValueError, match="token id out of range"):
        train(str(packed), str(tmp_path / "run"), vocab_size=16384, d_model=16,
              n_layers=1, n_heads=2, steps=1, device="cpu", checkpoint_every=0)


def test_training_accepts_a_shard_prepared_at_the_model_vocab(tmp_path):
    prepare_documents(DOCS, source="test", shard_id="ok", output_dir=tmp_path,
                      vocab_size=4096)
    packed = tmp_path / "ok-packed.jsonl"
    pack_shard(tmp_path / "ok.jsonl", packed, sequence_length=32)
    result = train(str(packed), str(tmp_path / "run"), vocab_size=4096, d_model=16,
                   n_layers=1, n_heads=2, steps=1, device="cpu", checkpoint_every=1)
    assert result["final_loss"] > 0


def test_datasheet_records_the_vocabulary_the_shard_is_valid_for(tmp_path):
    sheet = prepare_documents(DOCS, source="test", shard_id="v", output_dir=tmp_path,
                              vocab_size=4096)
    assert sheet["vocab_size"] == 4096
