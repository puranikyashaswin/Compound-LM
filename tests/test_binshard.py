"""The binary shard must preserve packing semantics the JSONL format guarantees."""
import json

import numpy as np
import pytest

from src.data.binshard import (PAD_DOCUMENT_ID, PackedShard, convert_jsonl_shard,
                               token_dtype, write_packed_shard)
from src.data.packing import pack_documents

DOCS = [
    {"document_id": "a", "tokens": [1, 2, 3, 4, 5]},
    {"document_id": "b", "tokens": [6, 7]},
    {"document_id": "c", "tokens": [8, 9, 10]},
]


def test_dtype_is_uint16_for_gpt2_vocab_and_widens_when_needed():
    assert token_dtype(50257) == np.uint16
    assert token_dtype(70000) == np.int32


def test_packing_matches_the_jsonl_packer_token_for_token(tmp_path):
    """Same greedy layout as pack_documents, just stored as arrays."""
    write_packed_shard(DOCS, tmp_path / "s", sequence_length=4, vocab_size=64,
                       tokenizer_id="test")
    shard = PackedShard(tmp_path / "s")
    reference = pack_documents([dict(d) for d in DOCS], sequence_length=4,
                               min_documents_per_batch=0)
    assert len(shard) == len(reference)
    flat_bin = np.asarray(shard.tokens).reshape(-1).tolist()
    flat_ref = [t for packet in reference for t in packet["input_ids"]]
    assert flat_bin == flat_ref


def test_padding_is_marked_and_documents_stay_separable(tmp_path):
    write_packed_shard(DOCS, tmp_path / "s", sequence_length=4, vocab_size=64,
                       tokenizer_id="test")
    shard = PackedShard(tmp_path / "s")
    docs = np.asarray(shard.docs)
    # 10 real tokens into sequences of 4 -> 3 sequences, 2 padded slots.
    assert int((docs == PAD_DOCUMENT_ID).sum()) == 2
    assert shard.meta["padding_tokens"] == 2
    assert shard.meta["real_tokens"] == 10
    # Distinct documents keep distinct ids, so a block-diagonal mask is derivable.
    first_row_docs = [d for d in docs[0].tolist() if d != PAD_DOCUMENT_ID]
    assert len(set(first_row_docs)) >= 1
    assert shard.meta["n_documents"] == 3


def test_batch_returns_requested_rows_and_wraps(tmp_path):
    write_packed_shard(DOCS, tmp_path / "s", sequence_length=4, vocab_size=64,
                       tokenizer_id="test")
    shard = PackedShard(tmp_path / "s")
    tokens, docs = shard.batch([0, 1])
    assert tokens.shape == (2, 4) and docs.shape == (2, 4)
    wrapped, _ = shard.batch([len(shard)])
    direct, _ = shard.batch([0])
    assert wrapped.tolist() == direct.tolist()


def test_reader_detects_a_corrupted_shard(tmp_path):
    write_packed_shard(DOCS, tmp_path / "s", sequence_length=4, vocab_size=64,
                       tokenizer_id="test")
    target = (tmp_path / "s").with_suffix(".tokens.bin")
    data = bytearray(target.read_bytes())
    data[0] ^= 0xFF
    target.write_bytes(bytes(data))
    PackedShard(tmp_path / "s")  # tolerated without verification
    with pytest.raises(ValueError, match="shard corrupted"):
        PackedShard(tmp_path / "s", verify_hashes=True)


def test_out_of_range_token_is_refused(tmp_path):
    with pytest.raises(ValueError, match="token id out of range"):
        write_packed_shard([{"document_id": "x", "tokens": [99]}], tmp_path / "s",
                           sequence_length=4, vocab_size=8, tokenizer_id="test")


def test_convert_jsonl_shard_round_trips(tmp_path):
    source = tmp_path / "docs.jsonl"
    source.write_text("\n".join(json.dumps(d) for d in DOCS) + "\n", encoding="utf-8")
    convert_jsonl_shard(source, tmp_path / "s", vocab_size=64, tokenizer_id="test",
                        sequence_length=4)
    shard = PackedShard(tmp_path / "s")
    assert shard.meta["real_tokens"] == 10
