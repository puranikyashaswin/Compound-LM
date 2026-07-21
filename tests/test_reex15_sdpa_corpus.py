"""Reex-1.5 must refuse multi-doc Hub/local corpora so SDPA stays enabled."""
import json

from src.data.binshard import write_packed_shard
from scripts.train_reex15 import (_corpus_allows_sdpa_fast_path, _invalidate_corpus)


DOCS = [
    {"document_id": "a", "tokens": [1, 2, 3, 4, 5]},
    {"document_id": "b", "tokens": [6, 7]},
]


def test_sdpa_gate_accepts_single_document_shards(tmp_path):
    write_packed_shard(DOCS, tmp_path / "train", sequence_length=4, vocab_size=64,
                       tokenizer_id="test", cross_document=False)
    write_packed_shard(DOCS, tmp_path / "heldout", sequence_length=4, vocab_size=64,
                       tokenizer_id="test", cross_document=False)
    assert _corpus_allows_sdpa_fast_path(tmp_path)


def test_sdpa_gate_rejects_multi_document_and_legacy_shards(tmp_path):
    write_packed_shard(DOCS, tmp_path / "train", sequence_length=4, vocab_size=64,
                       tokenizer_id="test", cross_document=True)
    assert not _corpus_allows_sdpa_fast_path(tmp_path)

    # Legacy metas omit the flag; missing means multi-doc (old default).
    meta_path = (tmp_path / "train").with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    del meta["cross_document"]
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    assert not _corpus_allows_sdpa_fast_path(tmp_path)


def test_invalidate_corpus_clears_marker_and_shards(tmp_path):
    write_packed_shard(DOCS, tmp_path / "train", sequence_length=4, vocab_size=64,
                       tokenizer_id="test", cross_document=True)
    (tmp_path / "corpus.json").write_text("{}")
    _invalidate_corpus(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_reusable_gate_rejects_undersized_single_doc_corpus(tmp_path):
    from scripts.train_reex15 import _corpus_is_reusable

    write_packed_shard(DOCS, tmp_path / "reex15-train-bin", sequence_length=4,
                       vocab_size=64, tokenizer_id="test", cross_document=False)
    write_packed_shard(DOCS, tmp_path / "reex15-heldout-bin", sequence_length=4,
                       vocab_size=64, tokenizer_id="test", cross_document=False)
    (tmp_path / "corpus.json").write_text(json.dumps({
        "train": {"packed": str(tmp_path / "reex15-train-bin"), "tokens": 60_000_000},
        "heldout": {"packed": str(tmp_path / "reex15-heldout-bin"), "tokens": 1},
    }))
    assert not _corpus_is_reusable(tmp_path, target_tokens=500_000_000)
    assert _corpus_is_reusable(tmp_path, target_tokens=60_000_000)
