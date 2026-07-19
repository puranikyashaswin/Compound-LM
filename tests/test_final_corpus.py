"""The final experiment's corpus builder must be cheap and still honest.

A 21-minute corpus build on Kaggle traced to two avoidable costs: the SimHash
near-duplicate filter (88% of preparation time, and redundant on an already
deduplicated source) and a packed-JSONL round trip that stores every token as
text and loads it as a Python int. Both are now off by default. These tests pin
that the cheaper path still produces the same data.
"""
import pytest

from scripts.final_experiment import build_corpus
from src.data.loader import is_binshard, open_shard

# allow_download=False keeps the suite hermetic: no network, and no dependence
# on an upstream dataset that can change under us.
CORPUS = dict(target_tokens=60_000, vocab_size=4096, sequence_length=128,
              seed=17, proxy_vocabulary=200, proxy_successors=2,
              allow_download=False)


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    return build_corpus(tmp_path_factory.mktemp("bin"), **CORPUS,
                        binary_shards=True, near_duplicate=False)


@pytest.fixture(scope="module")
def jsonl(tmp_path_factory):
    return build_corpus(tmp_path_factory.mktemp("jsonl"), **CORPUS,
                        binary_shards=False, near_duplicate=False)


def test_binary_shards_are_memory_mapped(binary):
    assert binary["format"] == "binshard"
    assert is_binshard(binary["train"]["packed"]), "trainer would fall back to the JSONL loader"


def test_both_formats_carry_the_same_tokens(binary, jsonl):
    assert binary["train"]["tokens"] == jsonl["train"]["tokens"]
    assert binary["heldout"]["tokens"] == jsonl["heldout"]["tokens"]


def test_both_formats_pack_to_the_same_sequence_count(binary, jsonl):
    assert len(open_shard(binary["train"]["packed"])) == \
        len(open_shard(jsonl["train"]["packed"]))


def test_shards_load_with_the_declared_sequence_length(binary):
    rows = open_shard(binary["train"]["packed"])
    assert rows.sequence_length == CORPUS["sequence_length"]
    ids, docs = rows.batch(0, 2)
    assert len(ids) == 2 and len(ids[0]) == CORPUS["sequence_length"]
    assert len(docs) == 2


def test_tokens_are_inside_the_declared_vocabulary(binary):
    """The fold must happen at preparation, not silently at training time."""
    rows = open_shard(binary["train"]["packed"])
    ids, _ = rows.batch(0, min(8, len(rows)))
    assert max(max(row) for row in ids) < CORPUS["vocab_size"]


def test_train_and_heldout_are_disjoint(binary):
    from src.data.contamination import assert_disjoint_shards
    assert_disjoint_shards(binary["train"]["packed"], binary["heldout"]["packed"])


def test_provenance_records_the_filter_choice(binary):
    """Skipping a dedup filter is a data decision and must be recorded."""
    assert binary["near_duplicate_filter"] is False
    assert binary["source"]


def test_heldout_split_is_non_empty(binary):
    assert binary["heldout"]["tokens"] > 0
    assert binary["heldout"]["documents"] > 0
