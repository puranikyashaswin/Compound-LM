"""Session-limit and Hub-pull behavior that protects multi-session jobs.

A Kaggle job survives on two guarantees: the loop stops INSIDE the time budget
with uploaded state, and a resume pull never mistakes a network failure for a
fresh start (which would retrain step 0 and overwrite the real progress at the
first push). Both were violated silently before; these tests pin the fixes.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.data.binshard import write_packed_shard
from src.train.reference import train

SEQ_LEN = 32
VOCAB = 512


@pytest.fixture
def shard(tmp_path):
    docs = [{"document_id": f"d-{i}", "tokens": [(i * 13 + j) % VOCAB for j in range(64)]}
            for i in range(64)]
    prefix = tmp_path / "corpus" / "train"
    write_packed_shard(docs, prefix, sequence_length=SEQ_LEN, vocab_size=VOCAB,
                       tokenizer_id="test")
    return str(prefix)


def _train(shard, out, **overrides):
    settings = dict(vocab_size=VOCAB, d_model=16, n_layers=1, n_heads=2,
                    steps=40, device="cpu", batch_size=2, checkpoint_every=20)
    settings.update(overrides)
    return train(shard, str(out), **settings)


def test_deadline_stops_at_first_optimizer_boundary(shard, tmp_path):
    # max_seconds already spent: the run must stop at the FIRST optimizer
    # boundary with an off-cadence checkpoint, not run to the next cadence
    # point (step 20) -- that overshoot is where the platform kills sessions.
    result = _train(shard, tmp_path / "run", grad_accum=2, max_seconds=0.0)
    assert result["stopped_early"] is True
    assert result["reached_step"] == 2
    assert (tmp_path / "run" / "checkpoint-00000002.pt").exists()


def test_deadline_is_enforced_even_without_cadence_checkpoints(shard, tmp_path):
    # checkpoint_every=0 used to make max_seconds unenforceable: the stop
    # check lived inside the cadence block, so the loop ran to completion.
    result = _train(shard, tmp_path / "run", checkpoint_every=0, max_seconds=0.0)
    assert result["stopped_early"] is True
    assert result["reached_step"] == 1


def test_checkpoint_cadence_must_align_with_grad_accum(shard, tmp_path):
    with pytest.raises(ValueError, match="multiple of grad_accum"):
        _train(shard, tmp_path / "run", grad_accum=2, checkpoint_every=5, steps=10)


def test_resume_coerces_rng_state_container(shard, tmp_path):
    # map_location=<device> on resume can hand set_rng_state a tensor that is
    # no longer a CPU ByteTensor (that exact TypeError killed a GPU session).
    # CUDA is not available here, so displace the dtype instead and require
    # resume to coerce it back; uint8 values round-trip float32 exactly.
    out = tmp_path / "run"
    _train(shard, out, steps=4, checkpoint_every=2)
    checkpoint = out / "checkpoint-00000002.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state["torch_rng_state"] = state["torch_rng_state"].to(torch.float32)
    torch.save(state, checkpoint)

    result = _train(shard, out, steps=4, checkpoint_every=2, resume=str(checkpoint))
    assert result["reached_step"] == 4


def test_pull_refuses_to_start_fresh_on_network_failure(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.utils import LocalEntryNotFoundError

    import src.train.hf_sync as hf_sync

    monkeypatch.setattr(hf_sync, "ensure_repo", lambda *a, **k: None)
    sync = hf_sync.HubSync("someone/repo", token="test-token")

    def unreachable(*args, **kwargs):
        raise LocalEntryNotFoundError("connection error")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unreachable)
    with pytest.raises(RuntimeError, match="Refusing to assume a fresh start"):
        sync.pull_resume_state(tmp_path)


def test_pull_starts_fresh_only_on_definitive_absence(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError

    import src.train.hf_sync as hf_sync

    monkeypatch.setattr(hf_sync, "ensure_repo", lambda *a, **k: None)
    sync = hf_sync.HubSync("someone/repo", token="test-token")

    def absent(*args, **kwargs):
        raise EntryNotFoundError("404: no resume/state.json")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", absent)
    assert sync.pull_resume_state(tmp_path) == (None, None)
