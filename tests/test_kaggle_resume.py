"""Auto-resume must engage from the script, not just from train().

A multi-hour GPU run that dies must cost one checkpoint interval, not the run.
These drive scripts/kaggle_validation.py as a real subprocess and SIGKILL it
mid-training -- the closest reachable analogue of the process dying on Kaggle.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.data.binshard import write_packed_shard
from src.ledger.writer import read_entries

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "kaggle_validation.py"
SEQ_LEN = 32
CHECKPOINT_EVERY = 2
# Small vocab keeps the tied embedding from dominating the parameter count,
# so a test-sized corpus still clears the budget gate honestly.
VOCAB = 512


@pytest.fixture
def corpus(tmp_path):
    """Tiny binary corpus big enough that the budget gate passes for a tiny model."""
    def docs(prefix, count):
        return [{"document_id": f"{prefix}-{i}", "text_sha256": f"{prefix}hash{i}",
                 "tokens": [(i * 13 + j) % VOCAB for j in range(64)]}
                for i in range(count)]
    out = tmp_path / "corpus"
    write_packed_shard(docs("t", 400), out / "train", sequence_length=SEQ_LEN,
                       vocab_size=VOCAB, tokenizer_id="test")
    write_packed_shard(docs("h", 40), out / "heldout", sequence_length=SEQ_LEN,
                       vocab_size=VOCAB, tokenizer_id="test")
    return out


def _command(corpus, tmp_path, steps, *extra):
    return [sys.executable, str(SCRIPT),
            "--corpus", str(corpus), "--device", "cpu",
            "--steps", str(steps), "--batch-size", "2", "--seq-len", str(SEQ_LEN),
            "--vocab-size", str(VOCAB),
            "--d-model", "16", "--n-layers", "1", "--n-heads", "2",
            "--checkpoint-every", str(CHECKPOINT_EVERY),
            "--ledger", str(tmp_path / "ledger.jsonl"),
            "--run-dir", str(tmp_path / "runs"),
            "--max-epochs", "1000", *extra]


def _run(command, timeout=300):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(command, cwd=ROOT, env=env, text=True, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _kill_after_checkpoints(command, checkpoint_dir: Path, wanted: int, timeout=180):
    """Start the script and SIGKILL it once it has written `wanted` checkpoints."""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    process = subprocess.Popen(command, cwd=ROOT, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            if len(list(checkpoint_dir.glob("checkpoint-*.pt"))) >= wanted:
                process.send_signal(signal.SIGKILL)
                break
            if process.poll() is not None:
                break
            time.sleep(0.2)
        else:
            process.kill()
            raise AssertionError("script never reached the requested checkpoint count")
    finally:
        output = process.communicate()[0]
    return output


def test_rerunning_after_a_kill_resumes_and_keeps_the_ledger(corpus, tmp_path):
    steps = 12
    command = _command(corpus, tmp_path, steps)
    first_run = tmp_path / "runs" / "baseline-s17"

    _kill_after_checkpoints(command, first_run, wanted=3)

    ledger = tmp_path / "ledger.jsonl"
    killed_rows = read_entries(ledger)
    killed_checkpoints = sorted(p.name for p in first_run.glob("checkpoint-*.pt"))
    assert killed_checkpoints, "no checkpoint survived the kill"
    assert killed_rows, "no ledger rows survived the kill"
    resume_from = killed_checkpoints[-1]

    # The same command again -- the operator's natural reaction to a dead run.
    result = _run(command)
    assert result.returncode == 0, result.stdout

    assert f"Resuming from: {resume_from}" in result.stdout, result.stdout
    # Progress kept, not restarted: rows only ever appended.
    final_rows = read_entries(ledger)
    assert len(final_rows) > len(killed_rows)
    before = [(r["run_id"], r["tokens"]) for r in killed_rows]
    after = [(r["run_id"], r["tokens"]) for r in final_rows]
    assert after[:len(before)] == before, "pre-kill ledger rows were altered"
    assert len(after) == len(set(after)), "duplicate ledger rows after resume"
    # And the run actually finished rather than resuming into a stall.
    assert (first_run / f"checkpoint-{steps:08d}.pt").exists()


def test_fresh_is_opt_in_so_a_plain_rerun_never_deletes_the_ledger(corpus, tmp_path):
    command = _command(corpus, tmp_path, 12)
    assert _run(command).returncode == 0
    ledger = tmp_path / "ledger.jsonl"
    first_rows = read_entries(ledger)
    assert first_rows

    # Default rerun: ledger must survive untouched.
    _run(command)
    assert read_entries(ledger) == first_rows, "a plain rerun mutated the ledger"

    # Only --fresh discards it.
    fresh = _run(_command(corpus, tmp_path, 12, "--fresh"))
    assert fresh.returncode == 0, fresh.stdout
    assert "--fresh: discarded previous checkpoints and ledger" in fresh.stdout
    assert "Resuming from" not in fresh.stdout
    assert len(read_entries(ledger)) == len(first_rows)


def test_a_truncated_checkpoint_is_skipped_rather_than_crashing_the_resume(corpus, tmp_path):
    """The crash under investigation can leave a half-written checkpoint."""
    command = _command(corpus, tmp_path, 12)
    first_run = tmp_path / "runs" / "baseline-s17"
    _kill_after_checkpoints(command, first_run, wanted=3)

    checkpoints = sorted(first_run.glob("checkpoint-*.pt"))
    newest = checkpoints[-1]
    newest.write_bytes(newest.read_bytes()[:64])  # truncate, as a dead torch.save would
    survivor = checkpoints[-2].name

    result = _run(command)
    assert result.returncode == 0, result.stdout
    # Resumes from the last good one, not the truncated newest.
    assert f"Resuming from: {survivor}" in result.stdout, result.stdout
    # The truncated file is discarded and then legitimately regenerated by the
    # replay, so it must exist again and be readable this time.
    import torch
    torch.load(newest, map_location="cpu", weights_only=False)


def test_ledger_without_a_checkpoint_refuses_instead_of_duplicating(corpus, tmp_path):
    command = _command(corpus, tmp_path, 12)
    assert _run(command).returncode == 0
    # Checkpoints lost (e.g. a wiped session) but the ledger kept.
    import shutil
    shutil.rmtree(tmp_path / "runs" / "baseline-s17")

    result = _run(command)
    assert result.returncode != 0
    assert "has ledger rows but no readable checkpoint" in result.stdout
    assert "--fresh" in result.stdout
