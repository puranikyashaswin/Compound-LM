"""Persist training state to the Hugging Face Hub so a session limit costs nothing.

A Kaggle committed run starts with an **empty** working directory and is killed
at the session limit. Local checkpoints therefore do not survive to the next
run: the usual `--resume` path, which looks for files on disk, finds nothing and
silently restarts from zero. For a multi-session job the Hub is not a backup, it
is the only durable state, so resume has to read from it.

Two artefacts, deliberately separated, because they have very different sizes
and very different lifetimes:

* **resume state** (`resume/state.pt`) -- model + optimizer + RNG + data cursor.
  At 193M parameters that is ~2.3GB, so it is uploaded on a *time* cadence
  rather than per checkpoint, and it **overwrites** the previous one. Only the
  newest matters; keeping history here would cost tens of gigabytes for nothing.

* **milestone weights** (`milestones/step-XXXXXXXX/`) -- weights only, in HF
  format, kept forever. ~0.77GB in fp32 and readable by `from_pretrained`, so
  every milestone is an evaluable model rather than an opaque blob.

Uploads are best-effort by design: a transient Hub failure must never kill a
run that is training correctly. Failures are reported and counted, and the loop
continues on local state.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESUME_PATH = "resume/state.pt"
RESUME_META = "resume/state.json"


def _api(token: str | None = None):
    from huggingface_hub import HfApi
    return HfApi(token=token or os.environ.get("HF_TOKEN"))


def ensure_repo(repo_id: str, *, token: str | None = None, private: bool = True) -> None:
    """Create the model repo if it does not exist yet."""
    from huggingface_hub import create_repo
    create_repo(repo_id, token=token or os.environ.get("HF_TOKEN"),
                private=private, exist_ok=True, repo_type="model")


@dataclass
class SyncStatus:
    uploads: int = 0
    failures: int = 0
    last_error: str | None = None
    bytes_sent: int = 0
    milestones: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"uploads": self.uploads, "failures": self.failures,
                "last_error": self.last_error, "bytes_sent": self.bytes_sent,
                "milestones": list(self.milestones)}


class HubSync:
    """Push resume state and milestones; pull the newest resume state back.

    ``min_interval_s`` throttles the expensive resume upload. Set it from how
    much progress you can afford to lose, not from how often checkpoints
    happen: at 2.3GB an upload takes minutes, and doing it every checkpoint
    would spend more time uploading than training.
    """

    def __init__(self, repo_id: str, *, token: str | None = None,
                 min_interval_s: float = 1800.0, private: bool = True,
                 enabled: bool = True):
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        self.min_interval_s = min_interval_s
        self.enabled = enabled and bool(repo_id)
        self.status = SyncStatus()
        self._last_push = 0.0
        if self.enabled:
            if not self.token:
                raise ValueError(
                    "no Hugging Face token: set HF_TOKEN (Kaggle: Add-ons -> Secrets) "
                    "or pass --no-hub-sync to train without durable state"
                )
            ensure_repo(repo_id, token=self.token, private=private)

    # -- resume state --------------------------------------------------

    def push_resume_state(self, local_path: str | Path, *, step: int,
                          force: bool = False) -> bool:
        """Upload the resume state, overwriting the previous one."""
        if not self.enabled:
            return False
        now = time.monotonic()
        if not force and (now - self._last_push) < self.min_interval_s:
            return False
        path = Path(local_path)
        try:
            import json

            api = _api(self.token)
            api.upload_file(path_or_fileobj=str(path), path_in_repo=RESUME_PATH,
                            repo_id=self.repo_id, repo_type="model",
                            commit_message=f"resume state @ step {step}")
            meta = json.dumps({"step": step, "bytes": path.stat().st_size,
                               "updated": time.time()}, indent=2).encode()
            api.upload_file(path_or_fileobj=meta, path_in_repo=RESUME_META,
                            repo_id=self.repo_id, repo_type="model",
                            commit_message=f"resume meta @ step {step}")
            self._last_push = now
            self.status.uploads += 1
            self.status.bytes_sent += path.stat().st_size
            print(f"[hub] resume state pushed at step {step} "
                  f"({path.stat().st_size/1e9:.2f} GB)")
            return True
        except Exception as error:  # noqa: BLE001 - a Hub outage must not kill training
            self.status.failures += 1
            self.status.last_error = f"{type(error).__name__}: {error}"
            print(f"[hub] resume upload FAILED ({error}); training continues on local state")
            return False

    def pull_resume_state(self, local_dir: str | Path) -> tuple[Path | None, int | None]:
        """Fetch the newest resume state, or (None, None) if the repo has none.

        Only a definitive "the file is not on the Hub" starts fresh. Anything
        else -- DNS failure, timeout, a truncated download -- raises instead.
        Uploads may be best-effort, but this read is not: if state exists and
        the pull is mistaken for a fresh start, the run retrains from step 0 and
        its first push OVERWRITES the real progress. A session that dies in its
        first minutes with a clear error costs a retry; a session that quietly
        restarts from zero costs the entire job.
        """
        if not self.enabled:
            return None, None
        import json

        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

        try:
            meta_path = hf_hub_download(self.repo_id, RESUME_META, repo_type="model",
                                        token=self.token)
        except LocalEntryNotFoundError as error:
            # Subclass of EntryNotFoundError, but means "could not reach the
            # Hub", not "the file is not there" -- so it must be caught first.
            raise RuntimeError(
                f"could not reach the Hub to check for resume state on {self.repo_id}: "
                f"{error}. Refusing to assume a fresh start; re-run when the network is back."
            ) from error
        except EntryNotFoundError:
            print(f"[hub] no resume state on {self.repo_id}; starting fresh")
            return None, None
        step = int(json.loads(Path(meta_path).read_text())["step"])
        try:
            state = hf_hub_download(self.repo_id, RESUME_PATH, repo_type="model",
                                    token=self.token,
                                    local_dir=str(local_dir))
        except Exception as error:
            raise RuntimeError(
                f"resume state for step {step} exists on {self.repo_id} but could not be "
                f"fetched ({type(error).__name__}: {error}). Refusing to start fresh over "
                f"real progress; re-run the session."
            ) from error
        print(f"[hub] resuming from step {step} (pulled from {self.repo_id})")
        return Path(state), step

    # -- milestones ----------------------------------------------------

    def push_milestone(self, model_dir: str | Path, *, step: int) -> bool:
        """Upload weights-only HF-format output, kept permanently."""
        if not self.enabled:
            return False
        try:
            _api(self.token).upload_folder(
                folder_path=str(model_dir), repo_id=self.repo_id, repo_type="model",
                path_in_repo=f"milestones/step-{step:08d}",
                commit_message=f"milestone weights @ step {step}")
            self.status.milestones.append(f"step-{step:08d}")
            print(f"[hub] milestone step-{step:08d} pushed")
            return True
        except Exception as error:  # noqa: BLE001
            self.status.failures += 1
            self.status.last_error = f"{type(error).__name__}: {error}"
            print(f"[hub] milestone upload FAILED ({error}); training continues")
            return False

    def push_file(self, local_path: str | Path, path_in_repo: str) -> bool:
        """Upload a small artefact (ledger, evidence) alongside the weights."""
        if not self.enabled:
            return False
        try:
            _api(self.token).upload_file(
                path_or_fileobj=str(local_path), path_in_repo=path_in_repo,
                repo_id=self.repo_id, repo_type="model",
                commit_message=f"update {path_in_repo}")
            return True
        except Exception as error:  # noqa: BLE001
            self.status.failures += 1
            self.status.last_error = f"{type(error).__name__}: {error}"
            return False

    # -- corpus cache --------------------------------------------------

    def push_corpus(self, corpus_dir: str | Path) -> bool:
        """Store the built shards so later sessions download instead of rebuild.

        A Kaggle session wipes its working directory, so an uncached corpus is
        re-derived from scratch every run -- for Reex-1.5 that is ~35 minutes of
        streaming, skipping and tokenising, repeated every session for a
        byte-identical result. Uploading the shards once turns that into a
        download.
        """
        if not self.enabled:
            return False
        try:
            _api(self.token).upload_folder(
                folder_path=str(corpus_dir), repo_id=self.repo_id, repo_type="model",
                path_in_repo="corpus", commit_message="corpus shards")
            print("[hub] corpus cached for later sessions")
            return True
        except Exception as error:  # noqa: BLE001
            self.status.failures += 1
            self.status.last_error = f"{type(error).__name__}: {error}"
            print(f"[hub] corpus upload FAILED ({error}); later sessions will rebuild")
            return False

    def pull_corpus(self, corpus_dir: str | Path) -> bool:
        """Restore cached shards, or return False if none are stored yet."""
        if not self.enabled:
            return False
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(self.repo_id, repo_type="model", allow_patterns="corpus/*",
                              local_dir=str(Path(corpus_dir).parent), token=self.token)
            return (Path(corpus_dir) / "corpus.json").exists()
        except Exception as error:  # noqa: BLE001
            print(f"[hub] no cached corpus ({type(error).__name__}); building it")
            return False
