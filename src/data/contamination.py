"""Data split contamination checks."""
from __future__ import annotations

import json
from pathlib import Path

def assert_disjoint_shards(train_path: str | Path, heldout_path: str | Path) -> None:
    """Ensure that no document text or hash from heldout_path appears in train_path."""
    t_path = Path(train_path)
    h_path = Path(heldout_path)

    # If these are packed files, look for their unpacked counterparts in the same directory
    if "-packed" in t_path.name:
        candidate = t_path.parent / t_path.name.replace("-packed", "")
        if candidate.exists():
            t_path = candidate
    if "-packed" in h_path.name:
        candidate = h_path.parent / h_path.name.replace("-packed", "")
        if candidate.exists():
            h_path = candidate

    train_texts = set()
    train_hashes = set()

    for line in t_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "text" in row and row["text"]:
            train_texts.add(row["text"].strip())
        if "text_sha256" in row and row["text_sha256"]:
            train_hashes.add(row["text_sha256"])

    for line in h_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "text" in row and row["text"]:
            t = row["text"].strip()
            if t in train_texts:
                raise ValueError(f"contamination detected: document text '{t}' is in both train and heldout")
        if "text_sha256" in row and row["text_sha256"]:
            h = row["text_sha256"]
            if h in train_hashes:
                raise ValueError(f"contamination detected: document hash '{h}' is in both train and heldout")
