"""Append-only JSONL ledger with duplicate protection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.provenance.core import canonical_json


REQUIRED_FIELDS = {
    "run_id", "config_hash", "commit", "scale", "levers_on", "tokens",
    "wall_clock_s", "gpu_type", "est_cost_usd", "final_loss", "eval_scores",
    "seed", "notes", "fully_accounted_cost_usd",
}


def read_entries(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_entry(path: str | Path, entry: dict[str, Any]) -> None:
    missing = REQUIRED_FIELDS - entry.keys()
    if missing:
        raise ValueError(f"ledger entry missing fields: {sorted(missing)}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = read_entries(target)
    # Check if duplicate is present (same run_id AND same tokens)
    if any(row["run_id"] == entry["run_id"] and row["tokens"] == entry["tokens"] for row in existing):
        raise ValueError(f"run_id at this token count already exists: {entry['run_id']} at {entry['tokens']}")
    target.open("a", encoding="utf-8").write(canonical_json(entry) + "\n")
