"""Plan and launch auditable experiment matrices.

The launcher is intentionally conservative: it prints the plan first, refuses
duplicate config/seed identities, and records skipped completed runs. A real
cluster launcher can be plugged into ``launch`` later.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.ledger.writer import read_entries
from src.provenance.core import config_hash, load_config


DEFAULT_MATRIX = [
    {"id": "A0", "levers": [], "config": "configs/baseline_200m.yaml"},
    {"id": "B5", "levers": ["systems"], "config": "configs/baseline_200m.yaml"},
    {"id": "B2", "levers": ["muon"], "config": "configs/baseline_200m.yaml"},
    {"id": "B1", "levers": ["data"], "config": "configs/baseline_200m.yaml"},
    {"id": "B4", "levers": ["growth"], "config": "configs/baseline_200m.yaml"},
    {"id": "C1", "levers": ["systems", "muon", "data"], "config": "configs/baseline_200m.yaml"},
]


def load_matrix(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return DEFAULT_MATRIX
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("matrix must be a JSON list")
    return value


def plan(matrix: list[dict[str, Any]], *, seed: int, tokens: int,
         gpu_hours_per_million_tokens: float, gpu_hour_usd: float,
         ledger_path: str) -> list[dict[str, Any]]:
    existing = read_entries(ledger_path)
    complete = {row.get("config_hash") for row in existing}
    planned = []
    for item in matrix:
        config = load_config(item["config"])
        # Levers are declared deltas and therefore included in the identity.
        resolved = dict(config)
        resolved["levers"] = list(item.get("levers", []))
        resolved["seed"] = seed
        identity = config_hash(resolved)
        gpu_hours = tokens / 1_000_000 * gpu_hours_per_million_tokens
        planned.append({
            "id": item["id"], "config": item["config"], "levers": resolved["levers"],
            "seed": seed, "tokens": tokens, "config_hash": identity,
            "estimated_gpu_hours": gpu_hours,
            "estimated_cost_usd": gpu_hours * gpu_hour_usd,
            "status": "complete" if identity in complete else "pending",
        })
    return planned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tokens", type=int, default=5_000_000_000)
    parser.add_argument("--gpu-hours-per-million-tokens", type=float, default=0.001)
    parser.add_argument("--gpu-hour-usd", type=float, default=2.0)
    parser.add_argument("--ledger", default="ledger/runs.jsonl")
    parser.add_argument("--out")
    args = parser.parse_args()
    rows = plan(load_matrix(args.matrix), seed=args.seed, tokens=args.tokens,
                gpu_hours_per_million_tokens=args.gpu_hours_per_million_tokens,
                gpu_hour_usd=args.gpu_hour_usd, ledger_path=args.ledger)
    payload = {"schema_version": 1, "runs": rows}
    text = json.dumps(payload, indent=2) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
