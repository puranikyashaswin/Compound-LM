"""Deterministic smoke trainer used to test the audit spine without GPUs."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from src.health.check import check_checkpoint
from src.ledger.writer import append_entry
from src.provenance.core import config_hash, load_config, make_manifest


def run(config_path: str, *, tokens: int, ledger_path: str = "ledger/runs.jsonl") -> dict:
    config = load_config(config_path)
    seed = int(config["seed"])
    rng = random.Random(seed)
    started = time.perf_counter()
    losses = []
    loss = 6.0
    for step in range(max(1, tokens // 1000)):
        loss *= 0.9995
        loss += (rng.random() - 0.5) * 0.0001
        losses.append(loss)
    run_id = f"smoke-{config_hash(config)[:12]}-{seed}"
    checkpoint_blob = json.dumps({"losses": losses}, separators=(",", ":")).encode()
    checkpoint_hash = hashlib.sha256(checkpoint_blob).hexdigest()
    report = check_checkpoint(
        loss=losses[-1], grad_norm=1.0, median_grad_norm=1.0,
        checkpoint_hash=checkpoint_hash, provenance_ok=True,
    )
    if report.status == "red":
        raise RuntimeError(report.failures)
    wall = time.perf_counter() - started
    fully_accounted = float(config.get("data", {}).get("prep_cost_usd", 0.0))
    entry = {
        "run_id": run_id, "config_hash": config_hash(config),
        "commit": make_manifest(config, run_id)["git_commit"],
        "scale": config["scale"], "levers_on": config.get("levers", []),
        "tokens": tokens, "wall_clock_s": wall, "gpu_type": "cpu-smoke",
        "est_cost_usd": 0.0, "fully_accounted_cost_usd": fully_accounted,
        "final_loss": loss, "eval_scores": {"smoke": max(0.0, 1.0 - loss / 6.0)},
        "seed": seed, "notes": "deterministic audit-spine smoke run",
        "checkpoint_hash": checkpoint_hash, "health": report.as_dict(),
        "provenance": make_manifest(config, run_id),
    }
    append_entry(ledger_path, entry)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokens", type=int, default=10_000)
    parser.add_argument("--ledger", default="ledger/runs.jsonl")
    args = parser.parse_args()
    print(json.dumps(run(args.config, tokens=args.tokens, ledger_path=args.ledger), indent=2))


if __name__ == "__main__":
    main()
