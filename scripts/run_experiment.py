"""Single entry point for a reproducible COMPOUND-LM experiment."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.provenance.core import config_hash, load_config, make_manifest
from src.train.systems import SystemsPolicy, inspect_runtime


def run(config_path: str, *, shard: str, output_dir: str, steps: int,
        ledger: str | None = None, use_muon: bool = False,
        device: str = "auto", evaluate: bool = False) -> dict:
    config = load_config(config_path)
    run_id = f"{config.get('run_name', 'experiment')}-{config_hash(config)[:12]}-{config.get('seed', 0)}"
    manifest = make_manifest(config, run_id)
    policy = SystemsPolicy(precision=config.get("training", {}).get("precision", "bf16"),
                           compile=config.get("training", {}).get("compile", False),
                           fp8=config.get("training", {}).get("fp8", False), device=device)
    runtime = inspect_runtime(policy)
    result = {"schema_version": 1, "run_id": run_id, "status": "blocked",
              "config_hash": config_hash(config), "manifest": manifest,
              "runtime": runtime, "started_at": time.time()}
    shard_path = Path(shard)
    if not shard_path.exists():
        result["reason"] = f"packed shard does not exist: {shard}"
    elif runtime["active"].get("torch") is not True:
        result["reason"] = ("PyTorch is unavailable; run `python scripts/bootstrap.py` "
                            "to install the training runtime and provision data")
    else:
        from src.train.reference import train
        result["status"] = "completed"
        result["training"] = train(shard, output_dir, steps=steps, seed=config.get("seed", 17),
                                    device=device if device != "auto" else "cpu", ledger_path=ledger,
                                    run_id=run_id, use_muon=use_muon)
        if evaluate:
            from evals.run import run as evaluate_checkpoint
            evaluation_path = Path(output_dir) / "evaluation-E-v1.json"
            result["evaluation"] = evaluate_checkpoint(
                result["training"]["checkpoint"], str(evaluation_path), device=device,
            )
    result["finished_at"] = time.time()
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    (target / "run-manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--shard", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--ledger"); parser.add_argument("--device", default="auto")
    parser.add_argument("--use-muon", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    result = run(args.config, shard=args.shard, output_dir=args.output_dir, steps=args.steps,
                 ledger=args.ledger, use_muon=args.use_muon, device=args.device,
                 evaluate=args.evaluate)
    print(json.dumps(result, indent=2))
    if result["status"] == "blocked":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
