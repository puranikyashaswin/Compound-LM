#!/usr/bin/env python3
"""Run Phase 0.5b validation at a real token budget (~20 tokens/param, ~450M tokens) on GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Ensure framework is in path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train.reference import train
from src.ledger.compounding import compounding_report, cost_to_score
from src.ledger.writer import read_entries
from src.data.packing import pack_shard

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--steps", type=int, default=13750, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (sequences per step)")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    
    # 1. Repack shards to the target sequence length (512+)
    print(f"\n--- Repacking shards to sequence length {args.seq_len} ---")
    data_dir = ROOT / "data" / "real-v1"
    train_packed = data_dir / f"train-packed-{args.seq_len}.jsonl"
    heldout_packed = data_dir / f"heldout-packed-{args.seq_len}.jsonl"

    pack_shard(data_dir / "train.jsonl", train_packed, sequence_length=args.seq_len)
    pack_shard(data_dir / "heldout.jsonl", heldout_packed, sequence_length=args.seq_len)
    
    # Model configuration (~22.4M parameters)
    MODEL_CONFIG = dict(
        vocab_size=50257,
        d_model=256,
        n_layers=12,
        n_heads=8
    )

    ledger_path = ROOT / "work" / "kaggle-ledger.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()

    checkpoint_every = 1250 # 11 checkpoints per run

    def run_one(name, seed, use_muon):
        run_id = f"gpu-{name}-s{seed}"
        out_dir = ROOT / "runs" / "kaggle" / f"{name}-s{seed}"
        print(f"\n==========================================")
        print(f"Starting run: {run_id}")
        print(f"Levers: {'Muon' if use_muon else 'baseline'}")
        print(f"==========================================")
        
        start = time.perf_counter()
        result = train(
            str(train_packed),
            str(out_dir),
            **MODEL_CONFIG,
            steps=args.steps,
            seed=seed,
            device=device,
            checkpoint_every=checkpoint_every,
            heldout_shard=str(heldout_packed),
            use_muon=use_muon,
            levers_on=["optimizer"] if use_muon else [],
            ledger_path=str(ledger_path),
            run_id=run_id,
            batch_size=args.batch_size
        )
        elapsed = time.perf_counter() - start
        print(f"Finished {run_id} in {elapsed:.1f}s.")
        print(f"Final loss: {result['final_loss']:.4f}, Final held-out Acc: {result['eval_scores']['val_acc']:.4f}")
        return result

    # Run the 4-run matrix
    run_one("baseline", 17, use_muon=False)
    run_one("baseline", 23, use_muon=False)
    run_one("optimizer", 17, use_muon=True)
    run_one("optimizer", 23, use_muon=True)

    # 4. Analysis and Compounding Report
    print("\n\n==========================================")
    print("ANALYSIS AND EVALUATION")
    print("==========================================")
    
    entries = read_entries(ledger_path)
    run_curves = {}
    for entry in entries:
        run_id = entry["run_id"]
        if run_id not in run_curves:
            run_curves[run_id] = []
        run_curves[run_id].append({
            "step": entry["tokens"] // (args.batch_size * args.seq_len),
            "cost": float(entry["train_flops"]),
            "score": float(entry["eval_scores"]["val_acc"]),
            "loss": float(entry["final_loss"])
        })

    for run_id in run_curves:
        run_curves[run_id] = sorted(run_curves[run_id], key=lambda x: x["step"])

    # Compute baseline accuracy details
    b17_final = run_curves["gpu-baseline-s17"][-1]["score"]
    b23_final = run_curves["gpu-baseline-s23"][-1]["score"]
    
    m17_final = run_curves["gpu-optimizer-s17"][-1]["score"]
    m23_final = run_curves["gpu-optimizer-s23"][-1]["score"]

    print(f"\nFinal top-1 accuracies:")
    print(f"  Baseline s17: {b17_final:.4f}")
    print(f"  Baseline s23: {b23_final:.4f}")
    print(f"  Muon s17:     {m17_final:.4f}")
    print(f"  Muon s23:     {m23_final:.4f}")

    # Explicitly find target score well clear of early noise
    target_score = 0.15  # Hardcoded capability target (15.0% accuracy), well clear of noise floor
    print(f"\nUsing hardcoded capability target score: {target_score:.4f}")

    # Generate compounding report at that target score
    comp_rows = []
    for run_id, curve in run_curves.items():
        cost = cost_to_score(curve, target_score)
        levers = ["optimizer"] if "optimizer" in run_id else []
        seed = 17 if "s17" in run_id else 23
        comp_rows.append({
            "name": run_id,
            "levers": levers,
            "seed": seed,
            "recipe_cost": cost,
            "reached": cost is not None
        })

    # Raise error if target_score is not reached by all runs
    for row in comp_rows:
        if not row["reached"]:
            raise ValueError(f"Run {row['name']} did not reach the target score of {target_score}. Increase steps or decrease target_score.")

    report = compounding_report(comp_rows, target_score=target_score)

    print("\n--- Compounding Report ---")
    for row in report["rows"]:
        levers_str = ", ".join(row["levers"]) or "baseline"
        overlap = row["overlap_coefficient"]
        overlap_str = f"{overlap:.3f}×" if overlap is not None else "n/a"
        print(f"  {row['name']:<22} levers={levers_str:<10} seed={row['seed']} cost={row['recipe_cost']:.3e} multiplier={row['observed_multiplier']:.3f}× overlap={overlap_str}")

    # Compute baseline seed spread in cost-to-target
    cost_b17 = next(r["recipe_cost"] for r in report["rows"] if r["name"] == "gpu-baseline-s17")
    cost_b23 = next(r["recipe_cost"] for r in report["rows"] if r["name"] == "gpu-baseline-s23")
    baseline_multiplier_spread = abs(1.0 - (cost_b17 / cost_b23))
    print(f"\nBaseline cost-to-target multiplier noise band: {baseline_multiplier_spread:.4f}")

    muon_s17_mult = next(r["observed_multiplier"] for r in report["rows"] if r["name"] == "gpu-optimizer-s17")
    muon_s23_mult = next(r["observed_multiplier"] for r in report["rows"] if r["name"] == "gpu-optimizer-s23")
    avg_muon_mult = (muon_s17_mult + muon_s23_mult) / 2.0
    print(f"Average Muon Multiplier: {avg_muon_mult:.3f}×")

    # Math self-consistency assert: A run's multiplier must be positive if reached
    for row in report["rows"]:
        reached = next(r["reached"] for r in comp_rows if r["name"] == row["name"])
        mult = row.get("observed_multiplier")
        if reached:
            assert mult is not None and mult > 0.0, f"Math contradiction: {row['name']} reached target but has invalid multiplier {mult}"
        else:
            assert mult is None, f"Math contradiction: {row['name']} did not reach target but has multiplier {mult}"

    # Consistent warning: check if final accuracy ranking aligns with cost-to-target ranking
    accuracy_agrees = (m17_final >= b17_final and muon_s17_mult >= 1.0) and (m23_final >= b23_final and muon_s23_mult >= 1.0)
    if not accuracy_agrees:
        print(f"\n[WARNING] final accuracy ranking and cost-to-target ranking disagree.")
        print(f"          This can occur if Muon starts slower but has higher learning capacity.")
    else:
        print(f"\nConsistent check (accuracy vs multiplier ranks agree): PASS")

    out_report = {
        "tokens_per_param": (args.steps * args.batch_size * args.seq_len) / 22.4e6,
        "baseline_spread": baseline_multiplier_spread,
        "target_score": target_score,
        "avg_muon_multiplier": avg_muon_mult,
        "clears_noise_band": avg_muon_mult > (1.0 + baseline_multiplier_spread),
        "accuracy_vs_multiplier_consistent": accuracy_agrees,
        "report": report
    }
    
    out_file = ROOT / "outputs" / "kaggle-validation-report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out_report, indent=2) + "\n", encoding="utf-8")
    print(f"\nValidation report saved to: {out_file}")

if __name__ == "__main__":
    main()
