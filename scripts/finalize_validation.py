#!/usr/bin/env python3
"""Finalize the four-run Kaggle ledger into one evidence-block markdown file."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from statistics import mean, pstdev
from src.ledger.compounding import cost_to_score_detail
from src.ledger.writer import read_entries

EXPECTED = {"gpu-baseline-s17", "gpu-baseline-s23", "gpu-optimizer-s17", "gpu-optimizer-s23"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--output", default="outputs/final-validation-evidence.md")
    ap.add_argument("--target", type=float, default=None)
    args = ap.parse_args()
    entries = read_entries(args.ledger)
    groups = {run_id: sorted([e for e in entries if e["run_id"] == run_id], key=lambda e: e["tokens"])
              for run_id in EXPECTED}
    missing = [k for k, v in groups.items() if not v]
    if missing:
        raise SystemExit(f"missing expected runs: {missing}")
    finals = {k: v[-1] for k, v in groups.items()}
    target = args.target if args.target is not None else .8 * min(float(e["eval_scores"]["val_acc"]) for e in finals.values())
    details = {}
    for run_id, rows in groups.items():
        points = [{"cost": float(e["train_flops"]), "score": float(e["eval_scores"]["val_acc"])} for e in rows]
        details[run_id] = cost_to_score_detail(points, target)
    baseline_costs = [d["cost"] for k, d in details.items() if "baseline" in k and d["cost"] is not None]
    muon_costs = [d["cost"] for k, d in details.items() if "optimizer" in k and d["cost"] is not None]
    multiplier = (mean(baseline_costs) / mean(muon_costs)) if len(baseline_costs) == 2 and len(muon_costs) == 2 else None
    baseline_scores = [float(finals[k]["eval_scores"]["val_acc"]) for k in finals if "baseline" in k]
    muon_scores = [float(finals[k]["eval_scores"]["val_acc"]) for k in finals if "optimizer" in k]
    lines = ["# Kaggle validation evidence", "", f"Target score: {target:.6f}", "", "| Run | Seed | Final accuracy | Cost status | Cost |", "|---|---:|---:|---|---:|"]
    for run_id in sorted(EXPECTED):
        e, d = finals[run_id], details[run_id]
        cost = "n/a" if d["cost"] is None else f"{d['cost']:.6e}"
        lines.append(f"| {run_id} | {e['seed']} | {float(e['eval_scores']['val_acc']):.4%} | {d['status']} | {cost} |")
    lines += ["", f"Baseline seed spread: {pstdev(baseline_scores):.4%} (population SD)",
              f"Muon seed spread: {pstdev(muon_scores):.4%} (population SD)",
              f"Mean cost multiplier (baseline / Muon): {'n/a' if multiplier is None else f'{multiplier:.4f}x'}",
              "Overlap coefficient: 1.0000x for the single optimizer lever (no multi-lever interaction measured).",
              "", "All four runs were required from the ledger; no result is accepted without both seeds."]
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text("\n".join(lines) + "\n")
    print(out)

if __name__ == "__main__":
    main()
