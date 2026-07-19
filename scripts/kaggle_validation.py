#!/usr/bin/env python3
"""Run Phase 0.5b validation at a real token budget (~20 tokens/param, ~450M tokens) on GPU."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Ensure framework is in path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train.reference import resumable_checkpoint, train
from src.ledger.compounding import (assert_costs_resolved, compounding_report,
                                    cost_to_score_detail)
from src.ledger.writer import read_entries
from src.data.binshard import PackedShard
from src.data.budget import check_token_budget


def count_params(model_config: dict, sequence_length: int, architecture: str = "reference-v1") -> int:
    from src.model.registry import build_model
    model = build_model(architecture, **model_config, max_seq_len=sequence_length)
    return sum(p.numel() for p in model.parameters())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--steps", type=int, default=13750, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (sequences per step)")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--corpus", default=str(ROOT / "data" / "real-v2"),
                        help="Directory holding binary 'train' and 'heldout' shards "
                             "(build with scripts/build_corpus.py)")
    parser.add_argument("--vocab-size", type=int, default=50257,
                        help="Must match the corpus tokenizer (GPT-2 = 50257)")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--precision", default="fp32",
                        help="fp32 (default, matches the existing ledger), or auto/bf16/fp16 "
                             "for the systems lever. 'auto' picks bf16 on Ampere+ and "
                             "fp16+GradScaler on Turing (Kaggle T4), never fp32 on a GPU.")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model; verify throughput per GPU class")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="Clip gradients to this norm (1.0 is standard). Default off, "
                             "which reproduces the existing ledger; the old call site "
                             "clipped at 1e9, i.e. measured the norm without clipping.")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Applied to 2-D tensors only; norms and biases are exempt")
    parser.add_argument("--eval-batch-size", type=int, default=32,
                        help="Held-out evaluation batch size; affects speed, not scores")
    parser.add_argument("--architecture", default="reference-v1",
                        help="Model architecture for the whole matrix: reference-v1 or "
                             "reex-v2 (RoPE + RMSNorm + SwiGLU). A non-reference "
                             "architecture is recorded as an 'architecture' lever, so "
                             "compare its ledger against a reference-v1 baseline ledger, "
                             "never within one.")
    parser.add_argument("--lr-schedule", action="store_true",
                        help="Warmup + cosine decay, applied identically to every arm")
    parser.add_argument("--max-epochs", type=float, default=4.0,
                        help="Refuse a run that would loop the corpus more than this")
    parser.add_argument("--checkpoint-every", type=int, default=1250,
                        help="Steps between checkpoints (~11 per run at defaults). "
                             "Also the most work a crash can cost, since resume "
                             "restarts from the last one.")
    parser.add_argument("--ledger", default=str(ROOT / "work" / "kaggle-ledger.jsonl"))
    parser.add_argument("--run-dir", default=str(ROOT / "runs" / "kaggle"))
    parser.add_argument("--fresh", action="store_true",
                        help="Discard existing checkpoints and ledger and start over. "
                             "Off by default: re-running resumes from the last good "
                             "checkpoint instead of destroying hours of GPU work.")
    parser.add_argument("--analyze-only", metavar="LEDGER",
                        help="Skip training; run the analysis against an existing "
                             "ledger jsonl (e.g. one downloaded from a Kaggle run)")
    args = parser.parse_args()

    if args.analyze_only:
        analyze(Path(args.analyze_only), args)
        return

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
    
    # 1. Open the binary corpus (memory-mapped; shard size does not bound RAM)
    corpus = Path(args.corpus)
    train_packed = corpus / "train"
    heldout_packed = corpus / "heldout"
    if not train_packed.with_suffix(".meta.json").exists():
        raise SystemExit(
            f"no binary corpus at {corpus}. Build one first, sized for the model:\n"
            f"  python scripts/build_corpus.py --for-params 22400000 "
            f"--sequence-length {args.seq_len} --out-dir {corpus}"
        )
    train_shard = PackedShard(train_packed)
    print(f"\nCorpus: {train_shard.meta['real_tokens']:,} unique train tokens, "
          f"{len(train_shard):,} sequences of {train_shard.sequence_length}")

    # Defaults give ~22.4M parameters; override to smoke-test small or to scale up.
    MODEL_CONFIG = dict(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )

    # 2. Token-budget gate BEFORE any GPU time is spent. The previous run of
    #    this script looped a 1.67M-token corpus 270 times for 3.6 GPU-hours
    #    and could only measure memorization; no other gate here catches that.
    n_params = count_params(MODEL_CONFIG, train_shard.sequence_length, args.architecture)
    budget = check_token_budget(unique_tokens=train_shard.meta["real_tokens"],
                                steps=args.steps, batch_size=args.batch_size,
                                sequence_length=train_shard.sequence_length,
                                n_params=n_params, max_epochs=args.max_epochs)
    print(f"\n--- Token budget ({n_params:,} params) ---")
    print(f"  epochs over corpus: {budget.epochs:.2f}x")
    print(f"  tokens/param:       {budget.tokens_per_param:.1f} "
          f"({budget.chinchilla_ratio:.2f}x Chinchilla)")
    print(f"  status:             {budget.status.upper()}")
    for warning in budget.warnings:
        print(f"  [warn] {warning}")
    if budget.status == "red":
        for failure in budget.failures:
            print(f"  [FAIL] {failure}")
        raise SystemExit("token_budget_gate: refusing to spend GPU hours on a void run")

    ledger_path = Path(args.ledger)
    run_dir = Path(args.run_dir)
    if args.fresh:
        ledger_path.unlink(missing_ok=True)
        shutil.rmtree(run_dir, ignore_errors=True)
        print("\n--fresh: discarded previous checkpoints and ledger")

    checkpoint_every = args.checkpoint_every

    arch_lever = [] if args.architecture == "reference-v1" else ["architecture"]

    def run_one(name, seed, use_muon):
        run_id = f"gpu-{name}-s{seed}" if not arch_lever else f"gpu-{args.architecture}-{name}-s{seed}"
        out_dir = run_dir / (f"{name}-s{seed}" if not arch_lever
                             else f"{args.architecture}-{name}-s{seed}")

        # Auto-resume: a crashed or timed-out session costs one checkpoint
        # interval, not the whole run. Re-running the same command continues.
        resume = None if args.fresh else resumable_checkpoint(out_dir)
        ledgered = {e["run_id"] for e in read_entries(ledger_path)}
        if resume is None and run_id in ledgered:
            raise SystemExit(
                f"{run_id} has ledger rows but no readable checkpoint to resume from. "
                f"Re-running would append duplicate rows for the same token counts. "
                f"Pass --fresh to discard the previous attempt and start over."
            )

        print(f"\n==========================================")
        print(f"Starting run: {run_id}")
        print(f"Levers: {'Muon' if use_muon else 'baseline'}")
        if resume is not None:
            print(f"Resuming from: {resume.name}")
        print(f"==========================================")

        start = time.perf_counter()
        result = train(
            str(train_packed),
            str(out_dir),
            resume=str(resume) if resume is not None else None,
            **MODEL_CONFIG,
            steps=args.steps,
            seed=seed,
            device=device,
            checkpoint_every=checkpoint_every,
            heldout_shard=str(heldout_packed),
            use_muon=use_muon,
            muon_lr=args.muon_lr,
            lr_schedule=args.lr_schedule,
            levers_on=arch_lever + (["optimizer"] if use_muon else []),
            ledger_path=str(ledger_path),
            run_id=run_id,
            batch_size=args.batch_size,
            architecture=args.architecture,
            precision=args.precision,
            compile_model=args.compile,
            grad_clip=args.grad_clip,
            weight_decay=args.weight_decay,
            eval_batch_size=args.eval_batch_size
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
    analyze(ledger_path, args)


def analyze(ledger_path: Path, args) -> None:
    print("\n\n==========================================")
    print("ANALYSIS AND EVALUATION")
    print("==========================================")

    entries = read_entries(ledger_path)
    if not entries:
        raise SystemExit(f"ledger is empty or missing: {ledger_path}")
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

    # Target derived from what every run demonstrably reached (never a guess
    # made before the data existed): 0.9 x the weakest run's best score, the
    # same rule scripts/run_protocol.py uses. A hardcoded target above the
    # reachable range would abort the analysis after hours of paid training.
    max_reached = min(max(pt["score"] for pt in curve) for curve in run_curves.values())
    if max_reached <= 0:
        raise SystemExit("no_capability_signal: every run scored 0 on held-out; nothing to compare")
    target_score = round(max_reached * 0.9, 4)
    print(f"\nDerived capability target score (0.9 x weakest run's best): {target_score:.4f}")

    # Generate compounding report at that target score
    details = {run_id: cost_to_score_detail(curve, target_score)
               for run_id, curve in run_curves.items()}
    # If every run cleared the target before its first checkpoint, no cost was
    # observed and the multipliers would all tie at 1.000x by construction.
    assert_costs_resolved(details)

    comp_rows = []
    for run_id in run_curves:
        cost = details[run_id]["cost"]
        levers = ["optimizer"] if "optimizer" in run_id else []
        seed = 17 if "s17" in run_id else 23
        comp_rows.append({
            "name": run_id,
            "levers": levers,
            "seed": seed,
            "recipe_cost": cost,
            "cost_status": details[run_id]["status"],
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
