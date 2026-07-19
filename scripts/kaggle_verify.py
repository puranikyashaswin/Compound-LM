"""One-shot GPU verification for Kaggle. Answers the only open question.

Everything in this repo is verified except one claim: that mixed precision is
~2x on CUDA tensor cores. That number is the load-bearing assumption behind the
whole cost plan, and it has never been measured on NVIDIA hardware.

This script settles it in roughly ten minutes of free Kaggle quota. It:

  1. reports the GPU and what precision the policy resolves to on it;
  2. measures fp32 vs mixed precision across several tensor sizes, because the
     Apple M2 measurement showed the speedup is strongly size-dependent
     (1.15x at batch 8 x seq 512, 0.89x at batch 2 x seq 256);
  3. checks the loss curve tracks fp32, so a "speedup" that costs accuracy is
     reported as a failure rather than a win;
  4. measures the FLOP-reduction levers (vocabulary, depth, Muon step cost);
  5. prints a verdict, including the honest total for THIS GPU.

Expected on a T4: bf16 unsupported (Turing), so the policy should select
fp16 + GradScaler. If it reports fp32, that is the exact bug this repo was
built to catch -- a run silently paying fp32 prices on tensor-core hardware.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "kaggle-verification.json"))
    args = parser.parse_args()

    print("=" * 70)
    print("KAGGLE GPU VERIFICATION")
    print("=" * 70)
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device. In the Kaggle notebook sidebar set "
            "Settings -> Accelerator -> GPU T4 x2, then re-run."
        )
    name = torch.cuda.get_device_name(0)
    bf16 = torch.cuda.is_bf16_supported()
    print(f"GPU:   {name}")
    print(f"torch: {torch.__version__}")
    print(f"bf16 supported: {bf16}")

    from src.train.systems import SystemsPolicy, resolve_precision
    resolved = resolve_precision(SystemsPolicy(precision="auto"), device="cuda")
    print(f"policy 'auto' resolves to: {resolved.autocast_dtype} "
          f"(GradScaler={resolved.use_grad_scaler})")
    for note in resolved.notes:
        print(f"  note: {note}")
    if resolved.autocast_dtype is None:
        raise SystemExit(
            "FAIL: 'auto' resolved to fp32 on a CUDA device. This is the bug the "
            "systems lever exists to prevent -- the run would forfeit the tensor cores."
        )
    print()

    from scripts.verify_speedup import bench_steps

    # Size sweep: the M2 result showed the precision speedup depends heavily on
    # tensor size, so a single shape would be a misleading measurement.
    shapes = [
        ("small  (b2  x s256, d256)", dict(batch=2, seq=256, d_model=256, n_layers=12, vocab=50257)),
        ("medium (b8  x s512, d512)", dict(batch=8, seq=512, d_model=512, n_layers=6, vocab=16384)),
        ("large  (b16 x s512, d768)", dict(batch=16, seq=512, d_model=768, n_layers=12, vocab=50257)),
    ]

    print("=" * 70)
    print("MIXED PRECISION vs fp32")
    print("=" * 70)
    print(f"{'shape':<28} {'fp32 ms':>9} {'amp ms':>9} {'speedup':>9} {'loss div':>10}")
    print("-" * 70)

    records = []
    for label, shape in shapes:
        try:
            fp32_ms, fp32_losses, _ = bench_steps("cuda", "fp32", steps=args.steps,
                                                  warmup=args.warmup, **shape)
            amp_ms, amp_losses, amp_resolved = bench_steps("cuda", "auto", steps=args.steps,
                                                          warmup=args.warmup, **shape)
        except RuntimeError as error:
            print(f"{label:<28} skipped: {error}")
            continue
        speedup = fp32_ms / amp_ms
        divergence = max(abs(a - b) / max(1.0, abs(a))
                         for a, b in zip(fp32_losses, amp_losses))
        records.append({"shape": label, "config": shape, "fp32_ms": fp32_ms * 1000,
                        "amp_ms": amp_ms * 1000, "speedup": speedup,
                        "loss_divergence": divergence,
                        "dtype": amp_resolved.autocast_dtype})
        print(f"{label:<28} {fp32_ms*1000:9.1f} {amp_ms*1000:9.1f} "
              f"{speedup:8.2f}x {divergence:10.2e}")

    if not records:
        raise SystemExit("every shape failed to run; nothing measured")

    print()
    print("=" * 70)
    print("FLOP-REDUCTION LEVERS (vocabulary, depth, Muon step cost)")
    print("=" * 70)
    from scripts.verify_levers import time_config

    base = dict(vocab=50257, d_model=256, n_layers=12, seq=512, batch=8,
                steps=args.steps, warmup=args.warmup)
    baseline = time_config("cuda", precision="fp32", **base)
    print(f"fp32 baseline: {baseline*1000:.1f} ms/step\n")
    levers = {}
    for label, override in (
        ("vocab 50257->16384", dict(vocab=16384)),
        ("depth 12->6", dict(n_layers=6)),
        ("Muon step cost", dict(use_muon=True)),
        ("vocab + depth + amp", dict(vocab=16384, n_layers=6, precision="auto")),
    ):
        config = dict(base, precision="fp32")
        config.update(override)
        elapsed = time_config("cuda", **config)
        levers[label] = baseline / elapsed
        print(f"  {label:<24} {elapsed*1000:8.1f} ms   {baseline/elapsed:5.2f}x")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    best = max(record["speedup"] for record in records)
    worst_divergence = max(record["loss_divergence"] for record in records)
    print(f"  mixed precision on {name}: up to {best:.2f}x")
    print(f"  max loss divergence vs fp32: {worst_divergence:.2e}")

    equivalence_ok = worst_divergence <= 0.05
    print(f"  equivalence (loss tracks fp32): {'PASS' if equivalence_ok else 'FAIL'}")

    # The honest composite for this GPU: precision x the FLOP levers that the
    # protocol has actually validated. Muon enters as a wall-clock figure.
    muon_step_cost = 1.0 / levers.get("Muon step cost", 1.0)
    muon_wall_clock = 1.80 / muon_step_cost if muon_step_cost else 1.80
    total = best * levers.get("vocab 50257->16384", 1.0) * levers.get("depth 12->6", 1.0)
    print()
    print(f"  precision {best:.2f}x  x  vocab {levers.get('vocab 50257->16384', 1):.2f}x  "
          f"x  depth {levers.get('depth 12->6', 1):.2f}x  =  {total:.2f}x")
    print(f"  Muon step cost measured {muon_step_cost:.2f}x  ->  "
          f"wall-clock benefit {muon_wall_clock:.2f}x (from a 1.80x FLOP win)")
    print(f"  COMPOSITE (throughput+shape+optimizer): {total * muon_wall_clock:.2f}x")
    print()
    if best < 1.5:
        print("  NOTE: precision gain below 1.5x. The 4x plan assumed ~2x here.")
        print("  Re-derive the plan with this number rather than the assumption.")

    payload = {"gpu": name, "torch": torch.__version__, "bf16_supported": bf16,
               "resolved": resolved.as_dict(), "precision_by_shape": records,
               "levers": levers, "composite": total * muon_wall_clock}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  report written: {args.out}")

    if not equivalence_ok:
        raise SystemExit("FAIL: mixed precision changed the loss curve beyond tolerance")


if __name__ == "__main__":
    main()
