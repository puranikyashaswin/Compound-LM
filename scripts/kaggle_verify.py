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
    # fp16 and bf16 are measured separately, never only via 'auto'. A T4 run
    # that tested 'auto' alone reported 0.75x and hid the cause: torch claimed
    # bf16 support that was really emulation, and the genuinely fast path
    # (fp16 tensor cores) was never exercised.
    print(f"{'shape':<26} {'fp32 ms':>9} {'fp16':>14} {'bf16':>14}")
    print("-" * 70)

    records = []
    for label, shape in shapes:
        try:
            fp32_ms, fp32_losses, _ = bench_steps("cuda", "fp32", steps=args.steps,
                                                  warmup=args.warmup, **shape)
        except RuntimeError as error:
            print(f"{label:<26} skipped: {error}")
            continue

        row = {"shape": label, "config": shape, "fp32_ms": fp32_ms * 1000, "variants": {}}
        cells = []
        for precision in ("fp16", "bf16"):
            try:
                amp_ms, amp_losses, amp_resolved = bench_steps(
                    "cuda", precision, steps=args.steps, warmup=args.warmup, **shape)
            except RuntimeError as error:
                cells.append(f"{'err':>14}")
                row["variants"][precision] = {"error": str(error)}
                continue
            speedup = fp32_ms / amp_ms
            divergence = max(abs(a - b) / max(1.0, abs(a))
                             for a, b in zip(fp32_losses, amp_losses))
            row["variants"][precision] = {
                "ms": amp_ms * 1000, "speedup": speedup,
                "loss_divergence": divergence,
                "resolved_dtype": amp_resolved.autocast_dtype}
            cells.append(f"{speedup:8.2f}x{'':>5}")
        records.append(row)
        print(f"{label:<26} {fp32_ms*1000:9.1f} {''.join(cells)}")

    print()
    for row in records:
        for precision, data in row["variants"].items():
            if "error" not in data:
                print(f"  {row['shape']:<26} {precision}: {data['speedup']:.2f}x  "
                      f"loss divergence {data['loss_divergence']:.2e}  "
                      f"(ran as {data['resolved_dtype']})")

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
    measured = [(row["shape"], precision, data)
                for row in records for precision, data in row["variants"].items()
                if "error" not in data]
    if not measured:
        raise SystemExit("no precision variant ran successfully")
    best_shape, best_precision, best_data = max(measured, key=lambda item: item[2]["speedup"])
    best = best_data["speedup"]
    worst_divergence = max(data["loss_divergence"] for _, _, data in measured)
    print(f"  fastest precision on {name}: {best_precision} at {best:.2f}x "
          f"({best_shape.strip()})")
    print(f"  max loss divergence vs fp32: {worst_divergence:.2e}")

    auto_dtype = resolved.autocast_dtype
    best_dtype = best_data["resolved_dtype"]
    if auto_dtype != best_dtype:
        print(f"  WARNING: policy 'auto' picks {auto_dtype} but {best_dtype} measured "
              f"faster here. resolve_precision needs updating for this GPU.")

    equivalence_ok = worst_divergence <= 0.05
    print(f"  equivalence (loss tracks fp32): {'PASS' if equivalence_ok else 'FAIL'}")

    # The headline is the DIRECTLY MEASURED combination, not a product of
    # separately-measured levers. Two reasons, both learned the hard way:
    #
    #   1. The per-lever numbers come from different shapes. The precision
    #      figure above is measured on its own sweep; the lever figures use the
    #      lever baseline shape. Multiplying across shapes is not a measurement
    #      of anything.
    #   2. Levers overlap. The product overstates the combination whenever they
    #      compete for the same bottleneck -- which is exactly the quantity this
    #      repo's compounding table exists to report.
    combined = levers.get("vocab + depth + amp")
    same_shape_product = (levers.get("vocab 50257->16384", 1.0)
                          * levers.get("depth 12->6", 1.0))

    print()
    print("  MEASURED TOGETHER (one shape, one run) -- the number to trust:")
    if combined:
        print(f"    vocab + depth + mixed precision: {combined:.2f}x")
        # Precision at the lever shape is implied by the combination, not the
        # standalone sweep, so overlap is computed against what is comparable.
        print(f"    product of vocab x depth alone:  {same_shape_product:.2f}x")
        print(f"    => precision contributed a further "
              f"{combined / same_shape_product:.2f}x on top of them")
    else:
        print("    combination did not run")

    muon_step_cost = 1.0 / levers.get("Muon step cost", 1.0)
    muon_wall_clock = 1.80 / muon_step_cost if muon_step_cost else 1.80
    print()
    print(f"  Muon: steps cost {muon_step_cost:.2f}x more here, so a 1.80x FLOP win")
    print(f"        is {muon_wall_clock:.2f}x in wall clock -- BUT that 1.80x is a")
    print("        toy-scale (64-word corpus) result. It is the weakest number on")
    print("        this page and must be re-measured at real scale before use.")

    if combined:
        print()
        print(f"  VERIFIED ON THIS GPU: {combined:.2f}x "
              f"(equivalence-checked, single shape)")
        print(f"  With Muon, IF its toy multiplier holds: "
              f"{combined * muon_wall_clock:.2f}x (unverified at scale)")

    print()
    if best < 1.5:
        print("  NOTE: best precision gain below 1.5x -- re-derive the plan with")
        print("  this number rather than the 2x assumption.")
    small = min(measured, key=lambda item: item[2]["speedup"])
    print(f"  Size sensitivity: precision ranged {small[2]['speedup']:.2f}x to "
          f"{best:.2f}x across shapes.")
    print("  Small batches forfeit most of the gain -- size the real run accordingly.")

    payload = {"gpu": name, "torch": torch.__version__, "bf16_supported": bf16,
               "resolved": resolved.as_dict(), "precision_by_shape": records,
               "levers": levers,
               "verified_combined": combined,
               "muon_wall_clock_if_toy_holds": muon_wall_clock}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  report written: {args.out}")

    if not equivalence_ok:
        raise SystemExit("FAIL: mixed precision changed the loss curve beyond tolerance")


if __name__ == "__main__":
    main()
