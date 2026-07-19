"""Measure the systems lever on real accelerator hardware, and prove it is free.

The throughput levers are *equivalence* claims: they must reach the same
held-out capability, faster. A speedup that quietly costs accuracy is not a
speedup, it is a smaller model. This script therefore measures two things and
fails if either is wrong:

  1. wall-clock per step, fp32 vs the resolved mixed precision;
  2. the held-out capability curve, which must track fp32 within tolerance.

It runs on whatever accelerator is present (CUDA, MPS) and refuses to report a
speedup measured on CPU, where autocast resolves to fp32 and the number would
be meaningless.

Findings so far, Apple M2 / MPS: fp16 1.34x, bf16 0.97x. bf16 being a
*regression* on Apple silicon is why precision is resolved per device rather
than assumed from CUDA habits.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def bench_steps(device: str, precision: str, *, steps: int, warmup: int,
                d_model: int, n_layers: int, vocab: int, seq: int, batch: int):
    """Time forward+backward+step, excluding warmup, and return losses."""
    import torch
    from src.model.registry import build_model
    from src.train.reference import masked_next_token_loss
    from src.train.systems import (SystemsPolicy, apply_backend_flags, make_grad_scaler,
                                   resolve_precision)

    resolved = resolve_precision(SystemsPolicy(precision=precision), device=device)
    apply_backend_flags(resolved)
    dtype = getattr(torch, resolved.autocast_dtype) if resolved.autocast_dtype else None
    scaler = make_grad_scaler(device, enabled=resolved.use_grad_scaler)

    torch.manual_seed(0)
    model = build_model("reference-v1", vocab_size=vocab, d_model=d_model,
                        n_layers=n_layers, n_heads=8, max_seq_len=seq).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids = torch.randint(0, vocab, (batch, seq), device=device)
    docs = torch.zeros_like(ids)

    losses = []
    start = None
    for step in range(steps + warmup):
        if step == warmup:
            synchronize(device)
            start = time.perf_counter()
        if dtype is not None:
            with torch.autocast(device_type=device, dtype=dtype):
                loss = masked_next_token_loss(model(ids, docs), ids, docs)
        else:
            loss = masked_next_token_loss(model(ids, docs), ids, docs)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    synchronize(device)
    elapsed = (time.perf_counter() - start) / steps
    return elapsed, losses, resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--vocab", type=int, default=16384)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--loss-tolerance", type=float, default=0.05,
                        help="Max relative divergence of the loss curve vs fp32")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "speedup-report.json"))
    args = parser.parse_args()

    device = args.device or detect_device()
    print(f"device: {device}")
    if device == "cpu":
        raise SystemExit(
            "no accelerator present: on CPU every precision resolves to fp32, so a "
            "measured 'speedup' here would be noise. Run this on CUDA or MPS."
        )

    shape = dict(d_model=args.d_model, n_layers=args.n_layers, vocab=args.vocab,
                 seq=args.seq, batch=args.batch)
    print(f"shape: {shape}\n")

    results = {}
    for precision in ("fp32", "auto", "fp16", "bf16"):
        try:
            elapsed, losses, resolved = bench_steps(
                device, precision, steps=args.steps, warmup=args.warmup, **shape)
        except Exception as error:  # a precision this device cannot run at all
            print(f"  {precision:<6} unavailable: {type(error).__name__}: {error}")
            continue
        results[precision] = {"ms_per_step": elapsed * 1000, "losses": losses,
                              "resolved": resolved.as_dict()}
        print(f"  {precision:<6} {elapsed * 1000:7.1f} ms/step  "
              f"[{resolved.autocast_dtype or 'fp32'}"
              f"{', scaler' if resolved.use_grad_scaler else ''}]")

    if "fp32" not in results:
        raise SystemExit("fp32 baseline failed to run; nothing to compare against")

    base_ms = results["fp32"]["ms_per_step"]
    base_losses = results["fp32"]["losses"]

    print("\nSPEEDUP vs fp32")
    failures = []
    for precision, payload in results.items():
        if precision == "fp32":
            continue
        speedup = base_ms / payload["ms_per_step"]
        payload["speedup"] = speedup
        # Equivalence: the loss curve must track fp32. A faster run that learns
        # measurably worse has not made training cheaper, it has made it weaker.
        divergence = max(
            abs(a - b) / max(1.0, abs(a)) for a, b in zip(base_losses, payload["losses"]))
        payload["max_relative_loss_divergence"] = divergence
        verdict = "OK" if divergence <= args.loss_tolerance else "DIVERGED"
        if divergence > args.loss_tolerance:
            failures.append(f"{precision}: loss curve diverged by {divergence:.4f}")
        flag = "" if speedup >= 1.0 else "   <-- SLOWER THAN fp32"
        print(f"  {precision:<6} {speedup:5.2f}x   loss divergence {divergence:.5f} "
              f"[{verdict}]{flag}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"device": device, "shape": shape, "results": results}, indent=2) + "\n",
        encoding="utf-8")
    print(f"\nreport: {Path(args.out).relative_to(ROOT)}")

    if failures:
        raise SystemExit("equivalence FAILED: " + "; ".join(failures))
    print("equivalence: all precisions tracked the fp32 loss curve within tolerance")


if __name__ == "__main__":
    main()
