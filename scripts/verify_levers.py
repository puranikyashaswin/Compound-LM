"""Measure every cost lever's wall-clock on real hardware, one at a time.

`cost_reduction_plan.py` argues from arithmetic. This measures. The two must
be reconciled before a plan is trusted, because arithmetic FLOP counts assume
the hardware converts FLOPs to seconds at a constant rate, and it does not:
small matmuls stall, and precision speedups depend entirely on whether the
device has dedicated units for that dtype.

Levers split by how portable the measurement is:

  - **FLOP-reduction levers** (vocabulary, depth) remove work. They transfer
    across hardware reasonably well, so measuring them here is informative
    for a GPU run.
  - **Precision levers** are pure hardware. An Apple M2 measurement says
    almost nothing about a T4's tensor cores. Reported, but flagged.

Every lever is timed against the same fp32 baseline on the same shape, so the
numbers compose.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PORTABLE = "FLOP reduction; transfers across hardware"
HARDWARE = "hardware-specific; does NOT transfer to CUDA"


def synchronize(device: str) -> None:
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_config(device: str, *, vocab: int, d_model: int, n_layers: int,
                seq: int, batch: int, precision: str, steps: int, warmup: int,
                use_muon: bool = False) -> float:
    """Median seconds per training step for one configuration."""
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

    muon = None
    if use_muon:
        from src.optim.muon import Muon, partition_named_parameters
        partition = partition_named_parameters(model.named_parameters())
        by_name = dict(model.named_parameters())
        muon = Muon([by_name[n] for n in partition.muon], lr=0.02)
        optimizer = torch.optim.AdamW([by_name[n] for n in partition.adamw], lr=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ids = torch.randint(0, vocab, (batch, seq), device=device)
    docs = torch.zeros_like(ids)

    per_step = []
    for index in range(steps + warmup):
        synchronize(device)
        start = time.perf_counter()
        if dtype is not None:
            with torch.autocast(device_type=device, dtype=dtype):
                loss = masked_next_token_loss(model(ids, docs), ids, docs)
        else:
            loss = masked_next_token_loss(model(ids, docs), ids, docs)
        optimizer.zero_grad(set_to_none=True)
        if muon is not None:
            muon.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            if muon is not None:
                scaler.unscale_(muon)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        if scaler.is_enabled():
            scaler.step(optimizer)
            if muon is not None:
                scaler.step(muon)
            scaler.update()
        else:
            optimizer.step()
            if muon is not None:
                muon.step()
        synchronize(device)
        if index >= warmup:
            per_step.append(time.perf_counter() - start)
    return statistics.median(per_step)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--vocab", type=int, default=50257)
    parser.add_argument("--target-vocab", type=int, default=16384)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "lever-measurements.json"))
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cpu":
        raise SystemExit("no accelerator: timings on CPU would not represent a GPU run")

    base = dict(vocab=args.vocab, d_model=args.d_model, n_layers=args.n_layers,
                seq=args.seq, batch=args.batch, steps=args.steps, warmup=args.warmup)

    print(f"device: {device}   torch {torch.__version__}")
    print(f"baseline shape: vocab={args.vocab} d_model={args.d_model} "
          f"n_layers={args.n_layers} seq={args.seq} batch={args.batch}\n")

    print("measuring fp32 baseline ...")
    baseline = time_config(device, precision="fp32", **base)
    print(f"  {baseline * 1000:.1f} ms/step\n")

    trials = [
        ("mixed precision", dict(base, precision="auto"), HARDWARE),
        (f"vocab {args.vocab}->{args.target_vocab}",
         dict(base, vocab=args.target_vocab, precision="fp32"), PORTABLE),
        (f"depth {args.n_layers}->{args.n_layers // 2} (growth phase)",
         dict(base, n_layers=args.n_layers // 2, precision="fp32"), PORTABLE),
        ("Muon optimizer step cost", dict(base, precision="fp32", use_muon=True), PORTABLE),
        ("ALL: mixed precision + small vocab + half depth",
         dict(base, vocab=args.target_vocab, n_layers=args.n_layers // 2,
              precision="auto"), "combined"),
    ]

    results = {"device": device, "baseline_ms": baseline * 1000, "levers": {}}
    print(f"{'lever':<48} {'ms/step':>9} {'speedup':>9}")
    print("-" * 70)
    for name, config, kind in trials:
        elapsed = time_config(device, **config)
        speedup = baseline / elapsed
        results["levers"][name] = {"ms_per_step": elapsed * 1000, "speedup": speedup,
                                   "kind": kind}
        print(f"{name:<48} {elapsed * 1000:9.1f} {speedup:8.2f}x")

    print()
    print("NOTES")
    for name, payload in results["levers"].items():
        print(f"  {name}: {payload['kind']}")

    print()
    print("Muon's number above is step *cost*, not benefit: Newton-Schulz makes")
    print("each step more expensive, and the lever wins by needing fewer steps.")
    print("Its capability gain is measured by the protocol, not by this script.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nreport: {Path(args.out).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
