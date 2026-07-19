"""Project GPU-hours for a larger model, calibrated on a run you actually did.

The multipliers in `cost_reduction_plan.py` were computed for the 22.4M
validation config and **do not transfer unchanged to 200M**. The reason is in
the FLOP breakdown: at d_model=256 the output head is ~51% of forward compute,
so right-sizing the vocabulary is a large lever. At d_model=1024 the head is
~20%, and the same change buys much less. A plan that carries the small-model
multiplier to the big model overstates the saving by roughly 25%.

Cost here is quoted in **GPU-hours**, because that is the quantity that is
actually scarce (Kaggle: 30h/week, 12h maximum per session). Dollars are a
derived number and only apply to rented hardware.

Calibrate against a finished run with --ref-params/--ref-tokens/--ref-hours;
the projection then inherits your real MFU instead of a guessed one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ledger.cost_model import analyze_model

CHINCHILLA = 20.0

# Realistic sustained throughput, not vendor peak. MFU for a small model on a
# well-fed pipeline lands around 30-45%; these are peak-tensor-core numbers
# already discounted to that range.
GPUS = {
    "T4 (Kaggle free)": 19.5e12,
    "2x T4 (Kaggle free)": 35.0e12,
    "P100 (Kaggle, no tensor cores)": 6.0e12,
    "L4": 36.0e12,
    "RTX 4090": 40.0e12,
    "A100 40GB": 130.0e12,
    "H100": 400.0e12,
}


def train_flops(params: int, tokens: int) -> float:
    """The 6ND rule: forward+backward over every parameter, per token."""
    return 6.0 * params * tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=float, default=200e6, help="Target model parameters")
    parser.add_argument("--tokens", type=float, default=None,
                        help="Training tokens (default: Chinchilla-optimal 20/param)")
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--ref-params", type=float, default=None,
                        help="A finished run's parameter count, to calibrate throughput")
    parser.add_argument("--ref-tokens", type=float, default=None)
    parser.add_argument("--ref-hours", type=float, default=None)
    args = parser.parse_args()

    params = int(args.params)
    tokens = int(args.tokens) if args.tokens else int(params * CHINCHILLA)
    flops = train_flops(params, tokens)

    breakdown = analyze_model(vocab_size=args.vocab_size, d_model=args.d_model,
                              n_layers=args.n_layers, sequence_length=args.seq_len)

    print("=" * 70)
    print("TARGET RUN")
    print("=" * 70)
    print(f"  parameters:      {params/1e6:.0f}M")
    print(f"  tokens:          {tokens/1e9:.2f}B  ({tokens/params:.0f} per param)")
    print(f"  training FLOPs:  {flops:.3e}")
    print(f"  output head:     {breakdown.head_fraction:.1%} of forward compute")
    print()

    if args.ref_params and args.ref_tokens and args.ref_hours:
        ref_flops = train_flops(int(args.ref_params), int(args.ref_tokens))
        achieved = ref_flops / (args.ref_hours * 3600)
        ratio = flops / ref_flops
        print("=" * 70)
        print("CALIBRATED ON YOUR FINISHED RUN")
        print("=" * 70)
        print(f"  reference:       {args.ref_params/1e6:.0f}M params, "
              f"{args.ref_tokens/1e9:.2f}B tokens, {args.ref_hours:.1f}h")
        print(f"  implied throughput: {achieved/1e12:.1f} TFLOP/s sustained")
        print(f"  target is {ratio:.1f}x that compute")
        print(f"  -> same hardware, same settings: {args.ref_hours * ratio:.1f}h")
        print()
        print("  Compute scales with params x tokens. Holding tokens/param fixed,")
        print("  it therefore scales with the SQUARE of the parameter count:")
        print(f"  {args.ref_params/1e6:.0f}M -> {params/1e6:.0f}M is "
              f"{params/args.ref_params:.1f}x the parameters but "
              f"{ratio:.1f}x the compute.")
        print()

    print("=" * 70)
    print("GPU-HOURS BY HARDWARE (mixed precision, no other levers)")
    print("=" * 70)
    for name, throughput in GPUS.items():
        hours = flops / throughput / 3600
        sessions = ""
        if "Kaggle" in name:
            sessions = f"  [{hours/12:.1f} x 12h sessions, {hours/30:.1f} weekly quotas]"
        print(f"  {name:<32} {hours:7.1f} h{sessions}")

    print()
    print("=" * 70)
    print("WHICH LEVERS STILL APPLY AT THIS SCALE")
    print("=" * 70)
    small = analyze_model(vocab_size=args.vocab_size, d_model=256, n_layers=12,
                          sequence_length=512)
    print(f"  mixed precision      ~2.00x   yes: hardware, scale-independent")
    print(f"  fused AdamW + TF32   ~1.10x   yes")
    print(f"  Muon                 ~1.3-1.8x yes, but re-measure at scale")
    print(f"  depth growth         ~1.15x   BETTER here: transformer is a larger "
          f"share ({1-breakdown.head_fraction:.0%} vs "
          f"{1-small.head_fraction:.0%})")
    print(f"  vocabulary cut       ~1.12x   WEAKER here: head is "
          f"{breakdown.head_fraction:.0%} of forward, was {small.head_fraction:.0%} at 22M")
    print(f"  width rebalance      ~1.00x   GONE: d_model={args.d_model} already "
          f"saturates tensor cores")
    print()
    print("  The 6.35x measured against the 22.4M config does not transfer.")
    print("  Re-derive it here before planning around it.")


if __name__ == "__main__":
    main()
