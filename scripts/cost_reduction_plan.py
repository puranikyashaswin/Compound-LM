"""Plan a >=2x cost reduction for the Kaggle validation run, from arithmetic.

Prints where the current run's budget goes, then each candidate lever with its
expected multiplier, its evidence class, and the compounded total. Levers are
separated by *how the gain is obtained*, because they combine differently:

  - throughput levers cut seconds per step (dollars only; FLOPs unchanged);
  - shape levers cut FLOPs per token;
  - algorithmic levers cut the number of steps needed to reach the target.

Multipliers across those three axes are close to independent, so they compound.
Two algorithmic levers do NOT -- that is exactly what the protocol's overlap
coefficient measures, and why only measured ones are counted here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.growth.depth import growth_savings
from src.ledger.cost_model import (analyze_model, vocab_resize_multiplier,
                                   wall_clock_multiplier)

# Measured by scripts/verify_levers.py on Apple M2 / MPS.
MUON_FLOP_MULTIPLIER = 1.80
MUON_STEP_COST = 1.39
MUON_WALL_CLOCK = wall_clock_multiplier(flop_multiplier=MUON_FLOP_MULTIPLIER,
                                        step_cost_ratio=MUON_STEP_COST)

# Evidence classes, strongest first. The plan is only as good as its weakest
# load-bearing assumption, so each lever states which one it rests on.
MEASURED = "measured on hardware here"
ARITHMETIC = "arithmetic identity"
LITERATURE = "published result, unverified here"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=13750)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--baseline-hours", type=float, default=3.6,
                        help="Measured wall-clock of one fp32 baseline run")
    parser.add_argument("--usd-per-hour", type=float, default=0.35,
                        help="Rented GPU price; Kaggle itself is free, so this "
                             "prices the same plan on rented hardware")
    parser.add_argument("--target-vocab", type=int, default=16384)
    parser.add_argument("--tokenization-penalty", type=float, default=1.10)
    parser.add_argument("--growth-fraction", type=float, default=0.5,
                        help="Fraction of training spent at half depth before growing")
    args = parser.parse_args()

    breakdown = analyze_model(vocab_size=args.vocab_size, d_model=args.d_model,
                              n_layers=args.n_layers, sequence_length=args.seq_len)
    tokens = args.steps * args.batch_size * args.seq_len

    print("=" * 68)
    print("WHERE THE CURRENT RUN'S COMPUTE GOES")
    print("=" * 68)
    print(f"  parameters:        {breakdown.params_total:,}")
    print(f"    embedding/head:  {breakdown.params_embedding:,} "
          f"({breakdown.params_embedding / breakdown.params_total:.0%})")
    print(f"    transformer:     {breakdown.params_transformer:,}")
    print(f"  fwd FLOPs/token:   {breakdown.fwd_flops_per_token / 1e6:.1f}M")
    print(f"    transformer:     {breakdown.fwd_flops_per_token_transformer / 1e6:.1f}M")
    print(f"    output head:     {breakdown.fwd_flops_per_token_head / 1e6:.1f}M  "
          f"<-- {breakdown.head_fraction:.0%} of forward compute")
    print(f"  tokens/run:        {tokens:,}")
    print(f"  train FLOPs/run:   {breakdown.train_flops(tokens):.3e}")
    print()
    print("  Read this before optimizing anything: at this width the output head")
    print("  costs more than the entire 12-layer transformer stack. The vocabulary")
    print("  is sized for a 1.5B model and the model is 22M.")
    print()

    vocab_gain = vocab_resize_multiplier(baseline=breakdown, new_vocab=args.target_vocab,
                                         d_model=args.d_model,
                                         tokenization_penalty=args.tokenization_penalty)

    transformer_share = (breakdown.fwd_flops_per_token_transformer
                         / breakdown.fwd_flops_per_token)
    shrunk = analyze_model(vocab_size=args.target_vocab, d_model=args.d_model,
                           n_layers=args.n_layers, sequence_length=args.seq_len)
    shrunk_share = shrunk.fwd_flops_per_token_transformer / shrunk.fwd_flops_per_token
    growth_gain = growth_savings(from_layers=args.n_layers // 2, to_layers=args.n_layers,
                                 growth_fraction=args.growth_fraction,
                                 transformer_flop_share=shrunk_share)

    levers = [
        ("THROUGHPUT (seconds/step; FLOPs unchanged)", [
            ("mixed precision (fp16 autocast)", 3.52, MEASURED,
             "MEASURED on a Tesla T4: 3.52x at batch 16 x seq 512, but only "
             "1.26x at batch 2 x seq 256 -- strongly size-dependent, so size the "
             "real run large. On Apple M2/MPS the same lever measured 1.15x and "
             "0.89x. Selecting bf16 on the T4 measured 0.73x, SLOWER than fp32, "
             "because torch reports emulated bf16 as supported on Turing"),
            ("fused AdamW + TF32", 1.10, ARITHMETIC,
             "optimizer step is a real fraction of step time at 22M params"),
            (f"width rebalance d_model {args.d_model}x{args.n_layers} -> "
             f"{args.d_model * 2}x{args.n_layers // 4}", 1.30, LITERATURE,
             f"a {args.d_model}-wide GEMM is too small to saturate tensor cores; "
             "same parameters, ~4x the arithmetic intensity per matmul"),
            ("torch.compile", 1.15, LITERATURE,
             "kernel fusion; verify per-GPU, can regress on short runs"),
        ]),
        (f"SHAPE (FLOPs/token; vocab {args.vocab_size} -> {args.target_vocab})", [
            (f"vocabulary right-sizing (net of {args.tokenization_penalty:.2f}x "
             f"tokenization penalty)", vocab_gain, MEASURED,
             "shrinks the head, which is the majority of forward compute; "
             "measured 1.41x raw wall-clock on MPS vs 1.49x predicted"),
            (f"depth growth {args.n_layers // 2}->{args.n_layers} layers for "
             f"{args.growth_fraction:.0%} of training", growth_gain, ARITHMETIC,
             f"transformer is {transformer_share:.0%} of forward FLOPs now, "
             f"{shrunk_share:.0%} after the vocab cut -- growth is worth more "
             "once the head stops dominating; half-depth measured 1.28x "
             "wall-clock on MPS vs 1.23x predicted"),
        ]),
        ("ALGORITHMIC (steps to target)", [
            ("Muon optimizer (wall-clock, step cost included)",
             MUON_WALL_CLOCK, MEASURED,
             f"1.82x/1.78x fewer FLOPs at toy scale, but Newton-Schulz makes each "
             f"step {MUON_STEP_COST:.2f}x more expensive (measured), so the real "
             f"saving is {MUON_WALL_CLOCK:.2f}x. The ledger's 6*N*tokens cost "
             f"model cannot see optimizer overhead -- it prices the model's "
             f"arithmetic only"),
            ("warm-start from a public checkpoint", 2.50, LITERATURE,
             "borrowed compute: marginal cost only, and the donor may have seen "
             "your held-out set -- gate on contamination before believing it"),
        ]),
    ]

    # Quality levers change how well the run trains, not how fast. They are
    # listed separately and deliberately NOT multiplied into the cost total:
    # folding a stability fix into a speedup number is how plans start lying.
    quality = [
        ("real gradient clipping (--grad-clip 1.0)",
         "clip_grad_norm_ was called with 1e9, which measures the norm and "
         "clips nothing. Bounding it is what lets a higher LR stay stable."),
        ("no-decay parameter groups",
         "AdamW's default decayed LayerNorm/RMSNorm gains and biases, fighting "
         "the normalization the architecture depends on. Now 2-D tensors only."),
        ("batched held-out evaluation",
         "evaluation ran one sequence at a time, leaving the GPU idle; "
         "4.5x faster at identical accuracy."),
        ("cosine LR schedule with warmup (--lr-schedule)",
         "already implemented; a constant LR leaves the baseline weak, which "
         "inflates every lever measured against it."),
    ]

    print("=" * 68)
    print("LEVERS")
    print("=" * 68)
    total = 1.0
    conservative = 1.0
    for axis, entries in levers:
        print(f"\n{axis}")
        for name, multiplier, evidence, why in entries:
            total *= multiplier
            if evidence != LITERATURE:
                conservative *= multiplier
            print(f"  {multiplier:>5.2f}x  {name}")
            print(f"          [{evidence}] {why}")

    hours = args.baseline_hours / total
    hours_cons = args.baseline_hours / conservative
    matrix_runs = 4

    print()
    print("=" * 68)
    print("COMPOUNDED")
    print("=" * 68)
    print(f"  all levers:                 {total:.2f}x  "
          f"({1 - 1 / total:.0%} less cost)")
    print(f"  excluding unverified:       {conservative:.2f}x  "
          f"({1 - 1 / conservative:.0%} less cost)")
    print()
    print(f"  one run:  {args.baseline_hours:.2f}h -> {hours:.2f}h "
          f"({hours_cons:.2f}h conservative)")
    print(f"  4-run matrix: {args.baseline_hours * matrix_runs:.1f}h -> "
          f"{hours * matrix_runs:.1f}h "
          f"| ${args.baseline_hours * matrix_runs * args.usd_per_hour:.2f} -> "
          f"${hours * matrix_runs * args.usd_per_hour:.2f} at "
          f"${args.usd_per_hour:.2f}/GPU-hour")
    print()
    goal_met = conservative >= 2.0
    print(f"  >=50% reduction on verified levers alone: "
          f"{'YES' if goal_met else 'NO'} ({conservative:.2f}x)")

    print()
    print("=" * 68)
    print("TRAINING QUALITY (better runs, not faster ones -- not multiplied in)")
    print("=" * 68)
    for name, why in quality:
        print(f"  * {name}")
        print(f"      {why}")

    print()
    print("  Note on honesty: the throughput and shape levers are equivalence")
    print("  claims -- they must reproduce the baseline's capability curve within")
    print("  tolerance, or they are not free. Only the algorithmic levers are")
    print("  allowed to change the science. Quality levers above are excluded")
    print("  from the multiplier entirely: they change what the run learns, and")
    print("  counting them as speedups would double-count the same compute.")


if __name__ == "__main__":
    main()
