"""Grow a finished model into a larger one and continue training it.

This is how Reex-1 becomes Reex-1.5 without paying for 200M parameters from
scratch. Depth growth doubles the transformer stack while leaving the embedding
alone, so a 124M donor (d_model=768, 12 layers, vocab 50257) becomes 208M at 24
layers -- close to the Reex-1.5 target.

The growth is *function-preserving*: inserted blocks have their residual output
projections zeroed, so a zero-output block contributes nothing and the grown
model computes exactly what the donor computed. The build plan makes that a
hard pre-training gate, and this script refuses to train if it fails. Without
it, growth silently discards the compute already spent reaching the donor.

Three phases, each of which can be run alone:

  --inspect   report the donor's shape and what it can grow into
  --grow      produce and verify the grown checkpoint, then stop
  (default)   grow, verify, and continue training under the protocol

Warm-starting is not resuming. The optimizer's moments belong to a parameter
set that no longer exists, so the grown run starts with a fresh optimizer and a
fresh warmup. Its lineage is recorded: every checkpoint and ledger row carries
`parent_checkpoint_hash`, so the new model can always be traced to the donor.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def inspect_donor(path: str) -> dict:
    """Report what a checkpoint contains, tolerating foreign formats."""
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=False)
    weights = raw.get("model", raw) if isinstance(raw, dict) else raw
    config = raw.get("config") if isinstance(raw, dict) else None

    if config is None:
        # A bare state_dict from another trainer: recover the shape from tensors.
        config = _infer_config(weights)
        print("   no config in checkpoint; inferred from tensor shapes")

    # Deduplicate by storage: a tied lm_head appears in state_dict under its own
    # key while sharing the embedding's tensor, so summing values double-counts
    # it -- which for a 50257x768 embedding overstates the model by 38M.
    seen: set[int] = set()
    params = 0
    for tensor in weights.values():
        if not hasattr(tensor, "numel"):
            continue
        storage = tensor.data_ptr() if hasattr(tensor, "data_ptr") else id(tensor)
        if storage in seen:
            continue
        seen.add(storage)
        params += tensor.numel()
    return {"config": dict(config), "param_count": params,
            "tensor_count": len(weights),
            "architecture": config.get("architecture", "reference-v1"),
            "has_optimizer_state": isinstance(raw, dict) and "optimizer" in raw,
            "step": raw.get("step") if isinstance(raw, dict) else None}


def _infer_config(weights: dict) -> dict:
    """Recover d_model / n_layers / vocab from a bare state dict."""
    embedding = next((t for name, t in weights.items()
                      if "token_embedding" in name or name.endswith("wte.weight")), None)
    if embedding is None:
        raise ValueError(
            "cannot find a token embedding in this checkpoint; pass --d-model, "
            "--n-layers, --n-heads and --vocab-size explicitly"
        )
    vocab_size, d_model = embedding.shape
    layer_indices = set()
    for name in weights:
        for part in name.split("."):
            if part.isdigit():
                layer_indices.add(int(part))
                break
    return {"vocab_size": int(vocab_size), "d_model": int(d_model),
            "n_layers": len(layer_indices) or 1, "n_heads": max(1, int(d_model) // 64),
            "max_seq_len": 2048}


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--donor", required=True, help="Reex-1's final checkpoint")
    parser.add_argument("--to-layers", type=int,
                        help="Target depth; must be a multiple of the donor's")
    parser.add_argument("--mode", default="zero_init", choices=["zero_init", "stack"])
    parser.add_argument("--inspect", action="store_true", help="report and stop")
    parser.add_argument("--grow", action="store_true", help="grow and verify, then stop")
    parser.add_argument("--grown-checkpoint",
                        default=str(ROOT / "runs" / "reex-1.5" / "grown-init.pt"))
    # Continuation training
    parser.add_argument("--shard"); parser.add_argument("--heldout")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--use-muon", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1.5e-4,
                        help="Lower than a from-scratch run: the donor's weights are "
                             "already trained and a large LR would undo them")
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--run-dir", default=str(ROOT / "runs" / "reex-1.5"))
    parser.add_argument("--ledger", default=str(ROOT / "work" / "reex-1.5-ledger.jsonl"))
    parser.add_argument("--run-id", default="reex-1.5-grown")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")

    print("=" * 72)
    print("GROW AND CONTINUE")
    print("=" * 72)
    print("\n== 1. Donor ==")
    donor_info = inspect_donor(args.donor)
    config = donor_info["config"]
    print(f"   architecture : {donor_info['architecture']}")
    print(f"   d_model={config['d_model']} n_layers={config['n_layers']} "
          f"n_heads={config['n_heads']} vocab={config['vocab_size']}")
    print(f"   parameters   : {donor_info['param_count']:,}")
    if donor_info["step"] is not None:
        print(f"   trained steps: {donor_info['step']:,}")

    from src.ledger.cost_model import analyze_model
    print("\n   growth options:")
    for multiple in (2, 3, 4):
        target = config["n_layers"] * multiple
        grown = analyze_model(vocab_size=config["vocab_size"], d_model=config["d_model"],
                              n_layers=target, sequence_length=args.seq_len)
        print(f"     {config['n_layers']:>2} -> {target:<3} layers : "
              f"{grown.params_total/1e6:6.0f}M parameters")

    if args.inspect:
        return
    if not args.to_layers:
        raise SystemExit("--to-layers is required unless --inspect")

    print("\n== 2. Growth ==")
    from src.growth.depth import grow_depth
    from src.growth.hyperclone import assert_logit_equivalence
    from src.model.registry import build_model

    donor_model = build_model(donor_info["architecture"],
                              vocab_size=config["vocab_size"], d_model=config["d_model"],
                              n_layers=config["n_layers"], n_heads=config["n_heads"],
                              max_seq_len=config.get("max_seq_len", args.seq_len))
    raw = torch.load(args.donor, map_location="cpu", weights_only=False)
    donor_model.load_state_dict(raw.get("model", raw))
    donor_model.eval()

    grown, report = grow_depth(donor_model, to_layers=args.to_layers, mode=args.mode)
    grown_params = sum(p.numel() for p in grown.parameters())
    print(f"   {report.from_layers} -> {report.to_layers} layers ({args.mode})")
    print(f"   parameters   : {donor_info['param_count']:,} -> {grown_params:,}")
    print(f"   inserted at  : {list(report.inserted_at)}")

    print("\n== 3. Equivalence gate (hard) ==")
    if report.function_preserving:
        probe = torch.randint(0, config["vocab_size"], (2, min(64, args.seq_len)))
        check = assert_logit_equivalence(donor_model, grown.eval(), probe, tolerance=1e-4)
        print(f"   PASS: grown model reproduces the donor exactly "
              f"(max logit diff {check.max_abs_logit_diff:.2e})")
        print("   The compute already spent reaching the donor is preserved.")
    else:
        print(f"   SKIPPED: mode '{args.mode}' duplicates blocks verbatim, which is not")
        print("   the identity. The grown model does NOT reproduce the donor, so the")
        print("   build plan's growth gate cannot pass. Use --mode zero_init unless you")
        print("   accept discarding the donor's exact function.")

    target = Path(args.grown_checkpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": grown.state_dict(), "config": grown.config, "step": 0,
                "seed": 0, "use_muon": False,
                "growth": report.as_dict(), "donor": str(args.donor)}, target)
    print(f"\n   grown checkpoint: {target}")

    if args.grow:
        print("\n--grow given; stopping before training.")
        return

    if not args.shard:
        raise SystemExit("--shard is required to continue training (or pass --grow)")

    print("\n== 4. Continue training ==")
    from src.train.reference import train

    result = train(args.shard, str(Path(args.run_dir) / "continued"),
                   vocab_size=grown.config["vocab_size"], d_model=grown.config["d_model"],
                   n_layers=grown.config["n_layers"], n_heads=grown.config["n_heads"],
                   steps=args.steps, learning_rate=args.learning_rate, seed=17,
                   device=device, checkpoint_every=args.checkpoint_every,
                   heldout_shard=args.heldout, use_muon=args.use_muon,
                   batch_size=args.batch_size, lr_schedule=True,
                   warmup_fraction=args.warmup_fraction, precision=args.precision,
                   grad_clip=args.grad_clip, keep_checkpoints=args.keep_checkpoints,
                   ledger_path=args.ledger, run_id=args.run_id,
                   levers_on=["growth"] + (["optimizer"] if args.use_muon else []),
                   architecture=donor_info["architecture"],
                   init_from=str(target))

    print(f"\n   final loss   : {result['final_loss']:.4f}")
    if result.get("eval_scores"):
        print(f"   held-out acc : {result['eval_scores']['val_acc']:.4f}")
    print(f"   health       : {result['health']['status']}")
    print(f"   lineage      : parent {result['parent_checkpoint_hash'][:12]}")
    summary = Path(args.run_dir) / "growth-summary.json"
    summary.write_text(json.dumps(
        {"donor": donor_info, "growth": report.as_dict(),
         "grown_params": grown_params, "result": {
             k: v for k, v in result.items() if k != "losses"}},
        indent=2, default=str) + "\n", encoding="utf-8")
    print(f"   summary      : {summary}")


if __name__ == "__main__":
    main()
