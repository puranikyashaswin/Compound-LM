"""Grow Reex-1 (116M) into Reex-1.5 (~193M) and prove nothing was lost.

Reex-1 is a 116M LlamaForCausalLM -- 12 layers, hidden 768, 12 query heads over
4 key/value heads, SwiGLU 2112, RMSNorm, RoPE, tied embeddings, vocab 50257,
context 1024 -- trained on ~2B FineWeb-Edu tokens.

Doubling its depth to 24 layers gives 193.2M parameters: the transformer stack
doubles while the 38.6M embedding does not. The growth is function-preserving,
so Reex-1.5 begins exactly where Reex-1 finished and the 2B tokens already paid
for are carried forward rather than re-earned.

    # 1. look, change nothing
    python scripts/grow_reex.py --inspect

    # 2. grow, verify equivalence, write the checkpoint
    python scripts/grow_reex.py --to-layers 24 --out runs/reex-1.5/init

Start from the PRETRAINED base, not the SFT checkpoint, if Reex-1.5 is meant to
be a base model: continuing pretraining on top of instruction tuning partially
undoes the tuning and muddles the lineage. The `hf_format/` export is the SFT
model; pass --donor with the base export if you have one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DONOR = "puranikyashaswinsharma/reex-1"


def describe(model) -> dict:
    config = model.config
    seen, params = set(), 0
    for tensor in model.parameters():
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        params += tensor.numel()
    return {
        "params": params,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", None),
    }


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--donor", default=DEFAULT_DONOR,
                        help="HF repo id or local path to the hf_format export")
    parser.add_argument("--subfolder", default="hf_format",
                        help="Subfolder inside the repo holding config.json")
    parser.add_argument("--to-layers", type=int, default=24)
    parser.add_argument("--mode", default="zero_init", choices=["zero_init", "stack"])
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "runs" / "reex-1.5" / "init"))
    parser.add_argument("--probe-tokens", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    from transformers import LlamaForCausalLM
    from src.model.llama_adapter import grow_llama_depth

    print("=" * 72)
    print("GROW REEX-1 -> REEX-1.5")
    print("=" * 72)
    print(f"\n== 1. Donor: {args.donor} ({args.subfolder}) ==")
    load_kwargs = {"dtype": torch.float32}
    if args.subfolder:
        load_kwargs["subfolder"] = args.subfolder
    donor = LlamaForCausalLM.from_pretrained(args.donor, **load_kwargs).eval()
    info = describe(donor)
    for key, value in info.items():
        print(f"   {key:<24} {value:,}" if isinstance(value, int) else
              f"   {key:<24} {value}")

    per_layer = ((info["params"] - info["vocab_size"] * info["hidden_size"]
                  - info["hidden_size"]) // info["num_hidden_layers"])
    print("\n   growth options (transformer scales, embedding does not):")
    for multiple in (2, 3):
        target = info["num_hidden_layers"] * multiple
        total = per_layer * target + info["vocab_size"] * info["hidden_size"] + info["hidden_size"]
        print(f"     {info['num_hidden_layers']} -> {target:<3} layers : {total/1e6:6.1f}M")

    if args.inspect:
        print("\n--inspect given; nothing was changed.")
        return

    print(f"\n== 2. Growth: {info['num_hidden_layers']} -> {args.to_layers} layers "
          f"({args.mode}) ==")
    grown, report = grow_llama_depth(donor, to_layers=args.to_layers, mode=args.mode)
    grown_info = describe(grown)
    print(f"   parameters  {info['params']:,} -> {grown_info['params']:,}")
    print(f"   inserted at {list(report.inserted_at)}")

    print("\n== 3. Equivalence gate (hard) ==")
    ids = torch.randint(0, info["vocab_size"], (2, args.probe_tokens))
    with torch.no_grad():
        before = donor(input_ids=ids).logits
        after = grown.eval()(input_ids=ids).logits
    difference = (before - after).abs().max().item()
    print(f"   max |logit difference| : {difference:.3e} (tolerance {args.tolerance:g})")
    if report.function_preserving:
        if difference > args.tolerance:
            raise SystemExit(
                f"GROWTH GATE FAILED: the grown model does not reproduce Reex-1 "
                f"({difference:.3e} > {args.tolerance:g}). Training from here would "
                f"discard the ~2B tokens already spent."
            )
        print("   PASS -- Reex-1.5 starts exactly where Reex-1 finished.")
    else:
        print("   NOT APPLICABLE: 'stack' duplicates blocks verbatim, which is not the")
        print("   identity. Reex-1's exact function is NOT preserved.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grown.save_pretrained(out)
    try:
        from transformers import AutoTokenizer
        tokenizer_kwargs = {"subfolder": args.subfolder} if args.subfolder else {}
        AutoTokenizer.from_pretrained(args.donor, **tokenizer_kwargs).save_pretrained(out)
        print(f"\n   tokenizer copied (Reex-1.5 must keep Reex-1's GPT-2 BPE)")
    except Exception as error:
        print(f"\n   [warn] tokenizer not copied: {error}")

    provenance = {"donor": args.donor, "subfolder": args.subfolder,
                  "donor_config": info, "grown_config": grown_info,
                  "growth": report.as_dict(),
                  "max_abs_logit_diff": difference, "tolerance": args.tolerance,
                  "equivalence_passed": bool(report.function_preserving
                                             and difference <= args.tolerance)}
    (out / "growth-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"   checkpoint  : {out}")
    print(f"   provenance  : {out / 'growth-provenance.json'}")
    print("\nNext: continue pretraining from this checkpoint on FineWeb-Edu.")
    print("New layers start as exact no-ops, so give them warmup before full LR.")


if __name__ == "__main__":
    main()
