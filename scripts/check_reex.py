"""Is Reex-1 actually a working model? Check before growing it.

Growth carries a donor's function forward exactly -- including its faults. If
Reex-1 is broken, mis-exported, or has a tokenizer mismatch, Reex-1.5 inherits
all of it and the GPU hours spent continuing it are wasted. The equivalence
gate in `grow_reex.py` proves the *grown* model matches the donor; it says
nothing about whether the donor was any good. This does.

Four checks, cheapest first, each independently meaningful:

  1. **Loads and reports its shape.** Catches a corrupt or partial download.
  2. **Loss against ln(vocab).** An untrained model scores ln(50257)=10.8. A
     trained one should be far below it. This single number separates "real
     model" from "randomly initialised weights in a nice wrapper".
  3. **Tokenizer round trip.** If the tokenizer does not match the weights,
     loss looks plausible while generation is gibberish -- the failure mode
     that is easiest to miss and most expensive to discover later.
  4. **Greedy generation.** Human-readable proof it is a language model.

Runs on CPU in a couple of minutes. No GPU required.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPTS = [
    "The capital of France is",
    "Photosynthesis is the process by which",
    "In 1969, humans first landed on",
    "Water boils at a temperature of",
]


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="puranikyashaswinsharma/reex-1")
    parser.add_argument("--subfolder", default="hf_format")
    parser.add_argument("--eval-documents", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "reex-1-health.json"))
    args = parser.parse_args()

    from transformers import AutoTokenizer, LlamaForCausalLM

    load = {"subfolder": args.subfolder} if args.subfolder else {}
    print("=" * 72)
    print(f"REEX-1 HEALTH CHECK -- {args.model}/{args.subfolder}")
    print("=" * 72)

    print("\n== 1. Load ==")
    model = LlamaForCausalLM.from_pretrained(args.model, dtype=torch.float32, **load).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load)
    seen, params = set(), 0
    for tensor in model.parameters():
        if id(tensor) not in seen:
            seen.add(id(tensor))
            params += tensor.numel()
    config = model.config
    print(f"   parameters      {params:,}")
    print(f"   layers/hidden   {config.num_hidden_layers} / {config.hidden_size}")
    print(f"   heads (q/kv)    {config.num_attention_heads} / "
          f"{getattr(config, 'num_key_value_heads', config.num_attention_heads)}")
    print(f"   vocab / context {config.vocab_size} / {config.max_position_embeddings}")
    report = {"model": args.model, "params": params,
              "config": {k: getattr(config, k) for k in
                         ("num_hidden_layers", "hidden_size", "num_attention_heads",
                          "vocab_size", "max_position_embeddings")}}

    print("\n== 2. Tokenizer round trip ==")
    sample = "The quick brown fox jumps over the lazy dog."
    ids = tokenizer.encode(sample)
    restored = tokenizer.decode(ids)
    round_trips = restored.strip() == sample.strip()
    print(f"   '{sample}'")
    print(f"   -> {len(ids)} tokens -> '{restored.strip()}'")
    print(f"   round trip: {'OK' if round_trips else 'MISMATCH'}")
    if len(tokenizer) != config.vocab_size:
        print(f"   [WARN] tokenizer has {len(tokenizer)} tokens but the model expects "
              f"{config.vocab_size}; generation will be wrong even if loss looks fine")
    report["tokenizer_round_trip"] = round_trips
    report["tokenizer_size"] = len(tokenizer)

    print("\n== 3. Held-out loss vs random ==")
    uniform = math.log(config.vocab_size)
    try:
        from datasets import load_dataset
        stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                              split="train", streaming=True)
        texts = []
        for record in stream:
            text = record["text"].strip()
            if len(text) > 200:
                texts.append(text)
            if len(texts) >= args.eval_documents:
                break
        source = "fineweb-edu"
    except Exception as error:
        print(f"   FineWeb-Edu unavailable ({type(error).__name__}); using fixed prose")
        texts = [
            "Education is the process of facilitating learning, or the acquisition of "
            "knowledge, skills, values, beliefs, and habits. Educational methods include "
            "teaching, training, storytelling, discussion and directed research.",
            "The water cycle describes the continuous movement of water on, above and "
            "below the surface of the Earth. Water can change states among liquid, "
            "vapour, and ice at various places in the water cycle.",
        ] * 10
        source = "builtin-prose"

    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=min(512, config.max_position_embeddings))
            input_ids = ids["input_ids"]
            if input_ids.shape[1] < 2:
                continue
            out = model(input_ids=input_ids, labels=input_ids)
            count = input_ids.shape[1] - 1
            total_loss += float(out.loss) * count
            total_tokens += count

    mean_loss = total_loss / max(1, total_tokens)
    print(f"   corpus            {source} ({total_tokens:,} scored tokens)")
    print(f"   mean loss         {mean_loss:.4f}")
    print(f"   perplexity        {math.exp(mean_loss):.1f}")
    print(f"   random baseline   {uniform:.4f}  (ln of vocab {config.vocab_size})")
    healthy = mean_loss < uniform - 3.0
    print(f"   verdict           {'TRAINED' if healthy else 'SUSPICIOUS'} -- "
          f"{uniform - mean_loss:.2f} nats below random")
    if not healthy:
        print("   [FAIL] A trained 116M model on clean prose should sit far below")
        print("   random. This close to it means the weights, the tokenizer, or the")
        print("   export is wrong -- do not grow it until that is understood.")
    report.update({"mean_loss": mean_loss, "perplexity": math.exp(mean_loss),
                   "uniform_baseline": uniform, "eval_source": source,
                   "scored_tokens": total_tokens, "looks_trained": healthy})

    print("\n== 4. Greedy generation ==")
    generations = []
    with torch.no_grad():
        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt")
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id or 0)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            generations.append({"prompt": prompt, "completion": text})
            print(f"   {prompt!r}\n     -> {text[len(prompt):].strip()[:120]!r}")
    report["generations"] = generations

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str) + "\n",
                              encoding="utf-8")
    print(f"\n   report: {args.out}")

    print("\n" + "=" * 72)
    if healthy and round_trips:
        print("REEX-1 LOOKS HEALTHY -- safe to grow.")
    else:
        print("REEX-1 FAILED A CHECK -- growing it would carry the fault into")
        print("Reex-1.5 exactly, because the growth is function-preserving.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
