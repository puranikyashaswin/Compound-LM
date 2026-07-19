"""The decisive run: measure Muon at real scale, on real data, on a GPU.

Everything else in this repo is now verified. One number is not: Muon's 1.82x,
which was measured on a 64-word toy cycle. Toy-scale multipliers are exactly
what this framework refuses to trust -- its own README documents three earlier
occasions where a healthy, fully-ledgered toy result measured nothing real.

This script settles it under the full protocol:

  * a real corpus, sized by `src/data/budget.py` so the run cannot be
    memorization in disguise;
  * a two-seed baseline (the plan's mandatory gate) plus two Muon seeds;
  * capability-at-cost curves in BOTH FLOPs and wall clock, because Muon's
    steps cost more and the 6*N*tokens ledger cannot see that;
  * a timing-noise control -- the two baseline seeds are identical configs, so
    any difference in their seconds-per-step bounds what a lever must beat;
  * every existing gate: contamination, health, budget, seed spread, and the
    refusal to report a multiplier built from unresolved lower bounds.

Sized for one Kaggle T4 session. Defaults are ~40M tokens over a 22M model at
mixed precision, which the verified 3.25x speedup brings inside a couple of
hours for all four runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Held-out is 5% of documents, so the budget gate's constraints apply to the
# remaining 95% -- sizing the *total* to the requirement leaves the train split
# just under it, and the gate (correctly) rejects the run after the corpus has
# already been built.
TRAIN_FRACTION = 0.95
DEDUP_HEADROOM = 1.10  # documents dropped as exact/near duplicates


def build_corpus(out_dir: Path, *, target_tokens: int, vocab_size: int,
                 sequence_length: int, seed: int, proxy_vocabulary: int = 8000,
                 proxy_successors: int = 4) -> dict:
    """Materialize a real corpus, preferring FineWeb-Edu, else a fallback.

    The fallback is deliberately *not* a trivial cycle: it is a procedurally
    generated text with long-range structure and a large vocabulary, so the
    task cannot be solved by memorizing a short rule. It is still a proxy, and
    the report labels it as one.
    """
    from src.data.packing import pack_shard
    from src.data.pipeline import prepare_documents

    out_dir.mkdir(parents=True, exist_ok=True)
    documents, source = _load_documents(target_tokens, seed,
                                        proxy_vocabulary, proxy_successors)

    split = int(len(documents) * TRAIN_FRACTION)
    shards = {}
    for name, subset in (("train", documents[:split]), ("heldout", documents[split:])):
        sheet = prepare_documents(subset, source=source, shard_id=f"final-{name}",
                                  output_dir=out_dir, tokenizer_id="fallback-v1",
                                  vocab_size=vocab_size)
        packed = out_dir / f"final-{name}-packed.jsonl"
        pack_shard(out_dir / f"final-{name}.jsonl", packed,
                   sequence_length=sequence_length)
        shards[name] = {"packed": str(packed), "tokens": sheet["token_count"],
                        "documents": sheet["document_count_kept"]}
    shards["source"] = source
    return shards


def _load_documents(target_tokens: int, seed: int,
                    proxy_vocabulary: int = 8000,
                    proxy_successors: int = 4) -> tuple[list[str], str]:
    """Real text if the environment allows it, else a structured proxy."""
    try:
        from datasets import load_dataset
        stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                              split="train", streaming=True)
        documents, total = [], 0
        for record in stream:
            text = record["text"].strip()
            if not text:
                continue
            documents.append(text)
            total += len(text.split())
            if total >= target_tokens:
                break
        if documents:
            print(f"   corpus: FineWeb-Edu, {len(documents):,} documents")
            return documents, "fineweb-edu"
    except Exception as error:
        print(f"   FineWeb-Edu unavailable ({type(error).__name__}); using proxy corpus")

    # Proxy: a word-level Markov text over a large vocabulary. Successor
    # prediction is learnable but not memorizable from a short rule, and
    # held-out documents are generated from the same process with unseen seeds.
    import random

    rng = random.Random(seed)
    vocabulary = [f"w{index:05d}" for index in range(proxy_vocabulary)]
    # A sparse successor graph: each word has a few likely followers, so the
    # task has real conditional structure rather than one deterministic cycle.
    # Ceiling accuracy is 1/proxy_successors, and the model must learn
    # proxy_vocabulary * proxy_successors transitions to reach it -- so these
    # two numbers set how much capacity and data the task actually demands.
    successors = {word: rng.sample(vocabulary, proxy_successors)
                  for word in vocabulary}
    documents, total = [], 0
    while total < target_tokens:
        current = rng.choice(vocabulary)
        words = []
        for _ in range(rng.randint(120, 260)):
            words.append(current)
            current = rng.choice(successors[current])
        documents.append(" ".join(words))
        total += len(words)
    print(f"   corpus: procedural proxy, {len(documents):,} documents")
    return documents, "procedural-markov-proxy"


def run_arm(name, *, shard, heldout, seed, use_muon, levers, model_config,
            steps, batch_size, sequence_length, device, precision, checkpoint_every,
            grad_clip, ledger, run_dir):
    from src.eval.intrinsic import evaluate
    from src.train.reference import resumable_checkpoint, train

    out_dir = run_dir / name
    resume = resumable_checkpoint(out_dir)
    if resume is not None:
        print(f"   resuming {name} from {resume.name}")

    started = time.perf_counter()
    result = train(shard, str(out_dir), **model_config, steps=steps, seed=seed,
                   device=device, checkpoint_every=checkpoint_every,
                   heldout_shard=heldout, use_muon=use_muon, levers_on=levers,
                   batch_size=batch_size, precision=precision, lr_schedule=True,
                   grad_clip=grad_clip, ledger_path=str(ledger), run_id=name,
                   resume=str(resume) if resume else None)
    wall_clock = time.perf_counter() - started

    curve = []
    n_params = _count_params(model_config, sequence_length)
    for checkpoint in sorted(out_dir.glob("checkpoint-*.pt")):
        step = int(checkpoint.stem.split("-")[1])
        scores = evaluate(str(checkpoint), heldout, device=device)
        curve.append({"step": step,
                      "cost": 6 * n_params * sequence_length * batch_size * step,
                      "score": scores["val_acc"], "val_nll": scores["val_nll"]})
    return {"name": name, "seed": seed, "levers": levers,
            "final": result["eval_scores"], "health": result["health"],
            "curve": curve, "wall_clock_s": wall_clock,
            "seconds_per_step": wall_clock / max(1, steps),
            "checkpoint_hash": result["checkpoint_hash"]}


def _count_params(model_config: dict, sequence_length: int) -> int:
    from src.model.registry import build_model
    model = build_model("reference-v1", **model_config, max_seq_len=sequence_length)
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--corpus-dir", default=str(ROOT / "data" / "final-v1"))
    parser.add_argument("--run-dir", default=str(ROOT / "runs" / "final"))
    parser.add_argument("--ledger", default=str(ROOT / "work" / "final-ledger.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "final-experiment.json"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--proxy-vocabulary", type=int, default=8000,
                        help="Distinct words in the fallback corpus when FineWeb-Edu "
                             "is unreachable; smaller makes the task learnable sooner")
    parser.add_argument("--proxy-successors", type=int, default=4,
                        help="Followers per word; ceiling accuracy is 1/this")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    model_config = dict(vocab_size=args.vocab_size, d_model=args.d_model,
                        n_layers=args.n_layers, n_heads=args.n_heads)
    n_params = _count_params(model_config, args.seq_len)
    consumed = args.steps * args.batch_size * args.seq_len

    print("=" * 72)
    print("FINAL EXPERIMENT -- Muon at real scale")
    print("=" * 72)
    print(f"device {device} | params {n_params:,} | token-positions {consumed:,}")

    print("\n== 1. Corpus ==")
    # The corpus must satisfy BOTH budget constraints, not just one:
    #   * at most ~4 epochs over the data  -> tokens >= consumed / 3
    #   * at least 1 unique token/parameter -> tokens >= n_params
    # Sizing on the first alone produced a corpus the gate then rejected, which
    # is the gate doing its job but wastes the corpus build. The 5% headroom
    # covers documents dropped by the deduplication filter.
    required_train = max(consumed / 3, n_params * 1.05)
    target_tokens = int(required_train / TRAIN_FRACTION * DEDUP_HEADROOM)
    print(f"   train split needs >= {int(required_train):,} tokens "
          f"(consumed/3 = {consumed // 3:,}, params = {n_params:,})")
    print(f"   sizing corpus for {target_tokens:,} tokens total")
    shards = build_corpus(Path(args.corpus_dir), target_tokens=target_tokens,
                          vocab_size=args.vocab_size, sequence_length=args.seq_len,
                          seed=17, proxy_vocabulary=args.proxy_vocabulary,
                          proxy_successors=args.proxy_successors)
    print(f"   train {shards['train']['tokens']:,} tokens | "
          f"heldout {shards['heldout']['tokens']:,} tokens")

    print("\n== 2. Budget gate (before any GPU time) ==")
    from src.data.budget import check_token_budget
    budget = check_token_budget(unique_tokens=shards["train"]["tokens"],
                                steps=args.steps, batch_size=args.batch_size,
                                sequence_length=args.seq_len, n_params=n_params)
    print(f"   epochs {budget.epochs:.2f}x | tokens/param {budget.tokens_per_param:.1f} "
          f"| status {budget.status.upper()}")
    for warning in budget.warnings:
        print(f"   [warn] {warning}")
    if budget.status == "red":
        for failure in budget.failures:
            print(f"   [FAIL] {failure}")
        raise SystemExit("budget_gate: this run would measure memorization, not learning")

    ledger = Path(args.ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)
    common = dict(shard=shards["train"]["packed"], heldout=shards["heldout"]["packed"],
                  model_config=model_config, steps=args.steps,
                  batch_size=args.batch_size, sequence_length=args.seq_len,
                  device=device, precision=args.precision,
                  checkpoint_every=args.checkpoint_every, grad_clip=args.grad_clip,
                  ledger=ledger, run_dir=run_dir)

    print("\n== 3. Two-seed baseline (AdamW) ==")
    a0 = run_arm("baseline-s17", seed=17, use_muon=False, levers=[], **common)
    a1 = run_arm("baseline-s23", seed=23, use_muon=False, levers=[], **common)
    spread = abs(a0["final"]["val_acc"] - a1["final"]["val_acc"])
    print(f"   acc {a0['final']['val_acc']:.4f} / {a1['final']['val_acc']:.4f} "
          f"spread {spread:.4f}")

    print("\n== 4. Muon lever ==")
    m0 = run_arm("optimizer-s17", seed=17, use_muon=True, levers=["optimizer"], **common)
    m1 = run_arm("optimizer-s23", seed=23, use_muon=True, levers=["optimizer"], **common)
    print(f"   acc {m0['final']['val_acc']:.4f} / {m1['final']['val_acc']:.4f}")

    print("\n== 5. Capability-at-cost ==")
    from src.ledger.compounding import (assert_costs_resolved, compounding_report,
                                        cost_to_score_detail)
    from src.ledger.cost_model import wall_clock_multiplier

    runs = [a0, a1, m0, m1]
    reached = min(max(point["score"] for point in run["curve"]) for run in runs)
    if reached <= 0:
        raise SystemExit("no_capability_signal: no run learned anything measurable")
    target = round(reached * 0.9, 4)
    details = {run["name"]: cost_to_score_detail(run["curve"], target) for run in runs}
    assert_costs_resolved(details)

    rows = [{"name": run["name"], "levers": run["levers"], "seed": run["seed"],
             "recipe_cost": details[run["name"]]["cost"],
             "cost_status": details[run["name"]]["status"]} for run in runs]
    if any(row["recipe_cost"] is None for row in rows):
        raise SystemExit(f"target {target} not reached by every run; extend --steps")

    report = compounding_report(rows, target_score=target)

    by_name = {run["name"]: run for run in runs}
    base_sps = by_name["baseline-s17"]["seconds_per_step"]
    noise = abs(by_name["baseline-s23"]["seconds_per_step"] / base_sps - 1.0)
    report["timing_noise"] = noise
    report["timing_trustworthy"] = noise <= 0.05
    print(f"   timing-noise control (identical baseline seeds): {noise:.1%}")
    if noise > 0.05:
        print("   [warn] wall-clock multipliers unreliable on this machine")

    print(f"   target held-out accuracy: {target}")
    for row in report["rows"]:
        ratio = by_name[row["name"]]["seconds_per_step"] / base_sps
        row["step_cost_ratio"] = ratio
        row["wall_clock_multiplier"] = wall_clock_multiplier(
            flop_multiplier=row["observed_multiplier"], step_cost_ratio=ratio)
        print(f"   {row['name']:<16} flops={row['observed_multiplier']:.3f}x "
              f"step_cost={ratio:.2f}x wall_clock={row['wall_clock_multiplier']:.3f}x")

    muon_rows = [r for r in report["rows"] if r["levers"] == ["optimizer"]]
    muon_flops = sum(r["observed_multiplier"] for r in muon_rows) / len(muon_rows)
    muon_wall = sum(r["wall_clock_multiplier"] for r in muon_rows) / len(muon_rows)
    baseline_band = abs(report["rows"][1]["observed_multiplier"] - 1.0)

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  Muon at {n_params/1e6:.1f}M params on {shards['source']}:")
    print(f"    FLOP multiplier:       {muon_flops:.3f}x  (toy scale claimed 1.80x)")
    print(f"    wall-clock multiplier: {muon_wall:.3f}x"
          f"{'' if report['timing_trustworthy'] else '  [timing noisy]'}")
    print(f"    baseline seed noise:   {baseline_band:.3f}x")
    decisive = muon_flops - 1.0 > 2 * baseline_band
    print(f"    exceeds seed noise:    {'YES' if decisive else 'NO -- inconclusive'}")
    if not decisive:
        print("    A lever inside the seed band is not a result. Add a third seed")
        print("    or extend the run before claiming anything.")

    evidence = {"schema_version": 1, "device": device, "params": n_params,
                "model": model_config, "sequence_length": args.seq_len,
                "steps": args.steps, "batch_size": args.batch_size,
                "precision": args.precision, "grad_clip": args.grad_clip,
                "corpus": shards, "budget": budget.as_dict(),
                "baseline_seed_spread_acc": spread,
                "runs": [{k: v for k, v in run.items() if k != "curve"} | {"curve": run["curve"]}
                         for run in runs],
                "compounding": report,
                "muon_flop_multiplier": muon_flops,
                "muon_wall_clock_multiplier": muon_wall,
                "decisive_vs_seed_noise": decisive}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"\n  evidence: {args.out}")


if __name__ == "__main__":
    main()
