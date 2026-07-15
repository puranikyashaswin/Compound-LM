"""Execute the COMPOUND-LM protocol end-to-end at toy scale with real numbers.

This drives the scientific spine of the build plan on the local CPU:

  1. build disjoint train / held-out shards from a corpus (real pipeline);
  2. run a two-seed baseline (A0/A1) and measure seed spread — the plan's
     mandatory gate before any lever is trusted;
  3. run the optimizer lever (Muon on eligible 2D weights, AdamW elsewhere);
  4. evaluate every checkpoint on the held-out set to build capability-at-cost
     curves and interpolate cost-to-target-score (M(S)), never extrapolating;
  5. compute the compounding / overlap table and write an evidence package.

Every score is a real held-out measurement from the trained model — nothing is
fabricated. Swap the corpus, config, and (on GPU) the eval harness to scale up;
the protocol structure is unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TOY_DOCUMENTS = [
    "the reference model learns to predict the next token in a packed sequence",
    "compound efficiency levers are measured against a fair reproducible baseline",
    "an append only ledger records every audited run with a checkpoint hash",
    "health checks terminate a run instead of rationalizing a broken result",
    "document boundaries prevent cross document attention inside a packed batch",
    "provenance manifests make a model change auditable and not merely repeatable",
    "the frozen evaluation contract is fixed before the first baseline run begins",
    "muon optimizes eligible two dimensional weights while adamw handles the rest",
    "a two seed baseline estimates the noise floor before any lever is trusted",
    "capability at equal cost defines the efficiency claim not validation loss alone",
]

# Toy model dimensions — small enough for CPU, large enough to learn the corpus.
MODEL = dict(vocab_size=4096, d_model=96, n_layers=3, n_heads=4)
SEQ_LEN = 96
STEPS = 400
CHECKPOINT_EVERY = 25


def build_shards():
    from src.data.packing import pack_shard
    from src.data.pipeline import prepare_documents

    out = ROOT / "data" / "protocol-v1"
    # In-distribution held-out: a subset of the same sentences the model trains
    # on. At toy scale this measures learned fit (a real held-out measurement),
    # not out-of-distribution generalization. On GPU the frozen E-v1 benchmark
    # suite replaces this with contamination-checked downstream tasks.
    heldout_docs = TOY_DOCUMENTS[:4]
    train_docs = [doc for doc in TOY_DOCUMENTS if doc not in heldout_docs]
    shards = {}
    for name, subset in (("train", train_docs), ("heldout", heldout_docs)):
        sheet = prepare_documents(subset, source="synthetic-protocol", shard_id=f"protocol-{name}",
                                  output_dir=out, tokenizer_id="fallback-v1")
        packed = out / f"protocol-{name}-packed.jsonl"
        pack_shard(out / f"protocol-{name}.jsonl", packed, sequence_length=SEQ_LEN)
        shards[name] = {"packed": str(packed), "tokens": sheet["token_count"]}
    return shards


def run_one(name, shard, heldout, *, seed, use_muon, levers, params):
    """Train a run and build its (cost, score) curve from every checkpoint."""
    from src.eval.intrinsic import evaluate
    from src.train.reference import train

    out_dir = ROOT / "runs" / name
    result = train(shard, str(out_dir), **MODEL, steps=STEPS, seed=seed, device="cpu",
                   checkpoint_every=CHECKPOINT_EVERY, heldout_shard=heldout,
                   use_muon=use_muon, levers_on=levers)
    # Real capability-at-cost curve: cost = 6·N·tokens = 6·N·(SEQ_LEN·step) FLOPs.
    curve = []
    for ckpt in sorted(out_dir.glob("checkpoint-*.pt")):
        step = int(ckpt.stem.split("-")[1])
        scores = evaluate(str(ckpt), heldout, device="cpu")
        curve.append({"step": step, "cost": 6 * params * SEQ_LEN * step,
                      "score": scores["val_acc"], "val_nll": scores["val_nll"]})
    return {"name": name, "seed": seed, "levers": levers, "use_muon": use_muon,
            "final": result["eval_scores"], "health": result["health"],
            "checkpoint_hash": result["checkpoint_hash"], "curve": curve}


def write_summary(evidence: dict) -> None:
    """Emit a markdown evidence summary and inject the table into the README."""
    b = evidence["baseline"]
    lines = [
        "# COMPOUND-LM protocol evidence (toy scale)",
        "",
        f"_Real held-out measurements from {evidence['params']:,}-parameter runs on CPU. "
        "Toy scale; the protocol is identical at 200M._",
        "",
        "## Two-seed baseline (A0/A1)",
        "",
        f"- seed 17 val_acc: {b['seeds'][0]['val_acc']:.4f}",
        f"- seed 23 val_acc: {b['seeds'][1]['val_acc']:.4f}",
        f"- seed spread: {b['seed_spread_acc']:.4f} "
        f"({'PASS' if b['gate_pass'] else 'FAIL'} at gate <= 0.15)",
        "",
        "## Compounding / capability-at-cost",
        "",
        f"Target held-out accuracy: {evidence['compounding'].get('target_score')}",
        "",
    ]
    table = ["| Run | Levers | Cost (FLOPs) | Multiplier | Overlap |",
             "|---|---|---:|---:|---:|"]
    for row in evidence["compounding"].get("rows", []):
        levers = ", ".join(row["levers"]) or "baseline"
        overlap = row["overlap_coefficient"]
        overlap_str = f"{overlap:.3f}×" if isinstance(overlap, (int, float)) else "n/a"
        table.append(f"| {row['name']} | {levers} | {row['recipe_cost']:.3e} | "
                     f"{row['observed_multiplier']:.3f}× | {overlap_str} |")
    table_block = "\n".join(table)
    lines += [table_block, "", f"_{evidence['note']}_", ""]
    (ROOT / "outputs" / "protocol-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- AUTOGEN:TABLE START -->", "<!-- AUTOGEN:TABLE END -->"
    if start in text and end in text:
        readme.write_text(text.split(start, 1)[0] + start + "\n" + table_block + "\n" + end +
                          text.split(end, 1)[1], encoding="utf-8")


def count_params():
    from src.model.reference import ReferenceLM
    m = ReferenceLM(**MODEL, max_seq_len=SEQ_LEN)
    return sum(p.numel() for p in m.parameters())


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "protocol-report.json"))
    args = parser.parse_args()

    from src.ledger.compounding import compounding_report, cost_to_score

    print("== 1. Build disjoint train / held-out shards ==")
    shards = build_shards()
    train_shard, heldout = shards["train"]["packed"], shards["heldout"]["packed"]
    print(f"   train tokens={shards['train']['tokens']} heldout tokens={shards['heldout']['tokens']}")

    params = count_params()
    print(f"   model params={params:,}")

    print("== 2. Two-seed baseline (A0/A1) ==")
    a0 = run_one("baseline-s17", train_shard, heldout, seed=17, use_muon=False, levers=[], params=params)
    a1 = run_one("baseline-s23", train_shard, heldout, seed=23, use_muon=False, levers=[], params=params)
    seed_spread = abs(a0["final"]["val_acc"] - a1["final"]["val_acc"])
    print(f"   acc s17={a0['final']['val_acc']:.4f} s23={a1['final']['val_acc']:.4f} "
          f"spread={seed_spread:.4f}")

    print("== 3. Optimizer lever (Muon) ==")
    opt0 = run_one("optimizer-s17", train_shard, heldout, seed=17, use_muon=True,
                   levers=["optimizer"], params=params)
    opt1 = run_one("optimizer-s23", train_shard, heldout, seed=23, use_muon=True,
                   levers=["optimizer"], params=params)
    print(f"   acc s17={opt0['final']['val_acc']:.4f} s23={opt1['final']['val_acc']:.4f}")

    print("== 4. Capability-at-cost — M(S) at common target ==")
    runs = [a0, a1, opt0, opt1]
    max_reached = min(max(pt["score"] for pt in r["curve"]) for r in runs)
    target = round(max_reached * 0.9, 4)  # a score both runs demonstrably reach
    comp_rows = []
    for r in runs:
        cost = cost_to_score(r["curve"], target)
        comp_rows.append({"name": r["name"], "levers": r["levers"], "seed": r["seed"], "recipe_cost": cost,
                          "reached": cost is not None})
    reachable = [r for r in comp_rows if r["reached"]]
    report = {"target_score": target}
    if len(reachable) == len(comp_rows):
        report = compounding_report(comp_rows, target_score=target)
        for row in report["rows"]:
            print(f"   {row['name']:<14} cost={row['recipe_cost']:.3e} "
                  f"multiplier={row['observed_multiplier']:.3f}× "
                  f"overlap={row['overlap_coefficient']}")
    else:
        print(f"   target {target} not reached by all runs; reporting raw curves only")

    evidence = {
        "schema_version": 1,
        "model": MODEL, "sequence_length": SEQ_LEN, "steps": STEPS,
        "params": params,
        "shards": shards,
        "baseline": {"seeds": [a0["final"], a1["final"]], "seed_spread_acc": seed_spread,
                     "gate_pass": seed_spread <= 0.15},
        "runs": [{"name": r["name"], "levers": r["levers"], "final": r["final"],
                  "health": r["health"], "checkpoint_hash": r["checkpoint_hash"],
                  "curve": r["curve"]} for r in runs],
        "compounding": report,
        "note": "held-out intrinsic scores; real measurements, toy scale. "
                "Replace corpus/config/eval-harness to scale to the 200M baseline.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    write_summary(evidence)
    print(f"\nEvidence package: {Path(args.out).relative_to(ROOT)}")
    print(f"Baseline seed-spread gate (<=0.15): "
          f"{'PASS' if evidence['baseline']['gate_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
