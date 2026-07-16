#!/usr/bin/env python3
"""Real-data CPU proxy validation; explicitly not a 22.4M result."""
from __future__ import annotations

import json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train.reference import train
from src.ledger.writer import read_entries

TRAIN = ROOT / "data/real-v1/train-packed.jsonl"
HELDOUT = ROOT / "data/real-v1/heldout-packed.jsonl"
OUT = ROOT / "outputs/sanity-check-v2"
LEDGER = OUT / "sanity-ledger.jsonl"
MODEL = dict(vocab_size=8192, d_model=64, n_layers=4, n_heads=4)
SEQ_LEN = 256
SEEDS = (17, 23)
TOKENS_PER_PARAM = 0.5

def probe():
    import torch
    from src.model.reference import ReferenceLM
    rows = [json.loads(x) for x in TRAIN.read_text().splitlines() if x.strip()]
    out = []
    for batch in (1, 2, 4, 8):
        model = ReferenceLM(**MODEL, max_seq_len=SEQ_LEN).to("cpu").train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        ids = torch.tensor([rows[i % len(rows)]["input_ids"] for i in range(batch)], dtype=torch.long) % MODEL["vocab_size"]
        docs = torch.tensor([[(-1 if x == "__pad__" else 0) for x in rows[i % len(rows)]["document_ids"]] for i in range(batch)], dtype=torch.long)
        # Warmup then time ten real forward/backward/update steps.
        for _ in range(2):
            loss = torch.nn.functional.cross_entropy(model(ids, docs)[:, :-1].reshape(-1, model.config["vocab_size"]), ids[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        start = time.perf_counter()
        for _ in range(10):
            loss = torch.nn.functional.cross_entropy(model(ids, docs)[:, :-1].reshape(-1, model.config["vocab_size"]), ids[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        sec = (time.perf_counter() - start) / 10
        out.append({"batch_size": batch, "sec_per_step": sec, "tokens_per_sec": batch * SEQ_LEN / sec})
    return out

def main():
    if not TRAIN.exists() or not HELDOUT.exists():
        raise FileNotFoundError("real-v1 packed shards are required")
    OUT.mkdir(parents=True, exist_ok=True)
    if LEDGER.exists():
        raise RuntimeError(f"refusing to overwrite existing ledger: {LEDGER}")
    probe_rows = probe()
    chosen = max(probe_rows, key=lambda r: r["tokens_per_sec"])
    params = 0
    from src.model.reference import ReferenceLM
    params = sum(p.numel() for p in ReferenceLM(**MODEL, max_seq_len=SEQ_LEN).parameters())
    target_tokens = math.ceil(TOKENS_PER_PARAM * params)
    steps = math.ceil(target_tokens / (chosen["batch_size"] * SEQ_LEN))
    plan = {"label": "sanity-check scale, not validated 22.4M result", "device": "cpu", "model": MODEL,
            "params": params, "probe": probe_rows, "selected_batch_size": chosen["batch_size"],
            "steps": steps, "tokens_per_run": steps * chosen["batch_size"] * SEQ_LEN,
            "tokens_per_param": steps * chosen["batch_size"] * SEQ_LEN / params,
            "estimated_seconds_per_run": steps * chosen["sec_per_step"], "seeds": list(SEEDS)}
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    results = []
    for kind, muon in (("baseline", False), ("optimizer", True)):
        for seed in SEEDS:
            run_id = f"sanity-{kind}-s{seed}"
            start = time.perf_counter()
            result = train(str(TRAIN), str(OUT / run_id), **MODEL, steps=steps, seed=seed,
                           device="cpu", checkpoint_every=max(1, steps // 2), heldout_shard=str(HELDOUT),
                           use_muon=muon, levers_on=["optimizer"] if muon else [],
                           ledger_path=str(LEDGER), run_id=run_id, batch_size=chosen["batch_size"])
            results.append({"run_id": run_id, "kind": kind, "seed": seed, "elapsed_s": time.perf_counter()-start,
                            "final_loss": result["final_loss"], "final_val_acc": result["eval_scores"].get("val_acc")})
            print(json.dumps(results[-1]), flush=True)
    entries = read_entries(LEDGER)
    finals = {r["run_id"]: r for r in results}
    for run_id in finals:
        points = [e for e in entries if e["run_id"] == run_id]
        finals[run_id]["checkpoints"] = [{"tokens": e["tokens"], "score": e["eval_scores"].get("val_acc"), "loss": e["final_loss"], "cost": e["train_flops"]} for e in points]
    report = {"label": plan["label"], "plan": plan, "runs": results, "ledger_entries": len(entries),
              "baseline_seed_spread": abs(results[0]["final_val_acc"] - results[1]["final_val_acc"]),
              "optimizer_seed_spread": abs(results[2]["final_val_acc"] - results[3]["final_val_acc"]),
              "note": "No multiplier is claimed unless a common non-saturated target is reached by all four runs."}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
