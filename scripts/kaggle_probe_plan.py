#!/usr/bin/env python3
"""Probe a Kaggle T4, then print the exact safe validation launch command.

Run this on Kaggle (not on the Mac): it deliberately refuses to guess GPU
throughput. The 20% margin reserves time for evaluation/checkpoint I/O.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model.reference import ReferenceLM

MODEL = dict(vocab_size=50257, d_model=256, n_layers=12, n_heads=8)
SEQ_LEN = 256
SESSION_SECONDS = 9 * 3600
SAFETY = 0.80
TARGET_TOKENS_PER_PARAM = 2.0
NUM_RUNS = 4  # baseline x2 seeds + Muon x2 seeds

def main():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("kaggle_probe_requires_cuda: run this script on the T4 notebook")
    data = ROOT / "data/real-v1/train-packed.jsonl"
    rows = [json.loads(x) for x in data.read_text().splitlines() if x.strip()]
    model = ReferenceLM(**MODEL, max_seq_len=SEQ_LEN).cuda().train()
    params = sum(p.numel() for p in model.parameters())
    probes = []
    for batch in (8, 16):
        ids = torch.tensor([rows[i]["input_ids"] for i in range(batch)], device="cuda", dtype=torch.long) % MODEL["vocab_size"]
        docs = torch.zeros_like(ids)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        for _ in range(3):
            loss = torch.nn.functional.cross_entropy(model(ids, docs)[:, :-1].reshape(-1, MODEL["vocab_size"]), ids[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        torch.cuda.synchronize(); start = time.perf_counter()
        for _ in range(10):
            loss = torch.nn.functional.cross_entropy(model(ids, docs)[:, :-1].reshape(-1, MODEL["vocab_size"]), ids[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        torch.cuda.synchronize(); sec = (time.perf_counter() - start) / 10
        probes.append({"batch_size": batch, "sec_per_step": sec, "tokens_per_sec": batch * SEQ_LEN / sec})
    chosen = max(probes, key=lambda x: x["tokens_per_sec"])
    target_tokens = math.ceil(TARGET_TOKENS_PER_PARAM * params)
    target_steps = math.ceil(target_tokens / (chosen["batch_size"] * SEQ_LEN))
    safe_seconds = SESSION_SECONDS * SAFETY
    estimated_seconds_per_run = target_steps * chosen["sec_per_step"]
    estimated_seconds_all_runs = NUM_RUNS * estimated_seconds_per_run
    if estimated_seconds_all_runs > safe_seconds:
        raise SystemExit(json.dumps({"status": "insufficient_t4_budget_for_2_tok_per_param_all_four_runs",
                                     "probes": probes, "target_steps": target_steps,
                                     "estimated_seconds_all_runs": estimated_seconds_all_runs,
                                     "safe_seconds_all_runs": safe_seconds}, indent=2))
    plan = {"device": torch.cuda.get_device_name(0), "model_params": params, "probes": probes,
            "selected_batch_size": chosen["batch_size"], "steps": target_steps,
            "tokens": target_steps * chosen["batch_size"] * SEQ_LEN,
            "tokens_per_param": target_steps * chosen["batch_size"] * SEQ_LEN / params,
            "estimated_seconds_per_run": estimated_seconds_per_run,
            "estimated_seconds_all_four_runs": estimated_seconds_all_runs,
            "estimated_hours_all_four_runs": estimated_seconds_all_runs / 3600,
            "session_limit_seconds": SESSION_SECONDS,
            "safe_budget_seconds_with_20pct_margin": safe_seconds,
            "remaining_margin_seconds": safe_seconds - estimated_seconds_all_runs,
            "num_runs": NUM_RUNS,
            "launch_command": f".venv/bin/python scripts/kaggle_validation.py --device cuda --seq-len {SEQ_LEN} --batch-size {chosen['batch_size']} --steps {target_steps}"}
    print(json.dumps(plan, indent=2))
    (ROOT / "outputs/kaggle-probe-plan.json").write_text(json.dumps(plan, indent=2) + "\n")

if __name__ == "__main__":
    main()
