"""Minimal checkpointable training loop for packed COMPOUND-LM shards."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from src.data.loader import open_shard
from src.health.check import check_checkpoint, RollingMedian
from src.ledger.writer import append_entry
from src.model.reference import require_torch
from src.provenance.core import config_hash, sha256_bytes


def masked_next_token_loss(logits, ids, document_ids):
    import torch
    # Standard next-token prediction alignment:
    # logits[:, :-1] predicts ids[:, 1:]
    # document_ids[:, 1:] specifies the document for each target.
    # Targets are ignored if they belong to padding (document_id < 0).
    pred_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
    targets = ids[:, 1:].reshape(-1)
    target_docs = document_ids[:, 1:].reshape(-1)

    ignore_index = -100
    masked_targets = targets.clone()
    masked_targets[target_docs < 0] = ignore_index

    return torch.nn.functional.cross_entropy(pred_logits, masked_targets, ignore_index=ignore_index)


def lr_at_step(step: int, *, total_steps: int, base_lr: float,
               warmup_fraction: float = 0.01, min_lr_fraction: float = 0.1) -> float:
    """Linear warmup then cosine decay -- the standard LM schedule.

    A constant LR leaves a real-scale baseline weaker than it should be, which
    inflates every lever measured against it. Applied identically to every arm.
    """
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if not 0.0 <= min_lr_fraction <= 1.0:
        raise ValueError("min_lr_fraction must be in [0, 1]")
    warmup_steps = int(total_steps * warmup_fraction)
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_fraction + (1.0 - min_lr_fraction) * cosine)


def train(shard: str, output_dir: str, *, vocab_size: int = 32768,
          d_model: int = 128, n_layers: int = 2, n_heads: int = 4,
          steps: int = 10, learning_rate: float = 3e-4, seed: int = 17,
          device: str = "cpu", resume: str | None = None,
          checkpoint_every: int = 10, ledger_path: str | None = None,
          run_id: str | None = None, use_muon: bool = False,
          muon_lr: float = 0.02,
          heldout_shard: str | None = None, levers_on: list[str] | None = None,
          batch_size: int = 1, lr_schedule: bool = False,
          warmup_fraction: float = 0.01, min_lr_fraction: float = 0.1) -> dict:
    require_torch()
    import torch
    from src.model.reference import ReferenceLM
    from src.data.contamination import assert_disjoint_shards
    
    # 1. Run automatic contamination check
    if heldout_shard:
        assert_disjoint_shards(shard, heldout_shard)

    random.seed(seed); torch.manual_seed(seed)
    rows = open_shard(shard)

    model = ReferenceLM(vocab_size, d_model, n_layers, n_heads, rows.sequence_length)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    muon_optimizer = None
    optimizer_groups = {"adamw": [name for name, _ in model.named_parameters()], "muon": []}
    if use_muon:
        from src.optim.muon import Muon, partition_named_parameters
        partition = partition_named_parameters(model.named_parameters())
        by_name = dict(model.named_parameters())
        muon_optimizer = Muon([by_name[name] for name in partition.muon], lr=muon_lr)
        optimizer = torch.optim.AdamW([by_name[name] for name in partition.adamw], lr=learning_rate)
        optimizer_groups = {"adamw": list(partition.adamw), "muon": list(partition.muon)}
    
    losses = []
    start_step = 0
    data_position = 0
    grad_history = RollingMedian(window=100)

    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        if bool(state.get("use_muon", False)) != use_muon:
            raise ValueError("resume checkpoint optimizer mode does not match requested use_muon")
        # The recorded loader provenance must match this call, or the resumed
        # curve would silently continue over different data than it began on.
        checkpoint_shard = state.get("shard")
        if checkpoint_shard is not None and checkpoint_shard != str(Path(shard).resolve()):
            raise ValueError(
                f"resume checkpoint was trained on shard {checkpoint_shard}, not {Path(shard).resolve()}"
            )
        checkpoint_batch_size = state.get("batch_size")
        if checkpoint_batch_size is not None and int(checkpoint_batch_size) != batch_size:
            raise ValueError(
                f"resume checkpoint used batch_size {checkpoint_batch_size}, not {batch_size}"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if muon_optimizer is not None and state.get("muon_optimizer"):
            muon_optimizer.load_state_dict(state["muon_optimizer"])
        losses = list(state.get("losses", []))
        start_step = int(state["step"])
        seed = int(state.get("seed", seed))
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])
        if "python_rng_state" in state:
            random.setstate(state["python_rng_state"])
        if "grad_history" in state:
            grad_history.history = list(state["grad_history"])
        data_position = int(state.get("data_position", (start_step * batch_size) % len(rows)))

    model.train()
    started = time.perf_counter()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(step: int) -> Path:
        target = out / f"checkpoint-{step:08d}.pt"
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "step": step, "seed": seed, "config": model.config, "losses": losses,
                   "use_muon": use_muon, "optimizer_groups": optimizer_groups,
                   "muon_optimizer": muon_optimizer.state_dict() if muon_optimizer else None,
                   "torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(),
                   "grad_history": grad_history.history,
                   # Loader provenance. data_position is the live cursor the loop
                   # reads, not a recomputation of it, so a resumed run cannot
                   # disagree with the checkpoint about where the data resumes.
                   "shard": str(Path(shard).resolve()),
                   "data_position": data_position,
                   "batch_size": batch_size}
        torch.save(payload, target)
        return target

    n_params = sum(p.numel() for p in model.parameters())

    for step in range(start_step, steps):
        batch_ids, batch_docs = rows.batch(data_position, batch_size)
        data_position = (data_position + batch_size) % len(rows)

        ids = torch.tensor(batch_ids, dtype=torch.long, device=device) % vocab_size
        docs = torch.tensor(batch_docs, dtype=torch.long, device=device)

        if lr_schedule:
            # Both optimizers follow the same shape, each scaled from its own
            # base LR, so a schedule cannot advantage one arm over the other.
            scale = lr_at_step(step, total_steps=steps, base_lr=1.0,
                               warmup_fraction=warmup_fraction,
                               min_lr_fraction=min_lr_fraction)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * scale
            if muon_optimizer is not None:
                for group in muon_optimizer.param_groups:
                    group["lr"] = muon_lr * scale

        logits = model(ids, docs)
        
        # Use masked target cross entropy loss
        loss = masked_next_token_loss(logits, ids, docs)
        
        optimizer.zero_grad(set_to_none=True)
        if muon_optimizer is not None:
            muon_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).detach().cpu())
        grad_history.add(grad_norm)
        
        optimizer.step()
        if muon_optimizer is not None:
            muon_optimizer.step()
        
        current_loss = float(loss.detach().cpu())
        losses.append(current_loss)
        
        # 5. Ledger records score AND cost at every scheduled checkpoint, not only at the end.
        if checkpoint_every > 0 and ((step + 1) % checkpoint_every == 0 or step + 1 == steps):
            checkpoint = save_checkpoint(step + 1)
            checkpoint_hash = sha256_bytes(checkpoint.read_bytes())
            
            # Check checkpoint health. If health is red, raise ValueError to terminate training.
            report = check_checkpoint(loss=current_loss, grad_norm=grad_norm, median_grad_norm=grad_history.median(),
                                      checkpoint_hash=checkpoint_hash, provenance_ok=True)
            if report.status == "red":
                raise ValueError(f"health check failed: {report.failures}")
                
            eval_scores: dict = {}
            if heldout_shard:
                from src.eval.intrinsic import evaluate
                eval_scores = evaluate(str(checkpoint), heldout_shard, device=device)
                eval_scores["smoke"] = eval_scores["val_acc"]  # higher-is-better headline score
            
            tokens = (step + 1) * batch_size * rows.sequence_length
            train_flops = 6.0 * n_params * tokens
            
            if ledger_path:
                append_entry(ledger_path, {
                    "run_id": run_id or f"reference-{config_hash(model.config)[:12]}-{seed}",
                    "config_hash": config_hash(model.config), "commit": "uncommitted",
                    "scale": f"{n_params}p",
                    "levers_on": levers_on or [], "tokens": tokens,
                    "wall_clock_s": time.perf_counter() - started, "gpu_type": device,
                    "train_flops": train_flops,
                    "est_cost_usd": train_flops, "fully_accounted_cost_usd": train_flops,
                    "final_loss": current_loss, "eval_scores": eval_scores, "seed": seed,
                    "notes": f"checkpoint-{step+1}", "checkpoint_hash": checkpoint_hash,
                    "health": report.as_dict(), "resume": resume,
                    "optimizer_groups": optimizer_groups,
                })

    # Prepare final result dictionary
    final_step = steps
    checkpoint_path = out / f"checkpoint-{final_step:08d}.pt"
    if not checkpoint_path.exists():
        checkpoint = save_checkpoint(final_step)
    else:
        checkpoint = checkpoint_path
    
    checkpoint_hash = sha256_bytes(checkpoint.read_bytes())
    final_loss = losses[-1] if losses else 0.0
    last_grad_norm = grad_history.history[-1] if grad_history.history else 1e-12
    
    report = check_checkpoint(loss=final_loss, grad_norm=last_grad_norm, median_grad_norm=grad_history.median(),
                              checkpoint_hash=checkpoint_hash, provenance_ok=True)
    
    eval_scores = {}
    if heldout_shard:
        from src.eval.intrinsic import evaluate
        eval_scores = evaluate(str(checkpoint), heldout_shard, device=device)
        eval_scores["smoke"] = eval_scores["val_acc"]
        
    tokens = final_step * batch_size * rows.sequence_length
    train_flops = 6.0 * n_params * tokens
    
    result = {"checkpoint": str(checkpoint), "steps": final_step, "losses": losses,
              "final_loss": final_loss, "seed": seed, "health": report.as_dict(),
              "checkpoint_hash": checkpoint_hash, "optimizer_groups": optimizer_groups,
              "use_muon": use_muon, "eval_scores": eval_scores, "train_flops": train_flops}
              
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=10); parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume"); parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--ledger"); parser.add_argument("--run-id")
    parser.add_argument("--use-muon", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(train(args.shard, args.output_dir, steps=args.steps, device=args.device,
                           resume=args.resume, checkpoint_every=args.checkpoint_every,
                           ledger_path=args.ledger, run_id=args.run_id,
                           use_muon=args.use_muon, batch_size=args.batch_size), indent=2))


if __name__ == "__main__":
    main()
