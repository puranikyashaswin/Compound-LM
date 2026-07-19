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
from src.ledger.writer import append_entry, read_entries
from src.model.reference import require_torch
from src.provenance.core import config_hash, sha256_bytes


def assert_token_ids_in_range(ids, vocab_size: int) -> None:
    """Refuse tokens the model has no embedding for.

    This used to be ``ids % vocab_size``. Folding is fine for the hash-based
    fallback tokenizer, but applying it to a real corpus silently remaps every
    out-of-range token onto a *different valid token*: training proceeds, the
    loss curve looks healthy, and the run measures nothing. It is precisely the
    failure mode of the vocabulary-reduction lever, where a 50257-vocab corpus
    meets a 16384-vocab model. The fold now happens once, explicitly, at data
    preparation time (``src/data/pipeline.token_ids``) and is recorded in the
    datasheet; here a mismatch is an error.
    """
    highest = int(ids.max())
    lowest = int(ids.min())
    if highest >= vocab_size or lowest < 0:
        raise ValueError(
            f"token id out of range: shard contains ids in [{lowest}, {highest}] but the "
            f"model's vocab_size is {vocab_size}. The shard was tokenized for a different "
            f"vocabulary -- rebuild it at this size rather than folding ids into range."
        )


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


# Floating-point results that agree to this many relative units are the same
# measurement. Exact equality was never a sound contract: summing a held-out
# set in a different batch order changes the last ~7 digits, as does a
# different GPU kernel. The build plan calls for a declared tolerance rather
# than byte equality, and this is it. It is tight enough that any real
# divergence -- a different data order, a different seed, a different
# precision -- lands orders of magnitude outside it.
REPLAY_TOLERANCE = 1e-6


def _measurements_agree(prior: object, current: object) -> bool:
    """Compare replayed measurements up to floating-point reduction order."""
    if isinstance(prior, dict) and isinstance(current, dict):
        return prior.keys() == current.keys() and all(
            _measurements_agree(prior[key], current[key]) for key in prior
        )
    if isinstance(prior, bool) or isinstance(current, bool):
        return prior == current
    if isinstance(prior, (int, float)) and isinstance(current, (int, float)):
        if math.isnan(prior) or math.isnan(current):
            return math.isnan(prior) and math.isnan(current)
        return abs(prior - current) <= REPLAY_TOLERANCE * max(1.0, abs(prior), abs(current))
    return prior == current


def _append_or_verify_replay(ledger_path: str, entry: dict) -> None:
    """Append a ledger row, tolerating an identical deterministic replay.

    Resuming restarts from the newest readable checkpoint, which may sit behind
    the newest ledger row if a ledgered checkpoint was later lost or truncated.
    The replayed steps then produce rows that already exist, and the append-only
    guard would reject them -- killing the recovery it exists to protect.

    A replay reproduces the measurements, so the recorded science must match.
    It is compared on loss and eval scores rather than ``checkpoint_hash``:
    resuming yields bitwise-equal weights but *not* a byte-identical file,
    because torch.save's container is not stable across a load/re-save cycle.
    The hash proves file integrity, not state identity, so comparing it here
    would flag every honest replay as a contradiction.
    """
    prior = next((row for row in read_entries(ledger_path)
                  if row["run_id"] == entry["run_id"] and row["tokens"] == entry["tokens"]), None)
    if prior is None:
        append_entry(ledger_path, entry)
        return
    measured = ("final_loss", "eval_scores")
    disagreements = {key: (prior.get(key), entry.get(key)) for key in measured
                     if not _measurements_agree(prior.get(key), entry.get(key))}
    if disagreements:
        raise ValueError(
            f"ledger contradiction: {entry['run_id']} at {entry['tokens']} tokens was already "
            f"recorded with different results {disagreements}. A resumed run must reproduce "
            f"the measurements it replays."
        )


def resumable_checkpoint(output_dir: str | Path) -> Path | None:
    """Newest checkpoint in ``output_dir`` that actually loads, or None.

    A crash during ``torch.save`` can leave a truncated file, and that file
    sorts newest -- so naively resuming from the last checkpoint would fail on
    exactly the runs that most need to recover. Unreadable candidates are
    deleted: they carry no recoverable state, and leaving them would let the
    final-checkpoint reuse path mistake one for a finished run.
    """
    require_torch()
    import torch

    directory = Path(output_dir)
    if not directory.is_dir():
        return None
    for candidate in sorted(directory.glob("checkpoint-*.pt"), reverse=True):
        try:
            torch.load(candidate, map_location="cpu", weights_only=False)
            return candidate
        except Exception:
            candidate.unlink(missing_ok=True)
    return None


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
          warmup_fraction: float = 0.01, min_lr_fraction: float = 0.1,
          architecture: str = "reference-v1", precision: str = "fp32",
          compile_model: bool = False, grad_clip: float | None = None,
          weight_decay: float = 0.01, eval_batch_size: int = 32) -> dict:
    require_torch()
    import torch
    from src.model.registry import build_model
    from src.data.contamination import assert_disjoint_shards
    from src.train.systems import (SystemsPolicy, apply_backend_flags, make_grad_scaler,
                                   maybe_compile, resolve_precision)

    # 1. Run automatic contamination check
    if heldout_shard:
        assert_disjoint_shards(shard, heldout_shard)

    random.seed(seed); torch.manual_seed(seed)
    rows = open_shard(shard)

    model = build_model(architecture, vocab_size=vocab_size, d_model=d_model,
                        n_layers=n_layers, n_heads=n_heads,
                        max_seq_len=rows.sequence_length)
    model.to(device)

    # Systems lever: same mathematics, fewer seconds per step. Default fp32
    # preserves every result already in the ledger; speed is opt-in.
    policy = SystemsPolicy(precision=precision, compile=compile_model, device=device)
    resolved = resolve_precision(policy, device=device)
    apply_backend_flags(resolved)
    autocast_dtype = getattr(torch, resolved.autocast_dtype) if resolved.autocast_dtype else None
    scaler = make_grad_scaler(device, enabled=resolved.use_grad_scaler)
    for note in resolved.notes:
        print(f"[systems] {note}")

    named = dict(model.named_parameters())

    def _decay_groups(names: list[str]) -> list[dict]:
        """Split into decayed matrices and undecayed norms/biases/1-D tensors.

        Weight decay on LayerNorm/RMSNorm gains and biases pulls them toward
        zero for no regularization benefit -- it fights the normalization the
        architecture depends on. Every serious recipe excludes them; PyTorch's
        AdamW default does not, so it has to be done here.
        """
        decay = [n for n in names if named[n].ndim >= 2]
        no_decay = [n for n in names if named[n].ndim < 2]
        return [{"params": [named[n] for n in decay], "weight_decay": weight_decay},
                {"params": [named[n] for n in no_decay], "weight_decay": 0.0}]

    def _adamw(names: list[str]):
        # Fused AdamW folds the elementwise update into one kernel; on a small
        # model the optimizer step is a real fraction of step time.
        groups = _decay_groups(names)
        if policy.fused_optimizer and device == "cuda":
            try:
                return torch.optim.AdamW(groups, lr=learning_rate, fused=True)
            except (RuntimeError, TypeError):
                pass
        return torch.optim.AdamW(groups, lr=learning_rate)

    optimizer = _adamw(list(named))
    muon_optimizer = None
    optimizer_groups = {"adamw": [name for name, _ in model.named_parameters()], "muon": []}
    if use_muon:
        from src.optim.muon import Muon, partition_named_parameters
        partition = partition_named_parameters(model.named_parameters())
        by_name = dict(model.named_parameters())
        muon_optimizer = Muon([by_name[name] for name in partition.muon], lr=muon_lr)
        optimizer = _adamw(list(partition.adamw))
        optimizer_groups = {"adamw": list(partition.adamw), "muon": list(partition.muon)}

    compiled_model, _ = maybe_compile(model, policy)

    losses = []
    start_step = 0
    data_position = 0
    grad_history = RollingMedian(window=100)

    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        if bool(state.get("use_muon", False)) != use_muon:
            raise ValueError("resume checkpoint optimizer mode does not match requested use_muon")
        # An architecture mismatch is a lineage break, not a resume.
        checkpoint_arch = state.get("config", {}).get("architecture", "reference-v1")
        if checkpoint_arch != architecture:
            raise ValueError(
                f"resume checkpoint architecture is {checkpoint_arch!r}, not {architecture!r}"
            )
        # The recorded loader provenance must match this call, or the resumed
        # curve would silently continue over different data than it began on.
        checkpoint_shard = state.get("shard")
        if checkpoint_shard is not None and checkpoint_shard != str(Path(shard).resolve()):
            raise ValueError(
                f"resume checkpoint was trained on shard {checkpoint_shard}, not {Path(shard).resolve()}"
            )
        # Precision changes the numerics, so a resumed curve that switched
        # precision is not the curve it claims to continue.
        checkpoint_precision = state.get("precision")
        if checkpoint_precision is not None and checkpoint_precision != resolved.autocast_dtype:
            raise ValueError(
                f"resume checkpoint was trained at precision {checkpoint_precision!r}, "
                f"not {resolved.autocast_dtype!r}"
            )
        if state.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(state["scaler"])
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

    def _write_checkpoint(payload: dict, target: Path, attempts: int = 3) -> None:
        """Write a checkpoint, retrying transient filesystem failures.

        torch.save has been observed failing with "Parent directory does not
        exist" for a directory that demonstrably exists moments later. The cause
        is unexplained, so this does not pretend to fix it -- it re-asserts the
        directory and retries, which turns a lost multi-hour run into a pause.
        A checkpoint that fails every attempt is removed rather than left as a
        partial file that a later resume would mistake for a good one.
        """
        for attempt in range(attempts):
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, target)
                return
            except (RuntimeError, OSError) as error:
                target.unlink(missing_ok=True)
                if attempt == attempts - 1:
                    raise RuntimeError(
                        f"checkpoint write to {target} failed after {attempts} attempts: {error}"
                    ) from error
                time.sleep(0.5 * (attempt + 1))

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
                   "batch_size": batch_size,
                   "precision": resolved.autocast_dtype,
                   "scaler": scaler.state_dict() if scaler.is_enabled() else None}
        _write_checkpoint(payload, target)
        return target

    n_params = sum(p.numel() for p in model.parameters())

    for step in range(start_step, steps):
        batch_ids, batch_docs = rows.batch(data_position, batch_size)
        data_position = (data_position + batch_size) % len(rows)

        ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
        assert_token_ids_in_range(ids, vocab_size)
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

        if autocast_dtype is not None:
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                logits = compiled_model(ids, docs)
                loss = masked_next_token_loss(logits, ids, docs)
        else:
            logits = compiled_model(ids, docs)
            loss = masked_next_token_loss(logits, ids, docs)

        optimizer.zero_grad(set_to_none=True)
        if muon_optimizer is not None:
            muon_optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Unscale before reading the gradient norm. fp16 loss scaling multiplies
        # gradients by ~2^16; measuring the norm first would feed the health gate
        # a number three orders of magnitude off and trip the spike check on a
        # perfectly healthy run. Muon is unscaled explicitly because GradScaler
        # only touches the optimizers it is handed.
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            if muon_optimizer is not None:
                scaler.unscale_(muon_optimizer)

        # 1e9 is not clipping, it is measuring: the returned norm is the value
        # before any clipping, so the default keeps the health gate's reading
        # while leaving updates untouched. A real grad_clip (1.0 is standard)
        # bounds the update and is what lets a higher learning rate stay
        # stable -- but it changes the numbers, so it stays opt-in.
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip if grad_clip is not None else 1e9).detach().cpu())
        grad_history.add(grad_norm)

        if scaler.is_enabled():
            scaler.step(optimizer)
            if muon_optimizer is not None:
                scaler.step(muon_optimizer)
            scaler.update()
        else:
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
                eval_scores = evaluate(str(checkpoint), heldout_shard, device=device,
                                     batch_size=eval_batch_size)
                eval_scores["smoke"] = eval_scores["val_acc"]  # higher-is-better headline score
            
            tokens = (step + 1) * batch_size * rows.sequence_length
            train_flops = 6.0 * n_params * tokens
            
            if ledger_path:
                _append_or_verify_replay(ledger_path, {
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
        eval_scores = evaluate(str(checkpoint), heldout_shard, device=device,
                                     batch_size=eval_batch_size)
        eval_scores["smoke"] = eval_scores["val_acc"]
        
    tokens = final_step * batch_size * rows.sequence_length
    train_flops = 6.0 * n_params * tokens
    
    result = {"checkpoint": str(checkpoint), "steps": final_step, "losses": losses,
              "final_loss": final_loss, "seed": seed, "health": report.as_dict(),
              "checkpoint_hash": checkpoint_hash, "optimizer_groups": optimizer_groups,
              "use_muon": use_muon, "eval_scores": eval_scores, "train_flops": train_flops,
              "architecture": architecture, "systems": resolved.as_dict(),
              "grad_clip": grad_clip, "weight_decay": weight_decay,
              "eval_batch_size": eval_batch_size}
              
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=10); parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume"); parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--ledger"); parser.add_argument("--run-id")
    parser.add_argument("--use-muon", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--architecture", default="reference-v1")
    args = parser.parse_args()
    print(json.dumps(train(args.shard, args.output_dir, steps=args.steps, device=args.device,
                           resume=args.resume, checkpoint_every=args.checkpoint_every,
                           ledger_path=args.ledger, run_id=args.run_id,
                           use_muon=args.use_muon, batch_size=args.batch_size,
                           architecture=args.architecture), indent=2))


if __name__ == "__main__":
    main()
