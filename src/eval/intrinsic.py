"""Intrinsic capability evaluation on a held-out packed shard.

This is a real, non-fabricated metric: it loads a trained checkpoint, runs a
forward pass over held-out sequences the model never trained on, and reports
next-token cross-entropy, bits-per-token, and top-1 accuracy over non-padding,
in-document positions. On hardware without the lm-eval benchmark suite this is
the honest capability score that feeds the capability-at-cost ledger; the same
runner is replaced by the frozen E-v1 harness once GPUs are available.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any

from src.data.loader import open_shard
from src.model.reference import require_torch
from src.train.reference import assert_token_ids_in_range


def evaluate(checkpoint: str | None, heldout_shard: str, *, device: str = "cpu",
             batch_size: int = 32, max_batches: int | None = None,
             model=None) -> dict[str, Any]:
    """Return intrinsic scores; higher ``val_acc`` is better.

    Held-out sequences are independent and the document mask is per-sequence,
    so batching changes throughput and nothing else -- ``batch_size`` must not
    move any reported score. It defaults to 32 because this ran one sequence at
    a time, which leaves a GPU almost entirely idle and made the scheduled
    evaluation a real fraction of a run's wall clock.

    Pass ``model`` (already on ``device``) to score the resident training weights
    without reloading a multi-GB checkpoint that also carries optimizer state.
    ``checkpoint`` is required when ``model`` is omitted.
    """
    require_torch()
    import torch
    from src.model.registry import model_from_config

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if model is None and not checkpoint:
        raise ValueError("evaluate requires checkpoint or model")

    owns_model = model is None
    if owns_model:
        # Loaded to CPU first: a training checkpoint also carries the optimizer's
        # moment tensors (2/3 of its bytes), and map_location=device would park
        # those on the GPU for the whole evaluation next to the resident training
        # model -- pure waste that at 193M params is ~1.5GB of headroom.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = state["config"]
        model = model_from_config(config)
        model.load_state_dict(state["model"])
        del state
        model.to(device).eval()
        vocab = config["vocab_size"]
    else:
        vocab = int(model.config["vocab_size"])
        was_training = model.training
        model.eval()

    # Same loader the trainer uses, so a held-out score is measured against the
    # data it claims regardless of which shard format is on disk.
    rows = open_shard(heldout_shard)
    if len(rows) == 0:
        raise ValueError("held-out shard is empty")

    total_nll = 0.0
    total_correct = 0
    total_tokens = 0
    # ``max_batches`` caps the scored prefix. A mid-training curve needs a
    # stable, comparable score, not the whole held-out set: scoring 2.2M tokens
    # at every checkpoint of a long run spends GPU-hours re-measuring what a
    # 260K-token prefix already pins to ~3 decimal places. The cap always takes
    # the FIRST batches, so every checkpoint scores the identical subset.
    limit = len(rows) if max_batches is None else min(len(rows), max_batches * batch_size)
    try:
        with torch.no_grad():
            for index in range(0, limit, batch_size):
                # Never wrap past the end: rows.batch is cyclic, so a final partial
                # batch would silently re-score sequences from the start of the
                # shard and weight them twice in the mean.
                take = min(batch_size, limit - index)
                batch_ids, batch_docs = rows.batch(index, take)
                ids = torch.as_tensor(batch_ids, dtype=torch.long, device=device)
                assert_token_ids_in_range(ids, vocab)
                docs = torch.as_tensor(batch_docs, dtype=torch.long, device=device)
                logits = model(ids, docs)
                # Predict position t+1 from t; only score targets inside a real document.
                pred_logits = logits[:, :-1].reshape(-1, vocab)
                targets = ids[:, 1:].reshape(-1)
                target_docs = docs[:, 1:].reshape(-1)
                keep = target_docs >= 0
                if keep.sum() == 0:
                    continue
                pred_logits, targets = pred_logits[keep], targets[keep]
                nll = torch.nn.functional.cross_entropy(pred_logits, targets, reduction="sum")
                total_nll += float(nll)
                total_correct += int((pred_logits.argmax(dim=-1) == targets).sum())
                total_tokens += int(keep.sum())
    finally:
        if not owns_model and was_training:
            model.train()

    if total_tokens == 0:
        raise ValueError("no scorable held-out tokens")
    mean_nll = total_nll / total_tokens
    return {
        "val_nll": mean_nll,
        "val_bits_per_token": mean_nll / math.log(2),
        "val_perplexity": math.exp(mean_nll),
        "val_acc": total_correct / total_tokens,
        "heldout_tokens": total_tokens,
        "heldout_sequences": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.heldout, device=args.device), indent=2))


if __name__ == "__main__":
    main()
