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


def evaluate(checkpoint: str, heldout_shard: str, *, device: str = "cpu") -> dict[str, Any]:
    """Return intrinsic scores; higher ``val_acc`` is better."""
    require_torch()
    import torch
    from src.model.reference import ReferenceLM

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = state["config"]
    model = ReferenceLM(config["vocab_size"], config["d_model"], config["n_layers"],
                        config["n_heads"], config["max_seq_len"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    vocab = config["vocab_size"]

    # Same loader the trainer uses, so a held-out score is measured against the
    # data it claims regardless of which shard format is on disk.
    rows = open_shard(heldout_shard)
    if len(rows) == 0:
        raise ValueError("held-out shard is empty")

    total_nll = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for index in range(len(rows)):
            batch_ids, batch_docs = rows.batch(index, 1)
            ids = torch.tensor(batch_ids, dtype=torch.long, device=device) % vocab
            docs = torch.tensor(batch_docs, dtype=torch.long, device=device)
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
