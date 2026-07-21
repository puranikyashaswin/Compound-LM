"""Document-aware attention helpers shared by every decoder.

Packing puts several documents in one row (or pads a single document). Attention
must not cross a document boundary. The fast path matters for wall-clock:

* one real document per row → causal SDPA / flash (``is_causal=True``), with a
  pad key mask only when padding is present
* mixed documents in a row → explicit block-diagonal mask (slower backends)
"""
from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None


def is_single_document_batch(document_ids) -> bool:
    """True when no sequence mixes two real documents (padding ignored)."""
    if torch is None:
        raise RuntimeError("PyTorch is required")
    real = document_ids >= 0
    values = document_ids.to(dtype=torch.float32)
    hi = values.masked_fill(~real, float("-inf")).max(dim=-1).values
    lo = values.masked_fill(~real, float("inf")).min(dim=-1).values
    all_pad = ~real.any(dim=-1)
    return bool((all_pad | (hi == lo)).all())


def set_assume_single_document(model, enabled: bool) -> None:
    """Propagate a packing hint so forwards can skip the per-step doc sync."""
    if model is None:
        return
    flag = bool(enabled)
    if hasattr(model, "assume_single_document"):
        model.assume_single_document = flag
    for module in model.modules():
        if hasattr(module, "assume_single_document"):
            module.assume_single_document = flag


def scaled_dot_product_attention_documented(q, k, v, document_ids=None,
                                            *, assume_single_document: bool = False):
    """SDPA with packing semantics; prefer the flash causal path when safe."""
    if torch is None:
        raise RuntimeError("PyTorch is required")
    if document_ids is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    single = assume_single_document or is_single_document_batch(document_ids)
    if single:
        # Full rows of real tokens: pure causal flash / memory-efficient path.
        if bool((document_ids >= 0).all()):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Pad keys must stay invisible; bool allow-mask keeps mem-efficient SDPA.
        length = q.shape[-2]
        causal = torch.ones(length, length, device=q.device, dtype=torch.bool).tril()
        key_ok = (document_ids >= 0)[:, None, None, :]
        allow = causal[None, None, :, :] & key_ok
        return F.scaled_dot_product_attention(q, k, v, attn_mask=allow)

    length = q.shape[-2]
    blocked = torch.triu(torch.ones(length, length, device=q.device, dtype=torch.bool),
                         diagonal=1)
    same = document_ids[:, None, :, None] == document_ids[:, None, None, :]
    blocked = blocked[None, None, :, :] | ~same
    return F.scaled_dot_product_attention(q, k, v, attn_mask=~blocked)
