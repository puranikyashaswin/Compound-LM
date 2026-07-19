"""Use a HuggingFace Llama checkpoint (Reex-1) inside this framework.

Reex-1 is a 116M LlamaForCausalLM: 12 layers, hidden 768, **12 query heads
against 4 key/value heads** (grouped-query attention), SwiGLU at 2112, RMSNorm,
RoPE theta 10000, tied embeddings, vocab 50257, context 1024.

That GQA head split is why `src/model/reex.py` cannot be used for it: ReexLM
projects Q, K and V at full width, so its `qkv` tensor is 3*768 wide where
Reex-1's is 768+256+256. Loading one into the other would either fail or, worse,
silently reinterpret the weights. Reex-1's own README also notes that its export
had to permute Q/K to convert between interleaved and split-half RoPE
conventions -- exactly the kind of detail that produces a model which loads
cleanly and is quietly broken.

So the model implementation is HuggingFace's, which the export was verified
against, and this framework contributes what it is actually for: the protocol.
Document-masked packing, health gates, the ledger, capability-at-cost, and
function-preserving growth all apply unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = object


def require_transformers():
    try:
        from transformers import LlamaForCausalLM
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError(
            "transformers is required to load a Llama checkpoint; pip install transformers"
        ) from error
    return LlamaForCausalLM


@dataclass(frozen=True)
class LlamaGrowthReport:
    mode: str
    from_layers: int
    to_layers: int
    inserted_at: tuple[int, ...]
    function_preserving: bool

    def as_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "from_layers": self.from_layers,
                "to_layers": self.to_layers, "inserted_at": list(self.inserted_at),
                "function_preserving": self.function_preserving}


def load_llama(path: str, *, dtype: str = "float32"):
    """Load a Llama checkpoint directory (the `hf_format/` export)."""
    LlamaForCausalLM = require_transformers()
    model = LlamaForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype))
    return model


def grow_llama_depth(model, *, to_layers: int, mode: str = "zero_init"):
    """Deepen a Llama model, preserving its function when ``mode='zero_init'``.

    Each decoder layer is applied as two residual additions:
    ``x = x + self_attn(...)`` then ``x = x + mlp(...)``. Zeroing the residual
    *output* projections -- ``self_attn.o_proj`` and ``mlp.down_proj`` -- makes a
    layer contribute exactly nothing, so the deeper stack computes precisely
    what the donor computed. Training then moves those projections off zero.

    New layers are interleaved rather than appended so the added capacity is
    spread through the stack instead of piled at the output end.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    if mode not in ("zero_init", "stack"):
        raise ValueError(f"unknown growth mode {mode!r}")

    layers = model.model.layers
    from_layers = len(layers)
    if to_layers <= from_layers:
        raise ValueError(f"to_layers ({to_layers}) must exceed current depth ({from_layers})")
    if to_layers % from_layers:
        raise ValueError(
            f"to_layers ({to_layers}) must be a multiple of {from_layers}; uneven "
            "growth has no canonical placement for the new layers"
        )

    grown = copy.deepcopy(model)
    repeats = to_layers // from_layers
    new_layers, inserted = [], []
    for block in grown.model.layers:
        new_layers.append(block)
        for _ in range(repeats - 1):
            clone = copy.deepcopy(block)
            if mode == "zero_init":
                with torch.no_grad():
                    clone.self_attn.o_proj.weight.zero_()
                    clone.mlp.down_proj.weight.zero_()
                    for projection in (clone.self_attn.o_proj, clone.mlp.down_proj):
                        if getattr(projection, "bias", None) is not None:
                            projection.bias.zero_()
            inserted.append(len(new_layers))
            new_layers.append(clone)

    grown.model.layers = nn.ModuleList(new_layers)
    grown.config.num_hidden_layers = to_layers
    # Layer indices are used for the KV cache; renumber so they stay unique.
    for index, block in enumerate(grown.model.layers):
        if hasattr(block, "self_attn") and hasattr(block.self_attn, "layer_idx"):
            block.self_attn.layer_idx = index
    return grown, LlamaGrowthReport(mode=mode, from_layers=from_layers,
                                    to_layers=to_layers, inserted_at=tuple(inserted),
                                    function_preserving=(mode == "zero_init"))


if torch is not None:
    class LlamaProtocolAdapter(nn.Module):
        """Give a Llama model this framework's forward contract.

        The trainer calls ``model(input_ids, document_ids)`` and expects logits.
        ``document_ids`` carries the packing boundaries: several documents share
        one sequence, and a token must not attend across a boundary. Llama's
        default mask is purely causal, which would let the end of one document
        read the one before it -- the exact leakage `src/data/packing.py` and its
        validation exist to prevent. A 4-D additive mask is built here instead,
        combining causality with document identity, and padding (negative
        document ids) is excluded.
        """

        def __init__(self, model):
            super().__init__()
            self.model = model
            config = model.config
            self.config = {
                "architecture": "llama-hf",
                "vocab_size": config.vocab_size,
                "d_model": config.hidden_size,
                "n_layers": config.num_hidden_layers,
                "n_heads": config.num_attention_heads,
                "n_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
                "intermediate_size": config.intermediate_size,
                "max_seq_len": config.max_position_embeddings,
                # Numeric details a rebuild must reproduce exactly. Reex-1 uses
                # rms_norm_eps 1e-5 while LlamaConfig defaults to 1e-6: close
                # enough to load, far enough that a session resumed through the
                # default would be a subtly different function than the one
                # trained -- the precise drift the lineage rules exist to stop.
                "rms_norm_eps": float(config.rms_norm_eps),
                "rope_theta": float(getattr(config, "rope_theta", 10000.0)),
            }

        def forward(self, input_ids, document_ids=None):
            if document_ids is None:
                return self.model(input_ids=input_ids).logits

            batch, length = input_ids.shape
            device = input_ids.device
            causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
            same_document = document_ids[:, None, :, None] == document_ids[:, None, None, :]
            real = (document_ids >= 0)[:, None, None, :]
            allowed = causal[None, None, :, :] & same_document & real

            dtype = self.model.get_input_embeddings().weight.dtype
            mask = torch.zeros(batch, 1, length, length, dtype=dtype, device=device)
            mask.masked_fill_(~allowed, torch.finfo(dtype).min)
            return self.model(input_ids=input_ids, attention_mask=mask).logits
