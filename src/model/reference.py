"""Compact decoder-only reference model.

PyTorch is an optional runtime dependency: the audit/data layers remain usable
without it, while real training fails explicitly with installation guidance.
"""
from __future__ import annotations

import math

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = object


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for model training; install torch for your platform")


if torch is not None:
    class CausalSelfAttention(nn.Module):
        def __init__(self, d_model: int, n_heads: int):
            super().__init__()
            if d_model % n_heads:
                raise ValueError("d_model must be divisible by n_heads")
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.out = nn.Linear(d_model, d_model)
            # Set by the trainer when the shard meta guarantees single-doc rows.
            self.assume_single_document = False

        def forward(self, x: torch.Tensor, document_ids: torch.Tensor | None = None) -> torch.Tensor:
            b, t, c = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            from src.model.document_attention import scaled_dot_product_attention_documented
            assume = bool(getattr(self, "assume_single_document", False))
            y = scaled_dot_product_attention_documented(
                q, k, v, document_ids, assume_single_document=assume)
            return self.out(y.transpose(1, 2).contiguous().view(b, t, c))


    class ReferenceLM(nn.Module):
        def __init__(self, vocab_size: int = 32768, d_model: int = 256,
                     n_layers: int = 6, n_heads: int = 8, max_seq_len: int = 2048):
            super().__init__()
            self.config = dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                               n_heads=n_heads, max_seq_len=max_seq_len)
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    "norm1": nn.LayerNorm(d_model),
                    "attn": CausalSelfAttention(d_model, n_heads),
                    "norm2": nn.LayerNorm(d_model),
                    "mlp": nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)),
                }) for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight
            self.apply(self._init_weights)
            # GPT-2's scaled init: with n_layers residual additions feeding the
            # stream, unscaled output projections make activations grow with
            # depth. Applied after _init_weights so it wins for these tensors.
            for name, parameter in self.named_parameters():
                if name.endswith("attn.out.weight") or name.endswith("mlp.2.weight"):
                    nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

        @staticmethod
        def _init_weights(module) -> None:
            """Small-std init.

            nn.Embedding defaults to N(0, 1); since lm_head is tied to the token
            embedding, that puts initial logits at std ~= sqrt(d_model), which is
            saturated well beyond a uniform distribution -- an untrained model
            scores *worse* than random and training spends its early budget
            merely undoing the initialization.
            """
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, input_ids: torch.Tensor, document_ids: torch.Tensor | None = None) -> torch.Tensor:
            _, t = input_ids.shape
            if t > self.config["max_seq_len"]:
                raise ValueError("sequence exceeds model max_seq_len")
            positions = torch.arange(t, device=input_ids.device)[None, :]
            x = self.token_embedding(input_ids) + self.position_embedding(positions)
            for block in self.blocks:
                x = x + block["attn"](block["norm1"](x), document_ids)
                x = x + block["mlp"](block["norm2"](x))
            return self.lm_head(self.norm(x))
