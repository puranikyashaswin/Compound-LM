"""Compact decoder-only reference model.

PyTorch is an optional runtime dependency: the audit/data layers remain usable
without it, while real training fails explicitly with installation guidance.
"""
from __future__ import annotations

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

        def forward(self, x: torch.Tensor, document_ids: torch.Tensor | None = None) -> torch.Tensor:
            b, t, c = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
            if document_ids is not None:
                same = document_ids[:, None, :, None] == document_ids[:, None, None, :]
                mask = mask[None, None, :, :] | ~same
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=~mask)
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
