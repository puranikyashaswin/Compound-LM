"""Reex-v2 decoder: RMSNorm + rotary positions + SwiGLU.

This is the modern architecture candidate for Reex-2 (build plan C3 /
model-switch procedure). It deliberately keeps the same contract as
ReferenceLM -- same constructor signature, same forward signature with
document-masked attention, same tied embedding/head, and the same init
discipline (initial loss pinned to ln(vocab), see tests/test_model_init.py's
history) -- so it can be swapped in as a config-selected lever rather than a
code fork.
"""
from __future__ import annotations

import math

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = object


if torch is not None:
    class RMSNorm(nn.Module):
        def __init__(self, d_model: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d_model))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            variance = x.float().pow(2).mean(dim=-1, keepdim=True)
            return (x.float() * torch.rsqrt(variance + self.eps)).type_as(x) * self.weight


    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


    class RotaryEmbedding(nn.Module):
        def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
            super().__init__()
            if head_dim % 2:
                raise ValueError("rotary embedding requires an even head dimension")
            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
            positions = torch.arange(max_seq_len).float()
            freqs = torch.outer(positions, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos", emb.cos(), persistent=False)
            self.register_buffer("sin", emb.sin(), persistent=False)

        def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            t = q.shape[-2]
            cos = self.cos[:t][None, None, :, :]
            sin = self.sin[:t][None, None, :, :]
            return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


    class ReexAttention(nn.Module):
        def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
            super().__init__()
            if d_model % n_heads:
                raise ValueError("d_model must be divisible by n_heads")
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)

        def forward(self, x: torch.Tensor, document_ids: torch.Tensor | None = None) -> torch.Tensor:
            b, t, c = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            q, k = self.rotary(q, k)
            mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
            if document_ids is not None:
                same = document_ids[:, None, :, None] == document_ids[:, None, None, :]
                mask = mask[None, None, :, :] | ~same
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=~mask)
            return self.out(y.transpose(1, 2).contiguous().view(b, t, c))


    class SwiGLU(nn.Module):
        def __init__(self, d_model: int):
            super().__init__()
            # 8/3 * d_model keeps parameter count near the reference 4*d MLP.
            hidden = int(8 * d_model / 3)
            self.gate = nn.Linear(d_model, hidden, bias=False)
            self.up = nn.Linear(d_model, hidden, bias=False)
            self.down = nn.Linear(hidden, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


    class ReexLM(nn.Module):
        architecture = "reex-v2"

        def __init__(self, vocab_size: int = 32768, d_model: int = 256,
                     n_layers: int = 6, n_heads: int = 8, max_seq_len: int = 2048):
            super().__init__()
            self.config = dict(architecture=self.architecture, vocab_size=vocab_size,
                               d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                               max_seq_len=max_seq_len)
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    "norm1": RMSNorm(d_model),
                    "attn": ReexAttention(d_model, n_heads, max_seq_len),
                    "norm2": RMSNorm(d_model),
                    "mlp": SwiGLU(d_model),
                }) for _ in range(n_layers)
            ])
            self.norm = RMSNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight
            self.apply(self._init_weights)
            # Residual output projections scaled by depth, same policy the
            # reference model needed to keep initial loss at ln(vocab).
            for name, parameter in self.named_parameters():
                if name.endswith("attn.out.weight") or name.endswith("mlp.down.weight"):
                    nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

        @staticmethod
        def _init_weights(module) -> None:
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
            x = self.token_embedding(input_ids)
            for block in self.blocks:
                x = x + block["attn"](block["norm1"](x), document_ids)
                x = x + block["mlp"](block["norm2"](x))
            return self.lm_head(self.norm(x))
