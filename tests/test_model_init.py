"""An untrained model must be exactly as good as guessing -- no worse.

With nn.Embedding's default N(0, 1) init and lm_head tied to it, initial logits
had std ~= sqrt(d_model): at d_model=256 the untrained model scored *worse* than
random (loss 25 against ln(50257)=10.8) and early training was spent undoing the
initialization rather than learning. Baselines carrying that handicap inflate
every lever measured against them.
"""
import math

import pytest
import torch

from src.model.reference import ReferenceLM
from src.train.reference import masked_next_token_loss

VOCAB = 50257


def _initial_loss(d_model: int, n_layers: int, n_heads: int, seed: int = 0) -> float:
    torch.manual_seed(seed)
    model = ReferenceLM(VOCAB, d_model, n_layers, n_heads, 128)
    ids = torch.randint(0, VOCAB, (2, 128))
    docs = torch.zeros_like(ids)
    return masked_next_token_loss(model(ids, docs), ids, docs).item()


@pytest.mark.parametrize("d_model,n_layers,n_heads", [(32, 2, 2), (256, 12, 8), (768, 12, 8)])
def test_initial_loss_is_uniform_entropy_at_every_width(d_model, n_layers, n_heads):
    """Must hold as d_model grows: the old bug scaled with sqrt(d_model)."""
    loss = _initial_loss(d_model, n_layers, n_heads)
    assert loss == pytest.approx(math.log(VOCAB), abs=0.6)


def test_initial_logits_are_not_saturated():
    torch.manual_seed(0)
    model = ReferenceLM(VOCAB, 256, 12, 8, 128)
    ids = torch.randint(0, VOCAB, (2, 128))
    logits = model(ids, torch.zeros_like(ids))
    # Old init produced std ~= 16 here; a sane init keeps logits near zero.
    assert logits.std().item() < 1.0


def test_deep_stacks_do_not_amplify_activations():
    """Residual output projections are scaled by 1/sqrt(2*n_layers)."""
    shallow = _initial_loss(256, 2, 8)
    deep = _initial_loss(256, 24, 8)
    assert abs(shallow - deep) < 0.5


def test_tied_head_still_shares_storage_after_init():
    model = ReferenceLM(VOCAB, 64, 2, 2, 32)
    assert model.lm_head.weight is model.token_embedding.weight
