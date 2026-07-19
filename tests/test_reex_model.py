"""Reex-v2 architecture: same honesty gates as the reference model.

The 3.6x Muon mirage came from a broken init; any new architecture must pass
the identical pin (initial loss == ln(vocab)) before its runs can be believed.
The registry tests protect lineage: an old checkpoint without an architecture
key stays reference-v1, and a resume across architectures is refused.
"""
import math

import pytest
import torch

from src.model.reex import ReexLM
from src.model.reference import ReferenceLM
from src.model.registry import build_model, model_from_config
from src.train.reference import masked_next_token_loss

VOCAB = 50257


def _initial_loss(d_model: int, n_layers: int, n_heads: int, seed: int = 0) -> float:
    torch.manual_seed(seed)
    model = ReexLM(VOCAB, d_model, n_layers, n_heads, 128)
    ids = torch.randint(0, VOCAB, (2, 128))
    docs = torch.zeros_like(ids)
    return masked_next_token_loss(model(ids, docs), ids, docs).item()


@pytest.mark.parametrize("d_model,n_layers,n_heads", [(32, 2, 2), (256, 12, 8), (768, 12, 8)])
def test_initial_loss_is_uniform_entropy_at_every_width(d_model, n_layers, n_heads):
    loss = _initial_loss(d_model, n_layers, n_heads)
    assert loss == pytest.approx(math.log(VOCAB), abs=0.6)


def test_deep_stacks_do_not_amplify_activations():
    shallow = _initial_loss(256, 2, 8)
    deep = _initial_loss(256, 24, 8)
    assert abs(shallow - deep) < 0.5


def test_tied_head_still_shares_storage_after_init():
    model = ReexLM(VOCAB, 64, 2, 2, 32)
    assert model.lm_head.weight is model.token_embedding.weight


def test_param_count_close_to_reference():
    """SwiGLU at 8/3 width keeps the comparison fair on parameters."""
    kwargs = dict(vocab_size=4096, d_model=96, n_layers=3, n_heads=4, max_seq_len=96)
    ref = sum(p.numel() for p in ReferenceLM(**kwargs).parameters())
    reex = sum(p.numel() for p in ReexLM(**kwargs).parameters())
    assert abs(reex - ref) / ref < 0.10


def test_document_mask_blocks_cross_document_attention():
    torch.manual_seed(0)
    model = ReexLM(256, 32, 2, 2, 16).eval()
    ids = torch.randint(0, 256, (1, 16))
    docs = torch.zeros(1, 16, dtype=torch.long)
    docs[:, 8:] = 1
    with torch.no_grad():
        base = model(ids, docs)
        perturbed_ids = ids.clone()
        perturbed_ids[0, 12] = (ids[0, 12] + 1) % 256
        perturbed = model(perturbed_ids, docs)
    # Changing a token in document 1 must not change document 0's logits.
    assert torch.allclose(base[0, :8], perturbed[0, :8], atol=1e-5)


def test_registry_builds_both_architectures():
    kwargs = dict(vocab_size=128, d_model=32, n_layers=2, n_heads=2, max_seq_len=16)
    assert isinstance(build_model("reference-v1", **kwargs), ReferenceLM)
    assert isinstance(build_model("reex-v2", **kwargs), ReexLM)
    with pytest.raises(ValueError, match="unknown architecture"):
        build_model("reex-v3", **kwargs)


def test_legacy_checkpoint_config_defaults_to_reference():
    config = dict(vocab_size=128, d_model=32, n_layers=2, n_heads=2, max_seq_len=16)
    assert isinstance(model_from_config(config), ReferenceLM)
    assert isinstance(model_from_config({**config, "architecture": "reex-v2"}), ReexLM)


def test_reex_config_declares_its_architecture():
    model = ReexLM(128, 32, 2, 2, 16)
    assert model.config["architecture"] == "reex-v2"


def test_muon_partition_covers_reex_hidden_matrices():
    from src.optim.muon import partition_named_parameters
    partition = partition_named_parameters(ReexLM(128, 32, 2, 2, 16).named_parameters())
    assert any("attn.qkv" in name for name in partition.muon)
    assert any("mlp.gate" in name for name in partition.muon)
    assert all("embedding" not in name and "lm_head" not in name for name in partition.muon)
    assert all("norm" not in name for name in partition.muon)
