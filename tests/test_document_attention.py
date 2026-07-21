"""Document-aware SDPA must keep packing safety and prefer the causal path."""
import torch

from src.model.document_attention import (is_single_document_batch,
                                          scaled_dot_product_attention_documented,
                                          set_assume_single_document)
from src.model.reex import ReexLM


def test_causal_path_matches_masked_path_on_single_document():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, 8)
    k = torch.randn(2, 4, 8, 8)
    v = torch.randn(2, 4, 8, 8)
    docs = torch.zeros(2, 8, dtype=torch.long)
    flash = scaled_dot_product_attention_documented(q, k, v, docs, assume_single_document=True)
    checked = scaled_dot_product_attention_documented(q, k, v, docs, assume_single_document=False)
    assert torch.allclose(flash, checked, atol=1e-5)


def test_reex_still_blocks_cross_document_after_fast_path():
    torch.manual_seed(0)
    model = ReexLM(256, 32, 2, 2, 16).eval()
    ids = torch.randint(0, 256, (1, 16))
    docs = torch.zeros(1, 16, dtype=torch.long)
    docs[:, 8:] = 1
    with torch.no_grad():
        base = model(ids, docs)
        perturbed = ids.clone()
        perturbed[0, 12] = (ids[0, 12] + 1) % 256
        out = model(perturbed, docs)
    assert torch.allclose(base[0, :8], out[0, :8], atol=1e-5)


def test_set_assume_single_document_flags_attention_modules():
    model = ReexLM(128, 32, 2, 2, 16)
    set_assume_single_document(model, True)
    assert all(block["attn"].assume_single_document for block in model.blocks)
    assert is_single_document_batch(torch.tensor([[1, 1, -1]]))
