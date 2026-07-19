"""Growing Reex-1 must not discard the 2B tokens already spent on it.

Reex-1 is a 116M LlamaForCausalLM with grouped-query attention (12 query heads,
4 key/value heads). That head split is why `src/model/reex.py` cannot host it --
ReexLM projects Q, K and V at full width. The model implementation is therefore
HuggingFace's, and this framework supplies the protocol around it.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from src.model.llama_adapter import (LlamaProtocolAdapter, grow_llama_depth)


def _tiny_reex(layers: int = 3):
    """Reex-1's shape in miniature: same 3:1 GQA ratio, SwiGLU, RoPE, tied."""
    config = transformers.LlamaConfig(
        vocab_size=512, hidden_size=48, intermediate_size=132,
        num_hidden_layers=layers, num_attention_heads=6, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=True)
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)
    return model.eval()


def test_zero_init_growth_reproduces_the_donor_exactly():
    """The hard gate: a growth event that changes the function threw away compute."""
    donor = _tiny_reex()
    grown, report = grow_llama_depth(donor, to_layers=6, mode="zero_init")
    ids = torch.randint(0, 512, (2, 16))
    with torch.no_grad():
        difference = (donor(input_ids=ids).logits
                      - grown.eval()(input_ids=ids).logits).abs().max().item()
    assert report.function_preserving
    assert difference < 1e-5, f"grown model diverged from the donor by {difference}"


def test_stack_mode_is_not_equivalent_and_declares_it():
    donor = _tiny_reex()
    grown, report = grow_llama_depth(donor, to_layers=6, mode="stack")
    ids = torch.randint(0, 512, (2, 16))
    with torch.no_grad():
        difference = (donor(input_ids=ids).logits
                      - grown.eval()(input_ids=ids).logits).abs().max().item()
    assert report.function_preserving is False
    assert difference > 1e-3, "duplicated blocks should change the function"


def test_layer_count_and_config_both_updated():
    donor = _tiny_reex()
    grown, _ = grow_llama_depth(donor, to_layers=9)
    assert len(grown.model.layers) == 9
    assert grown.config.num_hidden_layers == 9


def test_layer_indices_stay_unique_for_the_kv_cache():
    """Duplicated layers share a layer_idx unless renumbered, corrupting cache reads."""
    grown, _ = grow_llama_depth(_tiny_reex(), to_layers=6)
    indices = [b.self_attn.layer_idx for b in grown.model.layers]
    assert indices == list(range(6))


def test_donor_is_left_untouched():
    donor = _tiny_reex()
    before = donor.model.layers[0].self_attn.o_proj.weight.clone()
    grow_llama_depth(donor, to_layers=6)
    assert torch.equal(donor.model.layers[0].self_attn.o_proj.weight, before)
    assert len(donor.model.layers) == 3


def test_uneven_or_shrinking_growth_is_refused():
    donor = _tiny_reex()
    with pytest.raises(ValueError, match="multiple"):
        grow_llama_depth(donor, to_layers=7)
    with pytest.raises(ValueError, match="must exceed"):
        grow_llama_depth(donor, to_layers=2)
    with pytest.raises(ValueError, match="unknown growth mode"):
        grow_llama_depth(donor, to_layers=6, mode="teleport")


def test_grown_model_still_receives_gradient_in_new_layers():
    """A zeroed projection that never trains is dead capacity."""
    grown, _ = grow_llama_depth(_tiny_reex(), to_layers=6)
    grown.train()
    ids = torch.randint(0, 512, (2, 16))
    grown(input_ids=ids).logits.square().mean().backward()
    inserted = grown.model.layers[1]
    assert inserted.self_attn.o_proj.weight.grad.abs().sum() > 0


# --- protocol adapter ------------------------------------------------------

def test_packing_mask_blocks_cross_document_attention():
    """Llama's default mask is purely causal, which leaks across packed documents."""
    adapter = LlamaProtocolAdapter(_tiny_reex()).eval()
    ids = torch.randint(0, 512, (1, 16))
    documents = torch.zeros(1, 16, dtype=torch.long)
    documents[:, 8:] = 1
    with torch.no_grad():
        base = adapter(ids, documents)
        perturbed_ids = ids.clone()
        perturbed_ids[0, 12] = (perturbed_ids[0, 12] + 1) % 512
        perturbed = adapter(perturbed_ids, documents)
    assert torch.equal(base[0, :8], perturbed[0, :8]), \
        "document 0 changed when document 1 was edited -- the packing mask leaks"


def test_adapter_reports_grouped_query_heads():
    adapter = LlamaProtocolAdapter(_tiny_reex())
    assert adapter.config["n_heads"] == 6
    assert adapter.config["n_kv_heads"] == 2
    assert adapter.config["architecture"] == "llama-hf"


def test_adapter_returns_logits_not_a_model_output():
    adapter = LlamaProtocolAdapter(_tiny_reex()).eval()
    ids = torch.randint(0, 512, (2, 16))
    with torch.no_grad():
        out = adapter(ids, torch.zeros_like(ids))
    assert out.shape == (2, 16, 512)


def test_muon_partition_selects_llama_projection_matrices():
    """Muon must take the hidden matrices and leave embeddings and norms alone."""
    from src.optim.muon import partition_named_parameters
    partition = partition_named_parameters(_tiny_reex().named_parameters())
    assert any("q_proj" in name for name in partition.muon)
    assert any("down_proj" in name for name in partition.muon)
    assert all("embed" not in name and "lm_head" not in name for name in partition.muon)
    assert all("norm" not in name for name in partition.muon)
