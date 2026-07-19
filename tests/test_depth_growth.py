"""Depth growth must not throw away the capability already paid for.

The build plan makes logit equivalence a hard pre-training gate for growth:
a growth event that changes the model's function has silently discarded the
compute spent reaching that function.
"""
import pytest
import torch

from src.growth.depth import grow_depth, growth_savings
from src.growth.hyperclone import assert_logit_equivalence
from src.model.reex import ReexLM
from src.model.reference import ReferenceLM


def _trained_ish(model):
    """Move off initialization so a zeroed projection is a real constraint."""
    torch.manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)
    return model.eval()


@pytest.mark.parametrize("factory", [
    lambda: ReferenceLM(128, 32, 3, 2, 16),
    lambda: ReexLM(128, 32, 3, 2, 16),
])
def test_zero_init_growth_is_function_preserving(factory):
    donor = _trained_ish(factory())
    grown, report = grow_depth(donor, to_layers=6, mode="zero_init")
    ids = torch.randint(0, 128, (2, 16))
    assert report.function_preserving
    assert len(grown.blocks) == 6
    assert_logit_equivalence(donor, grown.eval(), ids, tolerance=1e-5)


def test_stack_mode_is_not_function_preserving_and_says_so():
    donor = _trained_ish(ReferenceLM(128, 32, 3, 2, 16))
    grown, report = grow_depth(donor, to_layers=6, mode="stack")
    assert report.function_preserving is False
    ids = torch.randint(0, 128, (2, 16))
    with pytest.raises(ValueError, match="growth equivalence failed"):
        assert_logit_equivalence(donor, grown.eval(), ids, tolerance=1e-5)


def test_grown_model_still_trains():
    """A zeroed projection must receive gradient, or the new blocks are dead."""
    donor = _trained_ish(ReferenceLM(128, 32, 2, 2, 16))
    grown, _ = grow_depth(donor, to_layers=4, mode="zero_init")
    grown.train()
    ids = torch.randint(0, 128, (2, 16))
    logits = grown(ids, torch.zeros_like(ids))
    logits.square().mean().backward()
    inserted = grown.blocks[1]
    grads = [p.grad for n, p in inserted.named_parameters() if n.endswith("attn.out.weight")]
    assert grads and grads[0] is not None
    assert grads[0].abs().sum() > 0, "zero-initialized block received no gradient"


def test_config_records_the_new_depth():
    donor = ReferenceLM(128, 32, 2, 2, 16)
    grown, _ = grow_depth(donor, to_layers=4)
    assert grown.config["n_layers"] == 4


def test_uneven_or_shrinking_growth_is_refused():
    donor = ReferenceLM(128, 32, 3, 2, 16)
    with pytest.raises(ValueError, match="multiple"):
        grow_depth(donor, to_layers=4)
    with pytest.raises(ValueError, match="must exceed"):
        grow_depth(donor, to_layers=2)
    with pytest.raises(ValueError, match="unknown growth mode"):
        grow_depth(donor, to_layers=6, mode="teleport")


def test_donor_is_left_untouched():
    donor = _trained_ish(ReferenceLM(128, 32, 2, 2, 16))
    before = donor.blocks[0]["attn"].out.weight.clone()
    grow_depth(donor, to_layers=4)
    assert torch.equal(donor.blocks[0]["attn"].out.weight, before)
    assert len(donor.blocks) == 2


def test_savings_account_for_the_output_head():
    """Ignoring the head's share is how this lever gets overstated."""
    honest = growth_savings(from_layers=6, to_layers=12, growth_fraction=0.5,
                            transformer_flop_share=0.37)
    inflated = growth_savings(from_layers=6, to_layers=12, growth_fraction=0.5,
                              transformer_flop_share=1.0)
    assert honest < inflated
    assert 1.0 < honest < 1.15


def test_savings_improve_once_the_vocabulary_is_right_sized():
    """The interaction: a smaller head makes depth growth worth more."""
    big_vocab = growth_savings(from_layers=6, to_layers=12, growth_fraction=0.5,
                               transformer_flop_share=0.37)
    small_vocab = growth_savings(from_layers=6, to_layers=12, growth_fraction=0.5,
                                 transformer_flop_share=0.56)
    assert small_vocab > big_vocab


def test_no_growth_fraction_means_no_saving():
    assert growth_savings(from_layers=6, to_layers=12, growth_fraction=0.0,
                          transformer_flop_share=0.5) == pytest.approx(1.0)
