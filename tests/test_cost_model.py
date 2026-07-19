"""The cost breakdown must not flatter a lever it is used to justify."""
import pytest

from src.ledger.cost_model import (analyze_model, vocab_resize_multiplier,
                                   wall_clock_multiplier)

KAGGLE = dict(vocab_size=50257, d_model=256, n_layers=12, sequence_length=512)


def test_head_outweighs_the_transformer_stack_at_this_width():
    """The finding the whole vocab lever rests on."""
    b = analyze_model(**KAGGLE)
    assert b.fwd_flops_per_token_head > b.fwd_flops_per_token_transformer


def test_param_count_matches_the_real_model():
    from src.model.registry import build_model
    b = analyze_model(**KAGGLE)
    actual = sum(p.numel() for p in build_model(
        "reference-v1", vocab_size=50257, d_model=256, n_layers=12, n_heads=8,
        max_seq_len=512).parameters())
    # Analytic count omits biases, norms, and position embeddings.
    assert abs(b.params_total - actual) / actual < 0.02


def test_swiglu_and_gelu_mlps_cost_the_same():
    """SwiGLU's 3 matrices at 8/3 width == GELU's 2 at 4x; the default holds."""
    gelu = analyze_model(**KAGGLE, mlp_ratio=4.0)
    reex = analyze_model(**KAGGLE, mlp_ratio=4.0)
    assert gelu.fwd_flops_per_token == reex.fwd_flops_per_token


def test_vocab_shrink_gain_is_net_of_tokenization_penalty():
    b = analyze_model(**KAGGLE)
    free = vocab_resize_multiplier(baseline=b, new_vocab=16384, d_model=256,
                                   tokenization_penalty=1.0)
    real = vocab_resize_multiplier(baseline=b, new_vocab=16384, d_model=256,
                                   tokenization_penalty=1.10)
    assert real < free, "ignoring the tokenization penalty overstates the lever"
    assert real == pytest.approx(free / 1.10, rel=1e-6)


def test_a_penalty_below_one_is_refused():
    """A smaller vocab compressing *better* would be free FLOPs from nowhere."""
    b = analyze_model(**KAGGLE)
    with pytest.raises(ValueError, match="tokenization_penalty"):
        vocab_resize_multiplier(baseline=b, new_vocab=16384, d_model=256,
                                tokenization_penalty=0.9)


def test_growing_the_vocab_is_reported_as_a_loss():
    b = analyze_model(**KAGGLE)
    assert vocab_resize_multiplier(baseline=b, new_vocab=100000, d_model=256,
                                   tokenization_penalty=1.0) < 1.0


def test_attention_term_scales_with_sequence_length():
    short = analyze_model(**{**KAGGLE, "sequence_length": 512})
    long = analyze_model(**{**KAGGLE, "sequence_length": 4096})
    assert long.fwd_flops_per_token_attention > short.fwd_flops_per_token_attention


def test_expensive_steps_shrink_a_flop_win():
    """Muon: 1.82x fewer FLOPs, but 1.39x costlier steps (measured)."""
    assert wall_clock_multiplier(flop_multiplier=1.82, step_cost_ratio=1.39) == \
        pytest.approx(1.31, abs=0.01)


def test_free_steps_leave_the_multiplier_alone():
    assert wall_clock_multiplier(flop_multiplier=1.5, step_cost_ratio=1.0) == 1.5


def test_a_flop_win_can_be_erased_by_step_cost():
    """A lever that halves FLOPs but doubles step time saves nothing."""
    assert wall_clock_multiplier(flop_multiplier=2.0, step_cost_ratio=2.0) == 1.0


def test_nonpositive_inputs_are_refused():
    with pytest.raises(ValueError):
        wall_clock_multiplier(flop_multiplier=0, step_cost_ratio=1.0)
    with pytest.raises(ValueError):
        wall_clock_multiplier(flop_multiplier=1.0, step_cost_ratio=0)


def test_dimensions_must_be_positive():
    with pytest.raises(ValueError):
        analyze_model(vocab_size=0, d_model=256, n_layers=12, sequence_length=512)
