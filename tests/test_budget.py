"""The gate that would have caught 270 epochs before 3.6 GPU-hours were spent."""
import pytest

from src.data.budget import check_token_budget, tokens_needed


def test_it_refuses_the_actual_kaggle_run_that_wasted_gpu_hours():
    """22.4M params, 13750x64x512 positions, over a 1.67M-token corpus."""
    report = check_token_budget(unique_tokens=1_671_566, steps=13_750, batch_size=64,
                                sequence_length=512, n_params=22_400_000)
    assert report.status == "red"
    assert report.epochs == pytest.approx(269.5, abs=0.5)
    assert any("corpus_repeated" in f for f in report.failures)
    assert any("corpus_too_small_for_model" in f for f in report.failures)


def test_a_well_sized_run_passes_clean():
    # 22.4M params at ~20 tokens/param, seen once.
    report = check_token_budget(unique_tokens=448_000_000, steps=13_672, batch_size=64,
                                sequence_length=512, n_params=22_400_000)
    assert report.status == "green"
    assert report.epochs == pytest.approx(1.0, abs=0.05)
    assert report.chinchilla_ratio == pytest.approx(1.0, abs=0.05)
    assert report.failures == []


def test_mild_repetition_warns_but_does_not_block():
    report = check_token_budget(unique_tokens=100_000_000, steps=6_000, batch_size=64,
                                sequence_length=512, n_params=5_000_000)
    assert report.status == "amber"
    assert report.failures == []
    assert any("corpus_repeated" in w for w in report.warnings)


def test_undertrained_model_warns_without_blocking():
    """4.5 tokens/param: legal (corpus seen once) but well under Chinchilla."""
    report = check_token_budget(unique_tokens=100_000_000, steps=3_000, batch_size=64,
                                sequence_length=512, n_params=22_400_000)
    assert report.status == "amber"
    assert report.failures == []
    assert any("under_chinchilla" in w for w in report.warnings)


def test_corpus_smaller_than_the_model_is_a_hard_failure():
    """0.45 tokens/param cannot train a 22.4M model regardless of epochs."""
    report = check_token_budget(unique_tokens=10_000_000, steps=300, batch_size=64,
                                sequence_length=512, n_params=22_400_000)
    assert report.status == "red"
    assert any("corpus_too_small_for_model" in f for f in report.failures)


def test_epoch_limit_is_configurable():
    kwargs = dict(unique_tokens=1_000_000, steps=1_000, batch_size=8,
                  sequence_length=512, n_params=200_000)
    assert check_token_budget(**kwargs, max_epochs=4.0).status == "red"
    assert check_token_budget(**kwargs, max_epochs=8.0).status != "red"


def test_tokens_needed_matches_chinchilla_reference():
    assert tokens_needed(n_params=22_400_000) == 448_000_000
    assert tokens_needed(n_params=100_000_000, tokens_per_param=20) == 2_000_000_000


def test_nonsense_inputs_are_refused():
    with pytest.raises(ValueError):
        check_token_budget(unique_tokens=0, steps=1, batch_size=1,
                           sequence_length=1, n_params=1)
    with pytest.raises(ValueError):
        check_token_budget(unique_tokens=1, steps=0, batch_size=1,
                           sequence_length=1, n_params=1)
