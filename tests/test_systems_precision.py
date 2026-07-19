"""The systems lever must be fast *and* honest about what it actually did.

Two failure modes this guards. First, a policy that asks for bf16 on hardware
without it (Kaggle's T4) silently running fp32 -- paying full price on a GPU
that had a 2-3x faster path available. Second, a resumed run switching
precision mid-curve, which continues a curve it did not produce.
"""
import pytest
import torch

from src.train.systems import (SystemsPolicy, apply_backend_flags, inspect_runtime,
                               resolve_precision)


def test_cpu_resolves_to_fp32_with_no_scaler():
    resolved = resolve_precision(SystemsPolicy(precision="auto"), device="cpu")
    assert resolved.autocast_dtype is None
    assert resolved.use_grad_scaler is False


def test_requesting_half_precision_on_cpu_is_recorded_not_silent():
    resolved = resolve_precision(SystemsPolicy(precision="bf16"), device="cpu")
    assert resolved.autocast_dtype is None
    assert resolved.notes, "degrading to fp32 must leave a recorded note"


def test_explicit_fp32_never_autocasts():
    resolved = resolve_precision(SystemsPolicy(precision="fp32"), device="cpu")
    assert resolved.autocast_dtype is None


def test_unknown_precision_is_rejected_on_every_device():
    with pytest.raises(ValueError, match="unknown precision"):
        resolve_precision(SystemsPolicy(precision="int4"), device="cpu")


def test_backend_flags_are_safe_without_cuda():
    apply_backend_flags(resolve_precision(SystemsPolicy(), device="cpu"))


def test_resolved_precision_serializes_for_the_ledger():
    payload = resolve_precision(SystemsPolicy(precision="auto"), device="cpu").as_dict()
    assert set(payload) >= {"autocast_dtype", "use_grad_scaler", "tf32", "requested", "notes"}
    assert isinstance(payload["notes"], list)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_gpu_auto_never_falls_back_to_fp32():
    """On any tensor-core GPU, 'auto' must select a half-precision path."""
    resolved = resolve_precision(SystemsPolicy(precision="auto"), device="cuda")
    assert resolved.autocast_dtype in ("bfloat16", "float16")
    assert resolved.use_grad_scaler == (resolved.autocast_dtype == "float16")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_bf16_request_on_pre_ampere_uses_fp16_not_fp32():
    resolved = resolve_precision(SystemsPolicy(precision="bf16"), device="cuda")
    if not torch.cuda.is_bf16_supported():
        assert resolved.autocast_dtype == "float16"
        assert resolved.use_grad_scaler is True
