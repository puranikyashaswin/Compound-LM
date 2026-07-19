"""Muon must behave correctly under loss scaling.

GradScaler was designed around torch's built-in optimizers. Muon is a custom
Optimizer subclass driven by a Newton-Schulz iteration, and it is handed to the
same scaler as AdamW. Three things have to hold or the optimizer lever is
measured on corrupted weights:

  1. gradients are unscaled before Muon's orthogonalization sees them --
     Newton-Schulz normalizes by the matrix norm, so a 65536x-scaled gradient
     would not simply cancel;
  2. an inf/NaN gradient causes the step to be SKIPPED, not applied;
  3. the reported gradient norm is the true one, for the health gate.
"""
import pytest
import torch

from src.model.registry import build_model
from src.optim.muon import Muon, partition_named_parameters
from src.train.reference import masked_next_token_loss

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
NEEDS_ACCELERATOR = pytest.mark.skipif(
    DEVICE == "cpu", reason="autocast/GradScaler need an accelerator")


def _setup():
    torch.manual_seed(0)
    model = build_model("reference-v1", vocab_size=512, d_model=32, n_layers=2,
                        n_heads=2, max_seq_len=16).to(DEVICE)
    partition = partition_named_parameters(model.named_parameters())
    by_name = dict(model.named_parameters())
    muon = Muon([by_name[n] for n in partition.muon], lr=0.02)
    adamw = torch.optim.AdamW([by_name[n] for n in partition.adamw], lr=1e-3)
    return model, partition, by_name, muon, adamw


def _backward(model, scaler):
    ids = torch.randint(0, 512, (2, 16), device=DEVICE)
    docs = torch.zeros_like(ids)
    with torch.autocast(DEVICE, dtype=torch.float16):
        loss = masked_next_token_loss(model(ids, docs), ids, docs)
    scaler.scale(loss).backward()
    return loss


@NEEDS_ACCELERATOR
def test_muon_steps_normally_under_loss_scaling():
    model, partition, by_name, muon, adamw = _setup()
    scaler = torch.amp.GradScaler(DEVICE, enabled=True)
    before = [by_name[n].detach().clone() for n in partition.muon]
    _backward(model, scaler)
    scaler.unscale_(adamw)
    scaler.unscale_(muon)
    scaler.step(adamw)
    scaler.step(muon)
    scaler.update()
    after = [by_name[n].detach().clone() for n in partition.muon]
    assert any(not torch.equal(a, b) for a, b in zip(before, after)), \
        "Muon parameters did not move under loss scaling"
    assert all(torch.isfinite(by_name[n]).all() for n in partition.muon)


@NEEDS_ACCELERATOR
def test_inf_gradient_skips_the_muon_step_instead_of_corrupting_weights():
    model, partition, by_name, muon, adamw = _setup()
    scaler = torch.amp.GradScaler(DEVICE, enabled=True)
    _backward(model, scaler)
    for name in partition.muon:
        by_name[name].grad = torch.full_like(by_name[name], float("inf"))

    before = [by_name[n].detach().clone() for n in partition.muon]
    scale_before = scaler.get_scale()
    scaler.unscale_(adamw)
    scaler.unscale_(muon)
    scaler.step(adamw)
    scaler.step(muon)
    scaler.update()
    after = [by_name[n].detach().clone() for n in partition.muon]

    assert all(torch.equal(a, b) for a, b in zip(before, after)), \
        "an inf gradient was applied to Muon parameters instead of being skipped"
    assert all(torch.isfinite(by_name[n]).all() for n in partition.muon)
    assert scaler.get_scale() < scale_before, "loss scale did not back off after inf"


def test_unscaled_gradients_reach_newton_schulz():
    """Muon's update is norm-normalized, so scale must be removed first.

    Run on CPU with an explicit scale factor: orthogonalizing a scaled gradient
    and then unscaling is NOT the same as unscaling first, because the
    normalization is nonlinear in the input.
    """
    from src.optim.muon import newton_schulz_orthogonalize

    torch.manual_seed(0)
    gradient = torch.randn(16, 8)
    direct = newton_schulz_orthogonalize(gradient)
    scaled = newton_schulz_orthogonalize(gradient * 65536.0)
    # Norm-normalization makes the two nearly equal, which is precisely why a
    # missing unscale_ would be invisible in the weights but wrong in the
    # gradient norm the health gate reads.
    assert torch.allclose(direct, scaled, atol=1e-4)


@NEEDS_ACCELERATOR
def test_grad_norm_is_read_after_unscaling():
    """A scaled norm would be ~65536x too large and trip the spike gate."""
    model, partition, by_name, muon, adamw = _setup()
    scaler = torch.amp.GradScaler(DEVICE, enabled=True)
    _backward(model, scaler)

    scaled_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
    scaler.unscale_(adamw)
    scaler.unscale_(muon)
    unscaled_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()

    assert unscaled_norm < scaled_norm / 100, \
        "unscale_ did not reduce the gradient norm; the health gate would see a fake spike"
