"""Runtime systems policy and truthful capability reporting.

The systems lever buys throughput without changing the mathematics: same
objective, same data order, same steps -- fewer seconds and dollars per step.
It is therefore the one lever whose gain is nearly certain, and the only one
that must prove *equivalence* rather than improvement.

Precision is resolved against the hardware actually present. This matters more
than it looks: Kaggle's T4 is Turing, which has fp16 tensor cores but no bf16.
A policy that asks for bf16 and silently falls back to fp32 there gets no
speedup at all while reporting that it tried -- which is how a run pays full
fp32 prices on hardware that could have gone 2-3x faster.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SystemsPolicy:
    precision: str = "bf16"
    compile: bool = False
    fp8: bool = False
    device: str = "auto"
    tf32: bool = True
    fused_optimizer: bool = True


def inspect_runtime(policy: SystemsPolicy) -> dict[str, Any]:
    """Return active capabilities; requested features are not assumed active."""
    report: dict[str, Any] = {"requested": asdict(policy), "active": {}, "warnings": []}
    try:
        import torch
    except ModuleNotFoundError:
        report["warnings"].append("PyTorch unavailable")
        report["active"] = {"torch": False, "compile": False, "fp8": False, "precision": "unavailable"}
        return report
    has_cuda = torch.cuda.is_available()
    dtype = policy.precision.lower()
    precision_active = dtype
    if dtype == "bf16" and not (has_cuda and torch.cuda.is_bf16_supported()):
        precision_active = "fp32"
        report["warnings"].append("bf16 requested but unsupported; using fp32")
    if dtype == "fp8":
        fp8_available = hasattr(torch, "float8_e4m3fn") and has_cuda
        if not fp8_available:
            report["warnings"].append("fp8 requested but unavailable; using bf16/fp32")
            precision_active = "bf16" if has_cuda else "fp32"
        report["active"]["fp8"] = fp8_available
    else:
        report["active"]["fp8"] = False
    compile_active = bool(policy.compile and hasattr(torch, "compile"))
    if policy.compile and not compile_active:
        report["warnings"].append("torch.compile requested but unavailable")
    report["active"].update({"torch": True, "cuda": has_cuda, "compile": compile_active,
                             "precision": precision_active})
    return report


@dataclass(frozen=True)
class ResolvedPrecision:
    """What the runtime will actually do, after checking the hardware."""
    autocast_dtype: str | None   # "bfloat16" | "float16" | None (fp32)
    use_grad_scaler: bool
    tf32: bool
    requested: str
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "notes": list(self.notes)}


def resolve_precision(policy: SystemsPolicy, device: str = "cuda") -> ResolvedPrecision:
    """Pick the fastest numerically-sound precision this device supports.

    ``auto`` is the recommended setting: bf16 where supported (no loss scaling
    needed), fp16 with a gradient scaler on older tensor-core GPUs, fp32 on
    CPU. Asking for a specific precision that the device lacks degrades with a
    recorded note rather than silently.
    """
    notes: list[str] = []
    requested = policy.precision.lower()
    # Validated before the device branch so a typo fails everywhere, not only
    # on the machine that happens to have a GPU.
    if requested not in ("auto", "bf16", "fp16", "fp32", "fp8"):
        raise ValueError(f"unknown precision {policy.precision!r}")
    try:
        import torch
    except ModuleNotFoundError:
        return ResolvedPrecision(None, False, False, requested, ("PyTorch unavailable",))

    if device == "mps" and torch.backends.mps.is_available():
        # Measured on an Apple M2 (scripts/verify_speedup.py): fp16 autocast is
        # ~1.15x faster than fp32, while bf16 gives no gain -- Apple silicon has
        # no bf16 acceleration path, so the CUDA habit of preferring bf16 is
        # actively wrong here.
        #
        # The scaler is kept even though dropping it measures ~4% faster:
        # without it, 0.38% of gradient entries underflow to exactly zero,
        # against 0.0019% with it -- 198x more silent gradient loss for 4%
        # wall clock. That is the trade loss scaling exists to refuse.
        if requested in ("auto", "fp16", "bf16", "fp8"):
            if requested == "bf16":
                notes.append("bf16 measured slower than fp32 on MPS (0.97x); "
                             "using fp16, which measured 1.34x")
            return ResolvedPrecision("float16", True, False, requested, tuple(notes))
        return ResolvedPrecision(None, False, False, requested, tuple(notes))

    if device == "cuda" and torch.cuda.is_available():
        # Do NOT trust torch.cuda.is_bf16_supported(): recent versions count
        # *emulated* bf16, so a Tesla T4 (Turing, SM 7.5) reports True. Emulated
        # bf16 runs without tensor cores and is slower than fp32 -- measured
        # 0.73-0.77x on a T4 across three tensor sizes, which is how this was
        # found. Native bf16 starts at Ampere (SM 8.0); below that, fp16 is the
        # fast path because Turing and Volta do have fp16 tensor cores.
        major, _minor = torch.cuda.get_device_capability()
        bf16_ok = major >= 8
        fp16_ok = major >= 7  # tensor cores from Volta onward
        if not bf16_ok and torch.cuda.is_bf16_supported():
            notes.append(
                f"torch reports bf16 supported on compute capability {major}.{_minor}, "
                "but that is emulation without tensor cores and measures slower than "
                "fp32; treating bf16 as unavailable")
    else:
        if requested not in ("fp32", "auto"):
            notes.append(f"{requested} requested but device is {device}; using fp32")
        return ResolvedPrecision(None, False, False, requested, tuple(notes))

    if requested in ("auto", "bf16"):
        if bf16_ok:
            chosen, scaler = "bfloat16", False
        elif requested == "bf16":
            chosen, scaler = "float16", True
            notes.append("bf16 unsupported on this GPU (pre-Ampere); using fp16 + GradScaler, "
                         "which is the fast path here -- fp32 would forfeit the tensor cores")
        else:
            chosen, scaler = "float16", True
            notes.append("auto: no native bf16 (pre-Ampere); selected fp16 + GradScaler")
        if chosen == "float16" and not fp16_ok:
            notes.append(
                "this GPU predates Volta and has no tensor cores at all; expect "
                "little or no speedup from mixed precision")
    elif requested == "fp16":
        chosen, scaler = "float16", fp16_ok
    elif requested == "fp32":
        chosen, scaler = None, False
    elif requested == "fp8":
        chosen, scaler = ("bfloat16", False) if bf16_ok else ("float16", True)
        notes.append("fp8 training is not implemented; using the best supported mixed precision")
    else:
        raise ValueError(f"unknown precision {policy.precision!r}")

    tf32 = bool(policy.tf32 and chosen is None)
    if policy.tf32 and chosen is not None:
        # Under autocast, matmuls already run in half precision; TF32 governs
        # the remaining fp32 matmuls and is harmless to leave enabled.
        tf32 = True
    return ResolvedPrecision(chosen, scaler, tf32, requested, tuple(notes))


def make_grad_scaler(device: str, *, enabled: bool):
    """Build a GradScaler across PyTorch versions.

    The device-generic ``torch.amp.GradScaler(device, ...)`` only exists from
    torch 2.4. Kaggle and Colab images lag, and a crash here would waste a
    whole GPU session on an API detail, so fall back to the older
    ``torch.cuda.amp.GradScaler``. A disabled scaler is a no-op passthrough, so
    the fallback is safe even when the device is not CUDA.
    """
    import torch

    try:
        return torch.amp.GradScaler(device, enabled=enabled)
    except (AttributeError, TypeError):
        if device == "cuda":
            return torch.cuda.amp.GradScaler(enabled=enabled)
        if enabled:
            raise RuntimeError(
                f"this PyTorch ({torch.__version__}) has no GradScaler for device "
                f"{device!r}; use --precision fp32 or upgrade torch"
            )
        return torch.cuda.amp.GradScaler(enabled=False)


def apply_backend_flags(resolved: ResolvedPrecision) -> None:
    """Enable TF32 matmul paths when the resolved policy asks for them."""
    try:
        import torch
    except ModuleNotFoundError:
        return
    if resolved.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def maybe_compile(model, policy: SystemsPolicy):
    """Compile only when explicitly requested and confirmed by the runtime."""
    report = inspect_runtime(policy)
    if report["active"].get("compile"):
        import torch
        return torch.compile(model), report
    return model, report
