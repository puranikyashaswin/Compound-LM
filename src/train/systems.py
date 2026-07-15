"""Runtime systems policy and truthful capability reporting."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SystemsPolicy:
    precision: str = "bf16"
    compile: bool = False
    fp8: bool = False
    device: str = "auto"


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


def maybe_compile(model, policy: SystemsPolicy):
    """Compile only when explicitly requested and confirmed by the runtime."""
    report = inspect_runtime(policy)
    if report["active"].get("compile"):
        import torch
        return torch.compile(model), report
    return model, report
