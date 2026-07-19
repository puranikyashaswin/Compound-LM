"""Verify every claim this repo makes, and label the ones it cannot verify.

The point is not to print "all green". It is to separate three things that are
easy to blur together:

  VERIFIED   executed here, on this machine, this run
  PORTABLE   measured here on a real accelerator; expected to transfer
  PROJECTED  arithmetic only -- no measurement backs it on any hardware

A plan built on PROJECTED numbers can still be right, but it must never be
reported with the same confidence as a VERIFIED one. The 2.0x mixed-precision
figure in the original plan was PROJECTED; measuring it on an M2 returned
1.15x, and bf16 -- which the plan assumed was the fast path -- was a
*regression*. That is the failure mode this script exists to catch.

Run with --quick to skip the training-based checks.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VERIFIED, PORTABLE, PROJECTED, FAILED = "VERIFIED", "PORTABLE", "PROJECTED", "FAILED"


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, name: str, status: str, detail: str) -> None:
        self.rows.append((name, status, detail))
        marker = {VERIFIED: "OK ", PORTABLE: "OK ", PROJECTED: "-- ", FAILED: "!! "}[status]
        print(f"  {marker}[{status:<9}] {name}: {detail}")

    @property
    def failed(self) -> list[tuple[str, str, str]]:
        return [row for row in self.rows if row[1] == FAILED]


def detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_tests(checks: Checks) -> None:
    print("\n[1] Test suite")
    started = time.perf_counter()
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header"],
                            cwd=ROOT, capture_output=True, text=True,
                            env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    tail = [line for line in result.stdout.strip().splitlines() if line.strip()][-1:]
    summary = tail[0] if tail else "no output"
    status = VERIFIED if result.returncode == 0 else FAILED
    checks.add("unit + integration tests", status,
               f"{summary} ({time.perf_counter() - started:.0f}s)")


def check_model_contracts(checks: Checks) -> None:
    print("\n[2] Model and lineage contracts")
    import math
    import torch
    from src.model.registry import build_model, model_from_config
    from src.train.reference import masked_next_token_loss

    for architecture in ("reference-v1", "reex-v2"):
        torch.manual_seed(0)
        model = build_model(architecture, vocab_size=50257, d_model=256,
                            n_layers=12, n_heads=8, max_seq_len=128)
        ids = torch.randint(0, 50257, (2, 128))
        loss = masked_next_token_loss(model(ids, torch.zeros_like(ids)), ids,
                                      torch.zeros_like(ids))
        expected = math.log(50257)
        ok = abs(loss.item() - expected) < 0.6
        checks.add(f"{architecture} initial loss == ln(V)",
                   VERIFIED if ok else FAILED,
                   f"{loss.item():.3f} vs ln(V)={expected:.3f}")

    legacy = model_from_config(dict(vocab_size=64, d_model=32, n_layers=2,
                                    n_heads=2, max_seq_len=16))
    checks.add("legacy checkpoint keeps reference-v1 lineage",
               VERIFIED if type(legacy).__name__ == "ReferenceLM" else FAILED,
               type(legacy).__name__)


def check_growth(checks: Checks) -> None:
    print("\n[3] Depth growth equivalence")
    import torch
    from src.growth.depth import grow_depth
    from src.growth.hyperclone import assert_logit_equivalence
    from src.model.registry import build_model

    for architecture in ("reference-v1", "reex-v2"):
        torch.manual_seed(0)
        donor = build_model(architecture, vocab_size=128, d_model=32, n_layers=3,
                            n_heads=2, max_seq_len=16)
        with torch.no_grad():
            for parameter in donor.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.02)
        donor.eval()
        grown, report = grow_depth(donor, to_layers=6, mode="zero_init")
        try:
            equivalence = assert_logit_equivalence(
                donor, grown.eval(), torch.randint(0, 128, (2, 16)), tolerance=1e-5)
            checks.add(f"{architecture} 3->6 layers is function-preserving", VERIFIED,
                       f"max logit diff {equivalence.max_abs_logit_diff:.2e}")
        except ValueError as error:
            checks.add(f"{architecture} 3->6 layers is function-preserving", FAILED, str(error))


def check_eval_batching(checks: Checks) -> None:
    print("\n[4] Evaluation batching is score-neutral")
    from src.eval.intrinsic import evaluate

    checkpoints = sorted((ROOT / "runs" / "baseline-s17").glob("checkpoint-*.pt"))
    heldout = ROOT / "data" / "protocol-v1" / "protocol-heldout-packed.jsonl"
    if not checkpoints or not heldout.exists():
        checks.add("batched eval == unbatched eval", PROJECTED,
                   "no protocol checkpoints on disk; run scripts/run_protocol.py")
        return
    one = evaluate(str(checkpoints[-1]), str(heldout), batch_size=1)
    many = evaluate(str(checkpoints[-1]), str(heldout), batch_size=64)
    same_acc = one["val_acc"] == many["val_acc"]
    same_tokens = one["heldout_tokens"] == many["heldout_tokens"]
    drift = abs(one["val_nll"] - many["val_nll"])
    checks.add("batched eval == unbatched eval",
               VERIFIED if (same_acc and same_tokens and drift < 1e-6) else FAILED,
               f"acc identical={same_acc}, tokens identical={same_tokens}, "
               f"nll drift={drift:.2e}")


def check_precision(checks: Checks, device: str) -> None:
    print("\n[5] Mixed precision on real hardware")
    if device == "cpu":
        checks.add("mixed-precision speedup", PROJECTED,
                   "no accelerator on this machine; ~2x on CUDA tensor cores is "
                   "arithmetic, not measurement")
        return

    report = ROOT / "outputs" / "speedup-report.json"
    if not report.exists():
        checks.add("mixed-precision speedup", PROJECTED,
                   "run scripts/verify_speedup.py to measure it here")
        return
    payload = json.loads(report.read_text())
    entries = {name: data for name, data in payload["results"].items() if name != "fp32"}
    best = max(entries.items(), key=lambda item: item[1].get("speedup", 0))
    worst_divergence = max(data.get("max_relative_loss_divergence", 0)
                           for data in entries.values())
    checks.add(f"mixed-precision speedup on {payload['device']}", PORTABLE,
               f"best {best[1]['speedup']:.2f}x ({best[0]}); "
               f"loss divergence <= {worst_divergence:.2e}")
    checks.add("mixed-precision speedup on CUDA tensor cores", PROJECTED,
               "~2x; no CUDA device available here to confirm")


def check_lever_measurements(checks: Checks) -> None:
    print("\n[6] Cost levers, measured")
    report = ROOT / "outputs" / "lever-measurements.json"
    if not report.exists():
        checks.add("per-lever wall-clock", PROJECTED,
                   "run scripts/verify_levers.py to measure them here")
        return
    payload = json.loads(report.read_text())
    for name, data in payload["levers"].items():
        status = PORTABLE if data["kind"].startswith("FLOP") else PORTABLE
        checks.add(f"lever: {name}", status,
                   f"{data['speedup']:.2f}x on {payload['device']}")


def check_protocol_reproduces(checks: Checks) -> None:
    print("\n[7] Published table matches the evidence package")
    report = ROOT / "outputs" / "protocol-report.json"
    readme = ROOT / "README.md"
    if not report.exists():
        checks.add("README table derives from the ledger", PROJECTED,
                   "no protocol report on disk")
        return
    payload = json.loads(report.read_text())
    rows = payload.get("compounding", {}).get("rows", [])
    text = readme.read_text()
    missing = [row["name"] for row in rows if row["name"] not in text]
    checks.add("README table derives from the ledger",
               VERIFIED if not missing and rows else FAILED,
               f"{len(rows)} runs published, {len(missing)} missing from README")

    gate = payload.get("baseline", {}).get("gate_pass")
    checks.add("two-seed baseline spread gate", VERIFIED if gate else FAILED,
               f"spread={payload.get('baseline', {}).get('seed_spread_acc'):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the test suite")
    args = parser.parse_args()

    import torch
    device = detect_device()
    print("=" * 72)
    print("COMPOUND-LM VERIFICATION")
    print("=" * 72)
    print(f"torch {torch.__version__}   accelerator: {device}")

    checks = Checks()
    if not args.quick:
        check_tests(checks)
    check_model_contracts(checks)
    check_growth(checks)
    check_eval_batching(checks)
    check_precision(checks, device)
    check_lever_measurements(checks)
    check_protocol_reproduces(checks)

    print()
    print("=" * 72)
    counts = {status: sum(1 for _, s, _ in checks.rows if s == status)
              for status in (VERIFIED, PORTABLE, PROJECTED, FAILED)}
    print(f"VERIFIED {counts[VERIFIED]}   PORTABLE {counts[PORTABLE]}   "
          f"PROJECTED {counts[PROJECTED]}   FAILED {counts[FAILED]}")
    print("=" * 72)
    if checks.failed:
        for name, _, detail in checks.failed:
            print(f"  FAILED: {name}: {detail}")
        raise SystemExit(1)
    if counts[PROJECTED]:
        print("\nUnverified claims remain. They are labelled PROJECTED above and")
        print("must not be reported with the confidence of a measurement.")


if __name__ == "__main__":
    main()
