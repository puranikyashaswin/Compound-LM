"""Run the frozen E-v1 evaluation contract.

The runner deliberately refuses to fabricate benchmark scores. In a production
environment it delegates to a pinned lm-eval-harness installation; without that
dependency it can still validate the contract and emit a deterministic report
with status ``unavailable``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.provenance.core import canonical_json, sha256_json


CONTRACT = Path(__file__).resolve().parents[1] / "contracts" / "eval_v1.yaml"


def _load_contract(path: str | Path = CONTRACT) -> dict[str, Any]:
    from src.provenance.core import load_config
    value = load_config(path)
    if value.get("contract_id") != "E-v1":
        raise ValueError("evaluation contract must be E-v1")
    return value


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    elif path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(child.relative_to(path)).encode())
            with child.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    else:
        raise FileNotFoundError(path)
    return digest.hexdigest()


def _harness_version() -> str | None:
    try:
        result = subprocess.run(
            ["lm_eval", "--version"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            return result.stdout.strip() or result.stderr.strip()
    except OSError:
        pass
    return None


def run(checkpoint: str, output: str, *, contract_path: str | Path = CONTRACT,
        model_args: str = "", device: str = "auto") -> dict[str, Any]:
    contract = _load_contract(contract_path)
    checkpoint_path = Path(checkpoint)
    report: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "contract_hash": sha256_json(contract),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _checkpoint_hash(checkpoint_path),
        "settings": contract["settings"],
        "tasks": contract["tasks"],
        "harness_version": _harness_version(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "status": "unavailable",
        "scores": {},
        "aggregate": None,
        "warnings": [],
    }

    if report["harness_version"] is None:
        report["warnings"].append(
            "lm_eval executable is unavailable; no benchmark scores were fabricated"
        )
    else:
        # Invoke only after the contract and checkpoint have been hashed. The
        # exact command is recorded so reports remain auditable.
        command = [
            "lm_eval", "--model", "hf", "--model_args",
            model_args or f"pretrained={checkpoint_path}",
            "--tasks", ",".join(contract["tasks"]),
            "--num_fewshot", str(contract["settings"]["num_fewshot"]),
            "--batch_size", str(contract["settings"]["batch_size"]),
            "--device", device,
            "--output_path", str(Path(output).with_suffix(".lm_eval")),
        ]
        report["command"] = command
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        report["returncode"] = completed.returncode
        report["stdout_sha256"] = hashlib.sha256(completed.stdout.encode()).hexdigest()
        report["stderr_sha256"] = hashlib.sha256(completed.stderr.encode()).hexdigest()
        if completed.returncode == 0:
            report["status"] = "completed"
        else:
            report["status"] = "failed"
            report["warnings"].append("lm_eval returned a non-zero exit code")

    report["report_hash"] = sha256_json({k: v for k, v in report.items() if k != "report_hash"})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--contract", default=str(CONTRACT))
    parser.add_argument("--model-args", default="")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = run(args.ckpt, args.out, contract_path=args.contract,
                 model_args=args.model_args, device=args.device)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
