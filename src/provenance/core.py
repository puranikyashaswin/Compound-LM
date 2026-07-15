"""Canonical serialization and provenance helpers."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal runtimes
    yaml = None


def _minimal_yaml(text: str) -> dict[str, Any]:
    """Parse the small, scalar/mapping YAML subset used by smoke configs.

    PyYAML remains the supported production dependency; this fallback keeps the
    audit spine runnable in a bare Python environment.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        if raw.strip().startswith("- "):
            while stack[-1][0] >= indent:
                stack.pop()
            if not isinstance(stack[-1][1], list):
                raise ValueError("minimal YAML parser encountered an invalid list")
            item = raw.strip()[2:].strip().strip("'\"")
            stack[-1][1].append(item)
            continue
        key, _, value = raw.strip().partition(":")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        value = value.strip()
        if not value:
            child: Any = [] if key == "tasks" else {}
            parent[key] = child
            stack.append((indent, child))
        elif value in ("[]", "{}"):
            parent[key] = [] if value == "[]" else {}
        elif value.lower() in ("true", "false"):
            parent[key] = value.lower() == "true"
        elif value.lower() in ("null", "none"):
            parent[key] = None
        else:
            try:
                parent[key] = float(value) if "." in value else int(value)
            except ValueError:
                parent[key] = value.strip("'\"")
    return root


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    value = yaml.safe_load(text) if yaml else _minimal_yaml(text)
    if not isinstance(value, dict):
        raise ValueError("configuration must be a YAML mapping")
    if value.get("schema_version") != 1:
        raise ValueError("unsupported or missing schema_version")
    return value


def config_hash(config: dict[str, Any]) -> str:
    return sha256_json(config)


def git_commit(cwd: str | Path = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def environment_snapshot() -> dict[str, str]:
    return {"python": platform.python_version(), "platform": platform.platform()}


def make_manifest(config: dict[str, Any], run_id: str, *, cwd: str | Path = ".") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "config_hash": config_hash(config),
        "git_commit": git_commit(cwd),
        "environment": environment_snapshot(),
        "model_impl_hash": sha256_json(config.get("model", {})),
        "tokenizer_hash": sha256_json({"tokenizer": config.get("model", {}).get("tokenizer")}),
        "data_manifest_hashes": [sha256_json(config.get("data", {}))],
        "evaluation_contract_hash": sha256_json(config.get("evaluation_contract")),
        "seed": config.get("seed"),
    }
