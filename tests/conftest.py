"""Shared fixtures.

Tests must be hermetic: runnable on a fresh clone with nothing generated. The
protocol shards under `data/protocol-v1/` are gitignored build artifacts, so
tests that consumed them passed locally (where a previous protocol run had left
them behind) and failed on a clean checkout -- which is exactly what happened
on Kaggle. The fixture below builds them on demand instead.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = ROOT / "data" / "protocol-v1"


@pytest.fixture(scope="session")
def protocol_shards() -> dict[str, str]:
    """Train/held-out toy shards, generated if absent.

    Session-scoped: building is cheap but not free, and every consumer wants
    the identical shard so that scores are comparable across tests.
    """
    train = PROTOCOL_DIR / "protocol-train-packed.jsonl"
    heldout = PROTOCOL_DIR / "protocol-heldout-packed.jsonl"
    if not (train.exists() and heldout.exists()):
        import sys
        sys.path.insert(0, str(ROOT))
        from scripts.run_protocol import build_shards
        build_shards()
    if not (train.exists() and heldout.exists()):  # pragma: no cover
        pytest.fail(f"protocol shards could not be built in {PROTOCOL_DIR}")
    return {"train": str(train), "heldout": str(heldout)}
