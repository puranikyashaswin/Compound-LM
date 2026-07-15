"""Deterministic curriculum and data-mixture scheduling."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Stage:
    name: str
    end_fraction: float
    weights: dict[str, float]


DEFAULT_STAGES = (
    Stage("foundation", 0.50, {"raw-v1": 0.70, "filtered-v1": 0.30}),
    Stage("capability", 0.85, {"filtered-v1": 0.55, "finemath-v1": 0.25, "stackedu-v1": 0.20}),
    Stage("premium", 1.00, {"filtered-v1": 0.35, "finemath-v1": 0.35, "stackedu-v1": 0.30}),
)


def _validate(stages: tuple[Stage, ...]) -> None:
    previous = 0.0
    for stage in stages:
        if not 0 < stage.end_fraction <= 1 or stage.end_fraction <= previous:
            raise ValueError("stage end_fraction must increase within (0, 1]")
        if not stage.weights or abs(sum(stage.weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"weights for {stage.name} must sum to 1")
        if any(weight < 0 for weight in stage.weights.values()):
            raise ValueError("curriculum weights cannot be negative")
        previous = stage.end_fraction
    if stages[-1].end_fraction != 1.0:
        raise ValueError("last curriculum stage must end at 1.0")


def build_schedule(total_tokens: int, *, stages: tuple[Stage, ...] = DEFAULT_STAGES) -> list[dict[str, Any]]:
    """Allocate integer token counts by stage/source with deterministic rounding."""
    if total_tokens < 1:
        raise ValueError("total_tokens must be positive")
    _validate(stages)
    output = []
    previous = 0.0
    allocated_total = 0
    for index, stage in enumerate(stages):
        stage_tokens = (total_tokens - allocated_total if index == len(stages) - 1
                        else round(total_tokens * (stage.end_fraction - previous)))
        allocation = {source: int(stage_tokens * weight) for source, weight in stage.weights.items()}
        # Put rounding remainder into the highest-weight source.
        remainder = stage_tokens - sum(allocation.values())
        top_source = max(stage.weights, key=stage.weights.get)
        allocation[top_source] += remainder
        output.append({"stage": stage.name, "tokens": stage_tokens, "sources": allocation,
                       "weights": stage.weights})
        allocated_total += stage_tokens
        previous = stage.end_fraction
    return output


def write_manifest(path: str | Path, total_tokens: int, *, stages: tuple[Stage, ...] = DEFAULT_STAGES) -> dict:
    schedule = build_schedule(total_tokens, stages=stages)
    manifest = {"schema_version": 1, "total_tokens": total_tokens,
                "stages": [asdict(stage) for stage in stages], "schedule": schedule}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(write_manifest(args.out, args.tokens), indent=2))


if __name__ == "__main__":
    main()
