"""Regenerate the README compounding table from the ledger."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.ledger.writer import read_entries


def _cost_to_score(row: dict, score: float) -> float | None:
    value = row.get("eval_scores", {}).get("smoke")
    if value is None or value < score:
        return None
    return float(row.get("fully_accounted_cost_usd", row.get("est_cost_usd", 0.0)))


def make_table(ledger_path: str, readme_path: str, target_score: float = 0.1) -> str:
    rows = read_entries(ledger_path)
    if not rows:
        table = "_No completed runs yet._"
    elif all(r.get("eval_scores", {}).get("smoke", 0.0) <= 0 for r in rows):
        # Publishing all-zero scores invites reading them as a null result. No
        # run demonstrated any capability, so there is nothing to compare.
        table = "_No run with a measured capability signal yet._"
    else:
        baseline = next((r for r in rows if not r.get("levers_on")), None)
        base_cost = _cost_to_score(baseline, target_score) if baseline else None
        lines = ["| Run | Levers | Score | Cost | Multiplier |", "|---|---|---:|---:|---:|"]
        for row in rows:
            score = row.get("eval_scores", {}).get("smoke", 0.0)
            cost = float(row.get("fully_accounted_cost_usd", row.get("est_cost_usd", 0.0)))
            mult = "n/a" if not base_cost or cost <= 0 else f"{base_cost / cost:.2f}×"
            lines.append(f"| {row['run_id']} | {', '.join(row.get('levers_on', [])) or 'baseline'} | {score:.4f} | {cost:.4f} | {mult} |")
        table = "\n".join(lines)
    readme = Path(readme_path).read_text(encoding="utf-8")
    start, end = "<!-- AUTOGEN:TABLE START -->", "<!-- AUTOGEN:TABLE END -->"
    if start not in readme or end not in readme:
        raise ValueError("README is missing AUTOGEN markers")
    before = readme.split(start, 1)[0] + start + "\n"
    after = "\n" + end + readme.split(end, 1)[1]
    Path(readme_path).write_text(before + table + after, encoding="utf-8")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="ledger/runs.jsonl")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--target-score", type=float, default=0.1)
    args = parser.parse_args()
    print(make_table(args.ledger, args.readme, args.target_score))
