"""Shard loading shared by the trainer and the evaluator.

Both must agree exactly on how a shard becomes batches: if they disagreed, a
held-out score would be measured against different data than it claims. Keeping
one implementation here is what makes that guarantee structural rather than a
convention two modules are trusted to follow.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def is_binshard(shard: str | Path) -> bool:
    return Path(shard).with_suffix(".meta.json").exists()


class JsonlLoader:
    """Whole-shard-in-RAM loader for the original packed JSONL format."""

    def __init__(self, shard: str | Path):
        rows = [json.loads(line) for line in Path(shard).read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError("packed shard is empty")
        for row in rows:
            row["doc_ids_int"] = [
                x if isinstance(x, int)
                else (int(hashlib.sha256(x.encode()).hexdigest()[:8], 16) if x != "__pad__" else -1)
                for x in row["document_ids"]
            ]
        self.rows = rows
        self.sequence_length = len(rows[0]["input_ids"])

    def __len__(self) -> int:
        return len(self.rows)

    def batch(self, position: int, batch_size: int):
        rows = [self.rows[(position + i) % len(self.rows)] for i in range(batch_size)]
        return ([row["input_ids"] for row in rows], [row["doc_ids_int"] for row in rows])


class BinLoader:
    """Memory-mapped loader; shard size no longer bounds RAM."""

    def __init__(self, shard: str | Path):
        from src.data.binshard import PackedShard
        self.shard = PackedShard(shard)
        self.sequence_length = self.shard.sequence_length

    def __len__(self) -> int:
        return len(self.shard)

    def batch(self, position: int, batch_size: int):
        tokens, docs = self.shard.batch(range(position, position + batch_size))
        return (tokens.tolist(), docs.tolist())


def open_shard(shard: str | Path):
    """Return the loader matching this shard's on-disk format."""
    return BinLoader(shard) if is_binshard(shard) else JsonlLoader(shard)
