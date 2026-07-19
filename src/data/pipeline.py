"""Small deterministic corpus pipeline for COMPOUND-LM.

The production adapter can replace the tokenizer and source reader, but the
invariants here are deliberately backend-independent: normalize, deduplicate,
hash, tokenize, and publish a datasheet before training consumes a shard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Iterable

from src.provenance.core import canonical_json, sha256_bytes, sha256_json


WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split()).strip()


def token_ids(text: str, vocab_size: int | None = None) -> list[int]:
    """Stable tokenizer fallback; real runs inject the frozen Reex tokenizer.

    The raw form emits 32-bit hashes. ``vocab_size`` folds them into range
    *here*, at preparation time, where the fold is recorded in the datasheet.
    It used to happen implicitly in the training loop as ``ids % vocab_size``,
    which silently remapped any out-of-range token -- including real tokenized
    corpora loaded against a smaller vocabulary. That turned a configuration
    error into corrupted data with a plausible-looking loss curve.
    """
    raw = [int(hashlib.sha256(piece.encode("utf-8")).hexdigest()[:8], 16)
           for piece in WORD_RE.findall(text)]
    if vocab_size is None:
        return raw
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    return [value % vocab_size for value in raw]


def _fingerprint(text: str, bands: int = 8) -> tuple[int, ...]:
    words = text.lower().split()
    if not words:
        return ()
    # Deterministic SimHash-like signature for cheap near-duplicate screening.
    bits = [0] * 64
    for word in words:
        digest = int(hashlib.sha256(word.encode()).hexdigest()[:16], 16)
        for bit in range(64):
            bits[bit] += 1 if digest & (1 << bit) else -1
    value = sum((1 << bit) for bit, score in enumerate(bits) if score >= 0)
    return tuple((value >> (i * (64 // bands))) & ((1 << (64 // bands)) - 1) for i in range(bands))


def prepare_documents(documents: Iterable[str], *, source: str, shard_id: str,
                      output_dir: str | Path, tokenizer_id: str = "fallback-v1",
                      near_duplicate: bool = True, vocab_size: int | None = None) -> dict:
    started = time.perf_counter()
    seen_exact: set[str] = set()
    seen_bands: set[tuple[int, ...]] = set()
    kept: list[dict] = []
    rejected = {"empty": 0, "exact_duplicate": 0, "near_duplicate": 0}
    for index, raw in enumerate(documents):
        text = normalize(raw)
        if not text:
            rejected["empty"] += 1
            continue
        digest = sha256_bytes(text.encode("utf-8"))
        if digest in seen_exact:
            rejected["exact_duplicate"] += 1
            continue
        bands = _fingerprint(text)
        if near_duplicate and bands in seen_bands:
            rejected["near_duplicate"] += 1
            continue
        seen_exact.add(digest)
        seen_bands.add(bands)
        if tokenizer_id == "reex-1":
            from src.data.tokenizer import ReexTokenizer
            tokenizer = ReexTokenizer("work/reex-tokenizer")
            ids = tokenizer.encode(text)
            if vocab_size is not None and ids and max(ids) >= vocab_size:
                raise ValueError(
                    f"tokenizer {tokenizer_id!r} emitted id {max(ids)} for a declared "
                    f"vocab_size of {vocab_size}. Folding a real tokenizer's ids into a "
                    f"smaller vocabulary silently maps distinct tokens onto each other; "
                    f"retrain the tokenizer at the target size instead."
                )
        else:
            ids = token_ids(text, vocab_size)
        kept.append({"document_id": f"{shard_id}-{index:08d}", "text": text, "text_sha256": digest, "tokens": ids})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shard_path = out / f"{shard_id}.jsonl"
    with shard_path.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(canonical_json(row) + "\n")
    datasheet = {
        "schema_version": 1,
        "shard_id": shard_id,
        "source": source,
        "tokenizer_id": tokenizer_id,
        # Recorded so a shard states the vocabulary its ids are valid for; the
        # trainer refuses a model whose vocab_size disagrees.
        "vocab_size": vocab_size,
        "document_count_input": index + 1 if 'index' in locals() else 0,
        "document_count_kept": len(kept),
        "token_count": sum(len(row["tokens"]) for row in kept),
        "rejected": rejected,
        "preprocessing_wall_clock_s": time.perf_counter() - started,
        "preprocessing_cost_usd": 0.0,
        "shard_sha256": sha256_bytes(shard_path.read_bytes()),
    }
    datasheet["datasheet_hash"] = sha256_json(datasheet)
    (out / f"{shard_id}.datasheet.json").write_text(json.dumps(datasheet, indent=2) + "\n", encoding="utf-8")
    return datasheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="UTF-8 file, one document per line")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--source", default="local")
    parser.add_argument("--tokenizer-id", default="fallback-v1")
    args = parser.parse_args()
    docs = Path(args.input).read_text(encoding="utf-8").splitlines()
    print(json.dumps(prepare_documents(docs, source=args.source, shard_id=args.shard_id,
                                       output_dir=args.output_dir, tokenizer_id=args.tokenizer_id), indent=2))


if __name__ == "__main__":
    main()
