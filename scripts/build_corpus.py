#!/usr/bin/env python3
"""Stream a real corpus into binary packed shards sized for the model.

The existing data/real-v1 shards hold 1.67M tokens -- 268x too few for the
22.4M-parameter run they were used for, which looped them 270 times. This
builds a corpus large enough to train once through, streaming from the source
so neither disk nor RAM has to hold it all at once.

Held-out documents are drawn from the same stream *after* global deduplication,
so the two splits are disjoint by construction; every document hash is recorded
next to the shard for the contamination gate to verify independently.

Typical use (on a machine with network, e.g. the Kaggle notebook):

    python scripts/build_corpus.py --target-tokens 460000000 \
        --heldout-tokens 4000000 --out-dir data/real-v2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.binshard import write_packed_shard
from src.data.budget import tokens_needed
from src.data.pipeline import _fingerprint, normalize
from src.provenance.core import sha256_bytes

DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG = "sample-10BT"


def _tokenizer(name: str):
    """GPT-2 BPE via tiktoken: same 50257 vocab as the existing shards."""
    import tiktoken
    encoding = tiktoken.get_encoding(name)
    return encoding


def stream_documents(dataset: str, config: str | None, text_key: str) -> Iterator[str]:
    from datasets import load_dataset
    stream = load_dataset(dataset, name=config, split="train", streaming=True)
    for record in stream:
        text = record.get(text_key)
        if text:
            yield text


def deduplicated(texts: Iterator[str], *, near_duplicate: bool = True) -> Iterator[dict]:
    """Normalize and drop exact/near duplicates, preserving the audited rules."""
    seen_exact: set[str] = set()
    seen_bands: set[tuple[int, ...]] = set()
    index = 0
    for raw in texts:
        text = normalize(raw)
        if not text:
            continue
        digest = sha256_bytes(text.encode("utf-8"))
        if digest in seen_exact:
            continue
        bands = _fingerprint(text)
        if near_duplicate and bands in seen_bands:
            continue
        seen_exact.add(digest)
        seen_bands.add(bands)
        yield {"document_id": f"doc-{index:09d}", "text": text, "text_sha256": digest}
        index += 1


def tokenized(documents: Iterator[dict], encoding, *, batch: int = 512) -> Iterator[dict]:
    """Tokenize in batches; tiktoken is far faster batched than per-document."""
    buffer: list[dict] = []

    def flush() -> Iterator[dict]:
        if not buffer:
            return
        for document, ids in zip(buffer, encoding.encode_ordinary_batch([d["text"] for d in buffer])):
            yield {**document, "tokens": ids}
        buffer.clear()

    for document in documents:
        buffer.append(document)
        if len(buffer) >= batch:
            yield from flush()
    yield from flush()


def take_tokens(documents: Iterator[dict], limit: int, counter: dict) -> Iterator[dict]:
    """Yield whole documents until ``limit`` tokens have been emitted."""
    total = 0
    for document in documents:
        if total >= limit:
            return
        total += len(document["tokens"])
        counter["tokens"] = total
        counter["documents"] = counter.get("documents", 0) + 1
        yield document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-tokens", type=int, default=None,
                        help="unique training tokens (default: 20/param for --for-params)")
    parser.add_argument("--for-params", type=int, default=None,
                        help="size the corpus for a model of this many parameters")
    parser.add_argument("--heldout-tokens", type=int, default=4_000_000)
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "real-v2"))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--encoding", default="gpt2")
    parser.add_argument("--no-near-dedup", action="store_true")
    args = parser.parse_args()

    if args.target_tokens is None and args.for_params is None:
        raise SystemExit("provide --target-tokens or --for-params")
    target = args.target_tokens or tokens_needed(n_params=args.for_params)

    out_dir = Path(args.out_dir)
    encoding = _tokenizer(args.encoding)
    vocab_size = encoding.n_vocab
    print(f"dataset={args.dataset} config={args.config} encoding={args.encoding} "
          f"vocab={vocab_size}")
    print(f"target train tokens={target:,} heldout tokens={args.heldout_tokens:,}")

    started = time.perf_counter()
    documents = tokenized(deduplicated(stream_documents(args.dataset, args.config, args.text_key),
                                       near_duplicate=not args.no_near_dedup), encoding)

    # Held-out is drawn first from the deduplicated stream; the training split
    # then continues from the same iterator, so no document can appear in both.
    heldout_counter: dict = {}
    heldout_meta = write_packed_shard(
        take_tokens(documents, args.heldout_tokens, heldout_counter),
        out_dir / "heldout", sequence_length=args.sequence_length, vocab_size=vocab_size,
        tokenizer_id=f"tiktoken-{args.encoding}", source=f"{args.dataset}:{args.config}",
        cross_document=False)
    print(f"heldout: {heldout_meta['real_tokens']:,} tokens, "
          f"{heldout_meta['n_documents']:,} docs, {heldout_meta['n_sequences']:,} sequences")

    train_counter: dict = {}
    train_meta = write_packed_shard(
        take_tokens(documents, target, train_counter),
        out_dir / "train", sequence_length=args.sequence_length, vocab_size=vocab_size,
        tokenizer_id=f"tiktoken-{args.encoding}", source=f"{args.dataset}:{args.config}",
        cross_document=False)
    print(f"train:   {train_meta['real_tokens']:,} tokens, "
          f"{train_meta['n_documents']:,} docs, {train_meta['n_sequences']:,} sequences")

    if train_meta["real_tokens"] < target * 0.95:
        print(f"\n[WARNING] the source ran dry at {train_meta['real_tokens']:,} tokens, "
              f"short of the {target:,} requested. Use a larger --config "
              f"(e.g. sample-100BT) or lower --for-params.")

    from src.data.contamination import assert_disjoint_shards
    assert_disjoint_shards(out_dir / "train", out_dir / "heldout")
    print("contamination gate: PASS (splits share no document hash)")

    elapsed = time.perf_counter() - started
    summary = {"train": train_meta, "heldout": heldout_meta,
               "requested_train_tokens": target, "wall_clock_s": elapsed,
               "dataset": args.dataset, "config": args.config}
    (out_dir / "corpus-summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                 encoding="utf-8")
    print(f"\nbuilt in {elapsed / 60:.1f} min -> {out_dir}/corpus-summary.json")


if __name__ == "__main__":
    main()
