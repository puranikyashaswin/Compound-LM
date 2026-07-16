"""Memory-mapped packed shard format for real-scale corpora.

The JSONL packed format (``src/data/packing.py``) keeps every token as a Python
int, so a Chinchilla-sized corpus for even a 22M model (~450M tokens) needs tens
of GB of RAM to load. This format stores the same packed sequences as flat
binary arrays read via ``numpy.memmap``: the OS pages in only the batches a step
touches, so shard size stops bounding memory.

Layout, for a shard at prefix ``P``:

  ``P.tokens.bin``  token ids,        uint16 (or int32 if vocab > 65535)
  ``P.docs.bin``    document index,   int32; ``-1`` marks padding
  ``P.meta.json``   schema, sequence_length, counts, sha256 of both arrays

Both arrays are ``n_sequences * sequence_length`` long and reshape to
``(n_sequences, sequence_length)``. Document index is a running integer per
document rather than a hash: attention only ever compares ids for equality
within a sequence, and integers cannot collide the way truncated hashes can.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from src.provenance.core import sha256_bytes

SCHEMA_VERSION = 1
PAD_DOCUMENT_ID = -1


def token_dtype(vocab_size: int) -> np.dtype:
    """Smallest lossless dtype for these ids; uint16 covers GPT-2's 50257."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    return np.dtype(np.uint16) if vocab_size <= np.iinfo(np.uint16).max + 1 else np.dtype(np.int32)


def _sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_packed_shard(documents: Iterable[dict], out_prefix: str | Path, *,
                       sequence_length: int, vocab_size: int,
                       tokenizer_id: str, source: str = "unknown",
                       drop_last_partial: bool = False) -> dict[str, Any]:
    """Greedily pack ``documents`` into fixed-length sequences on disk.

    Streams: never holds more than one sequence in memory, so the corpus may be
    far larger than RAM. Packing semantics match ``pack_documents`` -- documents
    are laid end to end, split across sequence boundaries when longer than
    ``sequence_length``, and every token carries its document's index so a
    trainer can build a block-diagonal mask. A trailing partial sequence is
    padded (``drop_last_partial`` discards it instead, for strict-variance runs).
    """
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    dtype = token_dtype(vocab_size)
    tokens_path = prefix.with_suffix(".tokens.bin")
    docs_path = prefix.with_suffix(".docs.bin")

    buffer_tokens: list[int] = []
    buffer_docs: list[int] = []
    n_sequences = 0
    n_real_tokens = 0
    n_pad_tokens = 0
    n_documents = 0

    # Document hashes are written alongside the arrays so the contamination gate
    # still has something to compare: a binary shard carries no document text,
    # and rehydrating a multi-GB corpus just to diff splits is not viable.
    hashes_path = prefix.with_suffix(".dochashes")
    hash_file = hashes_path.open("w", encoding="utf-8")

    with tokens_path.open("wb") as token_file, docs_path.open("wb") as doc_file:
        def flush(pad: bool) -> None:
            nonlocal buffer_tokens, buffer_docs, n_sequences, n_pad_tokens
            if not buffer_tokens:
                return
            if pad:
                padding = sequence_length - len(buffer_tokens)
                buffer_tokens.extend([0] * padding)
                buffer_docs.extend([PAD_DOCUMENT_ID] * padding)
                n_pad_tokens += padding
            token_file.write(np.asarray(buffer_tokens, dtype=dtype).tobytes())
            doc_file.write(np.asarray(buffer_docs, dtype=np.int32).tobytes())
            n_sequences += 1
            buffer_tokens, buffer_docs = [], []

        for document in documents:
            ids = document.get("tokens") or []
            if not ids:
                continue
            if max(ids) >= vocab_size or min(ids) < 0:
                raise ValueError(
                    f"token id out of range for vocab_size={vocab_size} in "
                    f"document {document.get('document_id')}"
                )
            document_index = n_documents
            n_documents += 1
            n_real_tokens += len(ids)
            digest = document.get("text_sha256")
            if digest:
                hash_file.write(f"{digest}\n")
            start = 0
            while start < len(ids):
                take = min(sequence_length - len(buffer_tokens), len(ids) - start)
                buffer_tokens.extend(ids[start:start + take])
                buffer_docs.extend([document_index] * take)
                start += take
                if len(buffer_tokens) == sequence_length:
                    flush(pad=False)
        if not drop_last_partial:
            flush(pad=True)
    hash_file.close()

    if n_sequences == 0:
        raise ValueError("no sequences were packed; corpus is empty or all documents were skipped")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "format": "binshard",
        "source": source,
        "tokenizer_id": tokenizer_id,
        "vocab_size": vocab_size,
        "sequence_length": sequence_length,
        "token_dtype": dtype.name,
        "n_sequences": n_sequences,
        "n_documents": n_documents,
        "real_tokens": n_real_tokens,
        "padding_tokens": n_pad_tokens,
        "attention_mode": "same_document_only",
        "tokens_sha256": _sha256_file(tokens_path),
        "docs_sha256": _sha256_file(docs_path),
        "document_hashes_file": hashes_path.name,
        "document_hashes_recorded": sum(1 for _ in hashes_path.open(encoding="utf-8")),
    }
    meta["meta_sha256"] = sha256_bytes(json.dumps(meta, sort_keys=True).encode())
    prefix.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


class PackedShard:
    """Read-only memory-mapped view over a binary packed shard."""

    def __init__(self, prefix: str | Path, *, verify_hashes: bool = False):
        self.prefix = Path(prefix)
        meta_path = self.prefix.with_suffix(".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"no binshard meta at {meta_path}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta.get("format") != "binshard":
            raise ValueError(f"not a binshard: {meta_path}")
        self.sequence_length = int(self.meta["sequence_length"])
        self.vocab_size = int(self.meta["vocab_size"])
        tokens_path = self.prefix.with_suffix(".tokens.bin")
        docs_path = self.prefix.with_suffix(".docs.bin")
        if verify_hashes:
            # Off by default: rehashing a multi-GB shard every run is wasteful,
            # but a ledgered run can demand it to prove the bytes are unchanged.
            for path, key in ((tokens_path, "tokens_sha256"), (docs_path, "docs_sha256")):
                actual = _sha256_file(path)
                if actual != self.meta[key]:
                    raise ValueError(f"shard corrupted: {path} sha256 {actual} != {self.meta[key]}")
        shape = (int(self.meta["n_sequences"]), self.sequence_length)
        self.tokens = np.memmap(tokens_path, dtype=np.dtype(self.meta["token_dtype"]),
                                mode="r", shape=shape)
        self.docs = np.memmap(docs_path, dtype=np.int32, mode="r", shape=shape)

    def __len__(self) -> int:
        return int(self.meta["n_sequences"])

    def batch(self, indices: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(tokens, docs)`` arrays for ``indices``, wrapped modulo length."""
        rows = [index % len(self) for index in indices]
        return (np.asarray(self.tokens[rows], dtype=np.int64),
                np.asarray(self.docs[rows], dtype=np.int64))

    def iter_documents(self) -> Iterator[int]:
        """Yield each distinct document index present in the shard, in order."""
        seen = -1
        for row in self.docs:
            for value in np.unique(row):
                if value > seen:
                    seen = int(value)
                    yield seen


def convert_jsonl_shard(jsonl_path: str | Path, out_prefix: str | Path, *,
                        vocab_size: int, tokenizer_id: str,
                        sequence_length: int, source: str = "converted-jsonl") -> dict[str, Any]:
    """Convert a documents JSONL (pre-packing) into a binary packed shard."""
    def stream() -> Iterator[dict]:
        with Path(jsonl_path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    return write_packed_shard(stream(), out_prefix, sequence_length=sequence_length,
                              vocab_size=vocab_size, tokenizer_id=tokenizer_id, source=source)
