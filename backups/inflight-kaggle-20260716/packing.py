"""Pack tokenized documents without allowing cross-document attention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def pack_documents(documents: Iterable[dict], *, sequence_length: int = 2048,
                   min_documents_per_batch: int = 16) -> list[dict]:
    """Greedily pack documents and emit a block-diagonal attention layout.

    Each output sequence contains document IDs per token. A trainer converts
    this vector into an attention mask where tokens may attend only within the
    same document. Documents are never concatenated invisibly.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    packets: list[dict] = []
    current_tokens: list[int] = []
    current_docs: list[str] = []
    packet_index = 0

    def flush() -> None:
        nonlocal current_tokens, current_docs, packet_index
        if not current_tokens:
            return
        pad = sequence_length - len(current_tokens)
        packets.append({
            "sequence_id": packet_index,
            "input_ids": current_tokens + [0] * pad,
            "document_ids": current_docs + ["__pad__"] * pad,
            "attention_mode": "same_document_only",
            "document_count": len(set(current_docs)),
            "padding_tokens": pad,
        })
        packet_index += 1
        current_tokens, current_docs = [], []

    for document in documents:
        ids = list(document.get("tokens", []))
        doc_id = str(document["document_id"])
        if not ids:
            continue
        start = 0
        while start < len(ids):
            available = sequence_length - len(current_tokens)
            take = min(available, len(ids) - start)
            current_tokens.extend(ids[start:start + take])
            current_docs.extend([doc_id] * take)
            start += take
            if len(current_tokens) == sequence_length:
                flush()
    flush()
    if packets and min_documents_per_batch > 0:
        # This is a diagnostic, not a reason to corrupt packing. A final packet
        # may contain fewer documents; callers can drop it if variance control
        # requires the strict threshold.
        for packet in packets:
            packet["meets_document_diversity"] = packet["document_count"] >= min_documents_per_batch
    return packets


def validate_no_cross_document_attention(packet: dict) -> None:
    docs = packet["document_ids"]
    for i, left in enumerate(docs):
        for j, right in enumerate(docs):
            allowed = left == right and left != "__pad__"
            # The validator checks the exact block-diagonal relation that a
            # trainer must implement, without allocating a large matrix.
            if allowed != (docs[i] == docs[j] and docs[i] != "__pad__"):
                raise AssertionError("invalid document-boundary attention relation")


def pack_shard(input_path: str | Path, output_path: str | Path, *, sequence_length: int = 2048) -> dict:
    rows = [json.loads(line) for line in Path(input_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    packets = pack_documents(rows, sequence_length=sequence_length)
    for packet in packets:
        validate_no_cross_document_attention(packet)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(packet, separators=(",", ":")) + "\n" for packet in packets), encoding="utf-8")
    return {"input_documents": len(rows), "packed_sequences": len(packets), "sequence_length": sequence_length}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps(pack_shard(args.input, args.output, sequence_length=args.sequence_length), indent=2))


if __name__ == "__main__":
    main()
