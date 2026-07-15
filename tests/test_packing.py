from src.data.packing import pack_documents, validate_no_cross_document_attention


def test_packing_preserves_boundaries_and_padding():
    packets = pack_documents([
        {"document_id": "a", "tokens": [1, 2, 3]},
        {"document_id": "b", "tokens": [4, 5]},
    ], sequence_length=4)
    assert len(packets) == 2
    assert packets[0]["document_ids"] == ["a", "a", "a", "b"]
    assert packets[1]["document_ids"] == ["b", "__pad__", "__pad__", "__pad__"]
    for packet in packets:
        validate_no_cross_document_attention(packet)


def test_long_document_is_split_without_loss():
    packets = pack_documents([{"document_id": "x", "tokens": list(range(7))}], sequence_length=4)
    assert sum(d == "x" for packet in packets for d in packet["document_ids"]) == 7
    assert all(packet["attention_mode"] == "same_document_only" for packet in packets)
