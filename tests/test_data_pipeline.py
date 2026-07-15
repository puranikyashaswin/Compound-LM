import json

from src.data.pipeline import prepare_documents


def test_data_pipeline_is_deduplicated_and_hashed(tmp_path):
    sheet = prepare_documents(
        ["A clean document.", "A clean document.", "", "A second document!"],
        source="fixture", shard_id="raw-test", output_dir=tmp_path,
    )
    assert sheet["document_count_input"] == 4
    assert sheet["document_count_kept"] == 2
    assert sheet["rejected"]["exact_duplicate"] == 1
    assert sheet["rejected"]["empty"] == 1
    assert sheet["token_count"] > 0
    assert len(sheet["shard_sha256"]) == 64
    saved = json.loads((tmp_path / "raw-test.datasheet.json").read_text())
    assert saved["datasheet_hash"] == sheet["datasheet_hash"]


def test_data_pipeline_normalizes_text(tmp_path):
    prepare_documents(["  hello\n\n world  "], source="fixture", shard_id="x", output_dir=tmp_path)
    row = json.loads((tmp_path / "x.jsonl").read_text())
    assert row["text"] == "hello world"
