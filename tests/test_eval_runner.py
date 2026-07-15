import json

from evals.run import run


def test_eval_runner_hashes_checkpoint_and_refuses_fake_scores(tmp_path):
    ckpt = tmp_path / "checkpoint.bin"
    ckpt.write_bytes(b"deterministic checkpoint")
    out = tmp_path / "report.json"
    report = run(str(ckpt), str(out))
    assert report["contract_id"] == "E-v1"
    assert report["checkpoint_sha256"]
    assert report["scores"] == {}
    assert report["status"] == "unavailable"
    saved = json.loads(out.read_text())
    assert saved["report_hash"] == report["report_hash"]
