import json

from scripts.run_experiment import run


def test_run_experiment_blocks_explicitly_without_shard(tmp_path):
    result = run("configs/baseline_200m.yaml", shard=str(tmp_path / "missing.jsonl"),
                 output_dir=str(tmp_path / "run"), steps=1)
    assert result["status"] == "blocked"
    assert "does not exist" in result["reason"]
    saved = json.loads((tmp_path / "run" / "run-manifest.json").read_text())
    assert saved["run_id"] == result["run_id"]
