# Real-training setup

The framework runs real experiments once the training environment is installed. `scripts/preflight.py` reports what is missing without fabricating results; `scripts/bootstrap.py` goes further and *prompts* for each missing piece, offering to install or provision it, then can drive one real end-to-end toy run.

## Interactive bootstrap (recommended)

Homebrew/system Python is externally managed (PEP 668), so use a virtualenv:

```bash
python3 -m venv .venv
.venv/bin/python scripts/bootstrap.py
```

The script walks three stages — infrastructure (torch/transformers/lm-eval), data, and a toy run — prompting before every install or execution. Nothing runs without confirmation. Non-interactive equivalents: `--assume-yes`, `--generate-toy-corpus`, `--run-toy`, `--no-input`.

A fully local smoke of the real pipeline (data prep → packing → CPU training → checkpoint → health → ledger) is:

```bash
.venv/bin/python -m pip install pyyaml torch
.venv/bin/python scripts/bootstrap.py --assume-yes --generate-toy-corpus --run-toy
PYTHONPATH=. .venv/bin/python ledger/make_table.py
```

## Full-scale setup

From a machine with a compatible CUDA/CPU PyTorch build:

```bash
python -m pip install -e '.[dev,training,data]'
python scripts/preflight.py
```

Then provide a local Reex tokenizer directory and create packed shards:

```bash
python -m src.data.pipeline --input corpus.txt --output-dir data/raw-v1 --shard-id raw-v1 --tokenizer-id reex-1
python -m src.data.packing --input data/raw-v1/raw-v1.jsonl --output data/raw-v1/raw-v1-packed.jsonl --sequence-length 2048
```

Run the audited experiment:

```bash
python scripts/run_experiment.py \
  --config configs/baseline_200m.yaml \
  --shard data/raw-v1/raw-v1-packed.jsonl \
  --output-dir runs/baseline \
  --steps 1000 --ledger ledger/runs.jsonl --evaluate
```

The command must report `ready_for_real_training: true` before any expensive run is considered valid.
