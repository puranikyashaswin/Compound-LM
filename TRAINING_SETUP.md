# Real-training setup

## Sizing the corpus to the model (read this first)

A run can pass every health gate and still measure nothing. The first GPU
validation trained a 22.4M-parameter model for 450M token-positions over a
corpus of **1.67M unique tokens** — 270 epochs over the same data, ~268× less
data than the model wanted. It cost 3.6 GPU-hours and could only measure
memorization: held-out accuracy plateaued at ~0.11 and Muon tied the baseline,
because in that regime the optimizer barely matters.

`src/data/budget.py` now refuses such a plan before any GPU time is spent, and
`scripts/kaggle_validation.py` calls it up front. Size the corpus first:

```bash
# ~20 tokens/param for a 22.4M model => ~450M unique tokens, streamed from
# FineWeb-Edu into memory-mapped binary shards (never held in RAM).
python scripts/build_corpus.py --for-params 22400000 \
    --sequence-length 512 --out-dir data/real-v2

# Then the four-run matrix; the budget gate must report green/amber first.
python scripts/kaggle_validation.py --corpus data/real-v2 --lr-schedule
```

Binary shards (`src/data/binshard.py`) exist because the JSONL loader holds
every token as a Python int: a 450M-token corpus would need ~30GB of RAM and
OOM a Kaggle instance. The binary format memory-maps, so shard size no longer
bounds memory. Both formats produce identical loss curves
(`tests/test_binshard_training.py`), so the swap does not change the science.

## Cutting the cost of the run (read before booking GPU time)

Every GPU run in this repo's history trained in **fp32**: `SystemsPolicy` was
defined but never wired into the training loop. Mixed precision, TF32, and
fused AdamW are now available and together roughly halve wall-clock at
identical mathematics.

```bash
# 'auto' = bf16 on Ampere+, fp16 + GradScaler on Turing (Kaggle T4).
# It never resolves to fp32 on a GPU -- the old bf16-or-nothing policy did,
# which is how a T4 run silently paid fp32 prices.
python scripts/kaggle_validation.py --corpus data/real-v2 --lr-schedule \
    --precision auto
```

Default stays `fp32` so existing ledger rows remain reproducible; speed is
opt-in. Precision is recorded per checkpoint and resuming across a precision
change is refused.

Training-quality flags, all opt-in for the same reason:

```bash
--grad-clip 1.0        # the old call site clipped at 1e9, i.e. never clipped
--weight-decay 0.01    # 2-D tensors only; norms and biases are now exempt
--eval-batch-size 32   # held-out eval ran one sequence at a time (4.5x slower)
``` See [docs/cost-reduction-plan.md](docs/cost-reduction-plan.md)
for the full ≥50% plan, and run `python scripts/cost_reduction_plan.py` to
regenerate its arithmetic — including the finding that at `d_model=256` with a
50257 vocabulary, the output head costs more than the whole transformer stack.

## Choosing the architecture

The trainer builds its model through `src/model/registry.py`. Two lineages
exist today:

- `reference-v1` (default): GPT-2-style — learned positions, LayerNorm, GELU.
- `reex-v2`: the Reex-2 candidate — RoPE, RMSNorm, SwiGLU, parameter-matched
  to the reference within ~1%.

```bash
# Same matrix, Reex-v2 lineage. Run/ledger IDs are prefixed with the
# architecture so it can never be silently merged into a reference baseline.
python scripts/kaggle_validation.py --corpus data/real-v2 --lr-schedule \
    --architecture reex-v2
```

Checkpoints record their architecture in `config`; old checkpoints without the
key load as `reference-v1`, and resuming across architectures is refused
(`tests/test_reex_model.py` pins the registry behavior and holds `reex-v2` to
the same initial-loss-equals-`ln(V)` gate as the reference model). Per the
build plan's model-switch procedure, a new architecture is a new lineage:
compare it against a reference baseline in a separate table, never inside one.

## If a run dies partway

**Re-run the identical command.** It resumes from the newest readable
checkpoint; a crash or a session timeout costs one `--checkpoint-every`
interval (~10 minutes of a 3.6-hour run at the defaults), not the run.

```bash
# Same command as before. Prints "Resuming from: checkpoint-000XXXXX.pt".
python scripts/kaggle_validation.py --corpus data/real-v2 --lr-schedule

# Only this discards prior checkpoints and the ledger:
python scripts/kaggle_validation.py --corpus data/real-v2 --lr-schedule --fresh
```

`--fresh` is opt-in precisely because the reflex after a dead run is to re-run
it, and that used to delete the ledger holding the hours already paid for.

What survives, and what does not:

- Checkpoints and ledger rows are written incrementally, so both survive a
  `SIGKILL`. `tests/test_kaggle_resume.py` kills the script mid-training and
  proves the re-run resumes with prior rows unaltered and no duplicates.
- A truncated checkpoint (what a dead `torch.save` leaves) is discarded and the
  resume falls back to the previous good one.
- **Residual gap:** a checkpoint written but not yet ledgered when the process
  dies loses that one point from its capability curve. The curve simply has one
  fewer sample; interpolation is unaffected.

There is an unexplained, non-deterministic `RuntimeError: Parent directory ...
does not exist` from `torch.save` for a directory that demonstrably exists. It
was seen only on local macOS and never reproduced in isolation. Checkpoint
writes now retry after re-asserting the directory, which makes it survivable
rather than fixed. If it fires on Kaggle, re-run the same command.

_On `checkpoint_hash`:_ it is a hash of the checkpoint **file**, so it proves
integrity, not state identity. A resumed run produces bitwise-equal weights but
a different file, because `torch.save`'s container is not byte-stable across a
load/re-save. Hashes therefore match across full reruns of the protocol but not
across a resume; the ledger compares replayed rows on `final_loss` and
`eval_scores` for that reason.

## Bootstrap and toy runs

The framework runs real experiments once the training environment is installed. `scripts/preflight.py` reports what is missing without fabricating results; `scripts/bootstrap.py` goes further and *prompts* for each missing piece, offering to install or provision it, then can drive one real end-to-end toy run.

### Interactive bootstrap (recommended)

Homebrew/system Python is externally managed (PEP 668), so use a virtualenv:

```bash
python3 -m venv .venv
.venv/bin/python scripts/bootstrap.py
```

The script walks three stages — infrastructure (torch/transformers/lm-eval), data, and a toy run — prompting before every install or execution. Nothing runs without confirmation. Non-interactive equivalents: `--assume-yes`, `--generate-toy-corpus`, `--run-toy`, `--no-input`.

A fully local smoke of the real pipeline (data prep → packing → CPU training → checkpoint → health → ledger) is:

```bash
.venv/bin/python -m pip install pyyaml numpy torch
.venv/bin/python scripts/bootstrap.py --assume-yes --generate-toy-corpus --run-toy
PYTHONPATH=. .venv/bin/python ledger/make_table.py
```

## Full-scale setup

From a machine with a compatible CUDA/CPU PyTorch build:

```bash
python -m pip install -e '.[dev,training,data]'
python scripts/preflight.py
```

For the corpus, use `scripts/build_corpus.py` (above) rather than the JSONL
pipeline below. The JSONL route materializes every token as a Python int, so it
is fine for toy shards and unusable at real scale — a 450M-token corpus needs
roughly 30GB of RAM to load and will OOM before the first step.

<details>
<summary>Legacy JSONL pipeline (toy scale only)</summary>

```bash
python -m src.data.pipeline --input corpus.txt --output-dir data/raw-v1 --shard-id raw-v1 --tokenizer-id reex-1
python -m src.data.packing --input data/raw-v1/raw-v1.jsonl --output data/raw-v1/raw-v1-packed.jsonl --sequence-length 2048
```

</details>

Run the audited experiment:

```bash
python scripts/run_experiment.py \
  --config configs/baseline_200m.yaml \
  --shard data/raw-v1/raw-v1-packed.jsonl \
  --output-dir runs/baseline \
  --steps 1000 --ledger ledger/runs.jsonl --evaluate
```

The command must report `ready_for_real_training: true` before any expensive run is considered valid.
