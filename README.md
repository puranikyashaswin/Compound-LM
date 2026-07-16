# COMPOUND-LM

An auditable experiment harness for measuring how language-model training efficiency levers compound.

The first implementation is the evidence spine: canonical configuration, provenance, append-only ledger, checkpoint health checks, deterministic smoke trainer, and automatic multiplier tables. Expensive model/data adapters are added only after these contracts pass.

`scripts/run_protocol.py` runs the protocol end-to-end at toy scale on CPU: a two-seed baseline plus a Muon optimizer lever, scored by held-out capability and compared by capability-at-cost. The same protocol scales to the 200M baseline by swapping the corpus, config, and (on GPU) the E-v1 benchmark harness.

The toy corpus is 64 words arranged in a fixed cycle, so each word deterministically predicts its successor. Held-out documents are windows the model never trained on (the contamination gate passes honestly), but every word transition inside them appears in training — so held-out accuracy measures learned generalization of the successor rule, not memorization. On that signal, two seeds of the Muon optimizer lever reach the common held-out target at roughly **3.6–3.8× less compute** than the AdamW baseline, with a baseline seed spread of 0.016 (gate ≤ 0.15).

_Provenance note:_ an earlier README published a table of uniform `1.000×` multipliers from a 69-token corpus on which every run scored 0 held-out accuracy. That was an absent measurement, not a null result — a target score of 0 makes every run tie by construction. The protocol now refuses to emit a table in that state (`no_capability_signal`), and `compounding_report` rejects a non-positive target outright.

<!-- AUTOGEN:TABLE START -->
| Run | Levers | Cost (FLOPs) | Multiplier | Overlap |
|---|---|---:|---:|---:|
| baseline-s17 | baseline | 2.785e+11 | 1.000× | 1.000× |
| baseline-s23 | baseline | 2.531e+11 | 1.100× | 1.100× |
| optimizer-s17 | optimizer | 7.411e+10 | 3.758× | 1.058× |
| optimizer-s23 | optimizer | 7.840e+10 | 3.552× | 1.000× |
<!-- AUTOGEN:TABLE END -->

## Quick start

```bash
# 1. install the training runtime (interactive; venv required on managed Python)
python3 -m venv .venv && .venv/bin/python -m pip install pyyaml torch
.venv/bin/python scripts/bootstrap.py            # prompts for each missing piece

# 2. run the full protocol end-to-end (real held-out scores + compounding table)
PYTHONPATH=. .venv/bin/python scripts/run_protocol.py

# 3. audit spine only (no torch needed)
PYTHONPATH=. python -m pytest
```

Outputs land in `outputs/protocol-report.json` (full evidence) and `outputs/protocol-summary.md`.
See [the build plan](outputs/compound-lm-build-plan.md) for the full experiment protocol and
[TRAINING_SETUP.md](TRAINING_SETUP.md) for scaling to the 200M baseline.
