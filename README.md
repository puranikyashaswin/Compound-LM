# COMPOUND-LM

An auditable experiment harness for measuring how language-model training efficiency levers compound.

The first implementation is the evidence spine: canonical configuration, provenance, append-only ledger, checkpoint health checks, deterministic smoke trainer, and automatic multiplier tables. Expensive model/data adapters are added only after these contracts pass.

The table below is generated from a **real** end-to-end protocol run (`scripts/run_protocol.py`) at toy scale on CPU: a two-seed baseline plus a Muon optimizer lever, scored by held-out capability and compared by capability-at-cost. Every number is measured, not fabricated. The same protocol scales to the 200M baseline by swapping the corpus, config, and (on GPU) the E-v1 benchmark harness.

<!-- AUTOGEN:TABLE START -->
| Run | Levers | Cost (FLOPs) | Multiplier | Overlap |
|---|---|---:|---:|---:|
| baseline-s17 | baseline | 1.063e+10 | 1.000× | 1.000× |
| baseline-s23 | baseline | 1.063e+10 | 1.000× | 1.000× |
| optimizer-s17 | optimizer | 1.063e+10 | 1.000× | 1.000× |
| optimizer-s23 | optimizer | 1.063e+10 | 1.000× | 1.000× |
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
