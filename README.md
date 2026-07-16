# COMPOUND-LM

An auditable experiment harness for measuring how language-model training efficiency levers compound.

The first implementation is the evidence spine: canonical configuration, provenance, append-only ledger, checkpoint health checks, deterministic smoke trainer, and automatic multiplier tables. Expensive model/data adapters are added only after these contracts pass.

`scripts/run_protocol.py` runs the protocol end-to-end at toy scale on CPU: a two-seed baseline plus a Muon optimizer lever, scored by held-out capability and compared by capability-at-cost. The same protocol scales to the 200M baseline by swapping the corpus, config, and (on GPU) the E-v1 benchmark harness.

The toy corpus is 64 words arranged in a fixed cycle, so each word deterministically predicts its successor. Held-out documents are windows the model never trained on (the contamination gate passes honestly), but every word transition inside them appears in training — so held-out accuracy measures learned generalization of the successor rule, not memorization. On that signal, two seeds of the Muon optimizer lever reach the common held-out target at **1.67× and 1.74× less compute** than the AdamW baseline (mean **1.71×**), against a baseline seed noise band of **0.12×**. Reruns reproduce every multiplier and checkpoint hash bit-identically.

Toy scale is a proxy, not a prediction: these multipliers are re-measured at each real scale before they are trusted.

### Corrections this instrument has made to its own results

Each of these was a green, healthy, fully-ledgered run that measured nothing real. They are recorded because the gates that now catch them exist only because these happened.

- **Uniform `1.000×` from a 69-token corpus.** Every run scored 0 held-out accuracy, making the target 0, which every run clears at its first checkpoint. An absent measurement, not a null result. Now gated: `no_capability_signal`, and `compounding_report` rejects a non-positive target.
- **`3.6–3.8×` for Muon.** Real scores, real curves — but the model's initial loss was 25.7 against `ln(V)=10.8`, because `nn.Embedding` defaults to `N(0,1)` and `lm_head` is tied to it, putting initial logits at std ≈ `√d_model`. The untrained model started *worse than random*, and Muon's apparent advantage was largely faster recovery from that broken init. With a correct init the advantage is 1.71×. Now gated: `tests/test_model_init.py` pins initial loss to `ln(V)` at every width.
- **`1.000×` again, after the init fix.** With a sane init the cycle rule is learned before the first checkpoint, so every cost was an unobserved lower bound and all runs tied by construction. Now gated: `assert_costs_resolved` refuses a comparison built entirely from lower bounds.

<!-- AUTOGEN:TABLE START -->
| Run | Levers | Cost (FLOPs) | Multiplier | Overlap |
|---|---|---:|---:|---:|
| baseline-s17 | baseline | 2.537e+10 | 1.000× | 1.000× |
| baseline-s23 | baseline | 2.878e+10 | 0.881× | 0.881× |
| optimizer-s17 | optimizer | 1.521e+10 | 1.667× | 0.957× |
| optimizer-s23 | optimizer | 1.456e+10 | 1.743× | 1.000× |
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
