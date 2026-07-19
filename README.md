# COMPOUND-LM

An auditable experiment harness for measuring how language-model training efficiency levers compound.

The first implementation is the evidence spine: canonical configuration, provenance, append-only ledger, checkpoint health checks, deterministic smoke trainer, and automatic multiplier tables. Expensive model/data adapters are added only after these contracts pass.

`scripts/run_protocol.py` runs the protocol end-to-end at toy scale on CPU: a two-seed baseline plus two levers — the Muon optimizer and the Reex-v2 architecture (RoPE + RMSNorm + SwiGLU, selected via `src/model/registry.py`, parameter-matched to the reference model within 2%) — each in isolation and then compounded, scored by held-out capability and compared by capability-at-cost with an overlap coefficient. The same protocol scales to the 200M baseline by swapping the corpus, config, and (on GPU) the E-v1 benchmark harness.

The toy corpus is 64 words arranged in a fixed cycle, so each word deterministically predicts its successor. Held-out documents are windows the model never trained on (the contamination gate passes honestly), but every word transition inside them appears in training — so held-out accuracy measures learned generalization of the successor rule, not memorization. On that signal, at this scale: the Muon optimizer lever reaches the common held-out target at **~1.8× less compute** than the AdamW baseline (seed noise band 0.07×); the Reex-v2 architecture lever alone is **slower** (~0.51×) on this corpus; and the compound arm (Reex-v2 + Muon) lands at **~1.45×** — an overlap coefficient of ~1.62, meaning Muon recovers far more of the architecture's toy-scale deficit than the independent product predicts. Per the build plan, an overlap above 1 triggers an audit before it is celebrated, and a toy-scale architecture result carries no weight for 200M either way — the cycle corpus rewards the reference model's learned absolute positions, which is exactly the kind of proxy artifact the re-measure-at-scale rule exists for.

Toy scale is a proxy, not a prediction: these multipliers are re-measured at each real scale before they are trusted.

### Corrections this instrument has made to its own results

Each of these was a green, healthy, fully-ledgered run that measured nothing real. They are recorded because the gates that now catch them exist only because these happened.

- **Uniform `1.000×` from a 69-token corpus.** Every run scored 0 held-out accuracy, making the target 0, which every run clears at its first checkpoint. An absent measurement, not a null result. Now gated: `no_capability_signal`, and `compounding_report` rejects a non-positive target.
- **`3.6–3.8×` for Muon.** Real scores, real curves — but the model's initial loss was 25.7 against `ln(V)=10.8`, because `nn.Embedding` defaults to `N(0,1)` and `lm_head` is tied to it, putting initial logits at std ≈ `√d_model`. The untrained model started *worse than random*, and Muon's apparent advantage was largely faster recovery from that broken init. With a correct init the advantage is 1.71×. Now gated: `tests/test_model_init.py` pins initial loss to `ln(V)` at every width.
- **`1.000×` again, after the init fix.** With a sane init the cycle rule is learned before the first checkpoint, so every cost was an unobserved lower bound and all runs tied by construction. Now gated: `assert_costs_resolved` refuses a comparison built entirely from lower bounds.

<!-- AUTOGEN:TABLE START -->
| Run | Levers | Cost (FLOPs) | FLOP mult. | Wall-clock mult. | Overlap |
|---|---|---:|---:|---:|---:|
| baseline-s17 | baseline | 2.406e+10 | 1.000× | noisy | 1.000× |
| baseline-s23 | baseline | 2.596e+10 | 0.927× | noisy | 0.927× |
| optimizer-s17 | optimizer | 1.323e+10 | 1.819× | noisy | 1.021× |
| optimizer-s23 | optimizer | 1.351e+10 | 1.781× | noisy | 1.000× |
| architecture-s17 | architecture | 4.664e+10 | 0.516× | noisy | 1.028× |
| architecture-s23 | architecture | 4.797e+10 | 0.502× | noisy | 1.000× |
| compound-s17 | architecture, optimizer | 1.658e+10 | 1.451× | noisy | 1.624× |
| compound-s23 | architecture, optimizer | 1.657e+10 | 1.452× | noisy | 1.625× |
<!-- AUTOGEN:TABLE END -->

## Verifying the claims

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_all.py     # everything, labelled
PYTHONPATH=. .venv/bin/python scripts/verify_speedup.py # precision, on a GPU
PYTHONPATH=. .venv/bin/python scripts/verify_levers.py  # per-lever wall-clock
```

`verify_all.py` sorts every claim into **VERIFIED** (executed here),
**PORTABLE** (measured on a real accelerator), and **PROJECTED** (arithmetic
only), and never reports the third with the confidence of the first. That
distinction is load-bearing: measuring on an M2 GPU showed the projected 2×
mixed-precision speedup was 1.15× at one tensor size and *0.89× — a
regression* at another, and that Muon's 1.82× FLOP win is 1.31× in wall clock
once its costlier steps are counted. See
[docs/cost-reduction-plan.md](docs/cost-reduction-plan.md).

## Quick start

```bash
# 1. install the training runtime (venv required on managed Python)
python3 -m venv .venv && .venv/bin/python -m pip install pyyaml numpy torch
.venv/bin/python scripts/bootstrap.py            # prompts for each missing piece

# 2. run the full protocol end-to-end (real held-out scores + compounding table)
PYTHONPATH=. .venv/bin/python scripts/run_protocol.py

# 3. run the tests (needs torch: the training and eval paths are exercised for real)
PYTHONPATH=. .venv/bin/python -m pytest
```

Outputs land in `outputs/protocol-report.json` (full evidence) and `outputs/protocol-summary.md`.
See [the build plan](docs/build-plan.md) for the full experiment protocol and
[TRAINING_SETUP.md](TRAINING_SETUP.md) for scaling to a real corpus on GPU.
