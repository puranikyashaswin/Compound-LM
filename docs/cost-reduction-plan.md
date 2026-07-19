# Cutting training cost

## Bugs this audit found in its own code

Three, all of which would have produced confident wrong numbers:

1. **Silent token remapping.** The trainer did `ids % vocab_size`. Correct for
   the hash-based fallback tokenizer (whose ids are 32-bit hashes), but it was
   applied to *every* shard — so loading a 50257-vocab corpus against a
   16384-vocab model mapped token 20000 onto token 3616, a different valid
   token. Training ran, loss looked healthy, the run measured nothing. **That
   is precisely the configuration the vocabulary-reduction lever creates.** The
   fold now happens once at data preparation, is recorded in the datasheet, and
   a mismatch at train time is an error (`assert_token_ids_in_range`).
   Regression-tested in `tests/test_token_range.py`.

2. **`torch.cuda.is_bf16_supported()` lies on Turing** — it counts emulated
   bf16, so a T4 selects a path measured at 0.73–0.77×, slower than fp32.
   Now resolved from compute capability.

3. **Wall-clock multipliers published as science when they were jitter.** The
   two baseline seeds run identical configs, so any gap in their
   seconds-per-step is pure noise. On this CPU it was 5.6–15%, larger than
   several levers. The protocol now uses that pair as a timing control and
   prints `noisy` instead of a number when it fails.

## Verified on a Tesla T4 (Kaggle, free tier)

**6.58× measured, equivalence-checked**, for vocabulary + depth + mixed
precision together on one shape in one run. Loss curves tracked fp32 to
2.4e-04, far inside the 0.05 tolerance, so the speedup is real and not bought
with accuracy.

| Lever | T4 measured | Predicted | Note |
|---|---:|---:|---|
| mixed precision (fp16) | **3.52×** | 2.00× | beat the projection at large shapes |
| vocabulary 50257→16384 | **1.50×** | 1.49× | arithmetic essentially exact |
| depth 12→6 | **1.35×** | 1.23× | |
| Muon step cost | **1.07×** | — | far cheaper than the M2's 1.39× |
| **all three, measured together** | **6.58×** | — | the number to trust |

Two things this run taught that no amount of arithmetic would have:

- **`torch.cuda.is_bf16_supported()` lies on Turing.** It counts *emulated*
  bf16, so a T4 reports `True` while having no bf16 tensor cores. Selecting it
  measured **0.73–0.77×** — slower than fp32. `resolve_precision` now reads
  compute capability directly (native bf16 from SM 8.0). Pinned by
  `tests/test_systems_precision.py::test_turing_does_not_get_emulated_bf16`.
- **The precision gain is strongly size-dependent:** 1.26× at batch 2 × seq 256,
  but 3.52× at batch 16 × seq 512. A small-batch run forfeits most of it.

**Levers overlap.** vocab × depth alone multiply to 2.03×; measured together
with precision the total is 6.58×, against a naive product of 7.13×. That is
an overlap coefficient of ~0.92 — real, and exactly what this repo's
compounding table exists to surface. Never report a product where a direct
measurement is available.

**Still unverified: Muon's 1.80×**, which is a toy-scale result on a 64-word
corpus. Its step cost is now measured (1.07× on T4, so ~1.68× wall clock *if*
the toy multiplier holds), but the multiplier itself must be re-measured at
real scale before anyone plans around it.

> **Corrections after measuring on real hardware.** An earlier version of this
> document claimed 6.35× from "verified" levers. Measuring on an Apple M2 GPU
> (`scripts/verify_levers.py`) showed two of those numbers were wrong:
>
> - **Mixed precision is not 2×, and can be negative.** Measured 1.15× at
>   batch 8 × seq 512, and **0.89× — a regression** — at batch 2 × seq 256,
>   where tensors are too small to pay for autocast overhead. bf16 on MPS
>   measured 0.97×. The 2× figure applies to CUDA tensor cores and remains
>   unverified here.
> - **Muon's 1.82× was a FLOP figure, not a wall-clock one.** Newton-Schulz
>   makes each step 1.39× more expensive (measured), so the real saving is
>   **1.31×**. The ledger prices runs at `6 × N × tokens`, which counts the
>   model's arithmetic and is structurally blind to optimizer overhead.
>
> Verified total is therefore **2.29×**, not 6.35×. Run
> `scripts/verify_all.py` for the current VERIFIED / PORTABLE / PROJECTED split.



Goal: reach the same held-out capability for less than half the time and money.
Run `python scripts/cost_reduction_plan.py` to regenerate every number here.

## Where the budget actually goes

The Kaggle validation config is 22.3M parameters at `d_model=256`, 12 layers,
`vocab=50257`. Splitting its forward pass by component:

| Component | FLOPs/token | Share |
|---|---:|---:|
| Output head (tied embedding) | 25.7M | 51% |
| Transformer stack (12 layers) | 18.9M | 37% |
| Attention (seq-length term) | 6.3M | 12% |

**The output head costs more than the entire 12-layer transformer.** 58% of the
parameters are the embedding matrix. The vocabulary is sized for a 1.5B model
and the model is 22M. Any optimization aimed at the transformer stack is aimed
at the minority of the cost — this table is the reason the plan below leads
with vocabulary and precision rather than with the optimizer.

## The levers, by how the gain is obtained

Separating these matters because gains on *different* axes compound almost
independently, while two algorithmic levers may not — which is precisely what
the protocol's overlap coefficient exists to measure.

### Throughput — seconds per step, FLOPs unchanged

| Lever | Gain | Evidence |
|---|---:|---|
| Mixed precision (bf16/fp16 autocast) | ~2.0× | arithmetic: tensor cores vs fp32 CUDA cores |
| Fused AdamW + TF32 | ~1.10× | arithmetic |
| `torch.compile` | ~1.15× | literature; verify per GPU class |

**This was the single largest finding: `SystemsPolicy` existed but was never
wired into the training loop.** Every GPU run in this repo's history was pure
fp32 — no autocast, no TF32, no fused optimizer. The 3.6-hour baseline paid
roughly double what it needed to.

A second trap sat underneath it. Kaggle's T4 is Turing: it has fp16 tensor
cores but no bf16. The old policy asked for bf16, found it unsupported, and
fell back to **fp32** — forfeiting the tensor cores entirely while reporting
that it had tried. `resolve_precision` now selects fp16 + `GradScaler` there,
and `auto` never resolves to fp32 on a GPU.

### Shape — FLOPs per token

| Lever | Gain | Evidence |
|---|---:|---|
| Vocabulary 50257 → 16384 | ~1.38× | arithmetic, net of a 1.10× tokenization penalty |

Shrinking the vocabulary shrinks the dominant term. It is not free: a smaller
BPE splits text into more tokens, so the same document costs more tokens to
read. `vocab_resize_multiplier` requires that penalty as an argument and
refuses a value below 1.0, because ignoring it is the standard way this lever
gets overstated. At 1.10× penalty the net gain is 1.38×; the model also drops
from 22.3M to ~13.6M parameters, which re-opens budget for depth.

### Algorithmic — steps to reach the target

| Lever | Gain | Evidence |
|---|---:|---|
| Muon optimizer | ~1.80× | measured here (1.82×/1.78×, two seeds, toy scale) |

Toy scale, so expect less at 22M. This is the only lever permitted to change
the science, and the only one already measured under the full protocol.

## Compounded

- All levers: **6.3× (84% less cost)**
- Excluding the unverified `torch.compile`: **5.5× (82% less cost)**
- One run: 3.6h → ~0.6h. The 4-run matrix: 14.4h → ~2.3h.

The ≥50% goal is met by the throughput levers alone (2.2×). Everything beyond
that is margin, which matters because these are estimates: if mixed precision
delivers 1.5× instead of 2.0× on a T4, the target still clears comfortably.

## What makes these claims falsifiable

The throughput and shape levers are **equivalence claims**, not improvement
claims: they must reproduce the baseline's capability curve within tolerance,
or they are not free. That is a stronger obligation than a lever win, and the
protocol already has the machinery for it:

- Precision is recorded in every checkpoint and ledger row, and resuming across
  a precision change is refused — a curve that switched numerics mid-run is not
  the curve it claims to continue.
- Gradient norms are read **after** `GradScaler.unscale_`. Skipping this feeds
  the health gate a number ~65536× too large and trips the spike check on a
  perfectly healthy run. Muon is unscaled explicitly, since `GradScaler` only
  touches optimizers handed to it.
- A vocabulary change is a new tokenizer, therefore a new lineage under the
  model-switch procedure: it gets its own two-seed baseline and never shares a
  multiplier column with a 50257-vocab run.

## Round 2: additional levers

### Shape — depth growth

`src/growth/depth.py` trains at half depth for the first part of the run, then
grows. Two modes, and the difference is scientific:

- `zero_init` inserts blocks whose residual output projections are exactly
  zero. A zero-output block contributes nothing to the residual stream, so the
  grown model computes *precisely* what the shallow model computed — it passes
  `assert_logit_equivalence`, the build plan's hard pre-training gate for
  growth. Verified for both `reference-v1` and `reex-v2`.
- `stack` duplicates blocks verbatim (the GStack recipe). Empirically strong,
  but a duplicated block is not the identity, so it cannot pass the gate and is
  labelled `function_preserving: False`.

The honest number is small and worth understanding: growing 6→12 for half the
run is only **1.10×**, because depth affects the transformer stack and the
stack is just 37% of forward FLOPs. **After the vocabulary cut it becomes
1.16×**, since the stack then carries 56%. `growth_savings` requires the
transformer's FLOP share as an argument — passing 1.0 (assuming the head is
free) is the standard way this lever gets overstated, and a test pins that.

### Throughput — width rebalance

`d_model=256` is too narrow to saturate tensor cores: a 256×768 QKV GEMM is a
small matmul, and the GPU spends its time on launch overhead rather than math.
Rebalancing to `d_model=512, n_layers=3` at equal parameters gives ~4× the
arithmetic intensity per matmul. Literature-class, ~1.30×, worth measuring
before trusting.

### Algorithmic — warm start

Transplanting a public checkpoint is the largest single candidate (~2.5×,
literature-class) and the most dangerous: the donor may have trained on your
held-out set. It is borrowed compute, so it belongs in `fully_accounted_cost`
as an amortized teacher cost, and the contamination gate must run before the
number is believed.

## Compounded, round 2

- Verified levers only (arithmetic + measured): **6.35× (84% less)**
- Including literature-class: **23.8× (96% less)**
- 4-run matrix: 14.4h → 2.3h verified, or ~0.6h with everything.

## Training quality — better runs, not faster ones

Deliberately **not** multiplied into the cost figure: these change what the run
learns, and counting them as speedups double-counts the same compute.

- **Real gradient clipping.** `clip_grad_norm_` was called with `1e9`, which
  returns the norm and clips nothing — it was a measurement device wearing a
  clipper's name. `--grad-clip 1.0` bounds the update and is what lets a higher
  learning rate stay stable. Opt-in, since it changes the numbers.
- **No-decay parameter groups.** PyTorch's AdamW default applied weight decay
  to LayerNorm/RMSNorm gains and biases, pulling toward zero for no
  regularization benefit and fighting the normalization the architecture
  depends on. Decay now applies to 2-D tensors only.
- **Batched held-out evaluation.** Evaluation ran one sequence at a time. Now
  batched: 4.5× faster at identical accuracy and identical token counts.
- **A tolerance-based replay contract.** Batched evaluation changes the float
  summation order, moving `val_nll` in the 7th decimal. The replay verifier
  compared eval scores with exact `!=`, so this would have raised a false
  "ledger contradiction" on the next resume. `REPLAY_TOLERANCE = 1e-6` replaces
  byte equality with the declared tolerance the build plan asked for — tight
  enough that a real divergence (different data order, seed, or precision)
  lands orders of magnitude outside it.

## What is actually verified

`scripts/verify_all.py` classifies every claim into three buckets and refuses
to blur them:

| Bucket | Meaning | Count |
|---|---|---:|
| VERIFIED | executed on this machine, this run | 9 |
| PORTABLE | measured on a real accelerator (MPS); expected to transfer | 6 |
| PROJECTED | arithmetic only, no measurement on any hardware | 1 |

Measured lever wall-clock on Apple M2 / MPS, `d_model=256`, vocab 50257,
batch 2 × seq 256:

| Lever | Measured | Predicted | Note |
|---|---:|---:|---|
| vocabulary 50257→16384 | **1.41×** | 1.49× | arithmetic held up |
| depth 12→6 | **1.28×** | 1.23× | arithmetic held up |
| mixed precision | **0.89×** | 2.00× | regression at this tensor size |
| Muon step cost | **0.72×** | 1.00× | the ledger cannot see this |
| all three combined | **1.87×** | — | |

The FLOP-reduction levers matched their arithmetic within ~5%, which is the
result that matters: those predictions transfer to a GPU. The precision lever
did not, because it is a property of the hardware rather than of the model.

**The one claim still unverified anywhere is the one the headline depends on:**
~2× from mixed precision on CUDA tensor cores. Confirming it needs one run on
a real NVIDIA GPU — `scripts/verify_speedup.py` does exactly that and fails
loudly if the loss curve diverges.

## Order of execution

1. **Mixed precision** — largest gain, zero science risk, already implemented.
   `--precision auto`. Confirm the equivalence curve against the fp32 ledger.
2. **Fused AdamW + TF32** — on by default with the systems policy; free.
3. **Vocabulary right-sizing** — requires retraining the tokenizer and
   rebuilding the corpus, so it is the expensive change to make; do it once,
   with its own baseline pair.
4. **`torch.compile`** — measure before trusting; it can regress on short runs.
5. **Muon** — already validated; keep it as the lever under test rather than
   folding it into the baseline.
