# COMPOUND-LM — Auditable Build Plan v0.2

## Outcome

Build an experimental framework that can make a defensible statement of the form:

> “At 200M parameters, for the frozen evaluation contract `E-v1`, recipe `R` reaches score `S` at **N× lower fully-accounted cost** than baseline `B-v1`.”

The first model is `Reex-2-200M-base`.  The primary product is the evidence package behind that claim: immutable run manifests, checkpoint records, evaluation reports, cost records, and the compounding/overlap table.

This plan retains the central idea in the v0.1 specification: the framework measures how efficiency levers *compound*, rather than declaring a win from one attractive run.

## Decisions adopted from the specification

- One YAML configuration is a run’s declared intent; levers are config deltas, never code forks.
- The baseline is a competent, reproducible 2023 dense recipe, not a deliberately weak straw man.
- Downstream capability at equal score—not validation loss alone—defines the efficiency claim.
- The evaluation contract is frozen before the first baseline run.
- The two-seed baseline is mandatory.
- Any unhealthy or provenance-incomplete run is excluded from the ledger comparison.
- Data preparation and teacher/synthetic-data costs are included in the economic record.
- Every checkpoint receives automatic health tests; a failing run is terminated and repaired, not rationalized into a result.

## Changes that make v0.1 scientifically stronger

### 1. Define cost unambiguously

Store three numbers for every run, rather than one ambiguous `est_cost_usd`:

| Field | Meaning | Use |
|---|---|---|
| `train_compute` | accelerator-seconds and estimated training FLOPs | algorithmic efficiency |
| `run_cost_usd` | actual rented hardware, storage, and egress cost | operational affordability |
| `fully_accounted_cost` | run cost + attributable preprocessing + teacher + sweep cost | headline “cheaper” claim |

Data preparation and teacher cost should be reported both as **amortized** and **unamortized**.  A reusable shard can be amortized across its declared experimental cohort, but never silently across unrelated future models.  The headline table must state which figure it uses.

### 2. Make a model change auditable, not merely reproducible

Every run and checkpoint receives an immutable `provenance.json` manifest containing:

```text
run_id, parent_run_id, config_hash, git_commit, dirty_diff_hash,
trainer_image_digest, Python/package lock hash, CUDA/driver/GPU details,
model_impl_hash, tokenizer_hash, vocabulary_hash,
data_manifest_hashes, data-license/datasheet IDs,
seed, distributed topology, precision policy,
checkpoint_sha256, optimizer/scheduler state hashes,
evaluation_contract_hash, health-report hash
```

`model_impl_hash` is calculated from the resolved architecture, module source hashes, parameter-shape manifest, initialization policy, and forward-pass feature flags. A new base model, attention implementation, tokenizer, packing policy, or objective creates a new **lineage**; it may be compared, but cannot be silently merged into an existing baseline column.

### 3. Replace “byte identical” with a realistic determinism contract

Within the same container, GPU class, world size, and deterministic-kernel policy, require identical sampled document IDs and loss arrays to a declared tolerance. Across GPU classes, use a statistical equivalence tolerance, not byte equality. Keep the two modes explicit:

- `replayable`: same environment; data order and checkpoint hashes must match.
- `comparable`: different approved environment; configuration and outcome must fall inside predeclared tolerances.

### 4. Add confidence intervals and decision rules

A single 200M run estimates a result; it does not establish it. Use the two baseline seeds to estimate baseline variation. For each claimed lever win, run a confirmatory seed if the estimated advantage is below 1.5 aggregate evaluation points or below 15% cost. Report bootstrap confidence intervals over tasks and seeds.

The efficiency multiplier at score `S` is calculated by interpolation over checkpointed evaluation curves:

`M(S) = fully_accounted_cost_baseline_to_S / fully_accounted_cost_recipe_to_S`.

Do not extrapolate beyond the last measured checkpoint. If a recipe never reaches `S`, report `not reached`, never an implied multiplier.

### 5. Promote architecture pack out of the core MVP

The specification says “MVP = 4 levers,” but the matrix currently contains five experimental levers: systems, optimizer, data, architecture, and growth. The recommended MVP is:

1. systems/numerics;
2. optimizer;
3. data/curriculum;
4. growth.

The architecture pack remains an explicitly optional v1.1 experiment after the four core levers work independently. This preserves the project’s most distinctive hypothesis—compounding measured against a fair baseline—without turning MVP into an uncontrolled architecture search.

### 6. Harden evaluation against leakage and benchmark overfitting

Keep the frozen public suite, but add a private holdout pack containing held-out domain data and manually reviewed prompts. Hash, encrypt, and access-control the private prompts. Apply n-gram and embedding-neighbour contamination checks to all training shards before each promoted run. Public benchmarks are useful for comparability; the holdout is the anti-overfitting gate.

## Repository design

```text
compound-lm/
├── configs/                  # resolved configs and small declarative deltas
├── contracts/                # frozen eval, data, cost, and health schemas
├── src/
│   ├── provenance/           # manifests, hashing, signing, lineage validation
│   ├── data/                 # data pipeline, datasheets, packing validation
│   ├── model/                # baseline definition + optional feature flags
│   ├── optim/                # Muon/AdamW hybrid and µP transfer rules
│   ├── growth/               # HyperClone and GStack with equivalence tests
│   ├── train/                # nanotron adapter, checkpoint lifecycle
│   ├── health/               # online checks and stop policies
│   └── ledger/               # append-only records and table generation
├── evals/                    # pinned runner, task manifests, reports
├── tests/                    # unit, integration, replay, and acceptance tests
├── scripts/                  # dry-run matrix planner and resumable launcher
├── ledger/                   # JSONL records, immutable run artifacts, tables
└── docs/                     # protocol, datasheets, experiment write-ups
```

The ledger stays append-only. Corrections are separate `supersedes_run_id` entries; old records are never edited in place.

## Build sequence

### Phase 0 — Freeze contracts (days 1–3)

Deliverables:

- baseline contract `B-v1` with parameter count, tokenizer, data tier, sequence length, schedule, hardware accounting, and target token budget;
- evaluation contract `E-v1`, exact task versions, prompt templates, few-shot settings, scorer versions, and aggregate formula;
- health contract `H-v1` with thresholds and automatic stop states;
- cost contract `C-v1` defining amortization and currency/GPU-hour conversion;
- `RunManifest` and `CheckpointManifest` JSON schemas.

Gate: a hand-written fake run set produces a correct table and rejects missing/invalid provenance.

### Phase 1 — Evidence and audit spine (days 4–7)

Build the configuration resolver, hashing service, ledger writer, artifact layout, and README/table generator before training code.

Every launch must create its manifest before compute begins. Every checkpoint is content-hashed and references its parent checkpoint. The launcher refuses a duplicate completed config/seed/environment combination unless `--rerun-reason` is present.

Gate: duplicate configs are rejected; a changed model flag, tokenizer, data shard, or package lock produces a new lineage/version; a checkpoint can be verified from ledger record to bytes on disk.

### Phase 2 — Frozen evaluation and leakage controls (days 8–12)

Wrap a pinned `lm-eval-harness` release. Save raw generations/log-likelihoods, per-task JSON, runner logs, and environment hash in an evaluation report. Build a calibration script for GPT-2, Pythia-160M, and Reex-116M.

Add:

- deterministic tokenization and prompt rendering tests;
- public-suite and private-holdout report separation;
- corpus contamination scanner;
- eval-score aggregation with per-task visibility, not only one mean.

Gate: repeated same-environment evaluation is identical; reference calibration is within the declared tolerance; a seeded deliberately contaminated document is detected by the scanner.

### Phase 3 — Data pipeline and proxy lab (days 13–20)

Produce immutable `raw-v1`, `filtered-v1`, and `curriculum-v1` datasets with document IDs, source IDs, license metadata, tokenizer version, deduplication outputs, classifier scores, and preprocessing cost. Validate intra-document masking and packing before any full run.

Run the 20M proxy matrix first: raw versus filtered, then filtered versus curriculum. It should use exactly the same evaluation contract and cost ledger as the 200M model.

Gate: datasheets reconcile token count within 0.1%; packing proves no cross-document attention; filtered produces a predeclared meaningful proxy advantage or the data pipeline is debugged before promotion.

### Phase 4 — Baseline trainer and health controller (days 21–28)

Adapt nanotron rather than rebuild distributed training. Implement checkpoint lifecycle:

1. save model/optimizer/RNG/dataloader state;
2. hash all artifacts;
3. run health checks;
4. run scheduled evaluation;
5. append signed ledger event;
6. promote, pause, or terminate according to the health policy.

Baseline health checks include loss, gradient/update norm, activation statistics, NaN/Inf rate, token/data-rate stalls, MFU, checkpoint recovery, and evaluation trend. Hard failures halt the job; soft warnings remain visible in the ledger.

Gate: 100M-token smoke test resumes bit-for-bit in replayable mode; health report and evaluation report link to the exact checkpoint; MFU meets the hardware-specific target.

### Phase 5 — Establish the reference (days 29–42)

Run A0/A1: two independently seeded full baseline runs. Store evaluation at regular cost fractions, preferably 5/10/25/50/75/100%, so score-to-cost interpolation is actually supported.

Before levers begin, publish internally:

- baseline mean and seed spread by task;
- loss/evaluation learning curves;
- throughput and fully-accounted cost;
- data, system, and provenance report.

Gate: if aggregate seed spread exceeds 1.5 points, increase the common token budget or add a third baseline seed; do not continue with an underpowered comparison.

### Phase 6 — Validate each core lever separately (days 43–63)

Build and test in this order:

1. **Systems (B5):** performance only; no intentional quality change. Require loss/eval equivalence within tolerance and a measured throughput gain.
2. **Optimizer (B2):** Muon for eligible 2D hidden weights, AdamW elsewhere; fixed-matrix numerical tests, µP transfer, and small verification sweep.
3. **Data (B1):** config-only selection of frozen shard versions; preprocessing and teacher costs enter the ledger.
4. **Growth (B4):** HyperClone/GStack. Function equivalence on fixed prompts is a hard pre-training gate. Preserve tokenizer, normalization, RoPE, parameter tying, and output semantics exactly.

For every lever, publish an individual result card: intent, implementation delta, tests, failed attempts, score/cost curve, and confidence interval.

Gate: a lever moves to compounding only if it beats the baseline under `E-v1` and passes all health/provenance rules. Otherwise record the null result and omit it from C3.

### Phase 7 — Compounding matrix (days 64–78)

Run only proven levers:

| Run | Configuration | Purpose |
|---|---|---|
| C1 | systems + optimizer + filtered data | first interaction measurement |
| C2 | C1 + growth | proposed Reex-2-200M-base |
| C3 optional | C2 + validated architecture pack | v1.1 architecture interaction |

For each row, calculate observed multiplier, product-of-isolated multipliers, and interaction/overlap:

`overlap = observed_multiplier / product(isolated_multipliers)`.

Values below one show overlapping gains; that is a valid and important result. A result above one triggers an audit of contamination, baseline fairness, cost accounting, and configuration drift before it is celebrated.

Gate: `C2` can be named Reex-2 only if it has complete lineage, passes the private holdout, has no unresolved health warning, and reaches the agreed baseline capability target with a confidence-supported cost advantage.

### Phase 8 — Release evidence package (days 79–84)

Generate the write-up solely from ledger artifacts. It includes method, exact baseline definition, all cost conventions, data sources, seed spread, individual lever results, compounding table, overlap coefficients, failures, and scale limits.

Release a reproducibility bundle containing resolved configs, package lock, public data manifests, code commit, checkpoint hashes, evaluation reports, and scripts that reproduce the tables.

## Checkpoint protocol

At every save, generate a status with one of four states:

| State | Meaning | Action |
|---|---|---|
| `green` | all hard and soft checks nominal | continue |
| `amber` | drift requiring review but recoverable | continue only to next scheduled review |
| `red` | invalid comparison or likely training failure | halt; do not enter result table |
| `quarantined` | incomplete provenance, leaked data, or corrupted artifact | preserve for debugging; exclude from all claims |

Hard checks:

- checkpoint and parent hashes validate;
- exact resolved config/lineage match;
- finite tensors and recoverable optimizer/RNG/dataloader state;
- no loss/gradient explosion threshold breach;
- measured throughput does not fall outside its declared variance band;
- no training/evaluation contamination finding;
- growth checkpoint passes logit equivalence before its first optimization step.

Soft checks:

- loss slope remains plausible;
- per-task evaluation trend is not diverging from aggregate score;
- data-source mix and document diversity match declared bounds;
- GPU utilization and dataloader latency remain stable.

## Model-switch procedure

When you change from one model implementation to another, do not overwrite the old result. Follow this protocol:

1. create a new `model_lineage_id` and resolved architecture manifest;
2. rerun the fixed forward-pass, tokenizer, packing, and checkpoint-resume tests;
3. run a small calibration training job under `B-v1` conditions;
4. compare calibration curve, MFU, and evaluation to the prior implementation;
5. label the result `comparable` only if it passes predeclared tolerances; otherwise start `B-v2` with two new baseline seeds;
6. retain cross-lineage comparisons in a separate table, never mix their multipliers.

This lets you aggressively try other models without contaminating the evidence.

## Initial acceptance checklist

- [ ] Config resolves to a canonical YAML/JSON representation and stable hash.
- [ ] Ledger is append-only and rejects unhashable/missing provenance.
- [ ] README table is regenerated solely from ledger records.
- [ ] A run can be replayed from a manifest and checkpoint chain.
- [ ] Evaluation contract is pinned, calibrated, and leak-checked.
- [ ] Cost has training, actual-dollar, and fully-accounted representations.
- [ ] Raw/filtered/curriculum data have frozen datasheets and version hashes.
- [ ] Baseline has two full seeds and recorded variance.
- [ ] Every claimed lever has an isolated, audited result before compounding.
- [ ] Reex-2 is named only after C2 meets the target and audit gates.

## Immediate first task

Implement Phase 0 and Phase 1 only. Do not begin data processing or model training until the audit spine can prove what a run was, what it consumed, which code executed, and whether a later model switch changed the scientific comparison.
