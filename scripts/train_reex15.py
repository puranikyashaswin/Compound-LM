"""Train Reex-1.5 across many Kaggle sessions, losing nothing between them.

Reex-1.5 is Reex-1 (116M, 12 layers) grown to 24 layers (193M) and pretrained
further. Reaching Chinchilla for 193M needs ~3.9B tokens against Reex-1's 2B,
so ~1.9B more -- about 25 GPU-hours at the throughput Reex-1 actually achieved.
That does not fit one session, so this script is built around being interrupted.

The design point that matters: **a Kaggle committed run starts with an empty
working directory.** Local checkpoints do not survive to the next session, so a
disk-based `--resume` finds nothing and silently restarts from zero, quietly
burning the quota again. The Hub is therefore not a backup but the live state:

    pull resume state from the Hub
      -> if none, grow Reex-1 and start there
      -> train until the time budget, checkpointing and uploading as it goes
      -> stop at a checkpoint boundary and upload
    (next session repeats, continuing exactly where this one stopped)

Run it identically every session. It works out where it is from the Hub.

The corpus deliberately skips the documents Reex-1 already trained on:
re-showing them would measure memorisation rather than buy new capability.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Set before torch initialises CUDA: the eager-attention path allocates and frees
# large per-layer tensors, and without this the freed blocks fragment until a
# large allocation fails despite enough total free memory.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REEX1 = "puranikyashaswinsharma/reex-1"


def build_corpus(out_dir: Path, *, target_tokens: int, skip_documents: int,
                 sequence_length: int, vocab_size: int, heldout_documents: int = 2000,
                 sync=None) -> dict:
    """Stream fresh FineWeb-Edu straight into memory-mapped shards.

    Deliberately never materialises the corpus. `prepare_documents` keeps every
    token as a Python int in a list, which at 500M tokens is ~18GB -- more RAM
    than a Kaggle GPU notebook has -- and additionally writes a ~5.5GB JSONL
    that would then be converted and discarded. Both are fatal at this scale:
    one OOMs, the other exhausts the ~19.5GB working directory once checkpoints
    are added.

    `write_packed_shard` accepts an iterable, so documents are tokenised and
    handed over one at a time and never all held at once. Peak memory is one
    document; peak disk is the shard itself.

    Held-out is taken as the FIRST `heldout_documents` documents and training
    from the rest, so the two are disjoint by construction and the split does
    not require knowing the corpus length in advance.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from src.data.binshard import write_packed_shard

    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "corpus.json"
    if marker.exists():
        print("   reusing corpus already built in this session")
        return json.loads(marker.read_text())

    # Kaggle wipes the working directory between sessions, so without this the
    # identical corpus is rebuilt every time -- ~35 minutes each, four times.
    # Downloading the shards back costs a couple of minutes instead.
    if sync is not None and sync.pull_corpus(out_dir):
        if marker.exists():
            print("   corpus restored from the Hub (skipped rebuild)")
            return json.loads(marker.read_text())

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    counts = {"train": 0, "heldout": 0}

    # ONE pass over the stream, not one per split. Opening a fresh stream per
    # split re-pays the 2.4M-document skip -- about five minutes, every session.
    # The iterator is held across both writes so the skip happens once.
    stream = iter(load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                               split="train", streaming=True))
    seen = 0
    for record in stream:
        seen += 1
        if seen % 500_000 == 0:
            print(f"     skipping documents Reex-1 saw: {seen:,}/{skip_documents:,}")
        if seen >= skip_documents:
            break

    def tokenize(text: str, which: str, index: int) -> dict | None:
        ids = [i for i in tokenizer.encode(text) if i < vocab_size]
        if not ids:
            return None
        counts[which] += len(ids)
        if counts[which] % 25_000_000 < len(ids):
            print(f"     {which}: {counts[which]/1e6:.0f}M tokens")
        return {"document_id": f"fwe-{which}-{index:08d}", "tokens": ids}

    # Held-out first, buffered: it is only ~2000 documents, and materialising it
    # lets the same iterator continue straight into the training split.
    heldout: list[dict] = []
    index = 0
    for record in stream:
        text = record["text"].strip()
        if not text:
            continue
        index += 1
        document = tokenize(text, "heldout", index)
        if document:
            heldout.append(document)
        if len(heldout) >= heldout_documents:
            break

    def train_documents():
        nonlocal index
        for record in stream:
            text = record["text"].strip()
            if not text:
                continue
            index += 1
            document = tokenize(text, "train", index)
            if document:
                yield document
            if counts["train"] >= target_tokens:
                return

    shards = {}
    for which, source in (("heldout", heldout), ("train", train_documents())):
        prefix = out_dir / f"reex15-{which}-bin"
        write_packed_shard(source, prefix, sequence_length=sequence_length,
                           vocab_size=vocab_size, tokenizer_id="hf:gpt2",
                           source="fineweb-edu")
        shards[which] = {"packed": str(prefix), "tokens": counts[which]}
        print(f"   {which}: {counts[which]:,} tokens -> {prefix.name}")

    marker.write_text(json.dumps(shards, indent=2))
    if sync is not None:
        sync.push_corpus(out_dir)
    return shards


def build_reex15(donor: str, subfolder: str, to_layers: int):
    """Grow Reex-1 and gate on exact equivalence before anything else."""
    import torch
    from transformers import LlamaForCausalLM

    from src.model.llama_adapter import LlamaProtocolAdapter, grow_llama_depth

    load = {"subfolder": subfolder} if subfolder else {}
    base = LlamaForCausalLM.from_pretrained(donor, dtype=torch.float32, **load).eval()
    grown, report = grow_llama_depth(base, to_layers=to_layers, mode="zero_init")

    ids = torch.randint(0, base.config.vocab_size, (2, 64))
    with torch.no_grad():
        difference = (base(input_ids=ids).logits
                      - grown.eval()(input_ids=ids).logits).abs().max().item()
    if difference > 1e-4:
        raise SystemExit(
            f"GROWTH GATE FAILED: grown model differs from Reex-1 by {difference:.3e}. "
            f"Training from here would discard the 2B tokens already spent."
        )
    print(f"   equivalence gate PASS (max logit diff {difference:.3e}) -- "
          f"Reex-1.5 starts exactly where Reex-1 finished")
    return LlamaProtocolAdapter(grown), report


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hub-repo", required=True,
                        help="Where Reex-1.5 lives, e.g. you/reex-1.5")
    parser.add_argument("--donor", default=REEX1)
    parser.add_argument("--donor-subfolder", default="hf_format")
    parser.add_argument("--to-layers", type=int, default=24)
    parser.add_argument("--target-tokens", type=float, default=1.86e9,
                        help="Tokens for the WHOLE job. Steps are derived from this, "
                             "because the loop counts MICRO-batches: with grad-accum 4 a "
                             "hand-computed step count under-trains by 4x.")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Override the derived step count (rarely needed)")
    parser.add_argument("--eval-batch-size", type=int, default=4,
                        help="Held-out eval batch. At seq 1024 with a 50257 head, batch 32 "
                             "materialises 6.6GB of logits on top of the resident training "
                             "model and OOMs the T4.")
    parser.add_argument("--max-hours", type=float, default=8.0,
                        help="Stop and upload before the session limit kills the run")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="MICRO-batch; the activation peak follows it. batch 8 x seq "
                             "1024 needs 13.6GB and OOMs a 14.6GB T4; batch 4 needs 8.4GB "
                             "and keeps the tensor cores fed far better than batch 2.")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="Micro-batches per optimizer step; batch 4 x accum 2 keeps "
                             "the effective batch at 8.")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.01)
    parser.add_argument("--checkpoint-every", type=int, default=2000,
                        help="Each checkpoint also evaluates the held-out set, "
                             "which at 500 was 18.5%% overhead (4.6h of a 25h job). "
                             "2000 costs 4.6%% and risks 13 minutes on a crash.")
    parser.add_argument("--milestone-every", type=int, default=50000,
                        help="Micro-steps between permanent HF-format milestones. Each is "
                             "0.77GB kept forever; at 5000 a full job would bank ~90 of "
                             "them (~69GB), brushing HF's private-storage quota.")
    parser.add_argument("--eval-max-batches", type=int, default=64,
                        help="Held-out batches scored per in-loop checkpoint (64 x 4 x 1024 "
                             "= 262K tokens, stable to ~3 decimals). Scoring the full 2.2M "
                             "tokens each time costs ~5 GPU-hours across the job. The "
                             "session-end eval always uses the full set.")
    parser.add_argument("--sync-interval-min", type=float, default=25.0)
    parser.add_argument("--corpus-tokens", type=int, default=60_000_000)
    parser.add_argument("--skip-documents", type=int, default=2_400_000,
                        help="Documents Reex-1 consumed; skipped so this trains on new text")
    parser.add_argument("--use-muon", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--no-hub-sync", action="store_true")
    parser.add_argument("--work-dir", default=str(ROOT / "runs" / "reex-1.5"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("=" * 72)
    print("REEX-1.5 -- resumable multi-session pretraining")
    print("=" * 72)
    tokens_per_step = args.batch_size * args.seq_len
    total_steps = args.total_steps or int(args.target_tokens / tokens_per_step)
    args.total_steps = total_steps
    effective_batch = args.batch_size * args.grad_accum
    print(f"device {device} | budget {args.max_hours:.1f}h this session")
    print(f"  micro-batch {args.batch_size} x accum {args.grad_accum} "
          f"= effective batch {effective_batch}")
    print(f"  {tokens_per_step:,} tokens/micro-step x {total_steps:,} steps "
          f"= {total_steps*tokens_per_step/1e9:.2f}B tokens")

    if args.milestone_every and args.milestone_every % args.checkpoint_every:
        # Milestones fire inside the checkpoint hook, so a cadence that is not
        # a multiple of checkpoint_every silently never fires.
        raise SystemExit(
            f"--milestone-every {args.milestone_every} must be a multiple of "
            f"--checkpoint-every {args.checkpoint_every}")

    from src.train.hf_sync import HubSync
    sync = HubSync(args.hub_repo, min_interval_s=args.sync_interval_min * 60,
                   enabled=not args.no_hub_sync)

    print("\n== 1. Where are we? ==")
    resume_path, resume_step = sync.pull_resume_state(work)

    print("\n== 2. Model ==")
    if resume_path is None:
        print(f"   first session: growing {args.donor} to {args.to_layers} layers")
        model, growth = build_reex15(args.donor, args.donor_subfolder, args.to_layers)
        print(f"   growth: {growth.as_dict()}")
    else:
        # Rebuild the same shape; train() loads the weights from the resume state.
        from transformers import LlamaConfig, LlamaForCausalLM

        from src.model.llama_adapter import LlamaProtocolAdapter
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        config = state["config"]
        model = LlamaProtocolAdapter(LlamaForCausalLM(LlamaConfig(
            vocab_size=config["vocab_size"], hidden_size=config["d_model"],
            num_hidden_layers=config["n_layers"],
            num_attention_heads=config["n_heads"],
            num_key_value_heads=config["n_kv_heads"],
            intermediate_size=config["intermediate_size"],
            max_position_embeddings=config["max_seq_len"],
            rms_norm_eps=config.get("rms_norm_eps", 1e-5),
            rope_theta=config.get("rope_theta", 10000.0),
            tie_word_embeddings=True)))
        print(f"   continuing from step {resume_step:,}")

    config = model.config
    params = sum({id(p): p.numel() for p in model.parameters()}.values())
    print(f"   {params:,} parameters, {config['n_layers']} layers, "
          f"{config['n_heads']}q/{config['n_kv_heads']}kv heads")

    print("\n== 3. Corpus (fresh text only) ==")
    shards = build_corpus(work / "corpus", target_tokens=args.corpus_tokens,
                          skip_documents=args.skip_documents,
                          sequence_length=args.seq_len,
                          vocab_size=config["vocab_size"], sync=sync)
    print(f"   train {shards['train']['tokens']:,} | "
          f"heldout {shards['heldout']['tokens']:,} tokens")

    from src.data.budget import check_token_budget
    budget = check_token_budget(unique_tokens=shards["train"]["tokens"],
                                steps=total_steps, batch_size=args.batch_size,
                                sequence_length=args.seq_len, n_params=params)
    print(f"   budget gate: {budget.status.upper()} "
          f"(epochs {budget.epochs:.2f}x, {budget.tokens_per_param:.1f} tokens/param)")
    for warning in budget.warnings:
        print(f"   [warn] {warning}")
    if budget.status == "red":
        for failure in budget.failures:
            print(f"   [FAIL] {failure}")
        raise SystemExit("budget_gate: this plan would measure memorization")

    print("\n== 4. Train ==")
    ledger = work / "reex15-ledger.jsonl"
    # The ledger is append-only history; each session must extend it, not
    # restart it. Without this pull the session begins with an empty file and
    # its final push OVERWRITES the Hub copy with only its own rows.
    if sync.enabled and not ledger.exists():
        try:
            from huggingface_hub import hf_hub_download
            fetched = hf_hub_download(sync.repo_id, "ledger/reex15-ledger.jsonl",
                                      repo_type="model", token=sync.token)
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.write_bytes(Path(fetched).read_bytes())
            print(f"   ledger restored from Hub "
                  f"({sum(1 for _ in ledger.open())} prior rows)")
        except Exception as error:
            print(f"   no prior ledger on Hub ({type(error).__name__}); starting one")

    session_t0 = time.time()
    base_step = resume_step or 0

    def on_checkpoint(step: int, checkpoint_path: Path) -> None:
        # Throughput and ETA from measured progress -- the one number no
        # pre-flight audit could produce, printed as soon as it exists.
        done = step - base_step
        if done > 0:
            rate = done * tokens_per_step / max(1.0, time.time() - session_t0)
            remaining_h = (total_steps - step) * tokens_per_step / rate / 3600
            print(f"[eta] step {step:,}/{total_steps:,} | {rate:,.0f} tok/s | "
                  f"~{remaining_h:.1f} GPU-hours left "
                  f"(~{remaining_h / args.max_hours:.1f} more sessions)")
        sync.push_resume_state(checkpoint_path, step=step)
        if args.milestone_every and step % args.milestone_every == 0:
            milestone = work / f"milestone-{step:08d}"
            model.model.save_pretrained(milestone)
            sync.push_milestone(milestone, step=step)
            # Local copy served its purpose either way; at 0.77GB apiece,
            # keeping them would eat the session's disk before its time.
            shutil.rmtree(milestone, ignore_errors=True)
        if ledger.exists():
            sync.push_file(ledger, "ledger/reex15-ledger.jsonl")

    from src.train.reference import train

    elapsed = time.time() - started
    result = train(
        shards["train"]["packed"], str(work / "run"),
        vocab_size=config["vocab_size"], d_model=config["d_model"],
        n_layers=config["n_layers"], n_heads=config["n_heads"],
        steps=total_steps, learning_rate=args.learning_rate, seed=17,
        device=device, checkpoint_every=args.checkpoint_every,
        heldout_shard=shards["heldout"]["packed"], use_muon=args.use_muon,
        batch_size=args.batch_size, lr_schedule=True,
        warmup_fraction=args.warmup_fraction, precision=args.precision,
        grad_clip=args.grad_clip, keep_checkpoints=3, grad_accum=args.grad_accum,
        eval_batch_size=args.eval_batch_size, eval_max_batches=args.eval_max_batches,
        # fp16 GPU training is not bit-replayable: backward atomics reorder, so
        # an honest crash-recovery replay lands ~1e-3 off. CPU default (1e-6)
        # would call that a contradiction and kill the recovery.
        replay_tolerance=5e-3,
        ledger_path=str(ledger), run_id="reex-1.5",
        levers_on=["growth"] + (["optimizer"] if args.use_muon else []),
        model_override=model,
        resume=str(resume_path) if resume_path else None,
        max_seconds=max(60.0, args.max_hours * 3600 - elapsed),
        on_checkpoint=on_checkpoint)

    print("\n== 5. Session summary ==")
    print(f"   reached step  : {result['reached_step']:,} / {total_steps:,} "
          f"({result['reached_step']*tokens_per_step/1e9:.3f}B tokens)")
    print(f"   final loss    : {result['final_loss']:.4f}")
    if result.get("eval_scores"):
        print(f"   held-out acc  : {result['eval_scores']['val_acc']:.4f}  "
              f"nll {result['eval_scores']['val_nll']:.4f}")
    print(f"   health        : {result['health']['status']}")
    print(f"   fp16 overflows: {result['overflow_steps']} "
          f"({result['overflow_rate']:.2%} of steps)")

    final = Path(result["checkpoint"])
    sync.push_resume_state(final, step=result["reached_step"], force=True)
    print(f"   hub uploads   : {sync.status.uploads} "
          f"({sync.status.bytes_sent/1e9:.1f} GB), failures {sync.status.failures}")

    if result["stopped_early"]:
        print("\n   TIME BUDGET REACHED -- state is on the Hub.")
        print("   Re-run this identical command in a new session to continue.")
    else:
        print("\n   TRAINING COMPLETE.")


if __name__ == "__main__":
    main()
