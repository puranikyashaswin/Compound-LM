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
import threading
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

# Single-document packing keeps every row free of document boundaries so the
# Llama adapter can stay on the 2-D SDPA/flash path. Multi-doc rows force a
# 4-D eager mask and were why Reex-1.5 projected ~99 GPU-hours instead of ~25.
CROSS_DOCUMENT_PACKING = False


def _corpus_allows_sdpa_fast_path(out_dir: Path) -> bool:
    """True only when every shard was packed with ``cross_document=False``."""
    metas = list(Path(out_dir).glob("*.meta.json"))
    if not metas:
        return False
    for meta_path in metas:
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            return False
        # Older shards omit the flag; treat missing as multi-doc (the old default).
        if meta.get("cross_document", True):
            return False
    return True


def _invalidate_corpus(out_dir: Path) -> None:
    """Drop a cached corpus so the next build cannot reuse an incompatible pack."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return
    for path in out_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _derive_total_steps(target_tokens: float, batch_size: int, seq_len: int,
                        grad_accum: int, override: int | None) -> int:
    tokens_per_step = batch_size * seq_len
    total_steps = override if override is not None else int(target_tokens / tokens_per_step)
    if total_steps % grad_accum:
        # End the job on an optimizer boundary: a trailing partial accumulation
        # window would run backward() without ever stepping -- wasted compute
        # and a final checkpoint whose optimizer never saw its last micro-steps.
        total_steps += grad_accum - total_steps % grad_accum
    return total_steps


def _corpus_train_tokens(out_dir: Path) -> int | None:
    marker = Path(out_dir) / "corpus.json"
    if not marker.exists():
        return None
    try:
        shards = json.loads(marker.read_text())
        return int(shards["train"]["tokens"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _corpus_is_reusable(out_dir: Path, *, target_tokens: int) -> bool:
    """Refuse multi-doc packs and corpora that undershoot the requested size."""
    if not _corpus_allows_sdpa_fast_path(out_dir):
        return False
    tokens = _corpus_train_tokens(out_dir)
    if tokens is None:
        return False
    # Allow 5% shortfall for stream exhaustion; refuse the 60M leftover when
    # the job asks for 500M.
    if tokens < int(target_tokens * 0.95):
        return False
    return True


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

    Packing is single-document (``cross_document=False``): short docs are padded
    rather than concatenated, so training stays on SDPA/flash instead of the
    eager 4-D document mask.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from src.data.binshard import write_packed_shard

    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "corpus.json"
    if marker.exists() and not _corpus_is_reusable(out_dir, target_tokens=target_tokens):
        have = _corpus_train_tokens(out_dir)
        print(f"   local corpus unusable for this run "
              f"(tokens={have}, need>={int(target_tokens * 0.95):,}); rebuilding")
        _invalidate_corpus(out_dir)
    if marker.exists():
        print("   reusing corpus already built in this session")
        return json.loads(marker.read_text())

    # Kaggle wipes the working directory between sessions, so without this the
    # identical corpus is rebuilt every time -- ~35 minutes each, four times.
    # Downloading the shards back costs a couple of minutes instead.
    if sync is not None and sync.pull_corpus(out_dir):
        if marker.exists() and _corpus_is_reusable(out_dir, target_tokens=target_tokens):
            print("   corpus restored from the Hub (skipped rebuild)")
            return json.loads(marker.read_text())
        if marker.exists():
            have = _corpus_train_tokens(out_dir)
            print(f"   Hub corpus unusable for this run "
                  f"(tokens={have}, need>={int(target_tokens * 0.95):,}); rebuilding")
            _invalidate_corpus(out_dir)

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
                           source="fineweb-edu",
                           cross_document=CROSS_DOCUMENT_PACKING)
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
    parser.add_argument("--batch-size", type=int, default=8,
                        help="MICRO-batch. With single-doc packing the adapter stays on "
                             "SDPA/flash, so batch 8 x seq 1024 fits a 14.6GB T4; the old "
                             "eager 4-D mask needed batch 4 to avoid OOM. Resuming an older "
                             "checkpoint auto-matches its stored batch_size.")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Micro-batches per optimizer step; batch 8 x accum 1 keeps "
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
    parser.add_argument("--corpus-tokens", type=int, default=500_000_000,
                        help="Unique training tokens to pack. 60M fails the budget gate "
                             "for the 1.86B-token Reex-1.5 job; 500M matches the prior "
                             "AMBER plan (~3.7 epochs).")
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
    print(f"device {device} | budget {args.max_hours:.1f}h this session")

    if args.checkpoint_every % args.grad_accum:
        # Checkpoints land on optimizer-step boundaries; a misaligned cadence
        # would silently checkpoint at a fraction of the requested rate.
        raise SystemExit(
            f"--checkpoint-every {args.checkpoint_every} must be a multiple of "
            f"--grad-accum {args.grad_accum}")
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
        # train() refuses a mismatched batch_size; adopt the checkpoint's so a
        # Hub resume of the pre-SDPA run (batch 4) does not die on the new default.
        checkpoint_batch = state.get("batch_size")
        if checkpoint_batch is not None and int(checkpoint_batch) != args.batch_size:
            print(f"   resume checkpoint used batch_size={checkpoint_batch}; "
                  f"matching it (CLI default was {args.batch_size})")
            args.batch_size = int(checkpoint_batch)
            # Pre-SDPA sessions used batch 4 x accum 2. Keep that effective
            # batch when the new defaults (8 x 1) are what landed us here.
            if args.batch_size == 4 and args.grad_accum == 1:
                args.grad_accum = 2
                print("   matching pre-SDPA grad_accum=2 for effective batch 8")
                if args.checkpoint_every % args.grad_accum:
                    raise SystemExit(
                        f"--checkpoint-every {args.checkpoint_every} must be a "
                        f"multiple of --grad-accum {args.grad_accum}")
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

    # Step count depends on micro-batch; derive after any resume batch adoption
    # so the ETA and token budget describe the run that will actually execute.
    override_steps = args.total_steps
    total_steps = _derive_total_steps(
        args.target_tokens, args.batch_size, args.seq_len, args.grad_accum, override_steps)
    args.total_steps = total_steps
    tokens_per_step = args.batch_size * args.seq_len
    effective_batch = args.batch_size * args.grad_accum
    print(f"  micro-batch {args.batch_size} x accum {args.grad_accum} "
          f"= effective batch {effective_batch}")
    print(f"  {tokens_per_step:,} tokens/micro-step x {total_steps:,} steps "
          f"= {total_steps*tokens_per_step/1e9:.2f}B tokens")

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
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
        try:
            fetched = hf_hub_download(sync.repo_id, "ledger/reex15-ledger.jsonl",
                                      repo_type="model", token=sync.token)
        except LocalEntryNotFoundError as error:
            # "Could not reach the Hub" is not "no ledger exists". Starting a
            # fresh ledger on a network blip would overwrite the recorded
            # history at the final push -- fail loud and cheap instead.
            raise SystemExit(f"could not reach the Hub for the prior ledger: {error}. "
                             f"Re-run rather than overwrite the recorded history.")
        except EntryNotFoundError:
            print("   no prior ledger on Hub; starting one")
        else:
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.write_bytes(Path(fetched).read_bytes())
            print(f"   ledger restored from Hub "
                  f"({sum(1 for _ in ledger.open())} prior rows)")

    session_t0 = time.time()
    base_step = resume_step or 0
    upload_thread: threading.Thread | None = None

    def on_checkpoint(step: int, checkpoint_path: Path) -> None:
        nonlocal upload_thread
        # Throughput and ETA from measured progress -- the one number no
        # pre-flight audit could produce, printed as soon as it exists.
        done = step - base_step
        if done > 0:
            rate = done * tokens_per_step / max(1.0, time.time() - session_t0)
            remaining_h = (total_steps - step) * tokens_per_step / rate / 3600
            print(f"[eta] step {step:,}/{total_steps:,} | {rate:,.0f} tok/s | "
                  f"~{remaining_h:.1f} GPU-hours left "
                  f"(~{remaining_h / args.max_hours:.1f} more sessions)")

        # Overlap Hub network I/O with the next training window. Local snapshot
        # work stays on this thread so weights cannot mutate mid-serialize.
        if upload_thread is not None:
            upload_thread.join()

        milestone = None
        if args.milestone_every and step % args.milestone_every == 0:
            milestone = work / f"milestone-{step:08d}"
            model.model.save_pretrained(milestone)

        def push() -> None:
            sync.push_resume_state(checkpoint_path, step=step)
            if milestone is not None:
                sync.push_milestone(milestone, step=step)
                shutil.rmtree(milestone, ignore_errors=True)
            if ledger.exists():
                sync.push_file(ledger, "ledger/reex15-ledger.jsonl")

        upload_thread = threading.Thread(target=push, name=f"hub-push-{step}", daemon=False)
        upload_thread.start()

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
        compile_model=True,
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
        on_checkpoint=on_checkpoint,
        # Kaggle workdirs and intentional SDPA corpus rebuilds change the
        # absolute shard path while keeping the reex15-*-bin leaf name.
        resume_shard_policy="same_name")

    if upload_thread is not None:
        upload_thread.join()
        print("   hub uploads flushed")

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
