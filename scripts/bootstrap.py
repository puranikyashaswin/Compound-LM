"""Interactive bootstrap for COMPOUND-LM real training.

`preflight.py` reports whether the execution infrastructure exists. This script
goes one step further: it *prompts* for each missing piece and offers to install
or provision it, then can drive one real end-to-end toy run so the pipeline
produces a genuine (small-scale) result. Nothing is installed, written, or run
without confirmation. No benchmark scores are ever fabricated.

Non-interactive automation:
  --assume-yes            answer "yes" to every install/provision prompt
  --generate-toy-corpus   provision a deterministic synthetic corpus instead of
                          asking for a real one
  --run-toy               after provisioning, execute a real reference run
  --no-input              never call input(); use defaults / flags only
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (import_name, pip_spec, why) — pip_spec None means "cannot pip-install here".
PACKAGES = [
    ("yaml", "pyyaml>=6.0", "canonical config parsing"),
    ("torch", "torch>=2.2", "the training runtime"),
    ("transformers", "transformers>=4.40", "the Reex tokenizer + HF model loading"),
    ("lm_eval", "lm-eval>=0.4.2", "the frozen E-v1 benchmark harness"),
]


class Prompter:
    def __init__(self, *, assume_yes: bool, allow_input: bool):
        self.assume_yes = assume_yes
        self.allow_input = allow_input

    def confirm(self, question: str) -> bool:
        if self.assume_yes:
            print(f"{question} [y/N] y (assume-yes)")
            return True
        if not self.allow_input:
            print(f"{question} [y/N] N (no-input)")
            return False
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")

    def ask(self, question: str, default: str = "") -> str:
        if not self.allow_input:
            return default
        answer = input(f"{question}{f' [{default}]' if default else ''} ").strip()
        return answer or default


def pip_install(spec: str) -> bool:
    print(f"  > {sys.executable} -m pip install {spec}")
    return subprocess.run([sys.executable, "-m", "pip", "install", spec]).returncode == 0


def ensure_packages(prompt: Prompter) -> list[str]:
    """Prompt for and optionally install each missing package. Returns blockers."""
    still_missing: list[str] = []
    for name, spec, why in PACKAGES:
        present = importlib.util.find_spec(name) is not None or shutil.which(name) is not None
        status = "present" if present else "MISSING"
        print(f"{name:<14} {status:<8} — {why}")
        if present:
            continue
        if prompt.confirm(f"  Install {spec} now?") and pip_install(spec):
            if importlib.util.find_spec(name) is None and shutil.which(name) is None:
                still_missing.append(name)
        else:
            still_missing.append(name)
    return still_missing


TOY_DOCUMENTS = [
    "the reference model learns to predict the next token in a packed sequence",
    "compound efficiency levers are measured against a fair reproducible baseline",
    "an append only ledger records every audited run with a checkpoint hash",
    "health checks terminate a run instead of rationalizing a broken result",
    "document boundaries prevent cross document attention inside a packed batch",
    "provenance manifests make a model change auditable and not merely repeatable",
    "the frozen evaluation contract is fixed before the first baseline run begins",
    "muon optimizes eligible two dimensional weights while adamw handles the rest",
]


def provision_corpus(prompt: Prompter, *, generate: bool) -> Path | None:
    """Return a path to a one-document-per-line corpus, or None to skip."""
    if not generate:
        given = prompt.ask("Path to a UTF-8 corpus (one document per line), blank to synthesize:")
        if given:
            path = Path(given)
            if path.exists():
                return path
            print(f"  corpus not found: {path}; falling back to synthetic")
        if not (generate or prompt.confirm("  Generate a deterministic synthetic toy corpus?")):
            return None
    corpus = ROOT / "data" / "toy-v1" / "corpus.txt"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text("\n".join(TOY_DOCUMENTS * 4) + "\n", encoding="utf-8")
    print(f"  wrote synthetic corpus: {corpus.relative_to(ROOT)} ({len(TOY_DOCUMENTS) * 4} lines)")
    return corpus


def build_shard(corpus: Path, *, sequence_length: int = 128) -> Path:
    """Run the real pipeline + packing on the corpus, returning the packed shard."""
    from src.data.packing import pack_shard
    from src.data.pipeline import prepare_documents

    out_dir = ROOT / "data" / "toy-v1"
    docs = corpus.read_text(encoding="utf-8").splitlines()
    datasheet = prepare_documents(docs, source="synthetic-toy", shard_id="toy-v1",
                                  output_dir=out_dir, tokenizer_id="fallback-v1")
    print(f"  datasheet: {datasheet['document_count_kept']} kept / "
          f"{datasheet['token_count']} tokens")
    packed = out_dir / "toy-v1-packed.jsonl"
    stats = pack_shard(out_dir / "toy-v1.jsonl", packed, sequence_length=sequence_length)
    print(f"  packed: {stats['packed_sequences']} sequences of length {sequence_length}")
    return packed


def run_toy(shard: Path, *, steps: int = 40) -> None:
    from src.train.reference import train

    out_dir = ROOT / "runs" / "toy"
    result = train(str(shard), str(out_dir), vocab_size=4096, d_model=64, n_layers=2,
                   n_heads=4, steps=steps, device="cpu",
                   ledger_path=str(ROOT / "ledger" / "runs.jsonl"), run_id="toy-v1")
    print(f"  final_loss={result['final_loss']:.4f} "
          f"health={result['health']['status']} "
          f"checkpoint={Path(result['checkpoint']).relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assume-yes", action="store_true")
    parser.add_argument("--no-input", action="store_true")
    parser.add_argument("--generate-toy-corpus", action="store_true")
    parser.add_argument("--run-toy", action="store_true")
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    prompt = Prompter(assume_yes=args.assume_yes, allow_input=not args.no_input)

    print("== 1. Execution infrastructure ==")
    blockers = ensure_packages(prompt)
    if blockers:
        print(f"\nStill missing: {', '.join(blockers)}. "
              "Install these before a real full-scale run.")

    torch_ready = importlib.util.find_spec("torch") is not None
    print("\n== 2. Data ==")
    corpus = provision_corpus(prompt, generate=args.generate_toy_corpus)

    if corpus is None:
        print("No corpus provisioned; stopping after infrastructure setup.")
        return
    shard = build_shard(corpus)

    print("\n== 3. Toy run ==")
    if not torch_ready:
        print("PyTorch unavailable; cannot execute a real run yet. Data is ready.")
        return
    if args.run_toy or prompt.confirm("Execute a real reference training run now?"):
        run_toy(shard, steps=args.steps)
        print("\nDone. A genuine (toy-scale) result is recorded in ledger/runs.jsonl.")
    else:
        print("Skipped run. Re-run with --run-toy to execute.")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
