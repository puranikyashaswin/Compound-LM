"""Subprocess worker used by the session-resume regression test."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from src.train.reference import train

ap = argparse.ArgumentParser()
ap.add_argument("--shard", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--steps", type=int, required=True)
ap.add_argument("--resume")
args = ap.parse_args()
result = train(args.shard, args.output, vocab_size=32768, d_model=16,
               n_layers=2, n_heads=2, batch_size=2, seed=41, device="cpu",
               checkpoint_every=6, use_muon=True, steps=args.steps,
               resume=args.resume)
print(json.dumps({"losses": result["losses"], "checkpoint": result["checkpoint"]}))
