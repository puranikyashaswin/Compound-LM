"""lm-eval adapter loading custom PyTorch state-dicts."""
from __future__ import annotations

import torch
from src.model.reference import ReferenceLM

def load_reference_checkpoint(path: str, device: str = "cpu") -> ReferenceLM:
    """Load ReferenceLM from a custom PyTorch state-dict checkpoint file."""
    state = torch.load(path, map_location=device, weights_only=False)
    config = state["config"]
    model = ReferenceLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"]
    )
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model
