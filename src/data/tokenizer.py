"""Frozen tokenizer adapter boundary."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class TokenizerAdapter:
    """Adapter used by the data pipeline; implementations must be deterministic."""
    tokenizer_id = "abstract"

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {"tokenizer_id": self.tokenizer_id}


class ReexTokenizer(TokenizerAdapter):
    tokenizer_id = "reex-1"

    def __init__(self, path: str | Path):
        self.path = str(path)
        if not Path(path).exists():
            raise FileNotFoundError(f"Reex tokenizer not found: {path}")
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as error:
            raise RuntimeError("transformers is required for ReexTokenizer") from error
        self._tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def metadata(self) -> dict[str, Any]:
        return {"tokenizer_id": self.tokenizer_id, "path": self.path,
                "vocab_size": len(self._tokenizer)}
