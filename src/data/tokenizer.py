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


class HuggingFaceTokenizer(TokenizerAdapter):
    """A pinned public sub-word tokenizer, e.g. ``gpt2``.

    Exists so a real run can measure language modelling rather than the
    fallback's hash-bucket prediction. The tokenizer's identity and vocabulary
    size go into the datasheet, so switching it is a visible lineage change
    rather than a silent one.
    """

    def __init__(self, name: str = "gpt2"):
        self.name = name
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "transformers is required for HuggingFaceTokenizer; install it or "
                "use tokenizer_id='fallback-v1'"
            ) from error
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer_id = f"hf:{name}"

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer)

    def metadata(self) -> dict[str, Any]:
        return {"tokenizer_id": self.tokenizer_id, "name": self.name,
                "vocab_size": self.vocab_size}


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
