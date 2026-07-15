"""Report whether the environment can execute real COMPOUND-LM training."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys


def preflight() -> dict:
    packages = {name: importlib.util.find_spec(name) is not None
                for name in ("torch", "transformers", "datasets", "datatrove", "lm_eval", "yaml")}
    commands = {name: shutil.which(name) is not None for name in ("lm_eval", "git")}
    blockers = []
    if not packages["torch"]: blockers.append("torch")
    if not packages["transformers"]: blockers.append("transformers")
    if not packages["lm_eval"] and not commands["lm_eval"]: blockers.append("lm-eval")
    return {"python": sys.version, "packages": packages, "commands": commands,
            "ready_for_real_training": not blockers, "blockers": blockers}


if __name__ == "__main__":
    print(json.dumps(preflight(), indent=2))
