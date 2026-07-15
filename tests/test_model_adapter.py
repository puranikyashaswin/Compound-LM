import pytest


def test_model_adapter_has_explicit_dependency_failure():
    from src.model.reference import torch, require_torch
    if torch is None:
        with pytest.raises(RuntimeError, match="PyTorch is required"):
            require_torch()
    else:
        require_torch()
