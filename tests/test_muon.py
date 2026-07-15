import pytest


def test_muon_has_explicit_dependency_gate():
    from src.optim.muon import require_torch
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(RuntimeError, match="PyTorch is required"):
            require_torch()
    else:
        require_torch()


def test_parameter_partition_is_conservative():
    from src.optim.muon import partition_named_parameters

    class P:
        def __init__(self, ndim): self.ndim = ndim

    partition = partition_named_parameters([
        ("blocks.0.mlp.weight", P(2)),
        ("token_embedding.weight", P(2)),
        ("blocks.0.norm.weight", P(1)),
        ("lm_head.weight", P(2)),
    ])
    assert partition.muon == ("blocks.0.mlp.weight",)
    assert set(partition.adamw) == {"token_embedding.weight", "blocks.0.norm.weight", "lm_head.weight"}
