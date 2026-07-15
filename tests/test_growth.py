def test_growth_requires_torch_explicitly():
    from src.growth.hyperclone import require_torch
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        try:
            require_torch()
        except RuntimeError as error:
            assert "PyTorch is required" in str(error)
        else:
            raise AssertionError("expected dependency failure")
    else:
        require_torch()


def test_growth_config_is_declared():
    from src.growth.hyperclone import expand_config
    result = expand_config({"model": {"d_model": 128}}, width_multiplier=2)
    assert result["model"]["d_model"] == 256
    assert result["growth"]["source_d_model"] == 128
