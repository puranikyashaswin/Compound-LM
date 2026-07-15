def test_reference_module_exports_training_entrypoint():
    from src.train.reference import train
    assert callable(train)
