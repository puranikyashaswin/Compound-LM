from scripts.preflight import preflight


def test_preflight_is_truthful():
    result = preflight()
    assert isinstance(result["ready_for_real_training"], bool)
    assert isinstance(result["blockers"], list)
    assert set(result["packages"]) >= {"torch", "transformers", "lm_eval"}
