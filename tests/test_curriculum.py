from src.data.curriculum import build_schedule


def test_curriculum_is_integer_conserving():
    schedule = build_schedule(101)
    assert sum(stage["tokens"] for stage in schedule) == 101
    assert sum(sum(stage["sources"].values()) for stage in schedule) == 101
    assert schedule[0]["stage"] == "foundation"
    assert schedule[-1]["stage"] == "premium"


def test_curriculum_rejects_invalid_budget():
    try:
        build_schedule(0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("expected invalid budget failure")
