from src.train.systems import SystemsPolicy, inspect_runtime


def test_systems_report_never_fakes_capabilities():
    report = inspect_runtime(SystemsPolicy(precision="fp8", compile=True))
    assert "active" in report and "warnings" in report
    assert isinstance(report["active"].get("compile"), bool)
    assert isinstance(report["active"].get("fp8"), bool)
