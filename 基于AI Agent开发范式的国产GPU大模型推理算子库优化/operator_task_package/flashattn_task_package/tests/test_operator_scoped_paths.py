from agent_system.paths import (
    latest_run_dir,
    list_run_dirs,
    operator_results_dir,
    prepare_run_dir,
    write_latest_run,
)


def test_operator_scoped_run_and_result_dirs():
    operator_id = "fused_moe_i8_tn"

    results_dir = operator_results_dir(operator_id)
    assert results_dir.name == operator_id
    assert results_dir.parent.name == "results"


def test_latest_run_can_be_operator_scoped(tmp_path, monkeypatch):
    import agent_system.paths as paths

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    run_dir = prepare_run_dir("run_20260717_000000_test", operator_id="fused_moe_i8_tn")
    write_latest_run(run_dir, operator_id="fused_moe_i8_tn")

    assert latest_run_dir(operator_id="fused_moe_i8_tn") == run_dir
    assert latest_run_dir() == run_dir
    assert list_run_dirs(operator_id="fused_moe_i8_tn") == [run_dir]
