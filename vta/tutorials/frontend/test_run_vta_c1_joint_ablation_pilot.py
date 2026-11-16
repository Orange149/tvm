"""Check the real-board pilot's frozen identity and qualification-before-timing gate."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_vta_c1_joint_ablation_pilot as pilot


def test_frozen_groups_reuse_identical_choices_and_preserve_joint_inputs(tmp_path):
    store = pilot.create(tmp_path / "pilot")
    contract = store.contract
    assert len(contract["programs"]) == 5
    groups = contract["groups"]
    assert groups["full"]["selected_program"] == groups["without_internal_features"]["selected_program"]
    assert groups["paper_style_baseline"]["selected_program"] == groups["without_cpu_pool"]["selected_program"]
    assert contract["extra_quantize_calls"] == 0
    assert contract["candidate_board_timings_consumed_for_selection"] is False
    assert len({r["input"]["sha256"] for r in contract["input_records"]}) == 3
    assert len({tuple(order) for order in contract["performance_orders"]}) == 3
    assert all(set(order) == set(contract["programs"]) for order in contract["performance_orders"])
    with pytest.raises(FileExistsError):
        pilot.create(tmp_path / "pilot")


@pytest.mark.parametrize("mode,runs", [("serial", "6"), ("pipeline", "6"), ("performance", "307")])
def test_native_arguments_preserve_full_thread_map_and_disable_timing_diagnostics(tmp_path, mode, runs):
    contract = {"remote": "/var/volatile/pilot", "programs": {"example": {"path": [
        ["cpu:00:02", 4], ["vta:03:17", 1], ["cpu:18:20", 2]]}}}
    manifest = {"programs": {name: {"input_names": ["edge"]} for name in ("cpu_00_02", "vta_03_17", "cpu_18_20")}}
    args = pilot.native_args(contract, manifest, "example", mode, "/var/volatile/pilot/session")
    assert args[args.index("--runs") + 1] == runs
    assert args[args.index("--stage0-runtime-num-threads") + 1] == "4"
    assert args[args.index("--stage1-runtime-num-threads") + 1] == "1"
    assert args[args.index("--stage2-runtime-num-threads") + 1] == "2"
    assert args[args.index("--stage2-cpu-affinity") + 1] == "0,1"
    assert ("--serial" in args) == (mode == "serial")
    assert ("--output-dump-dir" in args) == (mode != "performance")
    assert ("--c1-memory-audit" in args) == (mode != "performance")
    assert "--defer-result-write" in args
    assert "--shared-buffer-plan" not in args


def test_failed_qualification_prevents_every_performance_window(tmp_path, monkeypatch):
    output = tmp_path / "pilot"
    (output / "analysis").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(pilot, "create", lambda *args: SimpleNamespace(contract={}))
    monkeypatch.setattr(pilot, "build_programs", lambda store: {})
    monkeypatch.setattr(pilot, "deploy", lambda store, manifest: "boot")

    def collect(store, manifest, boot, key, mode, tag):
        calls.append((key, mode))
        if mode == "pipeline":
            raise ValueError("frontier differs")
        return {}

    monkeypatch.setattr(pilot, "collect_session", collect)
    with pytest.raises(ValueError, match="frontier differs"):
        pilot.run(output)
    assert calls == [("full_one_island", "serial"), ("full_one_island", "pipeline")]
    failure = json.loads((output / "analysis/failure.json").read_text())
    assert failure["performance_results_valid"] is False
    assert not (output / "analysis/result.json").exists()


def test_changed_artifact_is_rejected(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"before")
    reference = pilot.ref(path)
    path.write_bytes(b"after")
    with pytest.raises(ValueError, match="artifact drift"):
        pilot.verify(reference)


def test_service_summary_keeps_unknown_mutex_wait_unknown():
    from analyze_vta_c1_joint_ablation_pilot import resource_summary
    row = {"stage0_ms": 10, "stage1_ms": 20, "stage2_ms": 30,
           "stage1_run_ms": 18, "stage1_vta_mutex_wait_ms": None,
           "stage0_get_ms": 1, "stage1_set_ms": 2, "stage1_get_ms": 3, "stage2_set_ms": 4}
    result = resource_summary([row], [("cpu:00:02", 4), ("vta:03:17", 1), ("cpu:18:20", 2)])
    assert result["max_cpu_service_ms"] == 30
    assert result["vta_run_sum_ms"] == 18
    assert result["boundary_api_sum_ms"] == 10
    assert result["vta_mutex_wait_sum_ms"] is None
