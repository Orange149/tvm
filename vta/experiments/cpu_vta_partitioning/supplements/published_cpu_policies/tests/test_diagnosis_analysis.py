"""Native counter availability and diagnostic time-scope regression tests."""
from copy import deepcopy

import pytest

from cpu_policies.diagnosis.analysis import profile_quality, union_ms, _queue_rows


def affinity():
    return {"stages": [{"caller_tid": 10, "effective_affinities": [
        {"tid": 10, "cpus": [0], "cpu_time_delta_ms": 2.0},
        {"tid": 11, "cpus": [1], "cpu_time_delta_ms": 1.0}]}],
        "control_threads": [{"role": "producer", "tid": 20, "cpus": [0, 1, 2, 3]},
                            {"role": "collector", "tid": 21, "cpus": [0, 1, 2, 3]}]}


def profile(zero=False):
    threads = [{"tid": 10, "comm": "runner", "cpus": [0],
                "runtime_ns": 0 if zero else 2000000, "runqueue_wait_ns": 0,
                "timeslices": 0 if zero else 5, "voluntary_switches": 8, "nonvoluntary_switches": 2},
               {"tid": 11, "comm": "runner", "cpus": [1],
                "runtime_ns": 0 if zero else 1000000, "runqueue_wait_ns": 0,
                "timeslices": 0 if zero else 3, "voluntary_switches": 4, "nonvoluntary_switches": 1},
               {"tid": 21, "comm": "runner", "cpus": [0, 1, 2, 3],
                "runtime_ns": 0 if zero else 200000, "runqueue_wait_ns": 0,
                "timeslices": 0 if zero else 1, "voluntary_switches": 3, "nonvoluntary_switches": 0}]
    names = ("runtime_ns", "runqueue_wait_ns", "timeslices", "voluntary_switches", "nonvoluntary_switches")
    return {"schema": "vta_session_profile_v1", "scope": "whole_process_existing_threads",
            "started": True, "fatal_reason": None, "threads": threads,
            "aggregate": {name: sum(row[name] for row in threads) for name in names},
            "start_tid_count": 4, "end_tid_count": 3,
            "pmu": {name: {"available": False, "reason": "unsupported", "raw": 0,
                            "scaled": 0.0, "time_enabled": 0, "time_running": 0}
                    for name in ("cycles", "instructions", "cache_references", "cache_misses")}}


def test_zero_scheduler_metrics_are_unknown_not_proof_of_no_work():
    result = profile_quality(profile(zero=True), affinity(), sched_schedstats=0)
    assert result["scheduler_runtime_available"] is False
    assert result["scheduler_wait_available"] is False
    assert result["threads"][0]["runtime_ns"] == 0
    assert result["threads"][0]["runtime_ms"] is None
    assert result["stage_totals"][0]["runqueue_wait_ms"] is None
    assert result["coverage"]["missing_role_tids"] == [20]
    assert all(event["interpreted_scaled"] is None for event in result["pmu"].values())


def test_positive_runtime_does_not_make_all_zero_wait_available():
    result = profile_quality(profile(), affinity())
    assert result["scheduler_runtime_available"]
    assert not result["scheduler_wait_available"]
    assert result["stage_totals"][0]["runtime_ms"] == 3
    result = profile_quality(profile(), affinity(), sched_schedstats="1\n")
    assert result["scheduler_wait_available"]
    assert result["stage_totals"][0]["runqueue_wait_ms"] == 0
    assert result["stage_totals"][0]["complete_stage_thread_coverage"]
    assert result["threads"][1]["role"] == "worker"


def test_nonzero_wait_is_available_without_sysctl_assumption():
    doc = profile()
    doc["threads"][0]["runqueue_wait_ns"] = 3000000
    doc["aggregate"]["runqueue_wait_ns"] = 3000000
    result = profile_quality(doc, affinity())
    assert result["scheduler_wait_available"]
    assert result["threads"][0]["wait_fraction_of_runtime_plus_wait"] == .6


def test_pmu_available_but_never_running_is_unavailable():
    doc = profile()
    doc["pmu"]["cycles"].update(available=True, reason=None, time_enabled=100, time_running=0)
    result = profile_quality(doc, affinity())
    assert result["pmu"]["cycles"]["usable"] is False
    assert result["pmu"]["cycles"]["unavailable_reason"] == "counter_never_scheduled"
    doc["pmu"]["cycles"].update(raw=20, scaled=40.0, time_running=50)
    result = profile_quality(doc, affinity())
    assert result["pmu"]["cycles"]["usable"] is True
    assert result["pmu"]["cycles"]["running_fraction"] == .5


@pytest.mark.parametrize("change", ["schema", "negative", "aggregate", "affinity", "duplicate", "pmu_time"])
def test_corrupt_probe_is_rejected(change):
    doc = profile()
    if change == "schema":
        doc["schema"] = "unknown"
    elif change == "negative":
        doc["threads"][0]["runtime_ns"] = -1
    elif change == "aggregate":
        doc["aggregate"]["runtime_ns"] += 1
    elif change == "affinity":
        doc["threads"][0]["cpus"] = [1]
    elif change == "duplicate":
        doc["threads"].append(deepcopy(doc["threads"][0]))
    else:
        doc["pmu"]["cycles"]["time_running"] = 1
    with pytest.raises(ValueError):
        profile_quality(doc, affinity())


def test_native_fallback_is_unavailable_not_fabricated_zeros():
    result = profile_quality({"schema": "vta_session_profile_v1", "fatal_reason": "unsupported_platform"}, affinity())
    assert not result["usable"] and result["threads"] == []
    assert result["coverage"]["missing_role_tids"] == [10, 11, 20, 21]


def test_window_coverage_clips_all_events_and_merges_overlap():
    assert union_ms([(-1, 3), (2, 5), (6, 12), (20, 30)], 0, 10) == 9
    assert union_ms([], 0, 10) == 0


def test_queue_dequeue_can_precede_push_return():
    event = {"edge_index": 0, "push_request_ms": 1., "enqueue_ms": 2.,
             "dequeue_ms": 3., "push_return_ms": 4.}
    rows = _queue_rows({"diagnosis_detail": True, "diagnosis_case": "static_first", "diagnosis_block": "detailed"},
                       [{"frame_id": 0, "queue_timing_enabled": True, "queue_timings": [event]}])
    assert rows[0]["queue_residence_ms"] == 1.
    event["dequeue_ms"] = 1.5
    with pytest.raises(ValueError, match="causal"):
        _queue_rows({"diagnosis_detail": True, "diagnosis_case": "static_first", "diagnosis_block": "detailed"},
                    [{"frame_id": 0, "queue_timing_enabled": True, "queue_timings": [event]}])


def test_analysis_binds_raw_files_and_keeps_global_vs_frame_scopes(tmp_path, monkeypatch):
    import hashlib
    import json
    from cpu_policies.diagnosis import analysis
    from cpu_policies.open_resources.domain import candidate, component, component_key, stage

    root, bundle = tmp_path / "run", tmp_path / "bundle"
    plans = bundle / "plans"
    plans.mkdir(parents=True)
    root.mkdir()
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    cfg = candidate([stage("cpu", 0, 3, 1, [0]), stage("vta", 3, 20, 1, [0, 1, 2, 3]),
                     stage("cpu", 20, 21, 1, [1])])
    cost = {"service_ms": .6, "set_ms": .1, "run_ms": .4, "get_ms": .1,
            "run_core_ms": .4, "boundary_core_ms": .2, "set_core_ms": .1, "get_core_ms": .1}
    baseline = {component_key(component(row, cfg)): {"cost": cost} for row in cfg["stages"]}
    write(plans / "baseline_costs.json", baseline)
    write(plans / "programs.json", {})
    write(plans / "baseline_predictions.json", {cfg["id"]: {"score_ms": 5.}})
    write(root / "state.json", {"bundle": str(bundle), "bundle_id": "test", "boot_id": "boot"})
    write(root / "scheduler_capabilities.json", {"sched_schedstats": 1})
    affinity_data = {"stages": [
        {"caller_tid": 10 + i, "effective_affinities": [{"tid": 10 + i, "cpus": row["mask"], "cpu_time_delta_ms": 20.}]}
        for i, row in enumerate(cfg["stages"])], "control_threads": []}
    raw = root / "raw" / "boot" / "test-task"
    raw.mkdir(parents=True)
    frames = []
    for n in range(28):
        frame = {"frame_id": n, "completion_index": n, "input_index": 0, "completion_ms": n * 10. + 8.}
        for index, row in enumerate(cfg["stages"]):
            start = n * 10. + index * 2.
            frame.update({f"stage{index}_ms": 1.2, f"stage{index}_set_ms": .1,
                          f"stage{index}_run_ms": 1., f"stage{index}_get_ms": .1,
                          f"stage{index}_start_ms": start, f"stage{index}_end_ms": start + 1.2,
                          f"stage{index}_run_start_ms": start + .1, f"stage{index}_run_end_ms": start + 1.1,
                          f"stage{index}_queue_pop_ms": start})
            if row["device"] == "vta":
                frame.update({f"stage{index}_vta_mutex_acquire_ms": start,
                              f"stage{index}_vta_mutex_release_ms": start + 1.2})
        frames.append(frame)
    (raw / "result.jsonl").write_text("".join(json.dumps(frame) + "\n" for frame in frames))
    write(raw / "affinity.json", affinity_data)
    document = {"task_id": "test-task", "kind": "performance", "configuration": cfg,
                "warmup": 8, "scored": 16, "tail": 4, "diagnosis_case": "static_first",
                "diagnosis_block": "plain_before", "diagnosis_detail": False}
    files = {str(path.relative_to(root)): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
             for path in raw.iterdir()}
    receipt = root / "accepted" / "boot" / "test-task.json"
    write(receipt, {"task": document, "boot_id": "boot", "summary": {"ii_ms": 10.}, "files": files})
    write(root / "records" / "boot" / "performance.json", {"document": document, "boot_id": "boot",
          "summary": {"ii_ms": 10., "_receipt": {"path": str(receipt.relative_to(root)),
                      "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}}})
    monkeypatch.setattr(analysis, "_plot", lambda *args: None)
    result = analysis.analyze(root)
    report = json.loads((root / "analysis" / "analysis.json").read_text())
    assert result["performance_session_count"] == 1
    assert report["performance_windows"][0]["archived_predicted_ii_ms"] == 5.
    assert report["performance_windows"][0]["measured_to_archived_predicted_ii_ratio"] == 2.
    assert report["stage_comparison"][0]["pipeline_to_old_isolated_service_ratio"] == 2.
    assert report["stage_comparison"][0]["sample_frames"] == 16
    assert .11 < report["performance_windows"][0]["vta_mutex_held_coverage"] < .13
    (raw / "result.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="evidence changed"):
        analysis.analyze(root)
