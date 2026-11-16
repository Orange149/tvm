import copy

import pytest

from analyze_cpu_stage_overlap_validation import analyze_gate_rows, overlap, pair_effect, repeated_pair_inference
from run_cpu_stage_overlap_validation import adapters, condition
import run_cpu_vta_residual_validation as native


def rows(mode="exclusive"):
    result = []
    for frame in range(10):
        row = {"frame_id": frame, "mode": "pipeline", "stage_count": 3,
               "completion_ms": frame * 20 + 16, "cpu_stage_gate_mode": mode}
        for i, offset in enumerate((1, 6, 11)):
            start = frame * 20 + offset
            row.update({f"stage{i}_start_ms": start, f"stage{i}_end_ms": start + 4,
                        f"stage{i}_run_start_ms": start+1, f"stage{i}_run_end_ms": start+3,
                        f"stage{i}_run_ms": 2, f"stage{i}_ms": 4, f"stage{i}_process_cpu_ms": None})
            for name, value in (("request", start-1), ("acquire", start-0.5), ("release", start+4.5), ("wait", 0.5)):
                row[f"stage{i}_cpu_gate_{name}_ms"] = value if i != 1 else None
        result.append(row)
    return result


def test_wait_excluded_and_gate_bypasses_vta():
    m = analyze_gate_rows(rows(), "exclusive", discard=0)
    assert m["ii_ms"] == 20
    assert m["cpu_run_overlap_ms"] == 0
    assert m["stages"][0]["run_mean_ms"] == 2
    assert m["stages"][0]["executed_service_mean_ms"] == 4
    assert m["stages"][0]["gate_wait_mean_ms"] == 0.5
    assert m["stages"][1]["gate_wait_mean_ms"] is None


@pytest.mark.parametrize("fault", ["overlap", "wait_inside", "vta", "missing", "mode", "cpu_attribution"])
def test_rejects_invalid_gate_contract(fault):
    data = rows()
    if fault == "overlap":
        data[0]["stage0_cpu_gate_release_ms"] = 15.5
    elif fault == "wait_inside":
        data[0]["stage0_cpu_gate_acquire_ms"] = 2
    elif fault == "vta":
        data[0]["stage1_cpu_gate_request_ms"] = 4
    elif fault == "missing":
        data.pop(1)
    elif fault == "mode":
        data[0]["cpu_stage_gate_mode"] = "observe"
    else:
        data[0]["stage0_process_cpu_ms"] = 2
    with pytest.raises(ValueError):
        analyze_gate_rows(data, "exclusive", discard=0)


def test_cross_frame_overlap():
    assert overlap([(0, 4), (8, 12)], [(2, 10)]) == 4


def test_adapters_restore_legacy_helpers_even_on_exception():
    build, argv = native.build_runner, native.runner_argv
    with pytest.raises(RuntimeError):
        with adapters():
            assert native.build_runner is not build
            raise RuntimeError("fixture")
    assert native.build_runner is build and native.runner_argv is argv


def test_pair_rejects_thread_or_runtime_drift():
    a = {name: "fixture" for name in ("boot_id", "runner_sha256", "runtime_bindings", "program_identity",
                                     "numeric_input_sha256", "threads", "affinity", "spin_count", "queue_depth", "cpu_frequency_khz")}
    a.update(block=0, gate_metrics=analyze_gate_rows(rows("observe"), "observe", discard=0))
    b = copy.deepcopy(a)
    b["gate_metrics"]["gate_mode"] = "exclusive"
    assert pair_effect(a, b)["front_run_reduction_fraction"] == 0
    b["threads"] = [4, 1, 1]
    with pytest.raises(ValueError, match="threads"):
        pair_effect(a, b)


def test_cpu_thread_configuration_fixed():
    a, b = condition("A", "observe"), condition("B", "exclusive")
    assert a["threads"] == b["threads"] == [4, 1, 4]
    assert a["affinity"] == b["affinity"] == [[0, 1, 2, 3]] * 3


def repeated_fixture():
    pairs = []
    for boot in range(3):
        for block in range(2):
            a = {name: "fixture" for name in ("runner_sha256", "runtime_bindings", "program_identity",
                                              "numeric_input_sha256", "threads", "affinity", "spin_count", "queue_depth", "cpu_frequency_khz")}
            a.update(boot_id=str(boot), block=block, gate_metrics=analyze_gate_rows(rows("observe"), "observe", discard=0))
            a["gate_metrics"]["cpu_run_overlap_ms"] = 10
            b = copy.deepcopy(a)
            b["gate_metrics"]["gate_mode"] = "exclusive"
            b["gate_metrics"]["cpu_run_overlap_ms"] = 0
            b["gate_metrics"]["stages"][0]["run_mean_ms"] = 1
            pairs.append({"boot_id": str(boot), "block": block, "control": a, "exclusive": b})
    return pairs


def test_three_boot_inference_uses_boot_support():
    result = repeated_pair_inference(repeated_fixture())
    assert result["primary_gate_passed"]
    assert result["front_run_ci95"]["mean_ms"] == 1
    assert result["front_run_ci95"]["degrees_of_freedom"] == 2
    with pytest.raises(ValueError, match="three boots"):
        repeated_pair_inference(repeated_fixture()[:4])


def test_replication_rejects_cross_boot_frequency_drift():
    pairs = repeated_fixture()
    for arm in ("control", "exclusive"):
        pairs[-1][arm]["cpu_frequency_khz"] = 1
    with pytest.raises(ValueError, match="Cross-boot"):
        repeated_pair_inference(pairs)
