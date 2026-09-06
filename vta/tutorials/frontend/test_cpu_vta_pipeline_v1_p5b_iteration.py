#!/usr/bin/env python3
"""Unit tests for the P5B measured CPU-segment iteration."""

import iterate_cpu_vta_pipeline_v1_p5b as iteration
import review_cpu_vta_pipeline_v1_p5b_iteration as review


def test_pipeline_summary_uses_scored_completion_span(tmp_path):
    path = tmp_path / "result.jsonl"
    rows = []
    for index, end in enumerate((10.0, 20.0, 32.0, 44.0, 56.0)):
        rows.append(
            {
                "stage2_end_ms": end,
                "stage0_run_ms": 4.0,
                "stage1_run_ms": 5.0,
                "stage2_run_ms": 6.0,
                "raw_outputs": [{"fnv1a64": "same"}],
            }
        )
    path.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n")
    summary = iteration.summarize_pipeline(path, warmup=2)
    assert summary["scored_count"] == 3
    assert summary["cycle_ms"] == 12.0
    assert summary["fps"] == 1000.0 / 12.0


def test_segment_update_is_exact_and_does_not_mutate_input():
    context = {
        "segment_costs": {
            "cpu:00:02": {
                "service_ms_by_threads": {str(i): float(i) for i in range(1, 5)},
                "host_core_demand_ms_by_threads": {str(i): float(i * 2) for i in range(1, 5)},
            },
            "cpu:03:04": {
                "service_ms_by_threads": {"1": 7.0},
                "host_core_demand_ms_by_threads": {"1": 8.0},
            },
        }
    }
    measurements = [
        {
            "threads": i,
            "serial": {"stage0_run_ms": 10.0 + i, "stage0_run_process_cpu_ms": 20.0 + i},
        }
        for i in range(1, 5)
    ]
    revised, original = iteration.apply_segment_observations(context, measurements)
    assert context["segment_costs"]["cpu:00:02"]["service_ms_by_threads"]["4"] == 4.0
    assert revised["segment_costs"]["cpu:00:02"]["service_ms_by_threads"]["4"] == 14.0
    assert revised["segment_costs"]["cpu:03:04"] == context["segment_costs"]["cpu:03:04"]
    assert original["service_ms_by_threads"]["1"] == 1.0


def test_global_cpu_correction_updates_cpu_only_and_preserves_ratios():
    context = {
        "profile_segments": {
            "cpu:00:02": {"device": "cpu"},
            "cpu:03:04": {"device": "cpu"},
            "vta:05:06": {"device": "vta"},
        },
        "segment_costs": {
            "cpu:00:02": {
                "service_ms_by_threads": {str(i): 10.0 * i for i in range(1, 5)},
                "host_core_demand_ms_by_threads": {str(i): 20.0 * i for i in range(1, 5)},
            },
            "cpu:03:04": {
                "service_ms_by_threads": {str(i): 5.0 * i for i in range(1, 5)},
                "host_core_demand_ms_by_threads": {str(i): 8.0 * i for i in range(1, 5)},
            },
            "vta:05:06": {
                "service_ms_by_threads": {"1": 7.0},
                "host_core_demand_ms_by_threads": {"1": 2.0},
            },
        },
    }
    measurements = [
        {
            "threads": i,
            "serial": {"stage0_run_ms": 20.0 * i, "stage0_run_process_cpu_ms": 40.0 * i},
        }
        for i in range(1, 5)
    ]
    revised, correction = iteration.apply_global_cpu_thread_correction(context, measurements)
    assert revised["segment_costs"]["cpu:03:04"]["service_ms_by_threads"]["3"] == 30.0
    assert revised["segment_costs"]["cpu:03:04"]["host_core_demand_ms_by_threads"]["3"] == 48.0
    assert revised["segment_costs"]["vta:05:06"] == context["segment_costs"]["vta:05:06"]
    assert correction["updated_cpu_segment_count"] == 2


def test_validation_summary_uses_last_stage_for_five_stage_pipeline(tmp_path):
    path = tmp_path / "five_stage.jsonl"
    rows = []
    for index, end in enumerate((10.0, 20.0, 35.0, 50.0, 65.0)):
        row = {"stage_count": 5, "stage4_end_ms": end, "top1": 282}
        for stage in range(5):
            row["stage{}_run_ms".format(stage)] = float(stage + 1)
        rows.append(row)
    path.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n")
    result = review.summarize_validation(path, warmup=2)
    assert result["cycle_ms"] == 15.0
    assert result["stages"][4]["run_ms_median"] == 5.0
