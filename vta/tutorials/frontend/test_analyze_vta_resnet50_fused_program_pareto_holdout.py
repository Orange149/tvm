#!/usr/bin/env python3

from analyze_vta_resnet50_fused_program_pareto_holdout import (
    percent_saved,
    resource_totals,
    trials_to_target,
)


def test_resource_totals_count_actual_profiled_kernel_executions():
    profile = {
        "driver_run_calls": 2,
        "load_buffer_2d_bytes": 10,
        "store_buffer_2d_bytes": 3,
        "load_buffer_2d_calls": 4,
        "store_buffer_2d_calls": 1,
    }
    results = [{
        "correctness": [{"host_wall_ms": 7, "runtime_profile_complete": profile}],
        "timings": [{"host_wall_ms": 11, "runtime_profile_complete": profile}],
    }]
    totals = resource_totals(results)
    assert totals["candidate_dispatches"] == 1
    assert totals["graph_api_invocations"] == 2
    assert totals["fpga_kernel_invocations"] == 4
    assert totals["logical_dma_bytes"] == 26
    assert totals["logical_dma_calls"] == 10
    assert totals["host_call_wall_ms"] == 18
    assert percent_saved(2, 8) == 75


def test_trials_to_target_counts_rejected_candidates():
    rows = {
        "bad": {"status": "rejected_fail_closed"},
        "slow": {"status": "passed", "median_paired_latency_ratio": 1.2},
        "near": {"status": "passed", "median_paired_latency_ratio": 1.01},
    }
    assert trials_to_target(["bad", "slow", "near"], rows, 1.0, 0.02) == 3
    assert trials_to_target(["bad", "slow"], rows, 1.0, 0.02) is None
