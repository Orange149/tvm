#!/usr/bin/env python3

from analyze_vta_y10_same_space_online_ablation import (
    aggregate_strategy,
    invocation_cost,
    numeric_summary,
)


def test_invocation_cost_uses_driver_calls_and_logical_dma():
    profile = {
        "driver_run_calls": 3,
        "load_buffer_2d_bytes": 10,
        "store_buffer_2d_bytes": 2,
        "load_buffer_2d_calls": 4,
        "store_buffer_2d_calls": 1,
    }
    value = invocation_cost({
        "correctness": [{"host_wall_ms": 7.0, "runtime_profile_complete": profile}],
        "timings": [{"host_wall_ms": 11.0, "runtime_profile_complete": profile}],
    })
    assert value == {
        "graph_api_invocations": 2,
        "driver_kernel_invocations": 6,
        "logical_load_bytes": 20,
        "logical_store_bytes": 4,
        "logical_dma_bytes": 24,
        "logical_dma_calls": 10,
        "invocation_host_wall_ms": 18.0,
    }


def test_numeric_summary_reports_range_and_iqr():
    value = numeric_summary([1, 2, 3])
    assert value == {"median": 2.0, "min": 1.0, "max": 3.0, "q1": 1.5, "q3": 2.5}


def test_aggregate_keeps_missing_exact_target_as_censored():
    base = {
        "seed": 1,
        "outer_process_seconds_before_summary_write": 10.0,
        "first_exact_oracle": None,
        "first_oracle_plus_2pct": {"dispatch": 2, "outer_elapsed_seconds": 8.0},
        "total_cost": {
            "complete_candidate_build_action_seconds": 1.0,
            "graph_api_invocations": 2,
            "driver_kernel_invocations": 3,
            "logical_dma_bytes": 4,
            "logical_dma_calls": 5,
            "invocation_host_wall_ms": 6.0,
        },
        "observed_terminal_best": {"median_paired_latency_ratio": 1.0},
        "fixed_pool_terminal_best": {"regret_percent": 1.0},
    }
    value = aggregate_strategy([base])
    assert value["all_runs_hit_exact_oracle"] is False
    assert value["metrics"]["exact_oracle_dispatch"] is None
    assert value["metrics"]["oracle_plus_2pct_dispatch"]["median"] == 2.0
