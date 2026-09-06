#!/usr/bin/env python3
"""Tests for publication-valid controlled identification aggregation."""

import json

import run_ramps_resnet_identification as identification


def config():
    return {
        "config_id": "a01_q2_tuned",
        "candidate_id": "candidate",
        "stage_count": 3,
        "queue_depth": 2,
        "cpu_thread_policy": "tuned",
    }


def result(session, with_cpu_demand=True):
    row = {
        "config_id": "a01_q2_tuned",
        "session": session,
        "status": "ok",
        "passes_correctness_gate": True,
        "pipeline_throughput_fps": 10.0,
        "measured_cycle_ms": 100.0 + session,
        "stage_ms_summary": json.dumps([{"ms": 20.0}, {"ms": 30.0}, {"ms": 40.0}]),
    }
    if with_cpu_demand:
        row.update(
            {
                "stage_core_demand_ms_json": json.dumps([12.0 + session, 2.0, 18.0 + session]),
                "stage_cpu_time_scope": "exclusive_process_cpu_time",
            }
        )
    return row


def test_three_exclusive_cpu_demand_sessions_are_complete_but_remain_pilot():
    row = identification.aggregate_configs(
        [config()], [result(1), result(2), result(3)]
    )[0]
    assert row["measurement_complete"] is True
    assert row["publication_valid"] is False
    assert row["evidence_level"] == "pilot_fixed72"
    assert row["cpu_demand_valid"] is True
    assert json.loads(row["stage_core_demand_ms_json"]) == [14.0, 2.0, 20.0]


def test_missing_cpu_demand_is_not_publication_valid():
    row = identification.aggregate_configs(
        [config()], [result(1), result(2), result(3, with_cpu_demand=False)]
    )[0]
    assert row["publication_valid"] is False
    assert row["measurement_complete"] is False
    assert row["cpu_demand_valid"] is False
