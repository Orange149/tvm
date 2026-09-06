#!/usr/bin/env python3
"""Unit tests for CPU-VTA Pipeline V1 P2 fitting and cost accounting."""

import pytest

import run_cpu_vta_pipeline_v1_p2 as p2


def test_through_origin_fit_is_additive_and_has_no_intercept():
    model = p2.fit_through_origin(
        [
            {"gops": 1.0, "run_ms": 2.0},
            {"gops": 3.0, "run_ms": 6.0},
        ],
        "gops",
        "run_ms",
        "ms_per_logical_gop",
    )
    assert model["ms_per_logical_gop"] == pytest.approx(2.0)
    assert model["intercept_ms"] == 0.0
    assert model["fit_mape"] == pytest.approx(0.0)


def test_cost_ledger_clamps_legacy_negative_elapsed_without_hiding_it():
    commands = p2.normalize_cost_commands(
        [
            {"case_id": "old", "elapsed_s": -0.7, "status": "completed"},
            {"case_id": "new", "elapsed_s": 1.2, "status": "completed"},
        ]
    )
    assert commands[0]["elapsed_s"] == 0.0
    assert commands[0]["elapsed_s_raw"] == pytest.approx(-0.7)
    assert commands[0]["timing_anomaly"] == "negative_legacy_elapsed_clamped"
    assert commands[1]["elapsed_s"] == pytest.approx(1.2)
    assert "timing_anomaly" not in commands[1]
