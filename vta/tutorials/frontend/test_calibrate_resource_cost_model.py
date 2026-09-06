#!/usr/bin/env python3
"""Tests for strict RAMPS calibration aggregation."""

import pytest

import calibrate_resource_cost_model as calibration


def synthetic_measurements(sessions=3):
    rows = []
    for session in range(1, sessions + 1):
        for threads in (1, 2, 3, 4):
            for cases in calibration.CPU_BUCKET_CASES.values():
                for case_id in cases:
                    rows.append(
                        {
                            "session": session,
                            "thread_count": threads,
                            "cpu_num_threads_actual": threads,
                            "device": "arm_cpu",
                            "case_id": case_id,
                            "status": "ok",
                            "ok": "True",
                            "gops": float(threads),
                            "h2d_ms": 0.2,
                            "kernel_ms": 2.0 / threads,
                            "effective_memory_bandwidth_GBps": 0.75 * threads,
                            "d2h_ms": 0.1,
                            "batch": 1,
                            "channels_in": 64,
                            "height": 56,
                            "width": 56,
                        }
                    )
        for cases in calibration.VTA_BUCKET_CASES.values():
            for case_id in cases:
                rows.append(
                    {
                        "session": session,
                        "thread_count": 1,
                        "device": "vta",
                        "case_id": case_id,
                        "status": "ok",
                        "ok": "True",
                        "gops": 10.0,
                        "h2d_ms": 0.3,
                        "kernel_ms": 3.0,
                        "d2h_ms": 0.2,
                        "submit_ms": 0.1,
                        "sync_ms": 2.9,
                        "pack_ms": 0.05,
                    }
                )
    profiles = []
    for session in range(1, sessions + 1):
        for bucket, case_id in calibration.DMA_BUCKET_CASE.items():
            profiles.append(
                {
                    "session": session,
                    "bucket": bucket,
                    "case_id": case_id,
                    "ps_pl_load_bw_GBps": 1.2,
                    "ps_pl_store_bw_GBps": 0.8,
                    "avg_bytes_per_call": 4096.0,
                    "small_call_ratio": 0.1,
                    "strided_call_ratio": 0.2,
                    "padded_call_ratio": 0.3,
                    "fragmentation_penalty_ms_per_call": 0.002,
                    "load_bytes": 10000.0,
                    "store_bytes": 5000.0,
                    "load_calls": 2.0,
                    "store_calls": 1.0,
                    "device_wait_ms": 1.0,
                    "driver_submit_ms": 0.01,
                    "driver_sync_wait_ms": 0.9,
                }
            )
    return rows, profiles


def test_complete_aggregation_has_no_fallback_and_all_threads():
    rows, profiles = synthetic_measurements()
    model = calibration.aggregate_cost_model(
        rows, profiles, 3, [1, 2, 3, 4], "192.168.1.228", 9090, 5, 20
    )
    calibration.validate_complete_model(model, 3, [1, 2, 3, 4])
    assert model["source"] == "ramps_measured_calibration"
    assert model["cpu_eff_gops"]["residual_3x3_conv"]["3"] == 3.0
    assert set(model["dma"]) == set(calibration.DMA_BUCKET_CASE)


def test_missing_independent_session_is_rejected():
    rows, profiles = synthetic_measurements()
    profiles = [
        row
        for row in profiles
        if not (row["bucket"] == "padded_load" and row["session"] == 3)
    ]
    with pytest.raises(RuntimeError, match="lacks independent sessions"):
        calibration.aggregate_cost_model(
            rows, profiles, 3, [1, 2, 3, 4], "192.168.1.228", 9090, 5, 20
        )
