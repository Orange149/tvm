#!/usr/bin/env python3
"""Tests for model-independent resource feature reconstruction."""

import json

import calibrate_resource_cost_model as calibration
import resource_aware_dataset as dataset
from test_calibrate_resource_cost_model import synthetic_measurements


def measured_cost_model():
    rows, profiles = synthetic_measurements()
    return calibration.aggregate_cost_model(
        rows, profiles, 3, [1, 2, 3, 4], "192.168.1.228", 9090, 5, 20
    )


def test_yolo_candidate_uses_measured_hardware_services_and_physical_padding():
    candidate = {
        "candidate_id": "synthetic_yolo",
        "stage_devices": "cpu/vta/cpu",
        "stage_plan": [
            {"device": "cpu", "convs": [0, 1, 2]},
            {"device": "vta", "convs": [3, 4, 10]},
            {"device": "cpu", "convs": [12]},
        ],
        "island_count": 1,
        "effective_segment_count": 3,
        "boundary_bytes_est": 1024 * 1024,
        "runtime_config": {"stage0_threads": 3, "stage2_threads": 2, "queue_depth": 2},
    }
    record = dataset.yolo_record_from_candidate(
        candidate, calibration=measured_cost_model(), legacy_analysis=True
    )
    assert record.metadata["hardware_service_source"] == "measured_calibration"
    assert record.stages[0].memory_ms > 0.0
    assert record.stages[1].load_ms > 0.0
    assert record.stages[1].store_ms > 0.0
    assert record.stages[1].compute_ms > 0.0
    assert record.boundary_ms > 0.0


def test_resnet_group_key_holds_outer_span_and_island_count():
    row = {
        "vta_islands": [
            {"start_idx": 1, "end_idx": 4},
            {"start_idx": 8, "end_idx": 12},
        ]
    }
    assert dataset._resnet_group(row) == "resnet:i2_1_12"


def test_resnet_record_loads_direct_vta_profiler_counters(tmp_path):
    output_dir = tmp_path / "candidate"
    serial_dir = output_dir / "profile" / "serial"
    pipeline_dir = output_dir / "profile" / "pipeline"
    serial_dir.mkdir(parents=True)
    pipeline_dir.mkdir(parents=True)
    status = {
        "mem_copy_from_host_calls": 2,
        "mem_copy_from_host_bytes": 2000,
        "mem_copy_from_host_us": 4000,
        "mem_copy_to_host_calls": 1,
        "mem_copy_to_host_bytes": 1000,
        "mem_copy_to_host_us": 2000,
        "flush_cache_us": 600,
        "invalidate_cache_us": 400,
        "load_buffer_2d_bytes": 8000,
        "store_buffer_2d_bytes": 4000,
        "load_buffer_2d_calls": 8,
        "store_buffer_2d_calls": 4,
        "device_run_wait_us": 20000,
    }
    (serial_dir / "single_run_status.json").write_text(json.dumps(status))
    (pipeline_dir / "benchmark_totals_status.json").write_text(json.dumps(status))
    row = {
        "candidate_id": "synthetic_resnet",
        "output_dir": str(output_dir),
        "runs": 2,
        "pipeline_throughput_fps": 10.0,
        "vta_island_count": 1,
        "vta_islands": [{"start_idx": 1, "end_idx": 2}],
        "score_components_json": {
            "stages": [
                {
                    "name": "stage0_vta",
                    "device": "vta",
                    "threads": 1,
                    "static_run_ms_est": 5.0,
                    "static_vta_compute_ms_est": 4.0,
                }
            ]
        },
        "stage0_run_ms": 7.5,
    }
    record = dataset.resnet_record_from_row(row, str(tmp_path / "summary.json"))
    profile = record.metadata["direct_communication_profile"]
    assert profile["available"]
    assert profile["serial"]["direct_copy_ms"] == 6.0
    assert profile["pipeline"]["direct_copy_ms"] == 3.0
    assert profile["pipeline"]["direct_copy_bytes"] == 1500.0
    assert profile["pipeline"]["coherence_ms"] == 0.5
    assert record.metadata["measured_stage_run_ms"] == [7.5]
    provenance = record.metadata["score_input_provenance"]
    assert provenance["workload_features"]["kind"] == "compiler_static"
    assert provenance["service_parameters"]["kind"] == "compiler_static"
    assert provenance["candidate_outcome"]["use"] == "retrospective_evaluation_only"
