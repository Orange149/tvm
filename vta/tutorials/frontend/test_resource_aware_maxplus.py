#!/usr/bin/env python3
"""Unit tests for the resource-aware Max-Plus model."""

import math

import pytest

import resource_aware_maxplus as ramps


def record(stages, queue_depth=2, **kwargs):
    return ramps.PipelineRecord(
        model="synthetic",
        candidate_id=kwargs.pop("candidate_id", "candidate"),
        source_path="synthetic",
        group_id=kwargs.pop("group_id", "group"),
        measured_cycle_ms=kwargs.pop("measured_cycle_ms", None),
        queue_depth=queue_depth,
        stages=stages,
        effective_segment_count=len(stages),
        **kwargs,
    )


def test_cpu_roofline_uses_slower_compute_or_memory_path():
    compute_bound = ramps.cpu_service_ms(10e9, 1e6, 10.0, 1.0, launch_ms=1.0)
    memory_bound = ramps.cpu_service_ms(1e6, 1e9, 10.0, 1.0, launch_ms=1.0)
    assert math.isclose(compute_bound["service_ms"], 1001.0)
    assert math.isclose(memory_bound["service_ms"], 1001.0)


def test_physical_compute_rejects_black_box_effective_stage_gops():
    black_box = {
        "measurement_kind": "effective_stage_black_box",
        "effective_stage_gops": 12.0,
        "eligible_for_physical_compute_model": False,
    }
    with pytest.raises(ValueError, match="not eligible"):
        ramps.physical_compute_ms_from_calibration(1e9, black_box)


def test_physical_compute_accepts_only_component_level_rate():
    component = {
        "measurement_kind": "vta_compute_slope",
        "compute_gops": 10.0,
        "eligible_for_physical_compute_model": True,
    }
    assert math.isclose(ramps.physical_compute_ms_from_calibration(1e9, component), 100.0)


def test_vta_overlap_factor_interpolates_full_overlap_and_serial_execution():
    full_overlap = ramps.vta_service_ms(
        1e9, 10.0, 100e6, 100e6, 1.0, 1.0, overlap_factor=0.0
    )
    no_overlap = ramps.vta_service_ms(
        1e9, 10.0, 100e6, 100e6, 1.0, 1.0, overlap_factor=1.0
    )
    assert math.isclose(full_overlap["service_ms"], 100.0)
    assert math.isclose(no_overlap["service_ms"], 300.0)


def test_irregular_dma_calls_reduce_effective_bandwidth():
    contiguous = ramps.effective_dma_bandwidth_gbps(2.0, 16384)
    fragmented = ramps.effective_dma_bandwidth_gbps(
        2.0, 512, small_call_ratio=1.0, strided_call_ratio=0.5, padded_call_ratio=0.5
    )
    assert fragmented < contiguous


def test_balanced_three_stage_pipeline_cycle_is_one_stage_service():
    stages = [
        ramps.StageService("cpu0", "cpu", threads=1, compute_ms=10.0),
        ramps.StageService("vta1", "vta", compute_ms=10.0),
        ramps.StageService("cpu2", "cpu", threads=1, compute_ms=10.0),
    ]
    prediction = ramps.predict_record(record(stages, queue_depth=1))
    assert math.isclose(prediction["maxplus_cycle_ms"], 10.0)
    assert prediction["bottleneck_cycle"]["name"].startswith("stage:")


def test_multiple_vta_islands_share_one_serial_mutex_cycle():
    stages = [
        ramps.StageService("cpu0", "cpu", threads=1, compute_ms=8.0),
        ramps.StageService("vta1", "vta", compute_ms=7.0),
        ramps.StageService("cpu2", "cpu", threads=1, compute_ms=8.0),
        ramps.StageService("vta3", "vta", compute_ms=9.0),
        ramps.StageService("cpu4", "cpu", threads=1, compute_ms=8.0),
    ]
    prediction = ramps.predict_record(record(stages))
    assert math.isclose(prediction["maxplus_cycle_ms"], 16.0)
    assert prediction["bottleneck_cycle"]["resource"] == "vta_mutex"


def test_lightweight_scores_add_single_vta_and_communication_in_stages():
    stages = [
        ramps.StageService("cpu0", "cpu", compute_ms=8.0),
        ramps.StageService("vta1", "vta", compute_ms=7.0, load_ms=2.0),
        ramps.StageService("cpu2", "cpu", compute_ms=8.0),
        ramps.StageService("vta3", "vta", compute_ms=9.0, store_ms=3.0),
    ]
    boundaries = [
        ramps.BoundaryService(
            "b0",
            "cpu0",
            "vta1",
            "cpu_to_vta",
            2,
            "stage_dma",
            pack_ms=2.0,
        )
    ]
    scores = ramps.lightweight_cycle_scores(record(stages, boundaries=boundaries))
    assert math.isclose(scores["compute_balance_ms"], 16.0)
    assert scores["communication_aware_ms"] > scores["compute_balance_ms"]
    assert scores["components"]["heterogeneous_adapter_ms"] == 2.0


def test_zero_feedback_gate_ignores_labels_but_requires_declared_static_inputs():
    item = record(
        [ramps.StageService("cpu", "cpu", compute_ms=8.0, measured_ms=9.0)],
        measured_cycle_ms=10.0,
        metadata={
            "score_input_provenance": {
                "workload_features": {"kind": "compiler_static", "source": "tir"},
                "service_parameters": {
                    "kind": "independent_hardware_profile",
                    "source": "profile",
                },
            },
            "measured_stage_run_ms": [9.0],
        },
    )
    provenance = ramps.lightweight_cycle_scores(
        item, require_zero_feedback=True
    )["provenance"]
    assert provenance["zero_feedback_eligible"]
    assert provenance["candidate_measurement_fields_used_by_score"] == []
    assert "measured_cycle_ms" in provenance[
        "candidate_measurement_fields_present_for_evaluation"
    ]


def test_zero_feedback_gate_rejects_unspecified_service_source():
    with pytest.raises(RuntimeError, match="service_parameters:unspecified"):
        ramps.lightweight_cycle_scores(
            record([ramps.StageService("cpu", "cpu", compute_ms=8.0)]),
            require_zero_feedback=True,
        )


def test_ranking_metrics_reports_low_and_high_board_budgets():
    records = [
        record(
            [ramps.StageService("cpu", "cpu", compute_ms=float(index + 1))],
            candidate_id="c{}".format(index),
            measured_cycle_ms=float(index + 1),
        )
        for index in range(6)
    ]
    metrics = ramps.ranking_metrics(records, [6, 5, 4, 3, 2, 1])
    assert "regret_at_1" in metrics
    assert "regret_at_3" in metrics
    assert "regret_at_100" in metrics
    assert metrics["regret_at_100"] == 0.0
    assert "evaluations_to_oracle_95pct" in metrics
    assert "evaluations_to_oracle_98pct" in metrics


def test_measured_cpu_core_demand_is_not_wall_time_times_threads():
    stages = [
        ramps.StageService(
            "cpu0",
            "cpu",
            threads=4,
            compute_ms=20.0,
            core_demand_ms=10.0,
            core_demand_source="process_cpu_time",
        ),
        ramps.StageService(
            "cpu1",
            "cpu",
            threads=4,
            compute_ms=20.0,
            core_demand_ms=10.0,
            core_demand_source="process_cpu_time",
        ),
    ]
    prediction = ramps.predict_record(record(stages))
    cpu_pool = next(
        item
        for item in prediction["cycle_constraints"]
        if item["resource"] == "cpu_core_pool"
    )
    assert math.isclose(cpu_pool["service_ms"], 20.0)
    assert math.isclose(cpu_pool["cycle_ms"], 5.0)
    assert prediction["core_demand_sources"] == {
        "cpu0": "process_cpu_time",
        "cpu1": "process_cpu_time",
    }


def test_publication_mode_rejects_missing_cpu_core_demand():
    model = {
        "prediction_options": {
            "include_residual": False,
            "require_measured_core_demand": True,
        }
    }
    with pytest.raises(RuntimeError, match="lacks core_demand_ms"):
        ramps.predict_record(
            record([ramps.StageService("cpu", "cpu", threads=4, compute_ms=10.0)]),
            model,
        )


def test_deeper_queues_do_not_increase_finite_buffer_cycle():
    stages = [
        ramps.StageService("cpu0", "cpu", compute_ms=20.0),
        ramps.StageService("vta1", "vta", compute_ms=10.0),
        ramps.StageService("cpu2", "cpu", compute_ms=5.0),
    ]
    shallow = ramps.predict_record(record(stages, queue_depth=1))
    deep = ramps.predict_record(record(stages, queue_depth=4))
    assert deep["maxplus_cycle_ms"] <= shallow["maxplus_cycle_ms"]


def test_expanded_event_graph_karp_radius_matches_resource_constraints():
    stages = [
        ramps.StageService("cpu0", "cpu", threads=2, compute_ms=9.0),
        ramps.StageService("vta1", "vta", compute_ms=13.0),
        ramps.StageService("cpu2", "cpu", threads=1, compute_ms=7.0),
    ]
    payload = ramps.timed_event_graph_payload(record(stages, queue_depth=4))
    assert math.isclose(
        payload["karp_spectral_radius_ms"],
        payload["analytical_cycle_ms"],
        rel_tol=1e-10,
        abs_tol=1e-10,
    )
    assert payload["karp_validation_kind"] == "algebraic_encoding_consistency_only"
    assert payload["physical_event_graph_validation"].startswith("pending_stage5")
    assert payload["resource_demand_matrix"]["cpu_core_tokens"] == 4


def test_marked_graph_has_dependency_and_fifo_free_slot_channels():
    stages = [
        ramps.StageService("cpu0", "cpu", compute_ms=5.0),
        ramps.StageService("vta1", "vta", compute_ms=7.0),
        ramps.StageService("cpu2", "cpu", compute_ms=6.0),
    ]
    graph = ramps.timed_event_graph_payload(record(stages, queue_depth=2))[
        "marked_event_graph"
    ]
    data_channels = [item for item in graph["channels"] if item["kind"] == "data_dependency"]
    slot_channels = [item for item in graph["channels"] if item["kind"] == "fifo_free_slots"]
    assert len(data_channels) == 2
    assert len(slot_channels) == 2
    assert all(item["initial_tokens"] == 0 for item in data_channels)
    assert all(item["initial_tokens"] == 2 for item in slot_channels)


def test_cpu_and_vta_are_not_members_of_each_others_exclusive_resource():
    stages = [
        ramps.StageService("cpu", "cpu", compute_ms=9.0),
        ramps.StageService("vta", "vta", compute_ms=11.0),
    ]
    resources = {
        item["name"]: item
        for item in ramps.marked_event_graph_payload(record(stages))["resources"]
    }
    assert [item["stage"] for item in resources["vta_mutex"]["users"]] == ["vta"]
    assert [item["stage"] for item in resources["cpu_core_pool"]["users"]] == ["cpu"]


def test_ps_pl_load_and_store_have_independent_resource_cycles():
    stages = [
        ramps.StageService(
            "vta",
            "vta",
            compute_ms=1.0,
            load_ms=12.0,
            store_ms=4.0,
            overlap_factor=0.0,
        )
    ]
    constraints = ramps.build_cycle_constraints(record(stages))
    resources = {item.resource: item.cycle_ms for item in constraints}
    assert resources["ps_pl_load"] == 12.0
    assert resources["ps_pl_store"] == 4.0


def test_ps_pl_load_is_reported_when_it_is_co_critical_with_vta_service():
    item = record(
        [
            ramps.StageService(
                "vta",
                "vta",
                compute_ms=1.0,
                load_ms=12.0,
                overlap_factor=0.0,
            )
        ]
    )
    prediction = ramps.predict_record(item)
    resources = {cycle["resource"] for cycle in prediction["critical_cycles"]}
    assert "ps_pl_load" in resources
    assert "vta_mutex" in resources


def test_monotonic_uncertainty_scale_changes_risk_score_by_candidate():
    model = {
        "uncertainty": {
            "minimum_scale_ms": 0.1,
            "normalized_residual_q90": 1.0,
            "normalized_residual_q95": 2.0,
            "scale_model": {
                "intercept_ms": 1.0,
                "features": [
                    {
                        "name": "fragmentation_ms",
                        "knots": [0.0],
                        "coefficients": [2.0],
                    }
                ],
            },
        }
    }
    low = ramps.predict_record(record([], fragmentation_ms=0.0), model)
    high = ramps.predict_record(record([], fragmentation_ms=3.0), model)
    assert high["predicted_std_ms"] > low["predicted_std_ms"]
    assert high["risk_score_ms"] > low["risk_score_ms"]


def test_monotonic_residual_never_decreases_with_fragmentation():
    model = {
        "intercept_ms": -2.0,
        "features": [
            {"name": "fragmentation_ms", "knots": [0.0, 1.0], "coefficients": [2.0, 3.0]}
        ],
    }
    low = record([], fragmentation_ms=0.5)
    high = record([], fragmentation_ms=2.0)
    assert ramps.monotonic_residual(high, model) >= ramps.monotonic_residual(low, model)


def test_residual_is_disabled_by_default_and_requires_explicit_ablation():
    model = {
        "residual_model": {
            "intercept_ms": 5.0,
            "features": [],
        }
    }
    item = record([ramps.StageService("vta", "vta", compute_ms=10.0)])
    physical = ramps.predict_record(item, model)
    ablation = ramps.predict_record(item, model, include_residual=True)
    assert physical["prediction_mode"] == "physical"
    assert physical["predicted_cycle_ms"] == 10.0
    assert physical["available_residual_ms"] == 5.0
    assert ablation["prediction_mode"] == "physical_plus_residual"
    assert ablation["predicted_cycle_ms"] == 15.0


def test_synthetic_fit_produces_finite_predictions_and_nonnegative_hinges():
    records = []
    for index in range(24):
        cpu_ms = 10.0 + index
        vta_ms = 18.0 + (index % 5)
        fragmentation = float(index % 4)
        measured = max(cpu_ms, vta_ms) + 2.5 * fragmentation + 3.0
        records.append(
            record(
                [
                    ramps.StageService("cpu", "cpu", compute_ms=cpu_ms),
                    ramps.StageService("vta", "vta", compute_ms=vta_ms),
                ],
                candidate_id="c{}".format(index),
                group_id="g{}".format(index % 6),
                measured_cycle_ms=measured,
                fragmentation_ms=fragmentation,
                vta_island_count=1,
            )
        )
    model = ramps.fit_ramps_model(records, rank_weight=0.05)
    predictions = [
        ramps.predict_record(item, model, include_residual=True)["predicted_cycle_ms"]
        for item in records
    ]
    assert all(math.isfinite(item) and item > 0.0 for item in predictions)
    for feature in model["residual_model"]["features"]:
        assert all(value >= 0.0 for value in feature["coefficients"])
    assert ramps.ranking_metrics(records, predictions)["spearman"] > 0.8
