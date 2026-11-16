#!/usr/bin/env python3
"""Static correctness tests for the CPU-VTA Pipeline V1 P3 solver."""

import dataclasses
import json
from pathlib import Path

import pytest

import solve_cpu_vta_pipeline_v1_p3 as p3


OUTPUT = (
    Path(__file__).resolve().parent / "report_out/resource_aware_maxplus"
)


def load_context():
    profile = json.loads((OUTPUT / "v1_profile_manifest.json").read_text())
    costs = json.loads((OUTPUT / "v1_local_cost_table.json").read_text())
    return p3.build_context(profile, costs)


def test_cpu_thread_parameters_are_independent_per_stage():
    assert len(list(p3.positive_allocations(2))) == 16
    assert len(list(p3.positive_allocations(3))) == 64
    assert len(list(p3.positive_allocations(4))) == 256
    assert (3, 4) in set(p3.positive_allocations(2))


def test_pareto_dominance_requires_identical_state_at_call_site():
    base = p3.Label(
        position=3, vta_islands=1, last_device="cpu",
        max_cpu_stage_run_only_ms=9.0, max_cpu_stage_ms=10.0,
        last_cpu_stage_resource_ms=9.5,
        vta_service_sum_ms=5.0, vta_mutex_boundary_sum_ms=2.0,
        cpu_boundary_work_sum_ms=1.0, cpu_core_work_sum_ms=20.0,
        cpu_stage_core_work_sum_ms=18.0,
        cpu_stage_core_work_by_threads_ms=(0.0, 18.0, 0.0, 0.0),
        max_cpu_thread_parameter=2,
        boundary_host_core_work_sum_ms=1.0, shared_ddr_service_ms=4.0,
        cpu_ddr_logical_bytes=100, vta_ddr_physical_load_bytes=200,
        vta_ddr_physical_store_bytes=20, boundary_ddr_bytes=50,
        path=(("cpu:00:00", 2),),
    )
    worse = p3.Label(
        position=3, vta_islands=1, last_device="cpu",
        max_cpu_stage_run_only_ms=10.0, max_cpu_stage_ms=11.0,
        last_cpu_stage_resource_ms=10.5,
        vta_service_sum_ms=5.0, vta_mutex_boundary_sum_ms=3.0,
        cpu_boundary_work_sum_ms=2.0, cpu_core_work_sum_ms=24.0,
        cpu_stage_core_work_sum_ms=22.0,
        cpu_stage_core_work_by_threads_ms=(0.0, 22.0, 0.0, 0.0),
        max_cpu_thread_parameter=2,
        boundary_host_core_work_sum_ms=2.0, shared_ddr_service_ms=5.0,
        cpu_ddr_logical_bytes=120, vta_ddr_physical_load_bytes=220,
        vta_ddr_physical_store_bytes=30, boundary_ddr_bytes=60,
        path=(("cpu:00:00", 2),),
    )
    assert base.state == worse.state
    assert p3.dominates(base, worse)
    assert not p3.dominates(worse, base)


def test_prefix_mask_bound_checks_every_nested_capacity():
    label = dataclasses.replace(
        p3.initial_label(),
        cpu_core_work_sum_ms=12.0,
        cpu_stage_core_work_sum_ms=12.0,
        cpu_stage_core_work_by_threads_ms=(8.0, 0.0, 0.0, 4.0),
        max_cpu_thread_parameter=4,
    )
    assert label.cpu_prefix_mask_capacity_bounds_ms == pytest.approx(
        (8.0, 4.0, 8.0 / 3.0, 3.0)
    )
    assert label.cpu_prefix_mask_lower_bound_ms == pytest.approx(8.0)
    assert label.cpu_core_pool_lower_bound_ms == pytest.approx(8.0)


def test_ddr_demand_uses_cpu_bandwidth_and_physical_vta_traffic():
    context = load_context()
    assert context["vta_traffic_model"]["fit_point_count"] == 2
    assert context["vta_traffic_model"]["grouped_holdout_total_traffic_ape"] < 0.20
    assert context["vta_dma_qualification"]["fit_r_squared"] >= 0.99
    assert all(
        model["grouped_holdout_total_ape"] < 0.10
        for model in context["boundary_ownership_models"].values()
    )
    cpu = p3.stage_ddr_demand(
        context["profile_segments"]["cpu:00:00"], 4, context
    )
    assert cpu["cpu_logical_bytes"] > 0
    assert cpu["service_ms"] > 0
    vta = p3.stage_ddr_demand(
        context["profile_segments"]["vta:03:17"], 1, context
    )
    assert vta["vta_load_bytes"] > vta["vta_store_bytes"] > 0
    assert vta["service_ms"] > cpu["service_ms"]


def test_dp_top20_exactly_matches_full_static_enumeration():
    schema = json.loads((OUTPUT / "v1_unit_and_boundary_schema.json").read_text())
    context = load_context()
    dp_top, _ = p3.solve_k_best_label_setting(context, 20)
    oracle, _, topology_count, execution_count = p3.enumerate_oracle(schema, context, 20)
    assert topology_count == 4623
    assert execution_count == 972528
    assert [p3.candidate_id(item.path) for item in dp_top] == [
        p3.candidate_id(item.path) for item in oracle[:20]
    ]
    assert [item.score for item in dp_top] == pytest.approx(
        [item.score for item in oracle[:20]]
    )


def test_candidate_provenance_records_independent_overlapping_cpu_masks():
    context = load_context()
    top_one, _ = p3.solve_k_best_label_setting(context, 1)
    record = p3.candidate_record(top_one[0], 1, context)
    masks = [stage["cpu_core_mask"] for stage in record["stages"] if stage["device"] == "cpu"]
    assert masks == [list(range(thread)) for thread in record["cpu_stage_thread_parameters"]]
    assert record["cpu_thread_sum_is_a_constraint"] is False
    assert all(thread in p3.CPU_THREAD_CHOICES for thread in record["cpu_stage_thread_parameters"])
    assert record["cost_provenance"]["candidate_measured_throughput_consumed"] is False
    assert "sum(threads)" in record["cost_provenance"]["cpu_contention_scope_limitation"]
    assert record["cost_provenance"]["local_cost_table_artifact_sha256"] == context[
        "local_cost_table_artifact_sha256"
    ]
    assert record["resource_load_ms"]["direct_boundary_communication"] > 0
    assert record["resource_load_ms"]["cpu_owned_boundary_work_total"] > 0
    assert record["resource_load_ms"]["vta_mutex_boundary_work_total"] > 0
    assert record["resource_load_ms"]["cpu_core_pool_work"] > 0
    assert record["resource_load_ms"]["cpu_core_pool_lower_bound"] == pytest.approx(
        max(
            record["resource_load_ms"]["cpu_physical_pool_lower_bound"],
            record["resource_load_ms"]["cpu_prefix_mask_lower_bound"],
        )
    )
    assert record["resource_load_ms"]["cpu_prefix_mask_capacity"] == max(
        record["cpu_stage_thread_parameters"]
    )
    assert record["resource_load_ms"]["boundary_host_core_work_total"] > 0
    assert record["shared_ddr_traffic"]["vta_physical_load_bytes_est"] > 0
    assert all(
        boundary["stage_dma_additional_service_ms"] == 0.0
        for boundary in record["boundaries"]
    )
    assert all(boundary["host_accounting_ids"] for boundary in record["boundaries"])
    assert all(boundary["host_core_demand_ms"] > 0 for boundary in record["boundaries"])
    assert max(
        stage["resource_service_ms"]
        for stage in record["stages"]
        if stage["device"] == "cpu"
    ) == pytest.approx(record["resource_load_ms"]["max_cpu_stage"])


def test_middle_cpu_stage_owns_both_adjacent_boundary_components():
    context = load_context()
    path = (
        ("cpu:00:00", 1),
        ("vta:01:01", 1),
        ("cpu:02:02", 1),
        ("vta:03:14", 1),
        ("cpu:15:20", 2),
    )
    label = p3.label_from_path(path, context)
    record = p3.candidate_record(label, 1, context)
    middle_cpu = record["stages"][2]
    adjacent_cpu_work = (
        record["boundaries"][1]["cpu_owner_ms"]
        + record["boundaries"][2]["cpu_owner_ms"]
    )
    assert middle_cpu["owned_boundary_service_ms"] == pytest.approx(adjacent_cpu_work)
    assert middle_cpu["resource_service_ms"] == pytest.approx(
        middle_cpu["run_service_ms"] + adjacent_cpu_work
    )
    assert record["resource_load_ms"]["single_vta_plus_mutex_boundaries"] == pytest.approx(
        record["resource_load_ms"]["single_vta_service"]
        + record["resource_load_ms"]["vta_mutex_boundary_work_total"]
    )


@pytest.mark.parametrize("mode", p3.SCORING_MODES)
def test_scoring_mode_survives_every_cpu_and_vta_transition(mode):
    context = load_context()
    path = (("cpu:00:02", 4), ("vta:03:17", 1), ("cpu:18:20", 2))
    label = p3.initial_label(mode)
    for sid, threads in path:
        previous_bound = label.search_lower_bound
        label = p3.advance_label(label, context["profile_segments"][sid], threads, context)
        assert label.scoring_mode == mode
        assert label.search_lower_bound >= previous_bound
    rebuilt = p3.label_from_path(path, context, scoring_mode=mode)
    assert rebuilt == label
    assert label.score == label.search_lower_bound


@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("mode", p3.SCORING_MODES)
def test_search_queue_and_stop_threshold_use_only_selected_resources(monkeypatch, mode, top_k):
    # Inactive DDR/core costs must not hide the baseline winner behind the stop
    # threshold.  Two terminal branches isolate queue/stopping from graph costs.
    monkeypatch.setattr(p3, "build_outgoing", lambda context: {})

    def terminal_branches(label, context, outgoing):
        for thread, vta, core_work, ddr in [(1, 6.0, 80.0, 100.0), (2, 7.0, 32.0, 9.0)]:
            yield dataclasses.replace(
                label, position=len(p3.UNIT_ORDER), last_device="cpu", vta_islands=1,
                max_cpu_stage_ms=4.0, vta_service_sum_ms=vta,
                cpu_core_work_sum_ms=core_work,
                cpu_stage_core_work_by_threads_ms=(0.0, 0.0, 0.0, core_work),
                max_cpu_thread_parameter=4, shared_ddr_service_ms=ddr,
                path=(("cpu:00:20", thread),),
            )

    monkeypatch.setattr(p3, "expand_label", terminal_branches)
    ranked, stats = p3.solve_k_best_label_setting({}, top_k, scoring_mode=mode)
    expected_threads = [1, 2] if mode == "baseline" else [2, 1]
    expected_scores = {
        "baseline": [6.0, 7.0], "cpu_resource": [8.0, 20.0], "current": [9.0, 100.0],
    }
    assert [x.path[0][1] for x in ranked] == expected_threads[:top_k]
    assert [x.score for x in ranked] == expected_scores[mode][:top_k]
    assert stats["stop_threshold_ms"] == expected_scores[mode][top_k - 1]


def test_inactive_ddr_does_not_affect_pareto_ties():
    base = dataclasses.replace(p3.initial_label("baseline"), path=(("cpu:00:00", 1),))
    later = dataclasses.replace(base, shared_ddr_service_ms=1.0, path=(("cpu:00:00", 2),))
    assert p3.dominates(base, later)
    assert not p3.dominates(later, base)
    with pytest.raises(ValueError, match="different scoring modes"):
        p3.dominates(base, dataclasses.replace(later, scoring_mode="current"))


def test_invalid_scoring_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown scoring mode"):
        p3.initial_label("unsupported")


def test_comparison_refuses_to_overwrite_an_existing_directory(tmp_path):
    from compare_cpu_vta_pipeline_static_scores import compare

    with pytest.raises(FileExistsError, match="new directory"):
        compare(tmp_path / "missing-input", tmp_path)
