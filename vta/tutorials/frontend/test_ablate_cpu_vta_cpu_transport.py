"""Guard the independent factors and calibration isolation in the six-cell study."""

import copy
import json

import pytest

import ablate_cpu_vta_cpu_transport as ablation
import solve_cpu_vta_pipeline_v1_p3 as p3


@pytest.fixture(scope="module")
def inputs():
    costs = json.loads((p3.DEFAULT_OUTPUT / "v1_local_cost_table.json").read_text())
    profile = json.loads((p3.DEFAULT_OUTPUT / "v1_profile_manifest.json").read_text())
    return costs, p3.build_context(profile, costs)


def test_pooled_fit_excludes_holdout_and_deduplicates_thread_sweep(inputs):
    costs, _ = inputs
    original = ablation.fit_generic(costs)
    changed = copy.deepcopy(costs)
    for stage in changed["grouped_holdout_measurement"]["stages"]:
        stage["get_ms"] *= 1000
        stage["set_ms"] *= 1000
        stage["get_process_cpu_ms"] *= 1000
        stage["set_process_cpu_ms"] *= 1000
    assert ablation.fit_generic(changed) == original
    assert len(original["fit_points"]) == 4
    assert len({p["group"] for p in original["fit_points"]}) == 2
    points = original["fit_points"]
    assert original["total_ms_per_byte"] == pytest.approx(
        sum(p["logical_bytes"] * p["total_ms"] for p in points)
        / sum(p["logical_bytes"] ** 2 for p in points)
    )


def test_zeroing_transmission_removes_boundary_core_work_but_preserves_stage_dma(inputs):
    costs, base = inputs
    generic = ablation.fit_generic(costs)
    path = (("cpu:00:02", 4), ("vta:03:17", 1), ("cpu:18:20", 2))
    none = ablation.transport_context(base, "none", generic)
    before = p3.label_from_path(path, base, scoring_mode="cpu_resource")
    after = p3.label_from_path(path, none, scoring_mode="cpu_resource")
    assert before.boundary_host_core_work_sum_ms > 0
    assert before.cpu_boundary_work_sum_ms > 0
    assert before.vta_mutex_boundary_sum_ms > 0
    assert after.boundary_host_core_work_sum_ms == 0
    assert after.cpu_boundary_work_sum_ms == after.vta_mutex_boundary_sum_ms == 0
    assert after.cpu_core_work_sum_ms == pytest.approx(
        before.cpu_core_work_sum_ms - before.boundary_host_core_work_sum_ms)
    assert after.vta_service_sum_ms == before.vta_service_sum_ms
    assert after.cpu_stage_core_work_by_threads_ms == before.cpu_stage_core_work_by_threads_ms
    assert after.max_cpu_stage_run_only_ms == before.max_cpu_stage_run_only_ms
    assert after.score == pytest.approx(before.score_without_direct_boundary)
    record = ablation.record(after, 1, none, "D")
    assert all(edge["service_ms"] == edge["host_core_demand_ms"] == 0
               for edge in record["boundaries"])
    assert all(stage["run_service_ms"] > 0 for stage in record["stages"])
    assert "boundary_component_overrides" not in base


def test_generic_replaces_wall_and_core_coefficients_without_double_counting(inputs):
    costs, base = inputs
    generic = ablation.fit_generic(costs)
    context = ablation.transport_context(base, "generic", generic)
    for boundary in base["boundary_costs"].values():
        size = boundary["logical_bytes"]
        edge = p3.boundary_components(boundary, context)
        assert edge["cpu_owner_ms"] == 0
        assert edge["vta_mutex_ms"] == edge["total_ms"]
        assert edge["total_ms"] == pytest.approx(size * generic["total_ms_per_byte"])
        assert edge["host_core_demand_ms"] == pytest.approx(size * generic["host_core_ms_per_byte"])
    assert ablation.transport_context(base, "actual", generic) == base
    with pytest.raises(ValueError):
        ablation.transport_context(base, "typo", generic)


@pytest.mark.parametrize("cell", ablation.CELLS)
def test_each_factor_reaches_search_and_candidate_record(inputs, cell):
    costs, base = inputs
    cpu, transport = ablation.CELLS[cell]
    context = ablation.transport_context(base, transport, ablation.fit_generic(costs))
    mode = "cpu_resource" if cpu else "baseline"
    labels, _ = p3.solve_k_best_label_setting(context, 5, scoring_mode=mode)
    for label in labels:
        assert label.search_lower_bound == label.score
        rebuilt = p3.label_from_path(label.path, context, scoring_mode=mode)
        assert rebuilt == label
        row = ablation.record(label, 1, context, cell)
        assert row["cpu_pool_enabled"] == cpu
        assert row["transport_model"] == transport
        assert row["predicted_ii_ms"] == label.score
        assert "shared_ddr_physical_demand" not in row["active_resource_terms"]
        loads = row["resource_load_ms"]
        expected = max(loads["max_cpu_stage"], loads["single_vta_plus_mutex_boundaries"])
        if cpu:
            expected = max(expected, loads["cpu_core_pool_lower_bound"])
        assert label.score == expected
