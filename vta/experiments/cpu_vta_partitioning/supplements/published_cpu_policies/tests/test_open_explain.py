from copy import deepcopy

import pytest

from cpu_policies.open_resources.domain import candidate, component_key, requirements, stage
from cpu_policies.open_resources.explain import explain_pool


def fixture(masks=((0, 1), (1, 2)), gate="off", host="host_shared_1"):
    programs = {"cpu_00_00": {"device": "cpu", "begin": 0, "end": 1},
                "vta_01_01": {"device": "vta", "begin": 1, "end": 2},
                "cpu_02_02": {"device": "cpu", "begin": 2, "end": 3}}
    records = {}
    for key, cfg in requirements(programs).items():
        cpu = cfg["stages"][0]["device"] == "cpu"
        records[key] = {"kind": "measured", "configuration": cfg,
                        "cost": {"service_ms": 100. if cpu else 15.,
                                 "run_core_ms": 200. if cpu else 20.,
                                 "boundary_core_ms": 10. if cpu else 2.}}
    hostmask = (0, 1, 2, 3) if host == "shared4" else (int(host[-1]),)
    cfg = candidate([stage("cpu", 0, 1, len(masks[0]), masks[0]),
                     stage("vta", 1, 2, 1, hostmask),
                     stage("cpu", 2, 3, len(masks[1]), masks[1])], host, gate)
    ddr = {"stage_service_ms": {name: {str(n): 1. for n in range(1, 5)} for name in programs}}
    return cfg, records, ddr


def test_partial_overlap_and_pinned_host_explain_actual_core_hotspot():
    cfg, records, ddr = fixture()
    result = explain_pool({cfg["id"]: cfg}, records, ddr, {})
    row = result["configurations"][cfg["id"]]
    assert result["uses_pipeline_fps"] is False
    assert row["physical_cores"][1]["cpu_total_ms"] == 210.
    assert row["physical_cores"][1]["fixed_host_ms"] == 22.
    assert row["diagnostic_enhanced_score_ms"] == 232.
    assert "core_subset_1" in row["bottleneck_terms"]
    assert len(row["subset_capacity_bounds"]) == 15
    assert row["host_total_core_ms"] == 22.
    assert row["vta_service_sum_ms"] == 15.
    assert row["ddr_service_ms"] == 3.
    assert row["physical_cores"][0]["cpu_boundary_ms"] == 10.
    assert row["physical_cores"][1]["cpu_boundary_ms"] == 10.
    assert row["stages"][0]["compute_weights"] == [.5, .5]
    assert row["control_work_ms"] is None


def test_shared_host_is_not_falsely_split_onto_physical_cores():
    cfg, records, ddr = fixture(masks=((0, 1), (2, 3)), host="shared4")
    row = explain_pool([cfg], records, ddr, {})["configurations"][cfg["id"]]
    assert row["movable_host_core_work_ms"] == 22.
    assert all(core["fixed_host_ms"] == 0. for core in row["physical_cores"])
    assert all(core["shared_host_may_use_core"] for core in row["physical_cores"])
    full = row["subset_capacity_bounds"][-1]
    assert full["confined_host_core_work_ms"] == 22.
    assert full["capacity_bound_ms"] == (420. + 22.) / 4


def test_gate_off_group_sum_is_diagnostic_and_serial_gate_adds_real_constraint():
    opened, records, ddr = fixture(masks=((0, 1), (0, 1)), host="shared4")
    closed, _, _ = fixture(masks=((0, 1), (0, 1)), gate="resource_group", host="shared4")
    # Separate wall service from demand so the serial constraint is dominant.
    for value in records.values():
        value["cost"]["run_core_ms"] = 1.
        value["cost"]["boundary_core_ms"] = 0.
    a = explain_pool([opened], records, ddr, {})["configurations"][opened["id"]]
    b = explain_pool([closed], records, ddr, {})["configurations"][closed["id"]]
    assert a["isolated_service_sum_by_mask_ms"] == b["isolated_service_sum_by_mask_ms"] == {"0,1": 200.}
    assert a["group_sums_are_execution_constraints"] is False
    assert a["enabled_serial_group_services_ms"] == {}
    assert a["diagnostic_enhanced_score_ms"] == 100.
    assert b["group_sums_are_execution_constraints"] is True
    assert b["diagnostic_enhanced_score_ms"] == 200.
    assert b["bottleneck_terms"] == ["enabled_serial_group_0,1"]


def test_original_prefix_prediction_is_preserved_without_fps_or_input_mutation():
    cfg, records, ddr = fixture()
    rankings = {"methods": {"prefix": {"rankings": [dict(cfg, score_ms=99., static_rank=1,
                                                         measured_fps=123456.)]},
                            "enhanced": {"rankings": [dict(cfg, score_ms=232., static_rank=2)]}},
                "pure_static20": [dict(cfg, score_ms=232., static_rank=2)],
                "diverse20": [dict(cfg, score_ms=232., selection_reason="new_resource")]}
    inputs = deepcopy((cfg, records, ddr, rankings))
    result = explain_pool({cfg["id"]: cfg}, records, ddr, rankings)
    row = result["configurations"][cfg["id"]]
    assert row["selected_scores_by_method"]["prefix"]["score_ms"] == 99.
    assert row["diagnostic_enhanced_score_ms"] == 232.
    assert row["diagnostic_score_replaces_method_score"] is False
    assert row["selected_scores_by_method"]["enhanced_diverse"]["selection_reason"] == "new_resource"
    assert "123456" not in str(result)
    assert (cfg, records, ddr, rankings) == inputs


def test_unavailable_ddr_is_missing_not_measured_zero_and_bad_pool_fails():
    cfg, records, _ = fixture()
    row = explain_pool([cfg], records, None, {})["configurations"][cfg["id"]]
    assert row["ddr_available"] is False
    assert row["ddr_service_ms"] is None
    assert "shared_DDR_service" not in [t["term"] for t in row["resource_terms"]]
    with pytest.raises(ValueError, match="pool map key"):
        explain_pool({"wrong": cfg}, records, None, {})
    with pytest.raises(ValueError, match="duplicate configuration"):
        explain_pool([cfg, cfg], records, None, {})


def test_explanation_conserves_the_same_integer_work_as_the_scoring_model():
    cfg, records, _ = fixture(masks=((0, 1, 2), (1, 2, 3)), host="host_shared_2")
    for record in records.values():
        cpu = record["configuration"]["stages"][0]["device"] == "cpu"
        record["cost"] = {"service_ms": .0000005,
                          "run_core_ms": .0000015 if cpu else .0000005,
                          "boundary_core_ms": .0000005}
    row = explain_pool([cfg], records, None, {})["configurations"][cfg["id"]]
    assert row["diagnostic_enhanced_score_ns"] == 3
    assert [c["cpu_total_ns"] for c in row["physical_cores"]] == [2, 3, 1, 0]
    assert row["stages"][0]["cost"]["run_core_ms"] == .0000015
    assert row["stages"][0]["quantized_cost_ns"]["run_core"] == 2
    assert row["stages"][0]["allocated_compute_ns_by_slot"] == [1, 1, 0]
    assert row["subset_capacity_bounds"][1]["capacity_bound_ns"] == 3
