#!/usr/bin/env python3
"""Static gates for the frozen CPU-VTA Pipeline V1 P4R pilot."""

import shutil
from pathlib import Path

import pytest

import freeze_cpu_vta_pipeline_v1_p4r as p4r


OUTPUT = Path(__file__).resolve().parent / "report_out/resource_aware_maxplus"


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    isolated = tmp_path_factory.mktemp("p4r_replay")
    for name in (
        "v1_p4b_retrospective_report.json",
        "v1_unit_and_boundary_schema.json",
        "v1_profile_manifest.json",
        "v1_local_cost_table.json",
    ):
        shutil.copy2(OUTPUT / name, isolated / name)
    state = p4r.seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_execution_state",
            "protocol_id": p4r.PROTOCOL_ID,
            "current_stage": "V1-P4B",
        }
    )
    p4r.write_json(isolated / "v1_execution_state.json", state)
    return p4r.build_p4r(isolated)


def test_p4r_freezes_bounded_label_free_prospective_pool(outputs):
    plan = outputs["v1_p4r_prospective_plan.json"]
    manifest = outputs["v1_p4r_candidate_manifest.json"]
    assert plan["primary_comparison"]["primary_k"] == 3
    assert plan["budget"]["unique_candidate_board_limit"] == 20
    assert plan["budget"]["oracle_based_early_stopping_allowed"] is False
    assert plan["primary_comparison"]["claim_scope"].startswith("pilot evidence")
    assert plan["correctness_gate"]["p5a_independent_expected_tensor_and_comparator_required"] is True
    assert plan["correctness_gate"]["p5b_board_native_vs_reference_required_before_timing"] is True
    assert manifest["candidate_count"] == 20
    assert manifest["candidate_outcome_fields_present"] is False
    assert sorted(row["frozen_execution_order"] for row in manifest["rows"]) == list(range(1, 21))
    assert len({row["candidate_id"] for row in manifest["rows"]}) == 20
    assert all(row["candidate_outcome_fields_present"] is False for row in manifest["rows"])


def test_p4r_contains_all_comparison_arms_and_valid_thread_controls(outputs):
    manifest = outputs["v1_p4r_candidate_manifest.json"]
    role_rows = {}
    for row in manifest["rows"]:
        for role in row["roles"]:
            role_rows.setdefault(role["role"], []).append((row, role))
    for role in ("B0_uniform_random", "B2_single_frame", "B3_pipeline_max_load", "B4_v1"):
        assert {item[1]["source_rank"] for item in role_rows[role]} == {1, 2, 3}
    assert len(role_rows["B1_type_fixed"]) == 1
    assert len(role_rows["paired_thread_control"]) == 6
    reference = min(role_rows["B4_v1"], key=lambda item: item[1]["source_rank"])[0]
    reference_topology = reference["topology_id"]
    assert all(row["topology_id"] == reference_topology for row, _ in role_rows["paired_thread_control"])
    assert all(all(thread in (1, 2, 3, 4) for thread in row["stage_runtime_threads"]) for row in manifest["rows"])


def test_p4r_stops_before_compile_and_board_execution(outputs):
    state = outputs["v1_execution_state.json"]
    assert state["current_stage"] == "V1-P4R"
    assert state["current_stage_status"] == "completed_awaiting_user_confirmation"
    assert state["next_stage"] == "V1-P5A"
    assert state["next_stage_may_start"] is False
    assert (
        "independent_expected_tensor_and_raw_output_comparator_not_yet_demonstrated"
        in state["blocking_evidence"]
    )
