#!/usr/bin/env python3
"""Static tests for the CPU-VTA Pipeline V1 P4 compatibility gate."""

from pathlib import Path

import audit_cpu_vta_pipeline_v1_p4 as p4


OUTPUT = Path(__file__).resolve().parent / "report_out/resource_aware_maxplus"


def test_historical_pool_uses_independent_stage_threads_for_compatibility():
    outputs = p4.build_audit(OUTPUT)
    compatibility = outputs["v1_p4_measured_pool_compatibility.json"]
    labels = outputs["v1_p4_historical_labels.json"]
    historical = compatibility["historical_pool"]
    assert historical["record_count"] == 200
    assert historical["mapped_record_count"] == 200
    assert historical["cpu_thread_sum_counts_are_diagnostic_only"] is True
    assert compatibility["overlap"]["semantic_execution_candidate_count"] >= 40
    assert compatibility["overlap"]["current_runtime_exact_candidate_count"] == 0
    assert compatibility["overlap"]["topology_count"] > 0
    assert compatibility["comparison_eligible"] is True
    assert compatibility["current_runtime_policy_comparison_eligible"] is False
    assert compatibility["candidate_outcome_fields_present"] is False
    assert all("measured_cycle_ms" not in row for row in compatibility["rows"])
    assert labels["record_count"] == 200
    assert all("measured_cycle_ms" in row for row in labels["rows"])
    assert historical["semantic_cpu_thread_vector_counts"] == {
        "3,1,1,4": 13,
        "3,1,4": 44,
        "3,4": 13,
    }
    assert historical["per_stage_thread_search_coverage_complete"] is False


def test_p4_passes_compatibility_but_stops_before_unfrozen_baseline_scoring():
    outputs = p4.build_audit(OUTPUT)
    report = outputs["v1_p4_retrospective_report.json"]
    state = outputs["v1_execution_state.json"]
    assert report["status"] == "semantic_compatibility_passed_current_runtime_unverified"
    assert all(item["metrics"] is None for item in report["models"].values())
    assert report["gate"]["b4_low_k_regret_beats_b0_b3"] is None
    assert report["gate"]["historical_labels_comparable_to_current_runtime_b4"] is False
    assert report["gate"]["p5_may_start"] is False
    assert state["current_stage"] == "V1-P4"
    assert state["next_stage"] == "V1-P4B"
    assert state["next_stage_may_start"] is False


def test_p4_audit_remains_replayable_after_pipeline_advances_past_p4():
    current = p4.load_sealed_artifact(
        OUTPUT / "v1_execution_state.json", "cpu_vta_pipeline_v1_execution_state"
    )
    assert current["current_stage"] in {"V1-P4R", "V1-P5A", "V1-P5B", "V1-P6"}
    outputs = p4.build_audit(OUTPUT)
    assert outputs["v1_p4_measured_pool_compatibility.json"]["comparison_eligible"] is True
