#!/usr/bin/env python3
"""Static tests for the two-phase CPU-VTA Pipeline V1 P4B retrospective."""

from pathlib import Path

import evaluate_cpu_vta_pipeline_v1_p4b as p4b


OUTPUT = Path(__file__).resolve().parent / "report_out/resource_aware_maxplus"


def test_score_phase_contains_no_candidate_outcomes(tmp_path):
    for name in (
        "v1_p4_measured_pool_compatibility.json",
        "v1_profile_manifest.json",
        "v1_local_cost_table.json",
    ):
        (tmp_path / name).write_bytes((OUTPUT / name).read_bytes())
    assert not (tmp_path / "v1_p4_historical_labels.json").exists()
    outputs = p4b.build_frozen_scores(tmp_path)
    protocol = outputs["v1_p4b_scoring_protocol.json"]
    scores = outputs["v1_p4b_frozen_scores.json"]
    assert protocol["candidate_outcomes_used_by_scoring"] is False
    assert scores["candidate_outcome_fields_present"] is False
    assert scores["candidate_count"] == 69
    assert all("measured" not in str(row) for row in scores["rows"])
    assert all(
        set(row["scores"])
        == {"B2_single_frame_ms", "B3_pipeline_max_load_ms", "B4_v1_ms"}
        for row in scores["rows"]
    )


def test_type_fixed_baseline_is_not_replaced_by_a_nearest_candidate():
    scores = p4b.build_frozen_scores(OUTPUT)["v1_p4b_frozen_scores.json"]
    assert scores["type_fixed"]["present_in_semantic_pool"] is False
    assert scores["type_fixed"]["current_search_grammar_valid"] is True
    assert scores["type_fixed"]["requires_shortlist_native_compile"] is True


def test_evaluation_joins_labels_only_after_sealed_scores(tmp_path):
    score_outputs = p4b.build_frozen_scores(OUTPUT)
    for name, payload in score_outputs.items():
        p4b.write_json(tmp_path / name, payload)
    # Evaluation also requires the immutable compatibility artifact.
    (tmp_path / "v1_p4_measured_pool_compatibility.json").write_bytes(
        (OUTPUT / "v1_p4_measured_pool_compatibility.json").read_bytes()
    )
    (tmp_path / "v1_p4_historical_labels.json").write_bytes(
        (OUTPUT / "v1_p4_historical_labels.json").read_bytes()
    )
    outputs = p4b.evaluate_frozen_scores(tmp_path)
    report = outputs["v1_p4b_retrospective_report.json"]
    assert report["unique_candidate_count"] == 69
    assert report["duplicate_label_rule_applied_count"] == 1
    assert report["candidate_outcomes_used_to_fit_or_change_scores"] is False
    assert report["models"]["B1_type_fixed"]["metrics"] is None
    assert "top_5_recall" in report["models"]["B4_v1"]["metrics"]
    assert "top_5_recall" in report["models"]["B0_uniform_random"]["metrics"]
    assert report["gate"]["full_gate_passing_k_values"] == []
    assert report["gate"]["p5_may_start"] is False
    assert report["board_budget_diagnostic"]["b4_beats_uniform_random_median"] is False
    assert report["thread_search_coverage"][
        "per_stage_thread_joint_optimization_validated"
    ] is False
    assert report["random_baseline_comparison"]["uniform"][
        "probability_random_reaches_95pct_no_later_than_b4"
    ] > 0.5
    assert outputs["v1_execution_state.json"]["next_stage_may_start"] is False
    assert "historical_pool_does_not_cover_per_stage_thread_decisions" in outputs[
        "v1_execution_state.json"
    ]["blocking_evidence"]
