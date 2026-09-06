#!/usr/bin/env python3
"""Tests for the label-separated historical affinity replay."""

from pathlib import Path

import evaluate_resnet18_historical_affinity_models as replay


OUTPUT = Path(__file__).resolve().parent / "report_out/resource_aware_maxplus"


def test_historical_scores_are_label_free_and_cover_both_scopes():
    scores = replay.build_scores(OUTPUT)
    assert scores["candidate_outcomes_present"] is False
    assert scores["candidate_outcomes_used"] is False
    assert scores["historical_record_count"] == 200
    assert scores["unique_execution_candidate_count"] == 199
    assert scores["canonical_unique_candidate_count"] == 69
    assert all("measured_cycle_ms" not in row for row in scores["rows"])


def test_nested_prefix_bound_does_not_assume_equal_core_placement():
    cpu_stages = [
        {"threads": 4, "core_work_ms": 184.0},
        {"threads": 2, "core_work_ms": 68.0},
    ]
    assert replay._prefix_capacity_bounds(cpu_stages) == [0.0, 34.0, 68.0 / 3.0, 63.0]
    assert replay._uniform_per_core_loads(cpu_stages) == [80.0, 80.0, 46.0, 46.0]


def test_historical_evaluation_joins_all_labels_after_scoring():
    scores = replay.build_scores(OUTPUT)
    evaluation = replay.evaluate_scores(scores, OUTPUT)
    assert evaluation["candidate_outcomes_used_to_fit_or_change_scores"] is False
    assert evaluation["scopes"]["all_200_records_absolute_only"]["record_count"] == 200
    assert evaluation["scopes"]["all_199_unique_layouts"]["record_count"] == 199
    assert evaluation["scopes"]["canonical_69_unique_layouts"]["record_count"] == 69
    assert evaluation["interpretation_policy"]["production_model"] == "fluid_nested_prefix"
    assert evaluation["interpretation_policy"][
        "uniform_per_core_diagnostic_is_production_eligible"
    ] is False
    calibration = evaluation["cross_batch_absolute_calibration"]
    assert calibration["candidate_ranking_changed"] is False
    assert calibration["formal_prediction_eligible"] is False
    assert len(calibration["directions"]) == 2
    assert all(direction["train_count"] == 100 for direction in calibration["directions"])
    assert all(direction["test_count"] == 100 for direction in calibration["directions"])
    assert all(
        direction["methods"]["median_measured_to_predicted_ratio"]["test_error"][
            "mape"
        ]
        < 0.10
        for direction in calibration["directions"]
    )
    assert all(
        {
            "least_squares_through_origin",
            "median_measured_to_predicted_ratio",
            "mean_additive_offset",
            "median_additive_offset",
            "least_squares_affine",
        }
        <= set(direction["methods"])
        for direction in calibration["directions"]
    )
    summary = calibration["diagnostic_summary"]
    assert all(30.0 < value < 40.0 for value in summary["mean_additive_offsets_ms"])
    assert all(
        additive < proportional
        for additive, proportional in zip(
            summary["additive_test_mape"], summary["proportional_test_mape"]
        )
    )
    overlap = evaluation["current_top20_historical_overlap"]
    assert overlap["historical_labels_used_during_selection"] is False
    assert overlap["current_top20_count"] == 20
    assert overlap["current_unique_topology_count"] == 4
    assert overlap["exact_execution_match_count"] == 0
    assert overlap["topology_covered_candidate_count"] == 12
    assert overlap["topology_covered_unique_count"] == 3
    assert overlap["topology_uncovered_candidate_count"] == 8
