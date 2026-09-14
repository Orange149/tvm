"""Focused tests for the literature-inspired offline search replay."""

from analyze_vta_literature_guided_search import (
    deterministic_order,
    ranked_metrics,
    relative_delta,
    summarize_validity_order,
)


def test_validity_budget_counts_gross_attempts():
    rows = [
        {"candidate_id": "a", "status": "failed"},
        {"candidate_id": "b", "status": "ok"},
        {"candidate_id": "c", "status": "ok"},
        {"candidate_id": "d", "status": "failed"},
    ]
    summary = summarize_validity_order(rows)
    assert summary["4"] == {"valid": 2, "invalid": 2}
    assert summary["8"] == {"valid": 2, "invalid": 2}


def test_relative_delta_is_normalized_to_control():
    assert relative_delta({"x": 50}, {"x": 100}, ("x",)) == [-0.5]
    assert relative_delta({"x": 2}, {"x": 0}, ("x",)) == [2.0]


def test_regret_uses_best_observed_improvement():
    rows = [
        {"candidate_id": "a", "improvement_percent": -2.0},
        {"candidate_id": "b", "improvement_percent": 3.0},
        {"candidate_id": "c", "improvement_percent": 8.0},
    ]
    assert ranked_metrics(rows, 2)["regret_percentage_points"] == 5.0
    assert ranked_metrics(rows, 1)["all_selected_slow_down"]


def test_random_order_is_reproducible():
    rows = [{"candidate_id": value} for value in "abcd"]
    assert deterministic_order(rows, "x") == deterministic_order(rows, "x")
