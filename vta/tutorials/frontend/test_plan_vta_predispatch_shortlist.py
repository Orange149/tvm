"""Tests for the label-free C3 P5b pre-dispatch shortlist planner."""

import copy
import json
from pathlib import Path

import pytest

from plan_vta_predispatch_shortlist import (
    DEFAULT_BUDGETS,
    FEATURE_VERSION,
    build_shortlists,
    pareto_order,
)


BASE = (
    Path(__file__).resolve().parent
    / "report_out"
    / "stage_tile_cotuning"
    / "c3_dma_residency_autotune"
)
P4B = BASE / "04_dma_command_signatures" / "20260910_p4b_local_residency_pool_run01"


def _feature(candidate_id, total_bytes, load_calls, small_calls=0):
    metrics = {
        "total_dma_bytes": total_bytes,
        "total_dma_calls": load_calls + 1,
        "load_calls": load_calls,
        "store_calls": 1,
        "small_calls": small_calls,
        "strided_calls": 0,
        "padded_calls": 0,
        "input_reload": 1.0,
        "weight_reload": 1.0,
        "output_reload": 1.0,
        "negative_max_input_request_bytes": -1024,
        "negative_max_weight_request_bytes": -1024,
        "negative_max_output_request_bytes": -1024,
        "histogram_distinct_sizes": 1,
        "negative_histogram_min_request_bytes": -1024,
    }
    return {
        "candidate_id": candidate_id,
        "prefeature": {"metrics": metrics},
    }


def _load_real_inputs():
    pool = json.loads((P4B / "candidate_pool.json").read_text())
    records = [json.loads(line) for line in (P4B / "results.jsonl").read_text().splitlines()]
    return pool, records


def test_pareto_order_puts_dominated_candidate_in_later_front():
    better = _feature("a", 100, 2)
    tradeoff = _feature("b", 90, 3)
    dominated = _feature("c", 120, 4, small_calls=1)
    ordered = pareto_order([better, tradeoff, dominated])
    fronts = {candidate["candidate_id"]: candidate["pareto_front"] for candidate in ordered}
    assert fronts["a"] == fronts["b"] == 0
    assert fronts["c"] == 1


def test_real_pool_is_fully_joined_filtered_and_budgeted_by_unique_id():
    shortlist, features, failures, screening = build_shortlists(*_load_real_inputs())
    assert len(shortlist["workloads"]) == 10
    assert len(features) == screening["lower_successes_board_eligible"] == 165
    assert len(failures) == screening["lower_failures_filtered_before_board"] == 85
    assert screening["input_records"] == screening["unique_candidate_ids"] == 250
    assert screening["board_dispatches_consumed_by_screening"] == 0
    failed_ids = {failure["candidate_id"] for failure in failures}
    for workload in shortlist["workloads"].values():
        incumbent = workload["protected_original_incumbent"]
        for strategy in ("B4", "B5", "B6", "B7"):
            for budget in DEFAULT_BUDGETS:
                selected = workload["strategies"][strategy]["budgets"][str(budget)]
                assert selected["candidate_ids"][0] == incumbent
                assert len(selected["candidate_ids"]) == len(set(selected["candidate_ids"]))
                assert not failed_ids.intersection(selected["candidate_ids"])
                assert selected["effective_budget"] <= budget


def test_b5_orders_remainder_by_total_bytes_then_load_calls():
    shortlist, features, _, _ = build_shortlists(*_load_real_inputs())
    by_id = {feature["candidate_id"]: feature for feature in features}
    for workload in shortlist["workloads"].values():
        ranking = workload["strategies"]["B5"]["ranking_candidate_ids"]
        remainder = ranking[1:]
        observed = [
            (
                by_id[candidate_id]["metrics"]["total_dma_bytes"],
                by_id[candidate_id]["metrics"]["load_calls"],
                candidate_id,
            )
            for candidate_id in remainder
        ]
        assert observed == sorted(observed)


def test_b7_budget_four_protects_incumbent_and_maximizes_mode_diversity():
    shortlist, _, _, _ = build_shortlists(*_load_real_inputs())
    for workload in shortlist["workloads"].values():
        selected = workload["strategies"]["B7"]["budgets"]["4"]
        assert selected["candidate_ids"][0] == workload["protected_original_incumbent"]
        assert set(selected["mode_counts"]) == {
            "original",
            "input_stationary",
            "weight_stationary",
            "paper_inspired_hybrid",
        }


def test_feature_version_hashes_and_output_are_deterministic():
    first = build_shortlists(*_load_real_inputs())
    second = build_shortlists(*_load_real_inputs())
    assert first == second
    shortlist, features, _, _ = first
    assert shortlist["feature_version"] == FEATURE_VERSION
    assert len(shortlist["feature_set_sha256"]) == 64
    assert all(feature["feature_version"] == FEATURE_VERSION for feature in features)
    assert all(len(feature["feature_sha256"]) == 64 for feature in features)


@pytest.mark.parametrize("label_key", ["latency_ms", "correct", "costs_s", "fps"])
def test_rejects_post_dispatch_labels_instead_of_silently_using_them(label_key):
    pool, records = _load_real_inputs()
    poisoned = copy.deepcopy(records)
    poisoned[0][label_key] = 1
    with pytest.raises(ValueError, match="forbidden post-dispatch label"):
        build_shortlists(pool, poisoned)


def test_rejects_candidate_pool_result_identity_mismatch():
    pool, records = _load_real_inputs()
    with pytest.raises(ValueError, match="identity mismatch"):
        build_shortlists(pool, records[:-1])
