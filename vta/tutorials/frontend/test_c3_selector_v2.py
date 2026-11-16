import math

import numpy as np

from c3_selector_v2 import (
    budget_prefix,
    eligibility,
    fallback_key,
    fit_preprocessor,
    fixed_random_orders,
    rank_candidates,
    shortlist,
    train_selector,
    transform_rows,
)


def _row(task, index, target=0.0, missing=False):
    metrics = {
        "submissions": 1 + index % 2,
        "uop_load_bytes": None if missing else 32 + index,
        "total_request_bytes": 1000 + 10 * index,
        "total_request_calls": 20 + index,
        "instruction_bytes": 100 + index,
    }
    incumbent = dict(metrics)
    incumbent.update({"submissions": 1, "uop_load_bytes": 32, "total_request_bytes": 1000,
                      "total_request_calls": 20, "instruction_bytes": 100})
    return {
        "execution_task_id": task,
        "candidate_id": f"{task}-{index}",
        "candidate_mode": "input_stationary" if index % 2 else "original",
        "candidate_tile": {"tile_b": 1, "tile_h": 1 + index, "tile_w": 2,
                           "tile_ci": 1, "tile_co": 2, "oc_nthread": 1,
                           "h_nthread": 1},
        "candidate_metrics": metrics,
        "incumbent_metrics": incumbent,
        "target_log_latency_ratio": target,
        "is_incumbent": index == 0,
    }


def test_unknown_is_imputed_with_missing_indicator_not_zero_claim():
    rows = [_row("a", 0), _row("b", 1, missing=True)]
    prep = fit_preprocessor(rows)
    matrix = transform_rows(rows, prep)
    assert np.isfinite(matrix).all()
    # At least one missing-indicator column differs between the rows.
    half = 15 + 5 + 5
    assert matrix[1, half : 2 * half].sum() > matrix[0, half : 2 * half].sum()


def test_iqr_zero_uses_one():
    rows = [_row("a", 0), _row("b", 0)]
    prep = fit_preprocessor(rows)
    assert all(value > 0 for value in prep.iqrs)


def test_automatic_fallback_below_thresholds_and_unknown_last():
    rows = [_row("a", 0), _row("b", 1, missing=True)]
    contract = train_selector(rows)
    assert contract["selector_kind"] == "lexicographic_fallback"
    assert not eligibility(rows)["eligible"]
    assert fallback_key(rows[0]) < fallback_key(rows[1])


def test_loto_ridge_and_larger_alpha_tie_rule_is_serialized():
    rows = []
    for task_index in range(8):
        for index in range(5):
            rows.append(_row(f"t{task_index}", index,
                             target=math.log((100 + index) / 100)))
    contract = train_selector(rows)
    assert contract["selector_kind"] == "ridge"
    assert contract["ridge"]["alpha"] in (0.1, 1.0, 10.0, 100.0)
    assert "ties choose larger alpha" in contract["cross_validation"]["criterion"]
    ranked = rank_candidates(rows[:5], contract)
    assert [r["selector_v2_rank"] for r in ranked] == [1, 2, 3, 4, 5]


def test_shortlist_deduplicates_roles_and_caps_four():
    rows = [_row("a", i) for i in range(5)]
    result = shortlist(rows, train_selector(rows))
    assert result["shortlist_candidate_ids"][0] == "a-0"
    assert len(result["shortlist_candidate_ids"]) <= 4
    assert len(result["shortlist_candidate_ids"]) == len(set(result["shortlist_candidate_ids"]))


def test_invalid_candidate_consumes_budget_without_replacement():
    result = budget_prefix(["inc", "bad", "good", "later"], 2, "inc",
                           {"bad": False, "good": True, "later": True})
    assert result["attempted_new_candidate_ids"] == ["bad", "good"]
    assert result["timing_eligible_candidate_ids"] == ["inc", "good"]
    assert result["fpga_invalid_attempt_count"] == 1


def test_fixed_random_orders_are_reproducible_and_complete():
    left = fixed_random_orders(["c", "a", "b"], count=10)
    right = fixed_random_orders(["b", "c", "a"], count=10)
    assert left == right
    assert len(left) == 10
    assert all(sorted(order) == ["a", "b", "c"] for order in left)
