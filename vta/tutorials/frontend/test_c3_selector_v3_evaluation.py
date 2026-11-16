import copy
import math

import pytest

from c3_selector_v3_evaluation import evaluate_task_pool, fixed_random_orders


def candidate(candidate_id, target, bytes_, calls, valid=True, score=None):
    row = {
        "execution_task_id": "task",
        "candidate_set_id": candidate_id,
        "is_incumbent": candidate_id == "inc",
        "fpga_valid": valid,
        "target_log_latency_ratio": target,
        "aggregate_metrics": {
            "total_request_bytes": bytes_, "total_request_calls": calls,
            "submissions": calls, "uop_load_bytes": calls, "instruction_bytes": calls,
        },
    }
    if score is not None:
        row["predicted_graph_log_latency_ratio"] = score
        row["prediction_source"] = "frozen_completion_aware_ridge"
    return row


def pool():
    return [
        candidate("inc", 0.0, 100, 10),
        candidate("fast", math.log(0.8), 120, 8, score=-0.2),
        candidate("bytes", math.log(1.1), 40, 9, score=0.1),
        candidate("invalid", None, 50, 1, valid=False, score=-0.1),
    ]


def test_same_budget_methods_preserve_invalid_attempt_cost_and_pool_oracle():
    result = evaluate_task_pool(pool(), budgets=(1, 2), random_replicates=20)
    selector = result["deterministic_methods"]["selector_v3"]["B1"]
    assert selector["selected_candidate_set_id"] == "fast"
    assert selector["simple_regret_fraction"] == 0.0
    bytes_only = result["deterministic_methods"]["bytes_only"]["B1"]
    assert bytes_only["selected_candidate_set_id"] == "inc"
    calls = result["deterministic_methods"]["calls_only"]["B1"]
    assert calls["attempted_new_candidate_ids"] == ["invalid"]
    assert calls["fpga_invalid_attempt_count"] == 1
    assert calls["timing_eligible_candidate_ids"] == ["inc"]
    assert result["deterministic_methods"]["incumbent_only"]["B0"][
        "oracle_candidate_set_id"] == "fast"


def test_random_order_is_reproducible_and_target_does_not_change_static_order():
    ids = ["a", "b", "c"]
    assert fixed_random_orders(ids, seed=7, count=4) == fixed_random_orders(ids, seed=7, count=4)
    first = evaluate_task_pool(pool(), budgets=(1,), random_replicates=5)
    changed = copy.deepcopy(pool())
    changed[1]["target_log_latency_ratio"] = math.log(1.2)
    changed[2]["target_log_latency_ratio"] = math.log(0.7)
    second = evaluate_task_pool(changed, budgets=(1,), random_replicates=5)
    assert first["deterministic_methods"]["bytes_only"]["B1"][
        "attempted_new_candidate_ids"] == second["deterministic_methods"]["bytes_only"]["B1"][
            "attempted_new_candidate_ids"]


def test_untrusted_injected_scores_do_not_enable_selector_method():
    rows = pool()
    for row in rows:
        if not row["is_incumbent"]:
            row["prediction_source"] = "caller_injected"
    result = evaluate_task_pool(rows, budgets=(1,), random_replicates=2)
    assert "selector_v3" not in result["deterministic_methods"]


def test_valid_candidate_requires_latency_and_order_cannot_skip_pool_members():
    rows = pool()
    rows[1]["target_log_latency_ratio"] = None
    with pytest.raises(ValueError, match="lacks comparable latency"):
        evaluate_task_pool(rows, budgets=(1,), random_replicates=2)
