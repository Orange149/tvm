import copy
import math

import pytest

from c3_completion_policy import DEPENDENCY_TOKENS, TRANSITION_SAFE_FORCE_SERIAL
from c3_selector_v3_model import (
    SCHEMA,
    feature_names,
    row_numeric_features,
    score_graph_sets,
    train_selector,
)
from c3_selector_v3 import graph_set_measurement_plan


def row(task, index, scoped):
    policy = TRANSITION_SAFE_FORCE_SERIAL if scoped else DEPENDENCY_TOKENS
    candidate_bytes = 70 + index + (10 if scoped else 0)
    incumbent_bytes = 100 + index
    target = math.log((0.92 if scoped else 1.03) + index * 0.002)
    return {
        "execution_task_id": task,
        "candidate_set_id": "{}-{}".format(task, index),
        "label_scope": "exact_graph_candidate_set",
        "graph_context_qualified": True,
        "fpga_correct": True,
        "hardware_fingerprint_id": "hw",
        "measurement_protocol_id": "paired-six-round-v1",
        "members": [{"completion_policy": policy, "scoped_functions": ["f"]}],
        "aggregate_metrics": {
            "total_request_bytes": candidate_bytes,
            "total_request_calls": 5 + index,
            "instruction_bytes": 20 + index,
            "uop_load_bytes": 4,
            "submissions": 2 if scoped else 1,
            "sync_count": 1 if scoped else 0,
        },
        "incumbent_aggregate_metrics": {
            "total_request_bytes": incumbent_bytes,
            "total_request_calls": 7 + index,
            "instruction_bytes": 18 + index,
            "uop_load_bytes": 3,
            "submissions": 1,
            "sync_count": 0,
        },
        "target_log_latency_ratio": target,
    }


def eligible_rows():
    return [row("task{}".format(task), index, task < 4)
            for task in range(8) for index in range(5)]


def test_completion_aware_model_trains_with_grouped_policy_coverage():
    contract = train_selector(eligible_rows())
    assert contract["schema"] == SCHEMA
    assert contract["selector_kind"] == "completion_aware_ridge"
    assert contract["eligibility"]["execution_task_count"] == 8
    assert contract["eligibility"]["scoped_completion_task_count"] == 4
    assert contract["eligibility"]["dependency_only_task_count"] == 4
    assert len(contract["cross_validation"]["alphas"]) == 4
    assert "completion.scoped_completion_count" in feature_names()
    assert "completion.entry_drain_count" in feature_names()


def test_insufficient_completion_diversity_forces_static_fallback():
    rows = [row("task{}".format(task), index, True)
            for task in range(8) for index in range(5)]
    contract = train_selector(rows)
    assert contract["selector_kind"] == "static_fallback"
    assert "dependency_only_tasks=0<3" in contract["eligibility"]["reasons"]


def test_latency_is_not_a_feature_and_injected_prediction_is_overwritten():
    rows = eligible_rows()
    contract = train_selector(rows)
    target = copy.deepcopy(rows[0])
    target["target_log_latency_ratio"] = 999.0
    target["candidate_latency_ms"] = 999.0
    target["predicted_graph_log_latency_ratio"] = -999.0
    before = row_numeric_features(target)
    target["target_log_latency_ratio"] = -999.0
    target["candidate_latency_ms"] = 0.001
    assert row_numeric_features(target) == before
    scored = score_graph_sets([target], contract)[0]
    assert scored["predicted_graph_log_latency_ratio"] != -999.0
    assert scored["prediction_source"] == "frozen_completion_aware_ridge"


def test_scoring_fails_closed_without_incumbent_metrics():
    contract = train_selector(eligible_rows())
    target = copy.deepcopy(eligible_rows()[0])
    target.pop("incumbent_aggregate_metrics")
    with pytest.raises(ValueError, match="missing_incumbent_aggregate_metrics"):
        score_graph_sets([target], contract)


def test_scoring_fails_closed_on_hardware_or_nonproduction_policy():
    contract = train_selector(eligible_rows())
    target = copy.deepcopy(eligible_rows()[0])
    target["hardware_fingerprint_id"] = "different-hardware"
    with pytest.raises(ValueError, match="hardware fingerprint mismatch"):
        score_graph_sets([target], contract)

    target = copy.deepcopy(eligible_rows()[0])
    target["members"][0]["completion_policy"] = "legacy_unscoped_barrier_v1"
    with pytest.raises(ValueError, match="completion_policy_not_production_eligible"):
        score_graph_sets([target], contract)


def test_non_exact_or_mixed_hardware_training_never_unlocks_model():
    rows = eligible_rows()
    rows[0]["label_scope"] = "isolated_operator"
    rows[1]["hardware_fingerprint_id"] = "other-hw"
    contract = train_selector(rows)
    assert contract["selector_kind"] == "static_fallback"
    assert any(reason.startswith("rows_fail_exact_graph_contract=")
               for reason in contract["eligibility"]["reasons"])
    assert "hardware_fingerprint_count=2!=1" in contract["eligibility"]["reasons"]


def test_graph_plan_only_uses_predictions_computed_from_frozen_contract():
    contract = train_selector(eligible_rows())
    replacement = copy.deepcopy(eligible_rows()[0])
    replacement.update({"is_incumbent": False, "compatibility": {"compatible": True}})
    replacement["predicted_graph_log_latency_ratio"] = -999.0
    incumbent = {
        "candidate_set_id": "incumbent", "is_incumbent": True,
        "compatibility": {"compatible": True}, "aggregate_metrics": {},
    }
    plan = graph_set_measurement_plan([incumbent, replacement], model_contract=contract)
    assert plan["model_status"] == "COMPLETION_AWARE_MODEL"
    assert plan["performance_model_claim"] is True
    assert "selector_v3_model" in plan["roles_by_candidate_set_id"][replacement["candidate_set_id"]]

    incomplete = copy.deepcopy(replacement)
    incomplete.pop("incumbent_aggregate_metrics")
    fallback = graph_set_measurement_plan([incumbent, incomplete], model_contract=contract)
    assert fallback["model_status"] == "STATIC_FALLBACK"
    assert "missing_incumbent_aggregate_metrics" in fallback["model_fallback_reason"]
