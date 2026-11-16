import copy

from c3_composability import candidate_certificate, graph_context_certificate, static_set_certificate
from c3_selector_v3 import (
    candidate_set_deployment_gate,
    deployment_gate,
    graph_set_measurement_plan,
    graph_certificates_by_set,
    measurement_plan,
    qualify_pool,
)
from test_c3_composability import records


def fallback_contract():
    return {"selector_kind": "lexicographic_fallback"}


def row(candidate_id, mode, incumbent=False, correct=True):
    return {
        "candidate_id": candidate_id, "candidate_mode": mode, "is_incumbent": incumbent,
        "local_qualification_passed": correct,
        "candidate_metrics": {"submissions": 1, "uop_load_bytes": 2,
                              "instruction_bytes": 3, "total_request_bytes": 4,
                              "total_request_calls": 5},
    }


def graph_certificate_for(certificate):
    static = static_set_certificate(
        [certificate], {"graph_sha256": "g", "params_sha256": "p"},
        {"status": "passed", "global_vta_debug_flag": 0,
         "calls_by_function": {"target": 2}},
    )
    graph = graph_context_certificate(static, {
        "candidate_set_id": static["candidate_set_id"], "all_equal": True,
        "boot_id": "boot-one", "bitstream_reloaded": False,
        "mismatch_count_by_seed_and_output": {"0": [0], "1": [0], "2": [0]},
    })
    return static, graph


def test_pool_rejects_legacy_and_mode_ten_but_keeps_standard_and_mode_eleven():
    certificate = candidate_certificate(*records("m11", "w"))
    rows = [row("inc", "original", True), row("standard", "input_stationary"),
            row("m5", "input_weight_resident_barrier"),
            row("m10", "input_weight_resident_composable"),
            row("m11", "input_weight_resident_transition_safe")]
    result = qualify_pool(rows, {"m11": certificate})
    assert result["qualified_candidate_ids"] == ["inc", "standard", "m11"]
    assert result["rejected_candidates"]["m5"]["required_next_gate"] == "regenerate_as_mode11"
    assert result["rejected_candidates"]["m10"]["required_next_gate"] == "regenerate_as_mode11"


def test_measurement_plan_never_times_new_candidate_before_exact_graph_gate():
    certificate = candidate_certificate(*records("m11", "w"))
    static, graph = graph_certificate_for(certificate)
    rows = [row("inc", "original", True), row("m11", "input_weight_resident_transition_safe")]
    pending = measurement_plan(rows, fallback_contract(), {"m11": certificate})
    assert pending["timing_eligible_candidate_ids"] == ["inc"]
    assert pending["gate_by_candidate_id"]["m11"]["status"].startswith("GRAPH_CORRECTNESS")
    rows[1]["target_candidate_set_id"] = static["candidate_set_id"]
    qualified = measurement_plan(rows, fallback_contract(), {"m11": certificate}, [graph])
    assert qualified["timing_eligible_candidate_ids"] == ["inc", "m11"]


def test_deployment_requires_two_distinct_stable_boots():
    certificate = candidate_certificate(*records("m11", "w"))
    static, graph = graph_certificate_for(certificate)
    one_boot = [
        {"boot_id": "same", "median_improvement_pct": 3.0, "wins": 6, "rounds": 6},
        {"boot_id": "same", "median_improvement_pct": 3.1, "wins": 6, "rounds": 6},
    ]
    rejected = deployment_gate("m11", static["candidate_set_id"], [graph], one_boot)
    assert rejected["status"] == "FALL_BACK_TO_INCUMBENT"
    two_boots = copy.deepcopy(one_boot)
    two_boots[1]["boot_id"] = "other"
    accepted = deployment_gate("m11", static["candidate_set_id"], [graph], two_boots)
    assert accepted["status"] == "DEPLOY"


def test_unknown_completion_policy_fails_closed():
    result = qualify_pool([row("inc", "original", True), row("mystery", "new_magic")], {})
    assert result["rejected_candidates"]["mystery"]["required_next_gate"] == "classify_completion_policy"


def test_graph_gate_accepts_repeat_correctness_certificates_for_same_exact_set():
    certificate = candidate_certificate(*records("m11", "w"))
    static, first = graph_certificate_for(certificate)
    second = copy.deepcopy(first)
    second["graph_certificate_id"] = "another-boot-certificate"
    second["evidence"]["boot_id"] = "boot-two"
    indexed = graph_certificates_by_set([first, second])
    assert list(indexed) == [static["candidate_set_id"]]


def test_candidate_set_deployment_gate_binds_each_boot_to_exact_set():
    certificate = candidate_certificate(*records("m11", "w"))
    static, graph = graph_certificate_for(certificate)
    results = [
        {"candidate_set_id": static["candidate_set_id"], "boot_id": "one",
         "median_improvement_pct": 3.0, "wins": 6, "rounds": 6},
        {"candidate_set_id": static["candidate_set_id"], "boot_id": "two",
         "median_improvement_pct": 3.1, "wins": 5, "rounds": 6},
    ]
    accepted = candidate_set_deployment_gate(static["candidate_set_id"], [graph], results)
    assert accepted["status"] == "DEPLOY"
    assert accepted["deployment_unit"] == "exact_graph_candidate_set"
    results[1]["candidate_set_id"] = "another-set"
    rejected = candidate_set_deployment_gate(static["candidate_set_id"], [graph], results)
    assert rejected["status"] == "FALL_BACK_TO_INCUMBENT"
    assert "boot_1_candidate_set_identity_mismatch" in rejected["reasons"]


def test_graph_set_plan_uses_exact_set_certificates_not_member_certificates():
    certificate = candidate_certificate(*records("m11", "w"))
    static, graph = graph_certificate_for(certificate)
    incumbent = {
        "candidate_set_id": "incumbent-set", "is_incumbent": True,
        "aggregate_metrics": {}, "compatibility": {"compatible": True},
    }
    replacement = {
        "candidate_set_id": static["candidate_set_id"], "is_incumbent": False,
        "candidate_ids": ["m11"], "compatibility": {"compatible": True},
        "aggregate_metrics": {"total_request_bytes": 10, "total_request_calls": 2,
                              "submissions": 1, "uop_load_bytes": 3,
                              "instruction_bytes": 4},
        "predicted_graph_log_latency_ratio": -0.1,
    }
    pending = graph_set_measurement_plan([incumbent, replacement])
    assert pending["timing_eligible_candidate_set_ids"] == ["incumbent-set"]
    qualified = graph_set_measurement_plan([incumbent, replacement], [graph])
    assert qualified["timing_eligible_candidate_set_ids"] == [
        "incumbent-set", static["candidate_set_id"],
    ]
    assert qualified["selection_unit"] == "exact_graph_candidate_set"
    assert "selector_v3_static_fallback" in qualified["roles_by_candidate_set_id"][static["candidate_set_id"]]
    assert qualified["ranking_model"] == "static_command_fallback_no_completion_aware_model"
    assert qualified["performance_model_claim"] is False
    assert qualified["model_fallback_reason"] == "model_contract_not_provided"
