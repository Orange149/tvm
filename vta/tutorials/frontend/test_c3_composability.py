import copy

import pytest

from c3_composability import (
    GRAPH_STATUS,
    STATIC_SET_STATUS,
    candidate_certificate,
    candidate_disposition,
    graph_context_certificate,
    static_set_certificate,
)


def records(candidate_id="c0", workload_id="w0"):
    candidate = {
        "candidate_id": candidate_id, "workload_id": workload_id,
        "implementation_mode": 11,
        "residence_mode": "input_weight_resident_transition_safe",
        "parent_mode5_candidate_id": "p-" + candidate_id,
        "candidate_identity_fields": {"schedule_version": "schedule"},
    }
    static = {
        "candidate_id": candidate_id, "status": "ok", "lowered_tir_sha256": "tir",
        "storage_tir_sha256": "storage", "implementation_candidate_id": "impl",
        "arena_capacity_certificate": {"status": "passed"},
        "final_address_audit": {
            "status": "passed", "debug_mode_transitions": [32, 0],
            "sync_event_positions": [0, 10, 11],
        },
        "dependency_audit": {"balanced": True, "forbidden_direct_1_3_count": 0},
    }
    fsim = {
        "candidate_id": candidate_id, "status": "passed",
        "seeds": [{"seed": seed, "correct": True, "actual_sha256": str(seed)}
                  for seed in (0, 1, 2)],
        "command_features": {"status": "available_three_seed_consistent"},
    }
    return candidate, static, fsim


def test_three_certificate_levels_do_not_overclaim_graph_correctness():
    one = candidate_certificate(*records("c1", "w1"))
    two = candidate_certificate(*records("c2", "w2"))
    static = static_set_certificate(
        [one, two], {"graph_sha256": "graph", "params_sha256": "params", "model": "m"},
        {"status": "passed", "global_vta_debug_flag": 0,
         "calls_by_function": {"f1": 2, "f2": 2}},
    )
    assert static["status"] == STATIC_SET_STATUS
    assert static["graph_correctness_claimed"] is False
    graph = graph_context_certificate(static, {
        "candidate_set_id": static["candidate_set_id"], "all_equal": True,
        "boot_id": "boot", "bitstream_reloaded": False,
        "mismatch_count_by_seed_and_output": {"0": [0], "1": [0], "2": [0]},
        "complete_graph_run_calls": 6, "timing_collected": False,
        "performance_claim": False,
    })
    assert graph["status"] == GRAPH_STATUS
    assert graph["timing_eligible"] is True


def test_invalid_or_unclosed_completion_is_rejected():
    candidate, static, fsim = records()
    static["final_address_audit"]["debug_mode_transitions"] = [32]
    with pytest.raises(ValueError, match="qualification failed"):
        candidate_certificate(candidate, static, fsim)


def test_static_set_rejects_debug_scope_escape():
    certificate = candidate_certificate(*records())
    with pytest.raises(ValueError, match="exact enter/exit"):
        static_set_certificate(
            [certificate], {"graph_sha256": "g", "params_sha256": "p"},
            {"status": "passed", "global_vta_debug_flag": 0,
             "calls_by_function": {"target": 2, "escaped": 1}},
        )


def test_graph_gate_requires_exact_set_and_zero_mismatches():
    certificate = candidate_certificate(*records())
    static = static_set_certificate(
        [certificate], {"graph_sha256": "g", "params_sha256": "p"},
        {"status": "passed", "global_vta_debug_flag": 0,
         "calls_by_function": {"target": 2}},
    )
    with pytest.raises(ValueError, match="another candidate set"):
        graph_context_certificate(static, {"candidate_set_id": "wrong"})
    bad = {"candidate_set_id": static["candidate_set_id"], "all_equal": False,
           "boot_id": "boot", "bitstream_reloaded": False,
           "mismatch_count_by_seed_and_output": {"0": [1], "1": [0], "2": [0]}}
    with pytest.raises(ValueError, match="correctness gate failed"):
        graph_context_certificate(static, bad)


def test_selector_disposition_rejects_legacy_and_missing_certificates():
    incumbent = candidate_disposition({"candidate_id": "i", "is_incumbent": True}, {})
    assert incumbent["eligible"]
    legacy = candidate_disposition(
        {"candidate_id": "m5", "candidate_mode": "input_weight_resident_barrier"}, {}
    )
    assert not legacy["eligible"] and legacy["required_next_gate"] == "regenerate_as_mode11"
    missing = candidate_disposition(
        {"candidate_id": "m11", "candidate_mode": "input_weight_resident_transition_safe"}, {}
    )
    assert not missing["eligible"]
    cert = candidate_certificate(*records("m11", "w"))
    accepted = candidate_disposition(
        {"candidate_id": "m11", "candidate_mode": "input_weight_resident_transition_safe"},
        {"m11": cert},
    )
    assert accepted["eligible"] and accepted["required_next_gate"] == "exact_graph_context_correctness"

    predecessor_sensitive = candidate_disposition(
        {"candidate_id": "m10", "candidate_mode": "input_weight_resident_composable"}, {}
    )
    assert predecessor_sensitive == {
        "eligible": False,
        "reason": "predecessor_sensitive_mode10_not_transition_safe",
        "required_next_gate": "regenerate_as_mode11",
    }
