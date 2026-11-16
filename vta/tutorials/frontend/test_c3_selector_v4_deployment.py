import copy

import pytest

from c3_completion_policy import TRANSITION_SAFE_FORCE_SERIAL
from c3_selector_v4_deployment import production_static_order


def row(candidate_id, mode, request_bytes, *, incumbent=False, locally_correct=True):
    return {
        "candidate_id": candidate_id,
        "program_id": "program_" + candidate_id,
        "residence_mode": mode,
        "is_incumbent": incumbent,
        "local_qualification_passed": locally_correct,
        "performance_label": None,
        "timing_freeze_metrics": {
            "total_request_bytes": request_bytes,
            "total_request_calls": 10,
            "submissions": 3,
            "uop_load_bytes": 20,
            "instruction_bytes": 100,
        },
        "visible_features": {
            "tile_b": 1,
            "tile_h": 4,
            "tile_w": 4,
            "tile_ci": 1,
            "tile_co": 1,
            "oc_nthread": 1,
            "h_nthread": 1,
        },
    }


def certificate(candidate_id):
    return {
        "schema": "c3_composability_candidate_certificate_v1",
        "status": "LOCAL_SCOPED_COMPLETION_QUALIFIED",
        "identity": {
            "candidate_id": candidate_id,
            "completion_policy": TRANSITION_SAFE_FORCE_SERIAL,
        },
        "checks": {"qualified": True},
        "completion_contract": {
            "global_debug_flag_required": 0,
            "final_transition": [32, 0],
            "blocking_completion_between_enter_and_exit": True,
            "entry_drains_before_debug": 1,
        },
    }


def test_legacy_barrier_is_retained_for_reference_but_rejected_for_deployment():
    rows = [
        row("inc", "original", 500, incumbent=True),
        row("input", "input_stationary", 300),
        row("mode5", "input_weight_resident_barrier", 1),
        row("mode11", "input_weight_resident_transition_safe", 100),
    ]
    result = production_static_order(rows, {"mode11": certificate("mode11")})
    assert "mode5" in result["production_rejected_candidate_ids"]
    assert result["candidate_dispositions"]["mode5"]["required_next_gate"] == "regenerate_as_mode11"
    assert "program_mode5" not in result["ordered_program_ids"]
    assert result["budget_prefixes"]["B1"] == ["program_mode11"]


def test_transition_safe_candidate_fails_closed_without_certificate():
    rows = [
        row("inc", "original", 500, incumbent=True),
        row("mode11", "input_weight_resident_transition_safe", 100),
    ]
    with pytest.raises(ValueError, match="no production-eligible"):
        production_static_order(rows, {})


def test_nonresident_candidate_requires_local_correctness():
    rows = [
        row("inc", "original", 500, incumbent=True),
        row("input", "input_stationary", 100, locally_correct=False),
    ]
    with pytest.raises(ValueError, match="no production-eligible"):
        production_static_order(rows, {})


def test_latency_label_is_still_rejected_after_semantic_gate():
    rows = [
        row("inc", "original", 500, incumbent=True),
        row("input", "input_stationary", 100),
    ]
    rows[1]["latency_ms"] = 1.0
    with pytest.raises(ValueError, match="labels are forbidden"):
        production_static_order(rows, {})


def test_exactly_one_incumbent_and_unique_programs_are_required():
    rows = [
        row("a", "original", 500, incumbent=True),
        row("b", "input_stationary", 100, incumbent=True),
    ]
    with pytest.raises(ValueError, match="exactly one incumbent"):
        production_static_order(rows, {})
    rows[1]["is_incumbent"] = False
    rows[1]["program_id"] = rows[0]["program_id"]
    with pytest.raises(ValueError, match="program IDs"):
        production_static_order(rows, {})
