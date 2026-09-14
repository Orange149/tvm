"""Tests for P7R117 local no-leak extraction and skeleton construction."""

from __future__ import annotations

import json

import pytest

from run_vta_p7r117_yolo_no_leak_local import (
    DEFAULT_CANDIDATES,
    DEFAULT_CONTRACT,
    SEARCH_SCHEMA,
    build_no_leak_skeleton,
    normalize_command_structural,
    numeric_delta,
    static_worker_result,
    verify_inputs,
)


def test_verify_p7r115_frozen_input_and_gross_count():
    contract, candidates = verify_inputs(DEFAULT_CONTRACT, DEFAULT_CANDIDATES)
    assert contract["candidate_count"] == 36
    assert len(candidates) == 36
    assert {row["workload_id"] for row in candidates} == {"Y00", "Y01", "Y02"}


def test_numeric_delta_is_signed_mechanism_minus_original():
    control = {"status": "ok", "static_feature_vector": {"bytes": 100, "calls": 4}}
    mechanism = {"status": "ok", "static_feature_vector": {"bytes": 75, "calls": 5}}
    assert numeric_delta(mechanism, control) == {"bytes": -25.0, "calls": 1.0}
    assert numeric_delta({"status": "failed"}, control) is None


def test_skeleton_keeps_failures_in_gross_pool_and_pairs_family():
    candidates = [
        {
            "candidate_id": "o",
            "workload_id": "Y",
            "family_id": "YF00",
            "residence_mode": "original",
            "identity": {"complete_config_entity": {"entity": []}},
            "debug": {"config_index": 0},
        },
        {
            "candidate_id": "m",
            "workload_id": "Y",
            "family_id": "YF00",
            "residence_mode": "input_stationary",
            "identity": {"complete_config_entity": {"entity": []}},
            "debug": {"config_index": 0},
        },
    ]
    results = [
        {
            "candidate_id": "o",
            "status": "ok",
            "static_feature_vector": {"bytes": 100},
            "command_features": {"status": "requires_local_fsim_execution", "values": None},
        },
        {
            "candidate_id": "m",
            "status": "failed",
            "failure": {"category": "lower"},
        },
    ]
    skeleton = build_no_leak_skeleton(candidates, results)
    assert len(skeleton) == 2
    assert all(row["schema"] == SEARCH_SCHEMA for row in skeleton)
    assert skeleton[1]["same_tile_original_candidate_id"] == "o"
    assert skeleton[1]["same_tile_delta_static_features"] is None
    assert skeleton[1]["selection_eligibility"] == "failed_charged_no_replacement"
    assert "TopHub" in skeleton[0]["forbidden_feature_sources"]


def test_one_y01_static_worker_has_real_features_and_no_performance_label():
    _, candidates = verify_inputs(DEFAULT_CONTRACT, DEFAULT_CANDIDATES)
    candidate = next(
        row
        for row in candidates
        if row["workload_id"] == "Y01" and row["residence_mode"] == "original"
    )
    result = static_worker_result(candidate)
    assert result["status"] == "ok", json.dumps(result.get("failure"), indent=2)
    assert result["worker_wall_seconds"] > 0
    assert result["performance_label"] is None
    assert result["static_feature_vector"]["load_dma_calls"] > 0
    assert result["static_feature_vector"]["weight_dma_bytes"] > 0
    assert result["sram_features"]["certificate_passed"]
    assert result["command_features"]["status"] == "requires_local_fsim_execution"


def test_missing_original_fails_closed():
    candidate = {
        "candidate_id": "m",
        "workload_id": "Y",
        "family_id": "YF00",
        "residence_mode": "input_stationary",
        "identity": {"complete_config_entity": {"entity": []}},
        "debug": {"config_index": 0},
    }
    with pytest.raises(ValueError, match="missing same-tile original"):
        build_no_leak_skeleton([candidate], [{"candidate_id": "m", "status": "failed"}])


def test_command_normalization_ignores_only_process_global_submit_number():
    left = {"totals": {"insn_bytes": 32}, "submit_sequence": [{"submit": 7, "insn_bytes": 32}]}
    right = {"totals": {"insn_bytes": 32}, "submit_sequence": [{"submit": 99, "insn_bytes": 32}]}
    changed = {"totals": {"insn_bytes": 48}, "submit_sequence": [{"submit": 7, "insn_bytes": 48}]}
    assert normalize_command_structural(left) == normalize_command_structural(right)
    assert normalize_command_structural(left) != normalize_command_structural(changed)


if __name__ == "__main__":
    pytest.main([__file__])
