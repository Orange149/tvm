"""Accounting and isolation tests for the R18 literature-pool analyzer."""

from __future__ import annotations

import hashlib

import analyze_vta_c3_resnet18_literature_pool as analysis


def _id(text):
    return hashlib.sha256(text.encode()).hexdigest()


def candidate(name, mode="original"):
    return {
        "candidate_id": _id(name + mode),
        "workload_id": "W0",
        "family_id": name,
        "residence_mode": mode,
        "complete_config_entity": {"entity": []},
        "visible_features": {},
        "static_features": {},
        "workload_features": {},
    }


def phase_outcome(latency=1.0):
    return {
        "lower": {"status": "ok", "wall_seconds": 1.0, "compiler_attempts": 1},
        "fsim": {"status": "ok", "wall_seconds": 2.0},
        "compile": {"status": "ok", "wall_seconds": 3.0, "compiler_attempts": 1},
        "fpga_correctness": {
            "status": "ok",
            "wall_seconds": 4.0,
            "fpga_kernel_invocations": 3,
            "logical_dma_bytes": 100,
            "logical_dma_calls": 10,
        },
        "timing": {
            "status": "ok",
            "wall_seconds": 5.0,
            "latency_ms": latency,
            "fpga_kernel_invocations": 10,
            "logical_dma_bytes": 200,
            "logical_dma_calls": 20,
        },
    }


def test_target_metrics_records_first_trial_and_wall_to_quality():
    rows = [
        {
            "dispatch": 1,
            "candidate_id": "a",
            "latency_ms": 12.0,
            "cumulative_wall_seconds": 3.0,
        },
        {
            "dispatch": 2,
            "candidate_id": "b",
            "latency_ms": 10.1,
            "cumulative_wall_seconds": 7.0,
        },
    ]
    value = analysis.target_metrics(rows, {"latency_ms": 10.0})
    assert value["first_oracle_2_percent"]["dispatch"] == 2
    assert value["first_oracle_2_percent"]["best_candidate_id"] == "b"
    assert value["first_oracle_2_percent"]["cumulative_wall_seconds"] == 7.0
    assert value["final_best_candidate_id"] == "b"
    assert value["final_regret_fraction"] == 0.010000000000000009


def test_cheng_excludes_families_without_original_fallback():
    good = [candidate("F0", "original"), candidate("F0", "input_stationary")]
    missing = [candidate("F1", "weight_resident_barrier")]
    eligible, excluded = analysis.cheng_candidates(good + missing)
    assert {row["family_id"] for row in eligible} == {"F0"}
    assert excluded[0]["family_id"] == "F1"


def test_validity_metrics_report_f1_recall_and_top20_yield():
    labels = [True, False, True, False]
    probabilities = [0.9, 0.8, 0.7, 0.1]
    value = analysis.validity_metrics(labels, probabilities, top_k=2)
    assert value["tp"] == 2
    assert value["fp"] == 1
    assert value["fn"] == 0
    assert value["recall"] == 1.0
    assert value["f1"] == 0.8
    assert value["top_k_valid_yield"] == 0.5
    assert value["base_valid_yield"] == 0.5


def test_rieber_fsim_failure_counts_in_gross_and_wall_before_board():
    passed = candidate("pass")
    failed_id = _id("failedoriginal")
    payload = {
        "schema": "c3_literature_outcome_ledger_v1",
        "session_status": "complete",
        "pool_complete": True,
        "outcomes": {passed["candidate_id"]: phase_outcome()},
    }
    static = {failed_id: {"status": "ok", "wall_seconds": 1.5}}
    fsim = {failed_id: {"status": "failed", "wall_seconds": 2.5}}
    timeline, results, diagnostic = analysis.replay_rieber(
        [passed],
        payload,
        [failed_id, passed["candidate_id"]],
        static,
        fsim,
        [
            {
                **candidate("failed"),
                "candidate_id": failed_id,
                "is_valid": True,
                "wall_seconds": 1.5,
            },
            {**passed, "is_valid": True, "wall_seconds": 1.0},
        ],
        2,
        7,
        [],
    )
    assert len(results) == 2
    assert results[0]["is_valid"] is False
    assert results[0]["cumulative_wall_seconds"] == 5.0
    assert [row["phase"] for row in timeline[:3]] == ["lower", "lower", "fsim"]
    assert sum(row["phase"] == "lower" for row in timeline) == 2
    assert diagnostic["presampling_compiler_calls"] == 2
    assert diagnostic["presampling_wall_seconds"] == 2.5
    assert diagnostic["balanced_valid_e0_fsim_rejected"] == 1
