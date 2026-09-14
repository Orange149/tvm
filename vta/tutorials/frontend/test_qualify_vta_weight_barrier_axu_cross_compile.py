"""Tests for P4i weight-barrier AXU5EVB cross-compile selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qualify_vta_weight_barrier_axu_cross_compile import (
    expected_source_hashes,
    select_candidates,
    summarize,
    validate_p4g_artifacts,
)


REPO = Path(__file__).resolve().parents[3]
P4G = (
    Path(__file__).resolve().parent
    / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/04_dma_command_signatures"
    / "20260911_p4g_weight_barrier_pool_run01/results.jsonl"
)


def rows():
    return [json.loads(line) for line in P4G.read_text().splitlines() if line]


def test_frozen_p4g_artifacts_and_exact_selection():
    evidence = validate_p4g_artifacts(P4G)
    selected = select_candidates(rows(), expected_source_hashes(REPO))
    assert len(evidence["verified_output_sha256"]) == 10
    assert len(selected) == 32
    assert len({row["candidate_id"] for row in selected}) == 32
    assert all(row["fsim_qualification"]["overall_status"] == "passed" for row in selected)


def test_selection_rejects_stable_id_tamper():
    modified = rows()
    passing = next(row for row in modified if row["status"] == "ok")
    passing["identity"] = dict(passing["identity"])
    passing["identity"]["implementation_name"] = "tampered"
    with pytest.raises(ValueError, match="stable ID mismatch"):
        select_candidates(modified, expected_source_hashes(REPO))


def test_summary_retains_failure_vocabulary_and_claim_boundary():
    record = {
        "workload_id": "W00",
        "overall_status": "failed",
        "failure": {"category": "timeout", "subcategory": "candidate_worker_timeout"},
        "export": {"binary_retained": False},
    }
    summary = summarize([record], 1)
    assert summary["failure_categories"] == {"timeout": 1}
    assert summary["failure_subcategories"] == {"candidate_worker_timeout": 1}
    assert summary["board_access"] is False
    assert summary["performance_measurement"] == "not_collected"


if __name__ == "__main__":
    pytest.main([__file__])
