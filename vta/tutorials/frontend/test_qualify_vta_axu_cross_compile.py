"""Tests for the P4f AXU5EVB cross-compile qualification orchestrator."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from qualify_vta_axu_cross_compile import select_candidates, summarize


BASE = Path(__file__).resolve().parent / "report_out" / "stage_tile_cotuning"
P4 = BASE / "c3_dma_residency_autotune" / "04_dma_command_signatures"
P4B = P4 / "20260910_p4b_local_residency_pool_run01" / "results.jsonl"
P4E = P4 / "20260911_p4e_unified_local_qualification_run01" / "results.jsonl"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_real_selection_is_exact_165_and_splits_10_plus_155():
    selected = select_candidates(rows(P4B), rows(P4E))
    assert len(selected) == 165
    assert len({row["candidate_id"] for row in selected}) == 165
    roles = Counter(row["candidate_role"] for row in selected)
    assert roles == {
        "protected_original_incumbent": 10,
        "same_tile_original_control": 45,
        "residency_experiment": 110,
    }
    assert sum(row["candidate_role"] != "protected_original_incumbent" for row in selected) == 155


def test_selection_rejects_certificate_tir_mismatch():
    p4b = rows(P4B)
    p4e = rows(P4E)
    passing = next(row for row in p4e if row["local_status"] == "fsim_passed")
    modified = [dict(row) for row in p4e]
    target = next(row for row in modified if row["candidate_id"] == passing["candidate_id"])
    target["tir_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="TIR hash mismatch"):
        select_candidates(p4b, modified)


def test_summary_separates_fresh_and_referenced_evidence():
    common = {
        "residence_mode": "original",
        "overall_status": "passed",
        "failure": None,
        "export": {"binary_retained": False},
    }
    records = [
        {**common, "qualification_source": "referenced_p2_run02_cross_compile"},
        {**common, "qualification_source": "fresh_isolated_cross_compile"},
    ]
    result = summarize(records, 2)
    assert result["status"] == "completed"
    assert result["passed"] == 2
    assert result["referenced_p2_incumbents"] == 1
    assert result["fresh_cross_compile_records"] == 1
    assert result["binary_retained_count"] == 0


def test_summary_keeps_machine_readable_failure_categories():
    record = {
        "residence_mode": "input_stationary",
        "overall_status": "failed",
        "qualification_source": "fresh_isolated_cross_compile",
        "failure": {"category": "timeout", "subcategory": "candidate_worker_timeout"},
        "export": {"binary_retained": False},
    }
    result = summarize([record], 1)
    assert result["failure_categories"] == {"timeout": 1}
    assert result["failure_subcategories"] == {"candidate_worker_timeout": 1}


if __name__ == "__main__":
    pytest.main([__file__])
