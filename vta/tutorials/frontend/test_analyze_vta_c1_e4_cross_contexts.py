"""Regression tests for the E4 paired-context evidence summary."""

import json
from pathlib import Path

import pytest

import analyze_vta_c1_e4_cross_contexts as analyze


SOURCE = Path(__file__).resolve().parent / (
    "report_out/stage_tile_cotuning/compile_context_audit/"
    "c1_e4_cross_context_run1/summary.json"
)


def test_authoritative_e4_analysis_preserves_positive_and_negative_cases():
    result = analyze.analyze(SOURCE)
    summary = result["summary"]

    assert summary["requested_workloads"] == 12
    assert summary["complete_paired_contexts"] == 11
    assert summary["exact_frozen_dispatch"] == 11
    assert summary["same_tile_identity"] == 11
    assert summary["distinct_fusion_signatures"] == 11
    assert summary["base_dma_invariant"] == 11
    assert summary["positive_acc_load_residual"] == 11
    assert summary["failure_classes"] == {"acc_buffer_capacity_exceeded": 1}


def test_context_summary_tampering_is_rejected(tmp_path):
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload["rows"][0]["workload_id"] = "tampered"
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        analyze.analyze(path)
