#!/usr/bin/env python3
"""Semantic tests for the P7D formula freeze."""

import copy
from pathlib import Path

import pytest

import finalize_cpu_vta_pipeline_v1_p7d as p7d
from freeze_cpu_vta_pipeline_v1 import load_sealed_artifact
from solve_cpu_vta_pipeline_v1_p3 import build_context


OUTPUT = Path(__file__).resolve().parent / "report_out/resource_aware_maxplus"


def _p7b1():
    return load_sealed_artifact(
        OUTPUT / "v1_p7b1_three_boot_reproducibility.json",
        "cpu_vta_pipeline_v1_p7b1_three_boot_reproducibility",
    )


def test_memory_profile_admits_only_three_boot_streaming_read_write():
    rows = p7d.build_cpu_memory_profile(_p7b1())

    assert len(rows) == 36
    admitted = [row for row in rows if row["admitted_to_formula"]]
    assert len(admitted) == 8
    assert {row["pressure_class"] for row in admitted} == {"streaming"}
    assert {row["operation"] for row in admitted} == {"read", "write"}
    assert {row["threads"] for row in admitted} == {1, 2, 3, 4}
    assert all(row["cross_boot_cv"] <= 0.061 for row in rows)


def test_memory_profile_rejects_missing_matrix_entry():
    payload = copy.deepcopy(_p7b1())
    payload["memory_results"].pop()

    with pytest.raises(RuntimeError, match="memory matrix mismatch"):
        p7d.build_cpu_memory_profile(payload)


def test_component_admission_does_not_extrapolate_local_or_single_boot_signals():
    p7b1 = _p7b1()
    p7b2 = load_sealed_artifact(
        OUTPUT / "v1_p7b2_runtime_qualification_summary.json",
        "cpu_vta_pipeline_v1_p7b2_runtime_qualification_summary",
    )
    p7c = load_sealed_artifact(
        OUTPUT / "v1_p7c_qualification_summary.json",
        "cpu_vta_pipeline_v1_p7c_shared_ddr_qualification_summary",
    )

    admission = p7d.component_admission(p7b1, p7b2, p7c)

    assert admission["cpu_memory_streaming_bandwidth"]["admitted"]
    assert not admission["cpu_pair_slowdown"]["admitted"]
    assert admission["cpu_pair_slowdown"]["exact_observations_available"]
    assert not admission["runtime_frame_or_stage_intercept"]["admitted"]
    assert not admission["p7b2_boundary_slope"]["admitted"]
    assert not admission["cpu_vta_contention"]["admitted"]
    assert not admission["candidate_fitted_constant_or_scale"]["admitted"]


def test_streaming_override_changes_only_cpu_memory_rates():
    profile = load_sealed_artifact(
        OUTPUT / "v1_profile_manifest.json", "cpu_vta_pipeline_v1_profile_manifest"
    )
    local = load_sealed_artifact(
        OUTPUT / "v1_local_cost_table.json", "cpu_vta_pipeline_v1_local_cost_table"
    )
    context = build_context(profile, local)
    rows = p7d.build_cpu_memory_profile(_p7b1())
    revised = p7d.apply_streaming_memory_profile(context, rows)

    expected = next(
        row
        for row in rows
        if row["operation"] == "read" and row["threads"] == 1 and row["admitted_to_formula"]
    )
    assert revised["cpu_memory_models"][("read", 1)][
        "bandwidth_GBps_median"
    ] == pytest.approx(expected["bandwidth_GBps_median"])
    assert revised["segment_costs"] == context["segment_costs"]
    assert revised["boundary_ownership_models"] == context["boundary_ownership_models"]
    assert revised["vta_dma_qualification"] == context["vta_dma_qualification"]


def test_natural_top20_metrics_do_not_treat_pool_oracle_as_global():
    ranked = {
        "rows": [
            {"rank": rank, "candidate_id": "c{}".format(rank), "predicted_ii_ms": 9.0 + rank}
            for rank in range(1, 21)
        ]
    }
    board = {
        "summary": {"correctness_pass_count": 20},
        "rows": [
            {
                "candidate_id": "c{}".format(rank),
                "measurement": {"pipeline_fps": 1000.0 / (10.0 + rank)},
            }
            for rank in range(1, 21)
        ],
    }

    result = p7d.evaluate_natural_top20(ranked, board)

    assert result["complete_board_coverage"]
    assert "not the 972528-configuration global oracle" in result["measured_pool_scope"]
    assert result["throughput_regret_at_k"]["1"] == pytest.approx(0.0)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["score_tie_aware_spearman"] == pytest.approx(1.0)
