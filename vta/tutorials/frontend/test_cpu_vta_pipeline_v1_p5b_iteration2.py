#!/usr/bin/env python3
"""Tests for P5B Iteration2 CPU segment calibration."""

from pathlib import Path

import iterate_cpu_vta_pipeline_v1_p5b_iteration2 as iteration2


OUTPUT = Path(__file__).resolve().parent / "report_out/resource_aware_maxplus"


def test_atomic_profiles_are_reference_checked_and_deterministic():
    wall, core, sessions, reference = iteration2.summarize_atomic_profiles(OUTPUT)
    assert reference["passed"] is True
    assert len(sessions) == 4
    assert all(item["rows"] == 7 and item["scored_rows"] == 5 for item in sessions)
    assert all(item["top1_values"] == [282] for item in sessions)
    assert all(wall[index, threads] > 0 and core[index, threads] > 0
               for index in range(21) for threads in iteration2.THREADS)


def test_atomic_composition_passes_grouped_fused_segment_holdout():
    wall, _, _, _ = iteration2.summarize_atomic_profiles(OUTPUT)
    checks, metrics, gate = iteration2.build_grouped_holdouts(OUTPUT, wall)
    assert len(checks) == 23
    assert metrics["median_ape"] < 0.01
    assert metrics["p95_ape"] < 0.05
    assert metrics["max_ape"] < 0.20
    assert all(gate.values())


def test_measured_pool_replay_uses_no_fit_labels_and_preserves_order():
    profile = iteration2.load_sealed_artifact(
        OUTPUT / "v1_profile_manifest.json", "cpu_vta_pipeline_v1_profile_manifest"
    )
    local = iteration2.load_sealed_artifact(
        OUTPUT / "v1_local_cost_table.json", "cpu_vta_pipeline_v1_local_cost_table"
    )
    wall, core, _, _ = iteration2.summarize_atomic_profiles(OUTPUT)
    context, count = iteration2.apply_atomic_cpu_costs(
        iteration2.build_context(profile, local), wall, core
    )
    rows, replay = iteration2.rank_measured_pool(context, OUTPUT)
    assert count == 85
    assert len(rows) == 5
    assert replay["top1_match"] is True
    assert replay["exact_order_match"] is True
    assert "not used to fit" in replay["evidence_role"]
