#!/usr/bin/env python3
"""Tests for the local-only RAMPS source and budget audit."""

import audit_ramps_budgeted_search as audit
import resource_aware_maxplus as ramps


def synthetic_record(index):
    provenance = {
        "workload_features": {"kind": "compiler_static", "source": "test"},
        "service_parameters": {"kind": "legacy_default_static", "source": "test"},
    }
    return ramps.PipelineRecord(
        model="synthetic",
        candidate_id="c{}".format(index),
        source_path="test",
        group_id="g{}".format(index % 2),
        measured_cycle_ms=float(index + 1),
        queue_depth=2,
        stages=[ramps.StageService("cpu", "cpu", compute_ms=float(index + 1))],
        legacy_score_ms=float(index + 1),
        metadata={"score_input_provenance": provenance},
    )


def test_budget_audit_keeps_outcomes_as_labels_only():
    records = [synthetic_record(index) for index in range(6)]
    result = audit.evaluate_records(records, random_repetitions=10, seed=7)
    assert result["scope"] == "retrospective_measured_pool_only"
    assert result["models"]["m0_single_vta_compute"]["metrics"]["regret_at_1"] == 0.0
    assert "evaluations_to_oracle_98pct" in result["random_uniform"]["metrics"]
    provenance = audit.source_audit(records)
    assert provenance["zero_feedback_eligible_m0_m1_count"] == len(records)
    assert provenance["m2_measured_core_demand_count"] == 0
