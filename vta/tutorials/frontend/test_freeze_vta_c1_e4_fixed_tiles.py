"""Regression tests for the pre-context E4 tile freeze."""

import json

import pytest

import freeze_vta_c1_e4_fixed_tiles as freeze


def qualification_dirs():
    return {
        key: freeze.GROUPED / value
        for key, value in freeze.DEFAULT_QUALIFICATION_DIRS.items()
    }


def test_default_fixed_tiles_are_label_free_and_locally_qualified():
    result = freeze.build_freeze(freeze.DEFAULT_READINESS, qualification_dirs())

    assert result["status"] == "fixed_before_cross_context_lowering"
    assert len(result["rows"]) == 12
    assert result["gate_checks"]["all_unique_workloads"]
    assert result["gate_checks"]["all_static_and_fsim_qualified"]
    assert result["gate_checks"]["performance_labels_used"] is False
    assert result["gate_checks"]["cross_context_results_seen_before_freeze"] is False
    assert len({row["tile_identity_sha256"] for row in result["rows"]}) == 12


def test_readiness_tampering_is_rejected(tmp_path):
    payload = json.loads(freeze.DEFAULT_READINESS.read_text(encoding="utf-8"))
    payload["rows"][0]["workload_id"] = "tampered"
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        freeze.build_freeze(path, qualification_dirs())
