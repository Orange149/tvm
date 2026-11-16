"""Regression tests for the C1 E4 cross-model readiness freeze."""

import json

import pytest

import prepare_vta_c1_cross_model_e4 as e4


def test_default_e4_freeze_selects_six_per_model_and_fails_closed():
    result = e4.build_audit(
        [e4.GROUPED / name for name in e4.DEFAULT_CONTRACT_DIRS],
        e4.DEFAULT_C1_MODEL,
    )

    assert result["readiness_passed"]
    assert not result["e4_complete"]
    assert result["representative_counts"] == {
        "resnet50": 6,
        "yolov3_tiny": 6,
    }
    assert len(result["rows"]) == 12
    assert all(
        row["reuse_decision"] == "fail_closed_full_context_build_required"
        for row in result["rows"]
    )
    assert all(not row["checks"]["cross_placement_tile_stability_proven"] for row in result["rows"])
    assert all(not row["checks"]["normalized_fusion_context_available"] for row in result["rows"])


def test_contract_hash_tampering_is_rejected(tmp_path):
    source = e4.GROUPED / e4.DEFAULT_CONTRACT_DIRS[0]
    for name in ("contract.json", "candidates.jsonl", "artifact_hashes.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    contract = json.loads((tmp_path / "contract.json").read_text(encoding="utf-8"))
    contract["workload_id"] = "tampered"
    (tmp_path / "contract.json").write_text(
        json.dumps(contract, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="source hash mismatch"):
        e4.verify_contract_directory(tmp_path)
