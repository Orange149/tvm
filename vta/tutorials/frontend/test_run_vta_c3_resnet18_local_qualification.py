import copy
import json
from pathlib import Path

import run_vta_c3_resnet18_local_qualification as local


def pools():
    _, values = local.verify_p7r470(local.DEFAULT_INPUT)
    return {payload["workload_id"]: payload for _, payload in values}


def contextual(row, pool):
    return local.candidate_with_context(row, pool)


def test_frozen_input_is_three_label_blind_pools_with_288_identities():
    hashes, values = local.verify_p7r470(local.DEFAULT_INPUT)
    assert len(hashes) >= 9
    assert len(values) == 3
    assert sum(len(payload["candidates"]) for _, payload in values) == 288


def test_h2_weight_modes_are_not_applicable_and_fallback_to_exact_original():
    pool = pools()["R18-H2"]
    family = [contextual(row, pool) for row in pool["candidates"] if row["family_id"] == "R18-H2F00"]
    original = next(row for row in family if row["residence_mode"] == "original")
    for mode in ("weight_resident_barrier", "input_weight_resident_barrier"):
        candidate = next(row for row in family if row["residence_mode"] == mode)
        result = local.static_result(candidate, family)
        assert result["status"] == "not_applicable"
        assert result["fallback"]["candidate_id"] == original["candidate_id"]
        assert result["performance_label"] is None


def test_real_lowering_binds_tir_identity_and_hidden_dma_features():
    pool = pools()["R18-H1"]
    row = next(
        row
        for row in pool["candidates"]
        if row["debug"]["config_index"] == 0 and row["residence_mode"] == "original"
    )
    candidate = contextual(row, pool)
    family = [contextual(item, pool) for item in pool["candidates"] if item["family_id"] == row["family_id"]]
    result = local.static_result(candidate, family)
    assert result["status"] == "ok"
    assert len(result["lowered_tir_sha256"]) == 64
    assert len(result["implementation_candidate_id"]) == 64
    assert result["hidden_compiler_features"]["loop_count"] > 0
    assert result["hidden_compiler_features"]["tensorize_uop_sites"] > 0
    assert result["dma_features"]["logical_calls"]["load"] > 0
    assert result["dma_features"]["scope"].startswith("logical VTA")
    assert result["performance_label"] is None


def test_live_future_label_is_rejected(tmp_path):
    source = local.DEFAULT_INPUT
    for path in source.iterdir():
        if path.is_file():
            (tmp_path / path.name).write_bytes(path.read_bytes())
    target = tmp_path / "r18-h1_four_mode_96.json"
    payload = json.loads(target.read_text())
    payload["candidates"][0]["performance_label"] = 1.25
    target.write_text(json.dumps(payload))
    ledger_path = tmp_path / "artifact_hashes.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["artifacts"][target.name] = local.sha256_file(target)
    ledger_path.write_text(json.dumps(ledger))
    try:
        local.verify_p7r470(tmp_path)
    except ValueError as error:
        assert "future-label" in str(error)
    else:
        raise AssertionError("non-null target performance label was accepted")


def test_command_normalization_removes_process_global_submit_number():
    value = {"submit_sequence": [{"submit": 91, "reason": "explicit_sync"}]}
    normalized = local.normalize_command_signature(copy.deepcopy(value))
    assert "submit" not in normalized["submit_sequence"][0]
    assert normalized["submit_sequence"][0]["submit_ordinal_within_inference"] == 1
