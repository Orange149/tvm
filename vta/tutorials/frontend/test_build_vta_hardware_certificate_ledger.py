"""Tests for exact hardware-scoped VTA correctness certificates."""

import pytest

from build_vta_hardware_certificate_ledger import (
    certificate_identity,
    hardware_fingerprint,
    ingest,
    route,
)


def contract():
    return {
        "workload_id": "E99",
        "workload": ["conv", 1],
        "board": {"boot_id": "boot", "fpga_state": "operating", "udmabuf_bytes": 4096},
        "source_guards_sha256": {"schedule.py": "abc"},
    }


def row(correct, config_index=7, entity_value=1, tir_sha256="tir"):
    return {
        "mode": "input_stationary",
        "config_index": config_index,
        "complete_config_entity": {
            "index": config_index,
            "code_hash": None,
            "entity": [["tile_h", "sp", [-1, entity_value]]],
        },
        "tir_sha256": tir_sha256,
        "candidate_id": "candidate",
        "correct": correct,
        "seeds": [{"correct": correct, "mismatch_count": 0 if correct else 4}],
        "binary_sha256": "binary",
        "performance_measurement": "not_collected",
        "role": "probe",
    }


def test_routes_only_exact_passes_to_timing():
    assert route("passed") == "allow_timing"
    assert route("failed") == "reject_candidate"
    assert route("unknown") == "correctness_canary_only"


def test_rejects_contradictory_certificate_for_same_identity():
    ledger = {}
    ingest(contract(), [row(True)], ledger)
    with pytest.raises(ValueError, match="contradictory"):
        ingest(contract(), [row(False)], ledger)


def identity_key(item, target_contract=None):
    target_contract = contract() if target_contract is None else target_contract
    fingerprint = hardware_fingerprint(target_contract)
    return certificate_identity(target_contract, item, fingerprint["key"])[0]


def test_config_index_is_audit_only_and_renumbering_is_stable():
    first = row(True, config_index=7)
    renumbered = row(True, config_index=98123)
    assert identity_key(first) == identity_key(renumbered)

    ledger = {}
    ingest(contract(), [first, renumbered], ledger)
    assert len(ledger) == 1
    certificate = next(iter(ledger.values()))
    assert "config_index" not in certificate["identity"]
    assert certificate["audit"]["observed_config_indices"] == [7, 98123]
    assert certificate["observations"] == 2


def test_entity_tir_and_hardware_changes_change_identity():
    original = row(True)
    changed_entity = row(True, entity_value=2)
    changed_tir = row(True, tir_sha256="other-tir")
    changed_hardware = contract()
    changed_hardware["board"] = dict(changed_hardware["board"], boot_id="other-boot")

    original_key = identity_key(original)
    assert identity_key(changed_entity) != original_key
    assert identity_key(changed_tir) != original_key
    assert identity_key(original, changed_hardware) != original_key


def test_missing_or_incomplete_config_entity_fails_closed():
    missing = row(True)
    del missing["complete_config_entity"]
    with pytest.raises(ValueError, match="refusing legacy config_index-only identity"):
        identity_key(missing)

    incomplete = row(True)
    incomplete["complete_config_entity"] = {"index": 7, "code_hash": None}
    with pytest.raises(ValueError, match="mapping containing entity"):
        identity_key(incomplete)


def test_row_and_seed_correctness_contradiction_fails_closed():
    contradictory = row(True)
    contradictory["seeds"][0]["correct"] = False
    with pytest.raises(ValueError, match="contradicts per-seed evidence"):
        ingest(contract(), [contradictory], {})


def test_empty_seed_evidence_fails_closed():
    no_seeds = row(True)
    no_seeds["seeds"] = []
    with pytest.raises(ValueError, match="non-empty seed evidence"):
        ingest(contract(), [no_seeds], {})
