"""Tests for exact allowlist-derived VTA command-memory certificates."""

import pytest

from build_vta_command_resource_certificate import (
    build_certificate,
    canonical_hash,
    validate_certificate,
)


def candidate(candidate_id, config_index, mode="paper_inspired_hybrid"):
    return {
        "candidate_id": candidate_id,
        "workload_id": "W05",
        "public_mode": mode,
        "debug": {"config_index": config_index},
        "expected_tir_sha256": "tir-{}".format(candidate_id),
        "identity": {"complete_config_entity": {"index": config_index, "entity": []}},
    }


def signature(candidate_id, insn, uop, mode="paper_inspired_hybrid"):
    structural = {
        "peaks": {"insn_bytes": insn, "uop_bytes": uop},
        "submissions": 1,
        "finish": {"source_derived_count": 1},
    }
    return {
        "candidate_id": candidate_id,
        "workload_id": "W05",
        "public_mode": mode,
        "relowered_tir_sha256": "tir-{}".format(candidate_id),
        "outcome": "passed_local_signature",
        "execution_runtime": {"path": "/tmp/libvta_fsim.so", "sha256": "f" * 64},
        "command_signature": {
            "sha256": canonical_hash(structural),
            "structural": structural,
        },
    }


def dispatch(candidate_ids):
    return {
        "schema": "c3_vta_hardware_certified_dispatch_v1",
        "performance_labels_used": False,
        "records": [
            {"route": "allow_timing", "candidate_id": candidate_id}
            for candidate_id in candidate_ids
        ],
    }


def test_capacity_is_max_over_exact_allowlist_then_aligned():
    candidates = [candidate("a", 1), candidate("b", 2), candidate("unused", 3)]
    signatures = [
        signature("a", 3744, 1480),
        signature("b", 7200, 752),
        signature("unused", 23008, 10820),
    ]
    certificate = build_certificate(
        dispatch(["a", "b"]),
        candidates,
        signatures,
        {"TARGET": "generic-fpga", "LOG_UOP_BUFF_SIZE": 15},
        {"vta/runtime/runtime.cc": "runtime-hash"},
    )
    assert certificate["derivation"] == {
        "formula": "C_q(S)=align_A(max_{c in S,b in submits(c)} B_q(c,b))",
        "alignment_bytes": 4096,
        "insn_peak_bytes": 7200,
        "uop_peak_bytes": 1480,
        "insn_capacity_bytes": 8192,
        "uop_capacity_bytes": 4096,
        "total_capacity_bytes": 12288,
    }
    assert {row["candidate_id"] for row in certificate["identities"]} == {"a", "b"}
    assert certificate["boot_id_bound"] is False
    assert validate_certificate(certificate, required_candidate_ids=["a"])["environment"] == {
        "VTA_INSN_BUFFER_BYTES": "8192",
        "VTA_UOP_BUFFER_BYTES": "4096",
    }


def test_explicit_fallback_is_included_and_missing_signature_fails_closed():
    candidates = [candidate("timed", 1), candidate("fallback", 2, mode="original")]
    signatures = [signature("timed", 1024, 512)]
    with pytest.raises(ValueError, match="missing candidate/signature"):
        build_certificate(
            dispatch(["timed"]),
            candidates,
            signatures,
            {"TARGET": "generic-fpga"},
            {},
            explicit_ids=["fallback"],
        )


def test_exact_deployment_subset_changes_capacity_and_cannot_bypass_dispatch():
    candidates = [candidate("a", 1), candidate("b", 2), candidate("fallback", 3)]
    signatures = [
        signature("a", 3000, 1000),
        signature("b", 9000, 1000),
        signature("fallback", 3500, 1200),
    ]
    certificate = build_certificate(
        dispatch(["a", "b"]),
        candidates,
        signatures,
        {"TARGET": "generic-fpga"},
        {},
        explicit_ids=["fallback"],
        selected_ids=["a", "fallback"],
    )
    assert certificate["derivation"]["total_capacity_bytes"] == 8192
    with pytest.raises(ValueError, match="not dispatched"):
        build_certificate(
            dispatch(["a"]),
            candidates,
            signatures,
            {"TARGET": "generic-fpga"},
            {},
            selected_ids=["b"],
        )


def test_tampered_capacity_or_identity_is_rejected():
    certificate = build_certificate(
        dispatch(["a"]),
        [candidate("a", 1)],
        [signature("a", 4095, 2048)],
        {"TARGET": "generic-fpga"},
        {},
    )
    certificate["derivation"]["insn_capacity_bytes"] = 2048
    with pytest.raises(ValueError, match="derivation"):
        validate_certificate(certificate)


def test_finish_count_must_cover_each_submission():
    row = signature("a", 1024, 512)
    row["command_signature"]["structural"]["submissions"] = 2
    with pytest.raises(ValueError, match="FINISH/submission"):
        build_certificate(
            dispatch(["a"]),
            [candidate("a", 1)],
            [row],
            {"TARGET": "generic-fpga"},
            {},
        )


def test_missing_actual_execution_runtime_hash_fails_closed():
    row = signature("a", 1024, 512)
    row.pop("execution_runtime")
    with pytest.raises(ValueError, match="execution-runtime hash"):
        build_certificate(
            dispatch(["a"]),
            [candidate("a", 1)],
            [row],
            {"TARGET": "generic-fpga"},
            {},
        )
