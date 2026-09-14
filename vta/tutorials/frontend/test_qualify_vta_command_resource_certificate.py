"""Tests for upgrading local command signatures with real-FPGA evidence."""

import pytest

from build_vta_command_resource_certificate import build_certificate
from qualify_vta_command_resource_certificate import qualify_certificate


def provisional():
    dispatch = {
        "schema": "c3_vta_hardware_certified_dispatch_v2",
        "performance_labels_used": False,
        "records": [{"route": "allow_timing", "candidate_id": "a"}],
    }
    candidate = {
        "candidate_id": "a",
        "workload_id": "W",
        "public_mode": "original",
        "debug": {"config_index": 1},
        "expected_tir_sha256": "tir",
        "identity": {"complete_config_entity": {"entity": []}},
    }
    signature = {
        "candidate_id": "a",
        "workload_id": "W",
        "public_mode": "original",
        "relowered_tir_sha256": "tir",
        "outcome": "passed_local_signature",
        "execution_runtime": {"path": "/tmp/libvta_fsim.so", "sha256": "f" * 64},
        "command_signature": {
            "sha256": "signature",
            "structural": {
                "peaks": {"insn_bytes": 3000, "uop_bytes": 1000},
                "submissions": 1,
                "finish": {"source_derived_count": 1},
            },
        },
    }
    return build_certificate(
        dispatch,
        [candidate],
        [signature],
        {"TARGET": "test"},
        {"vta/runtime/runtime.cc": "runtime"},
    )


def evidence():
    summary = {
        "status": "passed",
        "capacity": {"insn_bytes": 4096, "uop_bytes": 4096},
        "claim_boundary": "single boot",
    }
    rows = [{
        "candidate_id": "a",
        "config_index": 1,
        "mode": "original",
        "tir_sha256": "tir",
        "binary_sha256": "binary",
        "correct": True,
        "seeds": [
            {"seed": seed, "correct": True, "mismatch_count": 0}
            for seed in (0, 1, 2)
        ],
    }]
    queue = {
        "schema": "vta_queue_capacity_status_v1",
        "insn_capacity_bytes": 4096,
        "uop_capacity_bytes": 4096,
        "insn_peak_bytes": 3000,
        "uop_peak_bytes": 1000,
    }
    contract = {
        "source_guards_sha256": {"vta/runtime/runtime.cc": "runtime"},
        "board": {
            "boot_id": "boot",
            "fpga_state": "operating",
            "udmabuf_bytes": 1 << 20,
            "remote_capacity_runtime_sha256": {"libvta.so": "runtime-binary"},
        },
    }
    return summary, rows, queue, contract


def test_three_seed_exact_board_evidence_upgrades_status():
    result = qualify_certificate(provisional(), *evidence())
    assert result["status"] == "qualified_reduced_capacity"
    assert result["board_qualification"]["correctness"][0]["passed_seed_count"] == 3
    assert result["board_qualification"]["boot_id_bound"] is True
    assert result["boot_id_bound"] is False


def test_incomplete_seeds_or_wrong_capacity_fail_closed():
    summary, rows, queue, contract = evidence()
    rows[0]["seeds"] = rows[0]["seeds"][:2]
    with pytest.raises(ValueError, match="three unique seeds"):
        qualify_certificate(provisional(), summary, rows, queue, contract)
    summary, rows, queue, contract = evidence()
    summary["capacity"]["insn_bytes"] = 8192
    with pytest.raises(ValueError, match="requested capacity"):
        qualify_certificate(provisional(), summary, rows, queue, contract)


def test_manifest_runtime_mismatch_fails_closed():
    summary, rows, queue, contract = evidence()
    contract["manifest_environment"] = {
        "VTA_COMMAND_MANIFEST_ID": "a" * 64,
        "VTA_REPLAY_POLICY": "disabled",
    }
    queue.update(command_manifest_id="b" * 64, replay_policy="disabled")
    with pytest.raises(ValueError, match="manifest/replay"):
        qualify_certificate(provisional(), summary, rows, queue, contract)
