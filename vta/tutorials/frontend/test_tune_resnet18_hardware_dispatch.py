"""Tests that hardware certificates filter candidates before AutoTVM measurement."""

import json

import pytest

from build_vta_command_resource_certificate import build_certificate
from tune_resnet18_vta import (
    load_certified_candidate_ids,
    load_certified_config_indices,
    load_command_resource_contract,
)


def write_dispatch(path, routes, label_free=True, workload_id="E00", mode="input_stationary"):
    path.write_text(json.dumps({
        "schema": "c3_vta_hardware_certified_dispatch_v1",
        "performance_labels_used": False if label_free else True,
        "records": [
            {"route": route, "config_index": index, "workload_id": workload_id, "mode": mode}
            for index, route in enumerate(routes)
        ],
    }))


def test_only_allow_timing_indices_survive(tmp_path):
    path = tmp_path / "dispatch.json"
    write_dispatch(path, ["reject_hardware", "correctness_canary_only", "allow_timing"])
    indices, digest = load_certified_config_indices(path, "E00", "input_stationary")
    assert indices == {2}
    assert len(digest) == 64


def test_refuses_empty_or_label_dependent_allowlist(tmp_path):
    empty = tmp_path / "empty.json"
    write_dispatch(empty, ["reject_hardware"])
    with pytest.raises(ValueError, match="no exact hardware-certified"):
        load_certified_config_indices(empty)
    leaked = tmp_path / "leaked.json"
    write_dispatch(leaked, ["allow_timing"], label_free=False)
    with pytest.raises(ValueError, match="label-free"):
        load_certified_config_indices(leaked)


def test_refuses_dispatch_for_a_different_workload_or_mode(tmp_path):
    workload = tmp_path / "workload.json"
    write_dispatch(workload, ["allow_timing"], workload_id="E03")
    with pytest.raises(ValueError, match="workload"):
        load_certified_config_indices(workload, "E00", "input_stationary")
    mode = tmp_path / "mode.json"
    write_dispatch(mode, ["allow_timing"], mode="weight_stationary")
    with pytest.raises(ValueError, match="mode"):
        load_certified_config_indices(mode, "E00", "input_stationary")


def test_exact_candidate_ids_and_command_resource_contract_are_linked(tmp_path):
    dispatch_path = tmp_path / "dispatch.json"
    dispatch_path.write_text(json.dumps({
        "schema": "c3_vta_hardware_certified_dispatch_v1",
        "performance_labels_used": False,
        "records": [{
            "route": "allow_timing",
            "candidate_id": "candidate-a",
            "config_index": 7,
            "workload_id": "E00",
            "mode": "input_stationary",
        }],
    }))
    candidate = {
        "candidate_id": "candidate-a",
        "workload_id": "E00",
        "public_mode": "input_stationary",
        "debug": {"config_index": 7},
        "expected_tir_sha256": "tir-a",
        "identity": {"complete_config_entity": {"index": 7, "entity": []}},
    }
    structural = {
        "peaks": {"insn_bytes": 3000, "uop_bytes": 1000},
        "submissions": 1,
        "finish": {"source_derived_count": 1},
    }
    signature = {
        "candidate_id": "candidate-a",
        "workload_id": "E00",
        "public_mode": "input_stationary",
        "relowered_tir_sha256": "tir-a",
        "outcome": "passed_local_signature",
        "execution_runtime": {"path": "/tmp/libvta_fsim.so", "sha256": "f" * 64},
        "command_signature": {"sha256": "signature-a", "structural": structural},
    }
    certificate = build_certificate(
        json.loads(dispatch_path.read_text()),
        [candidate],
        [signature],
        {"TARGET": "test"},
        {},
    )
    certificate_path = tmp_path / "command_resource_certificate.json"
    certificate_path.write_text(json.dumps(certificate))
    ids = load_certified_candidate_ids(dispatch_path)
    assert ids == {"candidate-a"}
    contract = load_command_resource_contract(certificate_path, ids)
    assert contract["environment"] == {
        "VTA_INSN_BUFFER_BYTES": "4096",
        "VTA_UOP_BUFFER_BYTES": "4096",
    }


def test_dispatch_without_exact_candidate_id_fails_closed(tmp_path):
    path = tmp_path / "dispatch.json"
    write_dispatch(path, ["allow_timing"])
    with pytest.raises(ValueError, match="lack exact candidate IDs"):
        load_certified_candidate_ids(path)
