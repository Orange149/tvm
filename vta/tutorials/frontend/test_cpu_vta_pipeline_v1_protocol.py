#!/usr/bin/env python3
"""Static tests for the CPU-VTA Pipeline V1 P0 artifacts."""

import argparse
import json

import freeze_cpu_vta_pipeline_v1 as v1
from deploy_classification_stage_pipeline_native import (
    resolve_stage_cpu_affinities,
    save_stage_to_cache,
    try_copy_stage_from_cache,
)
from split_resnet18_stages import SCHEMES


def write_preflight(tmp_path, passed=True):
    root = v1.repo_root()
    local_paths = {
        "bitstream": root / "apps/vta_rpc/firmware/vta_hpc.bit",
        "tvm_rpc": root / "build_axu_aarch64/tvm_rpc",
        "libtvm_runtime.so": root / "build_axu_aarch64/libtvm_runtime.so",
        "libvta.so": root / "build_axu_aarch64/libvta.so",
    }
    hashes = {name: v1.file_sha256(path) for name, path in local_paths.items()}
    gate_checks = {
        "all_component_hashes_match": passed,
        "required_rpc_functions_available": passed,
        "four_homogeneous_cpu_cores": passed,
        "runtime_num_threads_four": passed,
        "hpc_runtime_and_fpga_operating": passed,
        "native_vta_udmabuf_ready": passed,
    }
    payload = v1.seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_preflight_evidence",
            "protocol_id": v1.PROTOCOL_ID,
            "observed_at_utc": "2026-09-02T00:00:00+00:00",
            "passed": passed,
            "endpoint": {
                "ssh_target": "root@192.168.1.234",
                "ssh_options": [],
                "rpc_host": "192.168.1.234",
                "rpc_port": 9090,
            },
            "ssh": {"status": "connected" if passed else "failed"},
            "rpc": {"status": "connected" if passed else "failed"},
            "remote": {
                "fpga_firmware": "vta_hpc.bit",
                "cpu_frequencies_khz": "1066666,1066666,1066666,1066666",
                "cpu_governors": "userspace,userspace,userspace,userspace",
                "fpga_state": "operating" if passed else "unknown",
                "udmabuf0_size": "201326592" if passed else "unavailable",
                "udmabuf0_device": "present" if passed else "unavailable",
            },
            "expected_runtime": {
                "bitstream_name": "vta_hpc.bit",
                "bitstream_path": "/mnt/sd/tvm_deploy/firmware/vta_hpc.bit",
                "runtime_dir": "/mnt/sd/tvm_deploy/hpc",
            },
            "board_instance_proxy": {
                "device_tree_model": "fixture",
                "ethernet_mac": "00:11:22:33:44:55",
                "boot_image_sha256": "e" * 64,
                "identity_strength": "board_instance_proxy_not_silicon_serial",
            },
            "component_hashes": {
                "local": hashes,
                "remote": hashes,
                "matches": {name: passed for name in hashes},
            },
            "gate_checks": gate_checks,
        }
    )
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def complete_args(tmp_path, passed=True):
    return argparse.Namespace(
        output_dir=str(tmp_path),
        batch=1,
        image_size=224,
        board_host=v1.DEFAULT_BOARD_HOST,
        rpc_port=v1.DEFAULT_RPC_PORT,
        ssh_user=v1.DEFAULT_SSH_USER,
        preflight_evidence=str(write_preflight(tmp_path, passed=passed)),
    )


def test_unit_schema_freezes_real_resnet18_order_and_contracts():
    schema = v1.build_unit_and_boundary_schema()
    assert schema["model"]["atomic_unit_count"] == 21
    assert schema["model"]["cut_position_count"] == 20
    assert schema["model"]["unit_order"] == list(v1.UNIT_ORDER)
    assert all(item["contract_compatible"] for item in schema["legal_cut_positions"])
    assert schema["candidate_grammar"]["maximum_vta_islands"] == 3
    assert schema["candidate_grammar"]["cpu_thread_choices"] == [1, 2, 3, 4]
    assert schema["candidate_grammar"]["cpu_thread_sum_is_a_constraint"] is False
    v1.validate_artifact_sha256(schema)


def test_candidate_validator_accepts_independent_cpu_thread_parameters():
    schema = v1.build_unit_and_boundary_schema()
    stages = []
    for stage in SCHEMES["three_stage_e"]:
        frozen = dict(stage)
        frozen["threads"] = 3 if len(stages) == 0 else (4 if stage["device"] == "cpu" else 1)
        stages.append(frozen)
    assert v1.validate_candidate(stages, schema) == []


def test_candidate_validator_rejects_only_per_stage_thread_values_outside_domain():
    schema = v1.build_unit_and_boundary_schema()
    stages = []
    for stage in SCHEMES["three_stage_e"]:
        frozen = dict(stage)
        frozen["threads"] = 5 if stage["device"] == "cpu" else 1
        stages.append(frozen)
    errors = v1.validate_candidate(stages, schema)
    assert any("invalid threads" in error for error in errors)


def test_affinity_manifest_uses_independent_overlapping_prefix_masks():
    stages = [
        {"device": "cpu", "threads": 3},
        {"device": "vta", "threads": 1},
        {"device": "cpu", "threads": 4},
    ]
    manifest = v1.build_cpu_affinity_manifest(stages)
    assert manifest[0]["cpu_mask"] == [0, 1, 2]
    assert manifest[1]["cpu_mask"] == []
    assert manifest[2]["cpu_mask"] == [0, 1, 2, 3]
    assert manifest[0]["affinity_policy"] == "independent_overlapping_prefix_masks_v1"


def test_native_deploy_allows_overlapping_cpu_stage_affinity():
    records = [{"device": "cpu"}, {"device": "vta"}, {"device": "cpu"}]
    threads = {"stage0": 3, "stage1": 1, "stage2": 4}
    assert resolve_stage_cpu_affinities(records, threads, serial=False) == {
        "stage0": [0, 1, 2],
        "stage1": [],
        "stage2": [0, 1, 2, 3],
    }


def test_serial_profile_uses_the_same_per_stage_prefix_masks():
    records = [{"device": "cpu"}, {"device": "vta"}, {"device": "cpu"}]
    threads = {"stage0": 4, "stage1": 1, "stage2": 4}
    assert resolve_stage_cpu_affinities(records, threads, serial=True) == {
        "stage0": [0, 1, 2, 3],
        "stage1": [],
        "stage2": [0, 1, 2, 3],
    }


def test_stage_cache_miss_and_metadata_hit_are_unambiguous(tmp_path):
    cache_dir = tmp_path / "cache"
    stage_dir = tmp_path / "stage"
    restored_dir = tmp_path / "restored"
    stage_dir.mkdir()
    restored_dir.mkdir()
    assert try_copy_stage_from_cache(cache_dir, "key", restored_dir) is None

    for name in ["graph.json", "graphlib.so", "params.params"]:
        (stage_dir / name).write_bytes(name.encode("ascii"))
    record = {"cache_key": "key", "relay_ir_sha256": "a" * 64}
    save_stage_to_cache(cache_dir, "key", record, stage_dir)

    restored = try_copy_stage_from_cache(cache_dir, "key", restored_dir)
    assert restored == record
    assert (restored_dir / "stage_cache.json").exists()


def test_incomplete_remote_preflight_blocks_profile(tmp_path):
    args = complete_args(tmp_path, passed=False)
    outputs = v1.freeze_artifacts(args)
    fingerprint = outputs["v1_hardware_fingerprint.json"]
    protocol = outputs["v1_protocol.json"]
    assert fingerprint["profile_allowed"] is False
    assert fingerprint["stable_hardware_fingerprint_sha256"] is None
    assert outputs["v1_execution_state.json"]["next_stage_may_start"] is False


def test_complete_remote_identity_seals_stable_fingerprint(tmp_path):
    outputs = v1.freeze_artifacts(complete_args(tmp_path))
    fingerprint = outputs["v1_hardware_fingerprint.json"]
    protocol = outputs["v1_protocol.json"]
    assert fingerprint["profile_allowed"] is True
    assert len(fingerprint["stable_hardware_fingerprint_sha256"]) == 64
    assert protocol["status"] == "frozen"
    assert protocol["profile_allowed"] is True
    assert "total_host_core_ms/4" in protocol["objective"]["formula"]
    assert "current_stage" not in protocol
    assert "fingerprint_artifact_sha256" not in protocol["hardware"]
    assert outputs["v1_execution_state.json"]["current_stage"] == "V1-P0"
    assert (
        outputs["v1_execution_state.json"]["hardware_fingerprint_artifact_sha256"]
        == fingerprint["artifact_sha256"]
    )
    for payload in outputs.values():
        v1.validate_artifact_sha256(payload)
