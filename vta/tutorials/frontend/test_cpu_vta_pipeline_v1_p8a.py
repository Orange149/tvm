#!/usr/bin/env python3
"""Local tests for the P8A shared-memory feasibility audit."""

import json

import run_cpu_vta_pipeline_v1_p8a as p8a


def test_frozen_boundary_contract_is_exact():
    manifest = p8a.load_manifest(p8a.DEFAULT_PACKAGE)
    edge = p8a.boundary_contract(manifest)
    assert edge == {
        "shape": [1, 64, 56, 56],
        "dtype": "float32",
        "bytes": 802816,
        "producer_stage": 0,
        "consumer_stage": 1,
        "direction": "cpu_to_vta",
    }


def test_lowered_graph_requires_cpu_and_extdev_views():
    contracts = p8a.graph_boundary_contract(p8a.DEFAULT_PACKAGE)
    assert contracts["producer_output"]["device_index"] == 1
    assert contracts["consumer_input"]["device_index"] == 12
    assert contracts["producer_output"]["shape"] == contracts["consumer_input"]["shape"]
    assert contracts["producer_output"]["dtype"] == contracts["consumer_input"]["dtype"]


def test_reverse_boundary_contract_is_vta_to_cpu():
    manifest = p8a.load_manifest(p8a.DEFAULT_PACKAGE)
    edge = p8a.boundary_contract(manifest, 1)
    assert edge["direction"] == "vta_to_cpu"
    assert edge["bytes"] == 100352
    contracts = p8a.graph_boundary_contract(p8a.DEFAULT_PACKAGE, 1)
    assert contracts["producer_output"]["device_index"] == 12
    assert contracts["consumer_input"]["device_index"] == 1


def test_protocol_does_not_consume_candidate_throughput():
    manifest = p8a.load_manifest(p8a.DEFAULT_PACKAGE)
    protocol = p8a.build_protocol(manifest)
    encoded = json.dumps(protocol)
    assert protocol["candidate_throughput_used_for_fit"] is False
    assert "measured_fps" not in encoded
    assert "predicted_fps" not in encoded
    assert protocol["excluded_claims"][0] == "dual_slot_pipeline_throughput"
    assert protocol["correctness_scope"]["current_test"].startswith("ordinary_copy")


def test_runner_contains_required_zero_copy_guards():
    source = p8a.RUNNER_SOURCE.read_text(encoding="utf-8")
    for token in (
        "set_input_zero_copy",
        "set_output_zero_copy",
        "SameDType",
        "SameShape",
        "requires zero byte offsets",
        "VTAMemGetPhyAddr",
        "direct_vta_driver_cached",
        "baseline_materialization_bytes",
        "zero_copy_materialization_bytes",
        "reference_correctness_passed",
        "ordinary_copy_same_compiled_graph",
    ):
        assert token in source


def test_summary_rejects_direct_mapping_penalty_larger_than_copy_saved():
    row = {
        "reference_correctness_passed": True,
        "direction": "cpu_to_vta",
        "input_index": 0,
        "input_hash": "a",
        "reference_output_hashes": ["x"],
        "boundary_bytes": 16,
        "slot_virtual_address": "0x1000",
        "slot_physical_address": "0x2000",
        "same_virtual_address": True,
        "slot_alignment_bytes": 256,
        "baseline_materialization_bytes": 32,
        "zero_copy_materialization_bytes": 0,
        "baseline_edge_copy_ms": 1.0,
        "baseline_total_latency_ms": 10.0,
        "zero_copy_total_latency_ms": 11.1,
        "baseline_producer_run_ms": 3.0,
        "zero_copy_producer_run_ms": 4.1,
        "baseline_consumer_run_ms": 4.0,
        "zero_copy_consumer_run_ms": 4.0,
        "safe_copy_effective": False,
        "pair_order": "ordinary_then_zero_copy",
    }
    summary = p8a.summarize_control(
        "memcpy",
        [
            row,
            dict(
                row,
                input_index=1,
                input_hash="b",
                reference_output_hashes=["y"],
                pair_order="zero_copy_then_ordinary",
            ),
        ],
    )
    assert summary["direct_mapping_gate_passed"] is False
    assert summary["cpu_direct_output_penalty_ms"] > summary["baseline_edge_copy_ms_median"]
    assert summary["cpu_shared_mapping_penalty_ms"] == summary["cpu_direct_output_penalty_ms"]
    assert summary["slot_address_stable"] is True
    assert summary["slot_address_alignment_verified"] is True
    assert summary["pair_orders"] == [
        "ordinary_then_zero_copy",
        "zero_copy_then_ordinary",
    ]


def test_reverse_edge_gate_uses_cpu_consumer_penalty():
    row = {
        "reference_correctness_passed": True,
        "direction": "vta_to_cpu",
        "input_index": 0,
        "input_hash": "a",
        "reference_output_hashes": ["x"],
        "boundary_bytes": 16,
        "slot_virtual_address": "0x1000",
        "slot_physical_address": "0x2000",
        "same_virtual_address": True,
        "slot_alignment_bytes": 256,
        "baseline_materialization_bytes": 32,
        "zero_copy_materialization_bytes": 0,
        "baseline_edge_copy_ms": 1.0,
        "baseline_total_latency_ms": 10.0,
        "zero_copy_total_latency_ms": 11.1,
        "baseline_producer_run_ms": 3.0,
        "zero_copy_producer_run_ms": 3.0,
        "baseline_consumer_run_ms": 4.0,
        "zero_copy_consumer_run_ms": 5.1,
        "safe_copy_effective": False,
        "pair_order": "ordinary_then_zero_copy",
    }
    summary = p8a.summarize_control(
        "memcpy",
        [
            row,
            dict(
                row,
                input_index=1,
                input_hash="b",
                reference_output_hashes=["y"],
                pair_order="zero_copy_then_ordinary",
            ),
        ],
    )
    assert summary["cpu_direct_output_penalty_ms"] == 0.0
    assert summary["cpu_shared_mapping_penalty_ms"] > 1.0
    assert summary["direct_mapping_gate_passed"] is False


def test_allocator_metadata_is_not_used_as_hpc_proof():
    manifest = p8a.load_manifest(p8a.DEFAULT_PACKAGE)
    preflight = {
        "component_hashes": {
            "matches": {
                "bitstream": True,
                "libtvm_runtime.so": True,
                "libvta.so": True,
                "tvm_rpc": True,
            }
        }
    }
    evidence = p8a.build_coherence_evidence(
        {"udmabuf_dma_coherent": "0"}, preflight
    )
    assert evidence["allocator_metadata_attests_hardware_coherence"] is False
    assert evidence["vta_hpc_path_evidence_complete"] is True


def test_frozen_candidate_has_separate_independent_reference_evidence():
    evidence = p8a.build_correctness_evidence(
        "cpu-00-02_t4__vta-03-17_t1__cpu-18-20_t1"
    )
    assert evidence["current_test"] == "ordinary_copy_same_compiled_graph_equivalence"
    assert evidence["prior_independent_reference"]["passed"] is True
    assert evidence["prior_independent_reference"]["serial_pipeline_exact_frames"] == 22
