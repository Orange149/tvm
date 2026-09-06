#!/usr/bin/env python3
"""Static tests for CPU-VTA Pipeline V1 P1 manifests."""

import build_cpu_vta_pipeline_v1_p1 as p1
import freeze_cpu_vta_pipeline_v1 as v1


def test_reachable_manifest_rejects_adjacent_same_device_stages():
    schema = v1.build_unit_and_boundary_schema()
    candidates, segments, boundaries = p1.enumerate_reachable(schema)
    assert len(candidates) == 4623
    assert len(segments) == 172
    assert len(boundaries) == 27
    for scheme in candidates:
        devices = [stage["device"] for stage in scheme]
        assert all(left != right for left, right in zip(devices, devices[1:]))


def test_every_segment_maps_to_atomic_costs_and_only_heterogeneous_boundaries():
    schema = v1.build_unit_and_boundary_schema()
    _, segments, boundaries = p1.enumerate_reachable(schema)
    assert all(item["cost_mapping"]["unit_cost_keys"] for item in segments.values())
    assert all(item["logical_compute_ops_est"] > 0 for item in segments.values())
    assert all(item["logical_compute_gops_est"] > 0 for item in segments.values())
    assert {
        item["cost_mapping"]["model"] for item in segments.values()
    } == {"additive_atomic_logical_ops_v1"}
    assert {item["direction"] for item in boundaries.values()} == {
        "cpu_to_vta",
        "vta_to_cpu",
    }


def test_boundary_accounting_ids_are_globally_unique():
    schema = v1.build_unit_and_boundary_schema()
    _, _, boundaries = p1.enumerate_reachable(schema)
    ids = []
    for boundary in boundaries.values():
        for slot in boundary["slots"]:
            ids.extend([slot["host_accounting_id"], slot["dma_accounting_id"]])
    assert len(ids) == len(set(ids))
