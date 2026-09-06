#!/usr/bin/env python3
"""Local tests for P8B two-direction serial controls."""

import json

import run_cpu_vta_pipeline_v1_p8a as p8a
import run_cpu_vta_pipeline_v1_p8b as p8b


def test_protocol_freezes_both_heterogeneous_directions():
    manifest = p8a.load_manifest(p8a.DEFAULT_PACKAGE)
    protocol = p8b.build_protocol(manifest)
    assert [edge["direction"] for edge in protocol["edges"]] == [
        "cpu_to_vta",
        "vta_to_cpu",
    ]
    assert [edge["bytes"] for edge in protocol["edges"]] == [802816, 100352]
    assert protocol["candidate_throughput_used_for_fit"] is False
    assert "measured_fps" not in json.dumps(protocol)


def test_local_audit_checks_reverse_device_views(tmp_path):
    runner = tmp_path / "runner"
    runner.write_bytes(b"test")
    manifest = p8a.load_manifest(p8a.DEFAULT_PACKAGE)
    audit = p8b.build_local_audit(p8a.DEFAULT_PACKAGE, manifest, runner)
    assert audit["directions"] == ["cpu_to_vta", "vta_to_cpu"]
    assert all(
        edge["checks"]["graph_device_views_are_complementary_cpu_extdev"]
        for edge in audit["edge_audits"]
    )


def test_runner_accepts_generic_bidirectional_edge_option():
    source = p8a.RUNNER_SOURCE.read_text(encoding="utf-8")
    assert '--p8-edge" || key == "--p8a-edge' in source
    assert "BindP8HeterogeneousEdge" in source
    assert '"vta_to_cpu"' in source
