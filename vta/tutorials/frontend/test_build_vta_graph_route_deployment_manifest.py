import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import build_vta_graph_route_deployment_manifest as builder


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    candidate_id = "route-a"
    candidate_source = tmp_path / "candidate.json"
    candidate_hash = dump(candidate_source, {"candidates": [{
        "candidate_id": candidate_id,
        "public_mode": "input_stationary",
        "implementation_mode": 1,
        "identity": {"workload": ["conv2d_packed.vta"], "complete_config_entity": {"entity": [1]}},
    }]})
    route = {
        "candidate_id": candidate_id,
        "workload_id": "Y00",
        "public_mode": "input_stationary",
        "implementation_mode": 1,
        "candidate_source": str(candidate_source),
        "candidate_source_sha256": candidate_hash,
    }
    plan = tmp_path / "plan.json"
    dump(plan, {
        "schema": "c3_incumbent_protected_route_plan_v1",
        "status": "route_composition_complete",
        "selected_route_candidate_ids": [candidate_id],
        "decisions": [
            {"route": route, "accepted": True, "checks": {}, "evaluation": {}},
            {"route": {"candidate_id": "rejected"}, "accepted": False,
             "checks": {"minimum_improvement": False},
             "evaluation": {"median_paired_delta_ms": 1.0}},
        ],
    })
    build_dir = tmp_path / "build"
    variant_dir = build_dir / "selected"
    graph_hash = dump(variant_dir / "graph.json", {})
    params_hash = dump(variant_dir / "params.bin", {})
    lib_hash = dump(variant_dir / "graphlib.so", {})
    variant = {
        "deployment_variant": "selected", "candidate_ids": [candidate_id],
        "graph_sha256": graph_hash, "params_sha256": params_hash, "graphlib_sha256": lib_hash,
        "dispatch_audit": {"all_routes_scheduled": True,
                           "schedule_hits": [{"candidate_id": candidate_id}]},
    }
    dump(build_dir / "summary.json", {
        "status": "yolov3_tiny_two_route_factorial_cross_build_verified",
        "model": "yolov3-tiny", "variants": [variant],
    })
    board = tmp_path / "board.json"
    dump(board, {
        "status": "yolov3_tiny_multi_residency_board_correctness_and_timing_complete",
        "all_paired_outputs_equal": True, "all_correctness_outputs_nonzero": True,
        "boot_id": "boot", "variants": [variant],
        "board_after": {"FPGA": "operating", "UDMABUF": "201326592", "STORAGE_ERRORS": []},
    })
    clean = tmp_path / "clean.json"
    dump(clean, {"fresh_rpc": {"CWD": "/var/volatile/vta_c3_ram/runtime"},
                 "bitstream_reload": {"state": "operating", "sha256": "bit"}})
    return SimpleNamespace(
        selected_route_plan=str(plan), build_dir=str(build_dir), board_summary=str(board),
        clean_start=str(clean), workspace_root=str(tmp_path), output_dir=str(tmp_path / "out"),
    )


def test_binds_selected_route_and_graph(tmp_path):
    args = fixture(tmp_path)
    builder.run(args)
    manifest = json.loads((Path(args.output_dir) / "graph_route_manifest.json").read_text())
    assert manifest["selected_route_candidate_ids"] == ["route-a"]
    assert manifest["selected_deployment_variant"] == "selected"
    assert manifest["hardware_contract"]["udmabuf_bytes"] == 201326592


def test_rejects_mutated_graph_artifact(tmp_path):
    args = fixture(tmp_path)
    (Path(args.build_dir) / "selected" / "graphlib.so").write_text("mutated")
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        builder.run(args)
