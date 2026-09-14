#!/usr/bin/env python3
"""Invalidate a frozen holdout whose declared geometry disagrees with Relay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import prepare_vta_p7r115_yolo_confirmation as base
from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from prepare_vta_model_conv_adaptive_holdout import extract_resnet50_conv
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json
from vta_geometry_catalog import audit_sources


def read(path):
    return json.loads(Path(path).read_text())


def geometry_signature(record):
    return [
        record["input_shape_nchw"][0], record["ci"], record["height"], record["width"],
        record["co"], record["kernel"], record["kernel"],
        record["strides"][0], record["strides"][1], *record["padding"],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-contract", type=Path, required=True)
    parser.add_argument("--policy-contract", type=Path, required=True)
    parser.add_argument("--local-qualification", type=Path, required=True)
    parser.add_argument("--replacement-layer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    verified = {
        "target": len(verify_artifacts_compatible(args.target_contract)),
        "policy": len(verify_artifacts_compatible(args.policy_contract)),
        "local": len(verify_artifacts_compatible(args.local_qualification)),
    }
    target = read(args.target_contract / "contract.json")
    policy = read(args.policy_contract / "policy.json")
    local = read(args.local_qualification / "summary.json")
    if target.get("source_model") != "resnet50_v2":
        raise ValueError("audit is specific to resnet50_v2")
    if policy.get("workload_id") != target.get("workload_id"):
        raise ValueError("policy does not bind target")
    if local.get("board_contacted") or local.get("performance_labels_used"):
        raise ValueError("invalid target unexpectedly reached board or latency labels")

    actual = extract_resnet50_conv(target["source_layer"])
    replacement = extract_resnet50_conv(args.replacement_layer)
    declared = target["explicit_conv"]
    declared_compare = {
        "ci": declared["ci"], "co": declared["co"], "height": declared["height"],
        "width": declared["width"], "kernel": declared["kernel"],
        "stride": declared["stride"], "symmetric_padding": declared["padding"],
    }
    mismatches = {
        key: {"declared": value, "model": actual[key]}
        for key, value in declared_compare.items() if value != actual[key]
    }
    if not mismatches:
        raise ValueError("declared geometry unexpectedly agrees with Relay")

    rows = audit_sources(base.C3, base.EXPOSURE_SOURCES.values())
    actual_signature = geometry_signature(actual)
    replacement_signature = geometry_signature(replacement)
    actual_collisions = [
        row["source"] for row in rows if actual_signature in row["geometry_signatures"]
    ]
    replacement_collisions = [
        row["source"] for row in rows if replacement_signature in row["geometry_signatures"]
    ]
    result = {
        "schema": "c3_resnet50_geometry_misderivation_audit_v1",
        "status": "target_invalidated_before_board_due_model_geometry_mismatch",
        "invalidated_workload_id": target["workload_id"],
        "invalidated_source_layer": target["source_layer"],
        "declared_explicit_conv": declared,
        "relay_infertype_geometry": actual,
        "mismatches": mismatches,
        "actual_geometry_signature": actual_signature,
        "actual_geometry_reserved_source_count": len(actual_collisions),
        "actual_geometry_reserved_sources": actual_collisions,
        "replacement_layer": args.replacement_layer,
        "replacement_relay_infertype_geometry": replacement,
        "replacement_geometry_signature": replacement_signature,
        "replacement_reserved_source_count_before_new_freeze": len(replacement_collisions),
        "replacement_reserved_sources_before_new_freeze": replacement_collisions,
        "descendant_disposition": {
            "target_contract": "invalid for a true ResNet50 layer claim; retained immutably",
            "policy_contract": "must not be executed; bound to invalid target",
            "local_qualification": "synthetic operator-only local evidence; must not be dispatched or reported as ResNet50",
        },
        "board_contacted": False,
        "performance_labels_used": False,
        "bindings": {
            "target_manifest_sha256": sha256(args.target_contract / "artifact_hashes.json"),
            "policy_manifest_sha256": sha256(args.policy_contract / "artifact_hashes.json"),
            "local_manifest_sha256": sha256(args.local_qualification / "artifact_hashes.json"),
            "resnet_source_sha256": sha256(base.REPO / "python/tvm/relay/testing/resnet.py"),
        },
        "verified_artifact_counts": verified,
        "source_sha256": sha256(Path(__file__).resolve()),
        "claim_boundary": "Derivation audit only. It creates no board, latency, correctness, search, or replacement-target result.",
    }
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "summary.json", result)
    (args.output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    write_json(args.output_dir / "artifact_hashes.json", {"artifacts": {
        path.name: sha256(path) for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "status": result["status"], "mismatches": mismatches,
        "actual_geometry_reserved_source_count": len(actual_collisions),
        "replacement_reserved_source_count_before_new_freeze": len(replacement_collisions),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
