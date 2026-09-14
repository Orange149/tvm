#!/usr/bin/env python3
"""Freeze a full-graph fused-TIR DMA prediction before FPGA observation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_vta_resnet50_fused_tir_occurrence_delta import (
    RUNTIME_KEYS,
    function_inventory,
    vector_delta,
)
from build_vta_resnet50_relay_residency_dispatch_pair import load_candidate
from run_vta_resnet50_callsite_composed_graph_board import verify_build_artifacts
from run_vta_resnet50_generic_residency_pair_board import graph_workload_occurrences
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(args):
    audit = Path(args.tir_audit)
    audit_artifacts = verify_build_artifacts(audit)
    audit_summary = read_json(audit / "summary.json")
    if audit_summary.get("status") != "full_graph_fused_tir_pair_captured":
        raise RuntimeError("fused TIR audit is incomplete")
    builds = audit_summary["builds"]
    modes = [row["public_mode"] for row in builds]
    if modes != ["original", "input_stationary"]:
        raise RuntimeError("unexpected fused TIR modes")

    pair = Path(args.pair_build)
    pair_artifacts = verify_build_artifacts(pair)
    pair_summary = read_json(pair / "summary.json")
    if pair_summary.get("status") != "resnet50_generic_pair_cross_build_dispatch_verified":
        raise RuntimeError("full-graph pair build is incomplete")
    if [row["candidate_id"] for row in pair_summary["builds"]] != [
        row["candidate_id"] for row in builds
    ]:
        raise RuntimeError("pair and TIR audit candidate identities differ")
    left_graph = read_json(pair / modes[0] / "graph.json")
    right_graph = read_json(pair / modes[1] / "graph.json")
    if left_graph != right_graph:
        raise RuntimeError("full-graph A/B JSON differs")

    holdout = Path(args.holdout_contract)
    holdout_artifacts = verify_build_artifacts(holdout)
    holdout_row = read_json(holdout / "contract.json")
    if holdout_row.get("status") != "frozen_before_new_full_graph_build_or_callsite_latency":
        raise RuntimeError("call-site holdout contract is not frozen")
    candidate_ids = [row["candidate_id"] for row in builds]
    if [row["candidate_id"] for row in holdout_row["selected_pair"]] != candidate_ids:
        raise RuntimeError("holdout and build candidates differ")

    candidate = load_candidate(args.candidates, candidate_ids[0])
    workload = candidate["identity"]["workload"]
    occurrences = graph_workload_occurrences(left_graph, workload)
    inventories = {
        row["public_mode"]: function_inventory(row) for row in builds
    }
    rows = []
    aggregate = {key: 0 for key in RUNTIME_KEYS}
    for occurrence in occurrences:
        symbol = occurrence["func_name"]
        if any(symbol not in inventory for inventory in inventories.values()):
            raise RuntimeError("graph symbol missing from fused TIR: " + symbol)
        delta = vector_delta(
            inventories[modes[0]][symbol], inventories[modes[1]][symbol]
        )
        for key, value in delta.items():
            aggregate[key] += value
        rows.append({
            "graph_node": occurrence["graph_node"],
            "func_name": symbol,
            "original_dma": {
                key: inventories[modes[0]][symbol].get(key, 0) for key in RUNTIME_KEYS
            },
            "residency_dma": {
                key: inventories[modes[1]][symbol].get(key, 0) for key in RUNTIME_KEYS
            },
            "delta": delta,
        })
    if aggregate["load_buffer_2d_bytes"] >= 0:
        raise RuntimeError("label-free bundle rule abstains: fused-TIR LOAD bytes do not fall")
    if aggregate["load_buffer_2d_calls"] > 0:
        raise RuntimeError("label-free bundle rule abstains: fused-TIR LOAD calls rise")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    result = {
        "schema": "c3_vta_resnet50_fused_tir_occurrence_contract_v2",
        "status": "compiler_fused_tir_prediction_frozen_before_fpga",
        "modes": modes,
        "candidate_ids": candidate_ids,
        "graph_occurrence_count": len(occurrences),
        "graph_occurrences": rows,
        "full_graph_fused_tir_delta": aggregate,
        "exact_fields": list(RUNTIME_KEYS),
        "bundle_rule_pass": True,
        "bundle_rule": "negative total LOAD bytes and non-increasing total LOAD calls",
        "tir_audit_summary_sha256": sha256(audit / "summary.json"),
        "tir_audit_artifact_hash_manifest_sha256": sha256(audit / "artifact_hashes.json"),
        "verified_tir_audit_artifact_count": len(audit_artifacts),
        "pair_build_summary_sha256": sha256(pair / "summary.json"),
        "pair_build_artifact_hash_manifest_sha256": sha256(pair / "artifact_hashes.json"),
        "verified_pair_build_artifact_count": len(pair_artifacts),
        "holdout_contract_sha256": sha256(holdout / "contract.json"),
        "holdout_artifact_hash_manifest_sha256": sha256(holdout / "artifact_hashes.json"),
        "verified_holdout_artifact_count": len(holdout_artifacts),
        "freezer_source_sha256": sha256(Path(__file__).resolve()),
        "board_contacted": False,
        "full_graph_latency_observed": False,
        "partial_mask_latency_observed": False,
        "claim_boundary": (
            "Compiler-only prospective logical-DMA contract for one previously chosen tile; "
            "not FPGA correctness, latency, physical AXI, or a new operator-level holdout"
        ),
    }
    write_json(output / "fused_tir_occurrence_delta.json", result)
    (output / "command.txt").write_text(
        " ".join(__import__("sys").argv) + "\n", encoding="utf-8"
    )
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tir-audit", required=True)
    parser.add_argument("--pair-build", required=True)
    parser.add_argument("--holdout-contract", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
