#!/usr/bin/env python3
"""Freeze wave zero and control orders for correctness-driven Pareto peeling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from analyze_vta_operator_proxy_pareto_development import operator_proxy_program
from build_vta_resnet50_fused_program_pool import dominates, pareto_front
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def random_key(workload_id, candidate_id):
    material = "c3-invalid-peeling-random-v1|{}|{}".format(workload_id, candidate_id)
    return hashlib.sha256(material.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-contract", type=Path, required=True)
    parser.add_argument("--policy-contract", type=Path, required=True)
    parser.add_argument("--local-qualification", type=Path, required=True)
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
    if policy.get("status") != "frozen_before_target_lowering_fsim_fullgraph_fpga_or_latency":
        raise ValueError("peeling policy is not pristine")
    if local.get("status") != "completed_local_no_board" or local.get("board_contacted") or local.get("performance_labels_used"):
        raise ValueError("local qualification is incomplete or label-exposed")
    local_input = read(args.local_qualification / "contract.json")["input"]["path"]
    if Path(local_input).resolve() != (args.target_contract / "candidates.jsonl").resolve():
        raise ValueError("local qualification does not consume the target candidate contract")
    if target["workload_id"] != policy["workload_id"]:
        raise ValueError("workload binding mismatch")

    candidates = {row["candidate_id"]: row for row in read_rows(args.local_qualification / "candidates_v2.jsonl")}
    static = {row["candidate_id"]: row for row in read_rows(args.local_qualification / "static_results.jsonl")}
    eligible = [row["candidate_id"] for row in read_rows(args.local_qualification / "fsim_results.jsonl") if row.get("status") == "passed"]
    programs = [operator_proxy_program(candidates[cid], static[cid]) for cid in eligible]
    wave0 = sorted(pareto_front(programs), key=lambda row: row["candidate_id"])
    wave0_ids = [row["candidate_id"] for row in wave0]
    dominated_by = {
        row["candidate_id"]: sorted(other["candidate_id"] for other in programs if dominates(other, row))
        for row in programs if row["candidate_id"] not in wave0_ids
    }
    ids = [row["candidate_id"] for row in programs]
    by_id = {row["candidate_id"]: row for row in programs}
    orders = {
        "fixed_front": wave0_ids,
        "bytes_lazy": sorted(ids, key=lambda cid: (by_id[cid]["dma_bytes"], cid)),
        "calls_lazy": sorted(ids, key=lambda cid: (by_id[cid]["dma_calls"], cid)),
        "random_lazy": sorted(ids, key=lambda cid: (random_key(target["workload_id"], cid), cid)),
    }
    result = {
        "schema": "c3_invalid_dominator_peeling_front_v1",
        "status": "wave0_and_controls_frozen_before_fullgraph_fpga_or_latency",
        "workload_id": target["workload_id"],
        "eligible_candidate_count": len(programs),
        "operator_proxy_programs": programs,
        "proxy_axes": ["dma_bytes", "dma_calls", "extra_submissions"],
        "wave0_candidate_ids": wave0_ids,
        "dominated_by": dominated_by,
        "control_orders": orders,
        "peeling_update": policy["peeling_algorithm"],
        "label_visibility": {
            "target_full_graph": False, "target_fpga_correctness": False,
            "target_operator_latency": False, "target_full_graph_latency": False,
        },
        "bindings": {
            "target_manifest": sha256(args.target_contract / "artifact_hashes.json"),
            "policy_manifest": sha256(args.policy_contract / "artifact_hashes.json"),
            "local_manifest": sha256(args.local_qualification / "artifact_hashes.json"),
        },
        "verified_artifact_counts": verified,
        "source_sha256": sha256(Path(__file__).resolve()),
        "claim_boundary": "Frozen target-local wave and controls only; no R50F full-graph build, FPGA label, latency, oracle, or peeling success claim.",
    }
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "front.json", result)
    (args.output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    write_json(args.output_dir / "artifact_hashes.json", {"artifacts": {
        p.name: sha256(p) for p in args.output_dir.iterdir()
        if p.is_file() and p.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "eligible_candidate_count": len(programs), "wave0_candidate_ids": wave0_ids,
        "dominated_by": dominated_by, "control_orders": orders,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
