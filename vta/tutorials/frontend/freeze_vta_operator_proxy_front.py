#!/usr/bin/env python3
"""Freeze an operator-TIR DMA Pareto front before any target full-graph build."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from analyze_vta_operator_proxy_pareto_development import operator_proxy_program
from build_vta_resnet50_fused_program_pool import dominates, pareto_front
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def run(args):
    target_dir = Path(args.target_contract)
    policy_dir = Path(args.policy_contract)
    local_dir = Path(args.local_qualification)
    verified = {
        "target_contract": len(verify_artifacts_compatible(target_dir)),
        "policy_contract": len(verify_artifacts_compatible(policy_dir)),
        "local_qualification": len(verify_artifacts_compatible(local_dir)),
    }
    target = read_json(target_dir / "contract.json")
    policy = read_json(policy_dir / "contract.json")
    local = read_json(local_dir / "summary.json")
    if policy.get("status") != "frozen_before_target_lower_fsim_fullgraph_fpga_or_latency":
        raise RuntimeError("operator-proxy policy is not pristine")
    if policy["workload_id"] != target["workload_id"]:
        raise RuntimeError("policy/target mismatch")
    if local.get("status") != "completed_local_no_board":
        raise RuntimeError("local qualification is incomplete")
    if local.get("board_contacted") or local.get("performance_labels_used"):
        raise RuntimeError("target label leaked into local qualification")

    candidates = {row["candidate_id"]: row
                  for row in read_jsonl(local_dir / "candidates_v2.jsonl")}
    static = {row["candidate_id"]: row
              for row in read_jsonl(local_dir / "static_results.jsonl")}
    fsim = read_jsonl(local_dir / "fsim_results.jsonl")
    eligible_ids = [row["candidate_id"] for row in fsim if row.get("status") == "passed"]
    programs = [operator_proxy_program(candidates[cid], static[cid]) for cid in eligible_ids]
    front = sorted(pareto_front(programs), key=lambda row: row["candidate_id"])
    front_ids = [row["candidate_id"] for row in front]
    dominated = {
        row["candidate_id"]: sorted(
            other["candidate_id"] for other in programs if dominates(other, row)
        ) for row in programs if row["candidate_id"] not in front_ids
    }
    result = {
        "schema": "c3_vta_operator_proxy_front_contract_v1",
        "status": "operator_proxy_front_frozen_before_fullgraph_fpga_or_latency",
        "workload_id": target["workload_id"],
        "source_model": target["source_model"],
        "source_layer": target["source_layer"],
        "eligible_candidate_count": len(programs),
        "operator_proxy_programs": programs,
        "proxy_axes": ["dma_bytes", "dma_calls", "extra_submissions"],
        "proxy_front_candidate_ids": front_ids,
        "proxy_front_count": len(front_ids),
        "dominated_by": dominated,
        "label_visibility": {
            "target_full_graph_features": False,
            "target_fpga_correctness": False,
            "target_full_graph_latency": False,
        },
        "bound_inputs": {
            "target_contract_sha256": sha256(target_dir / "contract.json"),
            "policy_contract_sha256": sha256(policy_dir / "contract.json"),
            "local_summary_sha256": sha256(local_dir / "summary.json"),
            "local_artifact_manifest_sha256": sha256(local_dir / "artifact_hashes.json"),
        },
        "verified_artifact_counts": verified,
        "freezer_source_sha256": sha256(Path(__file__).resolve()),
        "board_contacted": False,
        "claim_boundary": (
            "The front uses only target operator lowering/FSim and logical DMA. It has not "
            "seen target full-graph features, FPGA correctness, or latency."
        ),
    }
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    write_json(output / "contract.json", result)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "status": result["status"],
        "eligible_candidate_count": len(programs),
        "proxy_front_count": len(front_ids),
        "proxy_front_candidate_ids": front_ids,
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-contract", required=True)
    parser.add_argument("--policy-contract", required=True)
    parser.add_argument("--local-qualification", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
