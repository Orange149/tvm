#!/usr/bin/env python3
"""Freeze a multi-strategy online execution suite before target board labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(args):
    roots = {
        "target": Path(args.target_contract),
        "policy": Path(args.policy_contract),
        "local": Path(args.local_qualification),
        "front": Path(args.front_contract),
    }
    verified = {name: len(verify_artifacts_compatible(path)) for name, path in roots.items()}
    target = read(roots["target"] / "contract.json")
    policy = read(roots["policy"] / "policy.json")
    local = read(roots["local"] / "summary.json")
    front = read(roots["front"] / "front.json")
    if policy.get("status") != "frozen_before_target_lowering_fsim_fullgraph_fpga_or_latency":
        raise ValueError("policy is not pristine")
    if local.get("status") != "completed_local_no_board" or local.get("board_contacted"):
        raise ValueError("local qualification is not pristine")
    if front.get("status") != "wave0_and_controls_frozen_before_fullgraph_fpga_or_latency":
        raise ValueError("front is not pristine")
    workload_ids = {target["workload_id"], policy["workload_id"], front["workload_id"]}
    if len(workload_ids) != 1:
        raise ValueError("workload binding mismatch")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    executor = Path(__file__).with_name("run_vta_invalid_dominator_peeling_online.py")
    board_adapter = Path(__file__).with_name("run_vta_resnet50_fused_program_pareto_board.py")
    stock_builder = Path(__file__).with_name("build_vta_resnet50_fused_program_pool.py")
    candidate_builder = Path(__file__).with_name(
        "build_vta_resnet50_generic_residency_tir_audit.py"
    )
    executions = [{
        "strategy": "peeling",
        "candidate_budget": None,
        "budget_seconds": args.budget_seconds,
        "admission_rule": "frozen Pareto wave; peel only after FPGA-invalid result",
    }]
    executions.extend({
        "strategy": strategy,
        "candidate_budget": args.control_candidate_budget,
        "budget_seconds": args.budget_seconds,
        "admission_rule": "immutable frozen-order prefix",
    } for strategy in ("calls_lazy", "bytes_lazy", "random_lazy"))
    suite = {
        "schema": "c3_vta_online_control_suite_v1",
        "status": "frozen_before_fullgraph_build_fpga_correctness_or_latency",
        "workload_id": target["workload_id"],
        "source_model": target["source_model"],
        "source_layer": target["source_layer"],
        "executions": executions,
        "execution_order": [row["strategy"] for row in executions],
        "bindings": {
            name + "_manifest": sha256(path / "artifact_hashes.json")
            for name, path in roots.items()
        },
        "verified_artifact_counts": verified,
        "executor_source_sha256": sha256(executor),
        "board_adapter_source_sha256": sha256(board_adapter),
        "stock_builder_source_sha256": sha256(stock_builder),
        "candidate_builder_source_sha256": sha256(candidate_builder),
        "board_contacted": False,
        "performance_labels_observed": False,
        "comparison_scope": (
            "Independent executions of method and three simple controls on the same frozen "
            "candidate identities. Control quality is evaluated only after all runs; method "
            "may replace an FPGA-invalid Pareto point under the same wall-clock admission rule."
        ),
        "claim_boundary": (
            "Orders and budgets are prospective; this contract contains no target full-graph "
            "correctness, latency, pool oracle, COCO mAP, or physical AXI observation."
        ),
    }
    write_json(output / "suite.json", suite)
    write_json(output / "artifact_hashes.json", {"artifacts": {
        "suite.json": sha256(output / "suite.json")
    }})
    print(json.dumps(suite, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-contract", required=True)
    parser.add_argument("--policy-contract", required=True)
    parser.add_argument("--local-qualification", required=True)
    parser.add_argument("--front-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-seconds", type=float, default=600.0)
    parser.add_argument("--control-candidate-budget", type=int, default=3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
