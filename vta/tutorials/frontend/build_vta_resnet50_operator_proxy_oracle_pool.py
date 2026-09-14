#!/usr/bin/env python3
"""Build candidates excluded by a prospective operator-proxy front for oracle completion."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import vta

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from build_vta_resnet50_fused_program_pool import fused_features
from build_vta_resnet50_generic_residency_tir_audit import build_one
from build_vta_resnet50_relay_residency_dispatch_pair import (
    make_relay_program,
    make_relay_testing_resnet50_program,
)
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
    front_contract_dir = Path(args.proxy_front_contract)
    front_pool_dir = Path(args.front_pool)
    front_board_dir = Path(args.front_board_result)
    verified = {
        "target_contract": len(verify_artifacts_compatible(target_dir)),
        "policy_contract": len(verify_artifacts_compatible(policy_dir)),
        "local_qualification": len(verify_artifacts_compatible(local_dir)),
        "proxy_front_contract": len(verify_artifacts_compatible(front_contract_dir)),
        "front_pool": len(verify_artifacts_compatible(front_pool_dir)),
        "front_board_result": len(verify_artifacts_compatible(front_board_dir)),
    }
    target = read_json(target_dir / "contract.json")
    local = read_json(local_dir / "summary.json")
    front_contract_path = front_contract_dir / (
        "front.json" if (front_contract_dir / "front.json").is_file() else "contract.json"
    )
    policy_path = policy_dir / (
        "policy.json" if (policy_dir / "policy.json").is_file() else "contract.json"
    )
    front_contract = read_json(front_contract_path)
    front_pool = read_json(front_pool_dir / "pool.json")
    front_board = read_json(front_board_dir / "summary.json")
    if local.get("status") != "completed_local_no_board":
        raise RuntimeError("local qualification is incomplete")
    if front_contract.get("status") not in {
        "operator_proxy_front_frozen_before_fullgraph_fpga_or_latency",
        "wave0_and_controls_frozen_before_fullgraph_fpga_or_latency",
    }:
        raise RuntimeError("proxy front contract mismatch")
    if front_pool.get("status") != "operator_proxy_front_fullgraphs_built_before_fpga":
        raise RuntimeError("prospective front pool mismatch")
    if front_board.get("status") != "prospective_final_fused_pareto_front_board_complete":
        raise RuntimeError("prospective front board wave is incomplete")
    front_pool_manifest = sha256(front_pool_dir / "artifact_hashes.json")
    if front_board["pool_artifact_manifest_sha256"] != front_pool_manifest:
        raise RuntimeError("front board result binds a different front pool")
    front_ids = list(front_contract.get(
        "proxy_front_candidate_ids", front_contract.get("wave0_candidate_ids", [])
    ))
    if front_ids != front_pool["pareto_front_candidate_ids"]:
        raise RuntimeError("built front differs from frozen operator proxy front")
    if front_ids != front_board["candidate_ids"]:
        raise RuntimeError("front board result did not execute the frozen front")

    candidates = {row["candidate_id"]: row
                  for row in read_jsonl(local_dir / "candidates_v2.jsonl")}
    eligible_ids = [row["candidate_id"] for row in read_jsonl(local_dir / "fsim_results.jsonl")
                    if row.get("status") == "passed"]
    remaining_ids = [cid for cid in eligible_ids if cid not in set(front_ids)]
    if len(eligible_ids) != front_contract["eligible_candidate_count"]:
        raise RuntimeError("eligible candidate count drift")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    copy_started = time.perf_counter()
    shutil.copytree(front_pool_dir / "stock_reference", output / "stock_reference")
    front_programs = {row["candidate_id"]: row for row in front_pool["programs"]}
    for cid in front_ids:
        shutil.copytree(front_pool_dir / cid, output / cid)
    copy_seconds = time.perf_counter() - copy_started

    env = vta.get_env()
    if front_pool.get("full_graph_source") == "relay.testing.resnet.get_workload(num_layers=50)":
        relay_program, params = make_relay_testing_resnet50_program(env)
    else:
        relay_program, params = make_relay_program(env, pretrained=True)
    stock = read_json(output / "stock_reference" / "build.json")
    stock_graph = read_json(output / "stock_reference" / "graph.json")
    stock_features = front_pool["stock_fused_features"]
    newly_built = {}
    completion_started = time.perf_counter()
    for position, candidate_id in enumerate(remaining_ids):
        row = candidates[candidate_id]
        root = output / candidate_id
        root.mkdir()
        build = build_one(row, relay_program, params, root, env)
        local_build = root / row["public_mode"]
        graph = read_json(local_build / "graph.json")
        if graph != stock_graph:
            raise RuntimeError("candidate Graph JSON differs from prospective stock")
        features = fused_features(build, graph, target["workload"])
        program = {
            "program_id": candidate_id,
            "candidate_id": candidate_id,
            "family_id": row["family_id"],
            "public_mode": row["public_mode"],
            "knobs": row["knobs"],
            "relative_dir": candidate_id + "/" + row["public_mode"],
            "build_seconds": build["build_seconds"],
            "dispatch_audit": build["dispatch_audit"],
            "graph_sha256": build["graph_sha256"],
            "params_sha256": build["params_sha256"],
            "graphlib_sha256": build["graphlib_sha256"],
            **features,
        }
        newly_built[candidate_id] = program
        write_json(local_build / "program.json", program)
        print(
            "oracle-build {}/{} {} {} B={} N={} S={}".format(
                position + 1, len(remaining_ids), row["family_id"], row["public_mode"],
                program["dma_bytes"], program["dma_calls"], program["extra_submissions"],
            ), flush=True,
        )
        gc.collect()
    completion_wall_seconds = time.perf_counter() - completion_started

    programs = [front_programs[cid] if cid in front_programs else newly_built[cid]
                for cid in eligible_ids]
    new_model_build_seconds = sum(row["build_seconds"] for row in newly_built.values())
    result = {
        "schema": "c3_vta_resnet50_operator_proxy_oracle_pool_v1",
        "status": "operator_proxy_oracle_completion_pool_after_front_labels",
        "workload_id": target["workload_id"],
        "source_model": target["source_model"],
        "source_layer": target["source_layer"],
        "full_graph_source": front_pool.get("full_graph_source"),
        "pretrained": front_pool.get("pretrained"),
        "eligible_program_count": len(programs),
        "programs": programs,
        "stock_reference": stock,
        "stock_fused_features": stock_features,
        "selection_kind": "pre_frozen_operator_proxy_front",
        "pareto_front_candidate_ids": front_ids,
        "pareto_front_count": len(front_ids),
        "remaining_candidate_ids": remaining_ids,
        "remaining_candidate_count": len(remaining_ids),
        "build_cost": {
            "prospective_stock_plus_front_seconds": front_pool["total_build_seconds"],
            "oracle_completion_new_candidate_model_build_seconds": new_model_build_seconds,
            "oracle_completion_process_wall_seconds": completion_wall_seconds,
            "copy_front_artifacts_seconds": copy_seconds,
            "counterfactual_stock_plus_all_candidate_model_build_seconds": (
                front_pool["total_build_seconds"] + new_model_build_seconds
            ),
        },
        "label_visibility": {
            "prospective_front_fpga_correctness": True,
            "prospective_front_full_graph_latency": True,
            "remaining_fpga_correctness": False,
            "remaining_full_graph_latency": False,
            "tophub_latency": False,
        },
        "prospective_front_source_artifact_manifest_sha256": front_pool_manifest,
        "prospective_front_board_artifact_manifest_sha256": sha256(
            front_board_dir / "artifact_hashes.json"
        ),
        "bound_inputs": {
            "target_contract_sha256": sha256(target_dir / "contract.json"),
            "policy_contract_sha256": sha256(policy_path),
            "local_artifact_manifest_sha256": sha256(local_dir / "artifact_hashes.json"),
            "proxy_front_artifact_manifest_sha256": sha256(
                front_contract_dir / "artifact_hashes.json"
            ),
        },
        "verified_artifact_counts": verified,
        "builder_source_sha256": sha256(Path(__file__).resolve()),
        "board_contacted_for_remaining": False,
        "claim_boundary": (
            "Post-front oracle completion. Front labels were already exposed, but cannot "
            "alter the frozen front or the remaining set; only remaining candidates are new builds."
        ),
    }
    write_json(output / "pool.json", result)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "status": result["status"],
        "eligible_program_count": len(programs),
        "front_count": len(front_ids),
        "remaining_count": len(remaining_ids),
        "build_cost": result["build_cost"],
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-contract", required=True)
    parser.add_argument("--policy-contract", required=True)
    parser.add_argument("--local-qualification", required=True)
    parser.add_argument("--proxy-front-contract", required=True)
    parser.add_argument("--front-pool", required=True)
    parser.add_argument("--front-board-result", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
