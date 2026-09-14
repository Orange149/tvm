#!/usr/bin/env python3
"""Build stock plus a pre-frozen operator-proxy front, and no other candidate."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
import time

import tvm
from tvm import relay
import vta

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from build_vta_resnet50_fused_program_pool import build_stock, fused_features
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
    outer_started = time.perf_counter()
    target_dir = Path(args.target_contract)
    policy_dir = Path(args.policy_contract)
    local_dir = Path(args.local_qualification)
    front_dir = Path(args.proxy_front_contract)
    verified = {
        "target_contract": len(verify_artifacts_compatible(target_dir)),
        "policy_contract": len(verify_artifacts_compatible(policy_dir)),
        "local_qualification": len(verify_artifacts_compatible(local_dir)),
        "proxy_front_contract": len(verify_artifacts_compatible(front_dir)),
    }
    target = read_json(target_dir / "contract.json")
    policy_path = policy_dir / ("policy.json" if (policy_dir / "policy.json").is_file()
                                else "contract.json")
    front_path = front_dir / ("front.json" if (front_dir / "front.json").is_file()
                              else "contract.json")
    policy = read_json(policy_path)
    local = read_json(local_dir / "summary.json")
    front = read_json(front_path)
    if policy.get("status") not in {
        "frozen_before_target_lower_fsim_fullgraph_fpga_or_latency",
        "frozen_before_target_lowering_fsim_fullgraph_fpga_or_latency",
    }:
        raise RuntimeError("policy contract is not pristine")
    if local.get("status") != "completed_local_no_board":
        raise RuntimeError("local qualification is incomplete")
    if front.get("status") not in {
        "operator_proxy_front_frozen_before_fullgraph_fpga_or_latency",
        "wave0_and_controls_frozen_before_fullgraph_fpga_or_latency",
    }:
        raise RuntimeError("operator proxy front is not pristine")
    if any(front["label_visibility"].values()) or front.get("board_contacted"):
        raise RuntimeError("target label leaked into proxy front")
    bound_local = front.get("bound_inputs", {}).get("local_artifact_manifest_sha256")
    if bound_local is None:
        bound_local = front.get("bindings", {}).get("local_manifest")
    if bound_local != sha256(local_dir / "artifact_hashes.json"):
        raise RuntimeError("proxy front does not bind local qualification")

    candidates = {row["candidate_id"]: row
                  for row in read_jsonl(local_dir / "candidates_v2.jsonl")}
    eligible = {row["candidate_id"] for row in read_jsonl(local_dir / "fsim_results.jsonl")
                if row.get("status") == "passed"}
    selected_ids = list(front.get("proxy_front_candidate_ids", front.get("wave0_candidate_ids", [])))
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("empty or duplicate proxy front")
    if not set(selected_ids).issubset(eligible):
        raise RuntimeError("proxy front contains a non-FSim-qualified candidate")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    env = vta.get_env()
    model_path = target.get("source_model_implementation", {}).get("path")
    if model_path == "python/tvm/relay/testing/resnet.py":
        relay_program, params = make_relay_testing_resnet50_program(env)
        full_graph_source = "relay.testing.resnet.get_workload(num_layers=50)"
        pretrained = False
    else:
        relay_program, params = make_relay_program(env, pretrained=True)
        full_graph_source = "mxnet.gluon.model_zoo.vision.resnet50_v2"
        pretrained = True
    stock = build_stock(relay_program, params, output, env)
    stock_graph = read_json(output / "stock_reference" / "graph.json")
    stock_features = fused_features(stock, stock_graph, target["workload"])
    stock.update(fused_features=stock_features)
    write_json(output / "stock_reference" / "build.json", stock)
    print("proxy-front stock complete {:.3f}s".format(stock["build_seconds"]), flush=True)

    programs = []
    for position, candidate_id in enumerate(selected_ids):
        row = candidates[candidate_id]
        root = output / candidate_id
        root.mkdir()
        build = build_one(row, relay_program, params, root, env)
        local_build = root / row["public_mode"]
        graph = read_json(local_build / "graph.json")
        if graph != stock_graph:
            raise RuntimeError("candidate Graph JSON differs from stock")
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
        programs.append(program)
        write_json(local_build / "program.json", program)
        print(
            "proxy-front {}/{} {} {} B={} N={} S={}".format(
                position + 1, len(selected_ids), row["family_id"], row["public_mode"],
                program["dma_bytes"], program["dma_calls"], program["extra_submissions"],
            ), flush=True,
        )
        gc.collect()

    result = {
        "schema": "c3_vta_resnet50_operator_proxy_front_pool_v1",
        "status": "operator_proxy_front_fullgraphs_built_before_fpga",
        "workload_id": target["workload_id"],
        "source_model": target["source_model"],
        "source_layer": target["source_layer"],
        "full_graph_source": full_graph_source,
        "pretrained": pretrained,
        "eligible_program_count": int(front["eligible_candidate_count"]),
        "built_program_count": len(programs),
        "programs": programs,
        "stock_reference": stock,
        "stock_fused_features": stock_features,
        "selection_axes": front["proxy_axes"],
        "pareto_front_candidate_ids": selected_ids,
        "pareto_front_count": len(selected_ids),
        "total_build_seconds": stock["build_seconds"] + sum(
            row["build_seconds"] for row in programs
        ),
        "label_visibility": {
            "fpga_correctness": False,
            "operator_latency": False,
            "full_graph_latency": False,
            "tophub_latency": False,
        },
        "bound_inputs": {
            "target_contract_sha256": sha256(target_dir / "contract.json"),
            "policy_contract_sha256": sha256(policy_path),
            "local_artifact_manifest_sha256": sha256(local_dir / "artifact_hashes.json"),
            "proxy_front_artifact_manifest_sha256": sha256(front_dir / "artifact_hashes.json"),
        },
        "verified_artifact_counts": verified,
        "builder_source_sha256": sha256(Path(__file__).resolve()),
        "board_contacted": False,
        "outer_process_seconds_through_pool_write": time.perf_counter() - outer_started,
        "claim_boundary": (
            "Actual prospective stock-plus-front build cost. Exact fused features are audited "
            "after selection and cannot add or remove candidates."
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
        "eligible_program_count": result["eligible_program_count"],
        "built_program_count": result["built_program_count"],
        "candidate_ids": selected_ids,
        "total_build_seconds": result["total_build_seconds"],
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-contract", required=True)
    parser.add_argument("--policy-contract", required=True)
    parser.add_argument("--local-qualification", required=True)
    parser.add_argument("--proxy-front-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
