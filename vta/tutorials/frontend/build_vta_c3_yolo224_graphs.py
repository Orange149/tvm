#!/usr/bin/env python3
"""Build stock and frozen-route YOLO-224 deployments for the C3 first screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import tvm
from tvm import relay
import vta

from build_vta_c3_evaluation_v5_graphs import (
    build_routed,
    build_stock,
    cross_compiler,
    load,
    load_jsonl,
    route_payload,
    sha256,
    write_json,
)
from build_vta_yolov3_tiny_relay_residency_dispatch_pair import make_relay_program
from freeze_vta_c3_selector_v4_production_timing import verify as verify_flat
from run_vta_yolov3_tiny_relay_residency_dispatch_pair_board import semantic_params_hash


VARIANTS = ("stock", "incumbent", "input_control", "selector_v4")


def same_tile(left, right):
    names = ("tile_b", "tile_h", "tile_w", "tile_ci", "tile_co",
             "oc_nthread", "h_nthread")
    return all(
        int(left["visible_features"][name]) == int(right["visible_features"][name])
        for name in names
    )


def selected_routes(pool_dir, timing_freeze_dir, analysis_dir):
    pool_dir = Path(pool_dir)
    timing_freeze_dir = Path(timing_freeze_dir)
    analysis_dir = Path(analysis_dir)
    verify_flat(pool_dir)
    verify_flat(timing_freeze_dir)
    verify_flat(analysis_dir)
    analysis = load(analysis_dir / "summary.json")
    if analysis.get("status") != "YOLO224_SINGLE_BOOT_OPERATOR_SCREEN_ANALYZED":
        raise ValueError("YOLO-224 operator analysis is incomplete")
    winners = load(analysis_dir / "selection_winners_frozen.json")
    policies = load(timing_freeze_dir / "task_policies.json")
    candidates = {row["program_id"]: row for row in load_jsonl(pool_dir / "candidates.jsonl")}
    if set(winners) != set(policies) or len(policies) != 3:
        raise ValueError("winner/task identity mismatch")
    result = {"incumbent": [], "input_control": [], "selector_v4": []}
    identities = {}
    for task_id in sorted(policies):
        policy = policies[task_id]
        incumbent = policy["incumbent_program_id"]
        winner = winners[task_id]["selector_v4_B3_winner"]
        if winner not in policy["production_program_pool"]:
            raise ValueError("selected winner is outside the production pool")
        if candidates[winner]["residence_mode"] != "input_weight_resident_transition_safe":
            raise ValueError("selected YOLO-224 deployment is not mode11")
        controls = [
            program for program in policy["production_program_pool"]
            if candidates[program]["residence_mode"] == "input_stationary"
            and same_tile(candidates[program], candidates[winner])
        ]
        if len(controls) != 1:
            raise ValueError("selected mode11 lacks one exact same-Tile input control")
        control = controls[0]
        result["incumbent"].append(route_payload(candidates[incumbent]))
        result["input_control"].append(route_payload(candidates[control]))
        result["selector_v4"].append(route_payload(candidates[winner]))
        identities[task_id] = {
            "incumbent_program_id": incumbent,
            "input_control_program_id": control,
            "selector_v4_program_id": winner,
            "selector_winner_source": "three dedicated selection samples only",
            "input_control_rule": "same task, same Tile/thread identity, input_stationary mode",
        }
    for name, routes in result.items():
        if len(routes) != 3 or len({row["execution_task_id"] for row in routes}) != 3:
            raise ValueError(name + " does not contain all three exact YOLO-224 routes")
    return result, identities


def save_artifacts(output, builds, args, shape, identities):
    if len({row["graph_sha256"] for row in builds}) != 1:
        raise RuntimeError("deployment variants produced different graph JSON")
    semantic = [
        semantic_params_hash(output / row["deployment_variant"] / "params.bin")
        for row in builds
    ]
    if any(value != semantic[0] for value in semantic[1:]):
        raise RuntimeError("deployment variants produced semantically different parameters")
    if any(not row["memory_contract"]["fits"] for row in builds):
        raise RuntimeError("a deployment exceeds the 192 MiB single-executor contract")
    for row in builds[1:]:
        if len(row.get("routes", [])) != 3:
            raise RuntimeError("a routed deployment did not bind exactly three tasks")
    summary = {
        "schema": "c3_yolo224_graph_build_v1",
        "status": "C3_FOUR_YOLO224_DEPLOYMENTS_BUILT",
        "input_shape": list(shape),
        "variants": builds,
        "route_identities": identities,
        "parameter_semantic_sha256": semantic[0][0],
        "parameter_inventory": semantic[0][1],
        "same_graph_json": True,
        "same_parameter_semantics": True,
        "non_target_build_context_identical": True,
        "post_evaluation_reselection": False,
        "assets": {
            "cfg": {"path": str(Path(args.cfg).resolve()), "sha256": sha256(args.cfg)},
            "weights": {"path": str(Path(args.weights).resolve()), "sha256": sha256(args.weights)},
            "darknet_lib": {
                "path": str(Path(args.darknet_lib).resolve()),
                "sha256": sha256(args.darknet_lib),
            },
            "board_pool": str(Path(args.board_pool_dir).resolve()),
            "timing_freeze": str(Path(args.timing_freeze_dir).resolve()),
            "operator_analysis": str(Path(args.analysis_dir).resolve()),
        },
        "execution_policy": "one executor and parameter set per fresh RPC process",
        "board_contacted": False,
        "claim_boundary": (
            "single-boot selected-route graph build and exact dispatch audit only; "
            "FPGA graph correctness and latency remain pending"
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(
        " ".join(__import__("sys").argv) + "\n", encoding="utf-8"
    )
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    return summary


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    for asset in (args.cfg, args.weights, args.darknet_lib):
        if not Path(asset).is_file():
            raise FileNotFoundError(asset)
    if sha256(args.weights) != "dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c":
        raise ValueError("YOLO weights do not match the frozen model")
    routes, identities = selected_routes(
        args.board_pool_dir, args.timing_freeze_dir, args.analysis_dir
    )
    output.mkdir(parents=True)
    env = vta.get_env()
    relay_program, params, shape = make_relay_program(env, args)
    if tuple(shape) != (1, 3, 224, 224):
        raise ValueError("derived graph is not YOLO-224")
    compiler = cross_compiler()
    builds = [build_stock(relay_program, params, output, env, compiler)]
    for name in VARIANTS[1:]:
        builds.append(build_routed(
            name, routes[name], relay_program, params, output, env, compiler
        ))
    summary = save_artifacts(output, builds, args, shape, identities)
    print(json.dumps({
        "status": summary["status"],
        "input_shape": summary["input_shape"],
        "variants": [{
            "name": row["deployment_variant"],
            "routes": len(row["routes"]),
            "build_seconds": row["build_seconds"],
            "static_contract_bytes": row["memory_contract"]["static_contract_bytes"],
        } for row in builds],
    }, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-pool-dir", type=Path, required=True)
    parser.add_argument("--timing-freeze-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--weights", default="/tmp/tvm_test_data/darknet/yolov3-tiny.weights")
    parser.add_argument("--darknet-lib", default="/home/orange/.tvm_test_data/darknet/libdarknet2.0.so")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
