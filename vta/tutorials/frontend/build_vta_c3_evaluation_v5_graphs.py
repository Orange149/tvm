#!/usr/bin/env python3
"""Build the sealed stock, incumbent, and selector-v4 YOLO-512 deployments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import tvm
from tvm import autotvm, relay
import vta
from vta.top.residency_dispatch import ExplicitResidencyDispatch

from build_vta_yolov3_tiny_relay_residency_dispatch_pair import (
    cross_compiler,
    make_relay_program,
    sha256,
    write_json,
)
from prepare_vta_c3_yolo_unseen_resolution import graph_memory_contract
from run_vta_yolov3_tiny_relay_residency_dispatch_pair_board import semantic_params_hash


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_POOL = GROUPED / "20260924_c3_yolo512_selector_v4_production_board_pool_freeze_run02"
DEFAULT_FREEZE = GROUPED / "20260926_c3_yolo512_evaluation_v5_freeze_run01"
DEFAULT_ANALYSIS = GROUPED / "20260926_c3_yolo512_evaluation_v5_cross_boot_analysis_run01"
DEFAULT_PREPARE = GROUPED / "20260924_c3_yolo_unseen_resolution_prepare_run05"
DEFAULT_OUTPUT = GROUPED / "20260926_c3_yolo512_evaluation_v5_graph_build_run01"
VARIANTS = ("stock", "incumbent", "selector_v4")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def route_payload(candidate):
    return {
        "candidate_id": candidate["program_id"],
        "workload_id": candidate["workload_id"],
        "execution_task_id": candidate["execution_task_id"],
        "identity": {
            "workload": candidate["identity"]["workload"],
            "complete_config_entity": candidate["complete_config_entity"],
            "public_mode": candidate["residence_mode"],
            "implementation_mode": candidate["implementation_mode"],
        },
        "representative_candidate_id": candidate["candidate_id"],
    }


def selected_routes(pool_dir, freeze_dir, analysis_dir):
    pool_dir, freeze_dir, analysis_dir = map(Path, (pool_dir, freeze_dir, analysis_dir))
    analysis = load(analysis_dir / "analysis.json")
    decision = load(analysis_dir / "graph_deployment_decision.json")
    if analysis.get("status") != "C3_EVALUATION_V5_CROSS_BOOT_ANALYSIS_COMPLETE":
        raise ValueError("cross-boot analysis is incomplete")
    if decision.get("status") != "GRAPH_BUILD_ELIGIBLE":
        raise ValueError("no cross-boot-confirmed mode11 deployment")
    policies = load(freeze_dir / "task_policies.json")
    candidates = {
        row["program_id"]: row for row in load_jsonl(pool_dir / "candidates.jsonl")
    }
    task_programs = decision["task_programs"]
    if set(task_programs) != set(policies):
        raise ValueError("deployment tasks do not match the frozen task set")
    if not any(row["stable_mode11"] for row in task_programs.values()):
        raise ValueError("graph build requires at least one confirmed mode11")
    result = {"incumbent": [], "selector_v4": []}
    for task_id in sorted(policies):
        policy = policies[task_id]
        frozen = task_programs[task_id]
        incumbent = frozen["incumbent_program_id"]
        deploy = frozen["deploy_program"]
        if incumbent != policy["incumbent_program_id"]:
            raise ValueError("incumbent identity drift")
        if deploy not in policy["production_program_pool"]:
            raise ValueError("deploy program is outside the production pool")
        if frozen["stable_mode11"]:
            if deploy != frozen["search_winner"]:
                raise ValueError("confirmed mode11 was replaced after selection")
            if candidates[deploy]["residence_mode"] != "input_weight_resident_transition_safe":
                raise ValueError("stable deployment is not mode11")
        elif deploy != incumbent:
            raise ValueError("unconfirmed task did not fall back to incumbent")
        result["incumbent"].append(route_payload(candidates[incumbent]))
        result["selector_v4"].append(route_payload(candidates[deploy]))
    for name, routes in result.items():
        if len(routes) != 2 or len({row["execution_task_id"] for row in routes}) != 2:
            raise ValueError(name + " does not contain the two exact frozen routes")
    return result


def save_build(local, graph, lib, params, compiler, result):
    (local / "graph.json").write_text(graph, encoding="utf-8")
    (local / "params.bin").write_bytes(relay.save_param_dict(params))
    lib.export_library(str(local / "graphlib.so"), compiler)
    result.update({
        "graph_sha256": sha256(local / "graph.json"),
        "params_sha256": sha256(local / "params.bin"),
        "graphlib_sha256": sha256(local / "graphlib.so"),
        "memory_contract": graph_memory_contract(graph),
    })
    write_json(local / "build.json", result)
    return result


def build_stock(relay_program, params, output, env, compiler, debug_flag=0):
    local = output / VARIANTS[0]
    local.mkdir()
    target = tvm.target.Target(env.target, host=env.target_host)
    relay.backend.te_compiler.get().clear()
    started = time.perf_counter()
    with autotvm.tophub.context(target):
        with vta.build_config(debug_flag=debug_flag, opt_level=3,
                              disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}):
            graph, lib, lowered_params = relay.build(relay_program, target=target, params=params)
    return save_build(local, graph, lib, lowered_params, compiler, {
        "deployment_variant": VARIANTS[0], "routes": [], "dispatch_audit": None,
        "build_seconds": time.perf_counter() - started,
        "vta_debug_flag": debug_flag,
    })


def build_routed(name, routes, relay_program, params, output, env, compiler, debug_flag=0):
    local = output / name
    local.mkdir()
    target = tvm.target.Target(env.target, host=env.target_host)
    relay.backend.te_compiler.get().clear()
    started = time.perf_counter()
    with autotvm.tophub.context(target):
        with ExplicitResidencyDispatch(routes) as dispatch:
            with vta.build_config(debug_flag=debug_flag, opt_level=3,
                                  disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}):
                graph, lib, lowered_params = relay.build(relay_program, target=target, params=params)
    audit = dispatch.summary()
    if not audit["all_routes_scheduled"] or len(audit["schedule_hits"]) != len(routes):
        raise RuntimeError(name + " did not schedule every exact route exactly once")
    return save_build(local, graph, lib, lowered_params, compiler, {
        "deployment_variant": name,
        "routes": routes,
        "dispatch_audit": audit,
        "build_seconds": time.perf_counter() - started,
        "vta_debug_flag": debug_flag,
    })


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    for asset in (args.cfg, args.weights, args.darknet_lib):
        if not Path(asset).is_file():
            raise FileNotFoundError(asset)
    if sha256(args.weights) != "dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c":
        raise ValueError("YOLO weights do not match the frozen model")
    routes = selected_routes(args.board_pool_dir, args.freeze_dir, args.analysis_dir)
    output.mkdir(parents=True)
    env = vta.get_env()
    relay_program, params, shape = make_relay_program(env, args)
    if tuple(shape) != (1, 3, 512, 512):
        raise ValueError("derived graph is not YOLO-512")
    compiler = cross_compiler()
    builds = [
        build_stock(relay_program, params, output, env, compiler),
        build_routed(VARIANTS[1], routes[VARIANTS[1]], relay_program, params, output, env, compiler),
        build_routed(VARIANTS[2], routes[VARIANTS[2]], relay_program, params, output, env, compiler),
    ]
    if len({row["graph_sha256"] for row in builds}) != 1:
        raise RuntimeError("deployment variants produced different graph JSON")
    semantic = [semantic_params_hash(output / row["deployment_variant"] / "params.bin")
                for row in builds]
    if any(value != semantic[0] for value in semantic[1:]):
        raise RuntimeError("deployment variants produced semantically different parameters")
    if any(not row["memory_contract"]["fits"] for row in builds):
        raise RuntimeError("a deployment exceeds the frozen 192 MiB single-executor contract")
    if any(len(row.get("routes", [])) != 2 for row in builds[1:]):
        raise RuntimeError("a routed deployment did not bind exactly two target tasks")
    summary = {
        "schema": "c3_evaluation_v5_yolo512_graph_build_v1",
        "status": "C3_EVALUATION_V5_THREE_YOLO512_DEPLOYMENTS_BUILT",
        "input_shape": list(shape),
        "variants": builds,
        "parameter_semantic_sha256": semantic[0][0],
        "parameter_inventory": semantic[0][1],
        "same_graph_json": True,
        "same_parameter_semantics": True,
        "non_target_build_context_identical": True,
        "post_oracle_reselection": False,
        "assets": {
            "cfg": {"path": str(Path(args.cfg).resolve()), "sha256": sha256(args.cfg)},
            "weights": {"path": str(Path(args.weights).resolve()), "sha256": sha256(args.weights)},
            "darknet_lib": {"path": str(Path(args.darknet_lib).resolve()), "sha256": sha256(args.darknet_lib)},
            "board_pool": str(Path(args.board_pool_dir).resolve()),
            "freeze": str(Path(args.freeze_dir).resolve()),
            "cross_boot_analysis": str(Path(args.analysis_dir).resolve()),
        },
        "execution_policy": "one executor and parameter set per fresh RPC process",
        "board_contacted": False,
        "claim_boundary": "cross-build and exact-route audit only; graph FPGA evidence pending",
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "status": summary["status"], "input_shape": summary["input_shape"],
        "variants": [{"name": row["deployment_variant"], "routes": len(row["routes"]),
                      "build_seconds": row["build_seconds"],
                      "static_contract_bytes": row["memory_contract"]["static_contract_bytes"]}
                     for row in builds],
    }, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-pool-dir", default=str(DEFAULT_POOL))
    parser.add_argument("--freeze-dir", default=str(DEFAULT_FREEZE))
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS))
    parser.add_argument("--cfg", default=str(DEFAULT_PREPARE / "resolution_512/yolov3-tiny-512.cfg"))
    parser.add_argument("--weights", default="/tmp/tvm_test_data/darknet/yolov3-tiny.weights")
    parser.add_argument("--darknet-lib", default="/home/orange/.tvm_test_data/darknet/libdarknet2.0.so")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
