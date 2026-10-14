#!/usr/bin/env python3
"""Build stock, frozen-incumbent, and selector-v2 YOLO-608 deployments."""

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
DEFAULT_POOLS = GROUPED / "20260924_c3_yolo608_candidate_pools_run01"
DEFAULT_LOCAL = GROUPED / "20260924_c3_yolo608_local_qualification_run01"
DEFAULT_ANALYSIS = GROUPED / "20260924_c3_holdout_cross_boot_analysis_v2_run02"
DEFAULT_TASKS = GROUPED / "20260924_c3_holdout_task_freeze_v2_run01"
DEFAULT_PREPARE = GROUPED / "20260924_c3_yolo_unseen_resolution_prepare_run02"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo608_graph_deployments_build_run02"
VARIANTS = ("stock_production", "known_incumbent", "selector_v2")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def route_payload(pool, candidate, program_id):
    mode = candidate["residence_mode"]
    return {
        "candidate_id": program_id,
        "workload_id": pool["workload_id"],
        "execution_task_id": pool["execution_task_id"],
        "identity": {
            "workload": pool["workload"],
            "complete_config_entity": candidate["complete_config_entity"],
            "public_mode": mode,
            "implementation_mode": candidate["implementation_mode"],
        },
        "representative_candidate_id": candidate["candidate_id"],
    }


def selected_routes(pools_dir, local_dir, analysis_dir, tasks_dir):
    analysis = load(Path(analysis_dir) / "analysis.json")
    stable = analysis["stable_deployment_task_ids"]
    if len(stable) != 2:
        raise ValueError("this frozen graph experiment requires exactly two stable tasks")
    decisions = {task_id: analysis["confirmation"][task_id] for task_id in stable}
    task_rows = {row["task_id"]: row for row in load(Path(tasks_dir) / "limited_budget_tasks.json")}
    programs = {(row["workload_id"], row["program_id"]): row
                for row in load_jsonl(Path(local_dir) / "qualified_programs.jsonl")}
    result = {"known_incumbent": [], "selector_v2": []}
    for task_id in stable:
        task = task_rows[task_id]
        pool = load(Path(pools_dir) / task["pool_file"])
        candidates = {row["candidate_id"]: row for row in pool["candidates"]}
        incumbent_candidate = candidates[task["incumbent_candidate_id"]]
        result["known_incumbent"].append(
            route_payload(pool, incumbent_candidate, decisions[task_id]["incumbent_program_id"])
        )
        selected_program_id = decisions[task_id]["selected_program_id"]
        qualified = programs[(task["workload_id"], selected_program_id)]
        selected_candidate = candidates[qualified["representative_candidate_id"]]
        result["selector_v2"].append(route_payload(pool, selected_candidate, selected_program_id))
    for name, routes in result.items():
        if len({json.dumps(row["identity"]["workload"], sort_keys=True) for row in routes}) != 2:
            raise ValueError(name + " routes do not address two distinct workloads")
    if any(row["identity"]["implementation_mode"] != 0 for row in result["known_incumbent"]):
        raise ValueError("incumbent deployment must use frozen mode-0 programs")
    if any(row["identity"]["implementation_mode"] != 5 for row in result["selector_v2"]):
        raise ValueError("selector deployment must use the confirmed mode-5 programs")
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
    routes = selected_routes(args.pools_dir, args.local_dir, args.analysis_dir, args.tasks_dir)
    output.mkdir(parents=True)
    env = vta.get_env()
    relay_program, params, shape = make_relay_program(env, args)
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
    summary = {
        "schema": "c3_yolo608_graph_deployments_build_v1",
        "status": "THREE_YOLO608_DEPLOYMENTS_CROSS_BUILT",
        "input_shape": list(shape),
        "variants": builds,
        "parameter_semantic_sha256": semantic[0][0],
        "parameter_inventory": semantic[0][1],
        "serialized_parameter_note": (
            "params.bin byte hashes may differ across schedules; semantic hash covers sorted "
            "name, shape, dtype, and tensor bytes"
        ),
        "execution_policy": "one executor per fresh RPC process; never co-reside three graphs",
        "assets": {
            "cfg": {"path": str(Path(args.cfg).resolve()), "sha256": sha256(args.cfg)},
            "weights": {"path": str(Path(args.weights).resolve()), "sha256": sha256(args.weights)},
            "darknet_lib": {"path": str(Path(args.darknet_lib).resolve()),
                            "sha256": sha256(args.darknet_lib)},
            "cross_boot_analysis": {"path": str(Path(args.analysis_dir).resolve()),
                                    "sha256": sha256(Path(args.analysis_dir) / "analysis.json")},
        },
        "board_contacted": False,
        "claim_boundary": "cross-build and exact dispatch audit only; full-graph FPGA evidence pending",
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
        "variants": [{"name": row["deployment_variant"],
                      "routes": len(row["routes"]),
                      "build_seconds": row["build_seconds"],
                      "static_contract_bytes": row["memory_contract"]["static_contract_bytes"]}
                     for row in builds],
    }, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools-dir", default=str(DEFAULT_POOLS))
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL))
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS))
    parser.add_argument("--tasks-dir", default=str(DEFAULT_TASKS))
    parser.add_argument("--cfg", default=str(DEFAULT_PREPARE / "resolution_608/yolov3-tiny-608.cfg"))
    parser.add_argument("--weights", default="/tmp/tvm_test_data/darknet/yolov3-tiny.weights")
    parser.add_argument("--darknet-lib", default="/home/orange/.tvm_test_data/darknet/libdarknet2.0.so")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
