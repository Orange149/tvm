#!/usr/bin/env python3
"""Build one-route YOLO-608 graphs to diagnose a failed two-route graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tvm
import vta

from build_vta_c3_yolo608_graph_deployments import (
    DEFAULT_ANALYSIS,
    DEFAULT_LOCAL,
    DEFAULT_POOLS,
    DEFAULT_PREPARE,
    DEFAULT_TASKS,
    build_routed,
    selected_routes,
)
from build_vta_yolov3_tiny_relay_residency_dispatch_pair import (
    cross_compiler,
    make_relay_program,
    sha256,
    write_json,
)
from run_vta_yolov3_tiny_relay_residency_dispatch_pair_board import semantic_params_hash


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_REFERENCE = GROUPED / "20260924_c3_yolo608_graph_deployments_build_run02"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo608_route_correctness_diagnostic_build_run01"


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    routes = selected_routes(args.pools_dir, args.local_dir, args.analysis_dir, args.tasks_dir)["selector_v2"]
    routes = sorted(routes, key=lambda row: row["execution_task_id"])
    env = vta.get_env()
    relay_program, params, shape = make_relay_program(env, args)
    compiler = cross_compiler()
    builds = []
    for route in routes:
        name = "selector_only_" + route["execution_task_id"][:12]
        builds.append(build_routed(name, [route], relay_program, params, output, env, compiler))
    reference = Path(args.reference_build).resolve()
    expected_graph = sha256(reference / "stock_production/graph.json")
    expected_semantic = json.loads((reference / "summary.json").read_text())["parameter_semantic_sha256"]
    for row in builds:
        local = output / row["deployment_variant"]
        if row["graph_sha256"] != expected_graph:
            raise RuntimeError("one-route diagnostic graph JSON differs from frozen graph")
        if semantic_params_hash(local / "params.bin")[0] != expected_semantic:
            raise RuntimeError("one-route diagnostic parameters differ semantically")
    summary = {
        "schema": "c3_yolo608_route_correctness_diagnostic_build_v1",
        "status": "ONE_ROUTE_CORRECTNESS_DIAGNOSTIC_BUILT",
        "input_shape": list(shape),
        "variants": builds,
        "reference_build": str(reference),
        "board_contacted": False,
        "performance_claim": False,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({"status": summary["status"], "variants": [
        {"name": row["deployment_variant"], "route": row["routes"][0]["candidate_id"]}
        for row in builds]}, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools-dir", default=str(DEFAULT_POOLS))
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL))
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS))
    parser.add_argument("--tasks-dir", default=str(DEFAULT_TASKS))
    parser.add_argument("--reference-build", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--cfg", default=str(DEFAULT_PREPARE / "resolution_608/yolov3-tiny-608.cfg"))
    parser.add_argument("--weights", default="/tmp/tvm_test_data/darknet/yolov3-tiny.weights")
    parser.add_argument("--darknet-lib", default="/home/orange/.tvm_test_data/darknet/libdarknet2.0.so")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
