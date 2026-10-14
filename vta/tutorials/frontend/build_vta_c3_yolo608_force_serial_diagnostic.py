#!/usr/bin/env python3
"""Build the frozen two-route YOLO-608 graph with VTA force-serial enabled."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo608_force_serial_diagnostic_build_run01"
DEBUG_FORCE_SERIAL = 1 << 5
VARIANT = "selector_v2_force_serial"


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    routes = selected_routes(
        args.pools_dir, args.local_dir, args.analysis_dir, args.tasks_dir
    )["selector_v2"]
    env = vta.get_env()
    relay_program, params, shape = make_relay_program(env, args)
    build = build_routed(
        VARIANT,
        routes,
        relay_program,
        params,
        output,
        env,
        cross_compiler(),
        debug_flag=DEBUG_FORCE_SERIAL,
    )
    reference = Path(args.reference_build).resolve()
    reference_summary = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    if build["graph_sha256"] != sha256(reference / "stock_production/graph.json"):
        raise RuntimeError("force-serial diagnostic changed graph JSON")
    semantic = semantic_params_hash(output / VARIANT / "params.bin")[0]
    if semantic != reference_summary["parameter_semantic_sha256"]:
        raise RuntimeError("force-serial diagnostic changed semantic parameters")
    summary = {
        "schema": "c3_yolo608_force_serial_diagnostic_build_v1",
        "status": "FORCE_SERIAL_TWO_ROUTE_DIAGNOSTIC_BUILT",
        "input_shape": list(shape),
        "variant": build,
        "vta_debug_flag": DEBUG_FORCE_SERIAL,
        "reference_build": str(reference),
        "board_contacted": False,
        "timing_collected": False,
        "performance_claim": False,
        "diagnostic_question": (
            "Does force-serial execution restore correctness for the frozen two-route graph?"
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "status": summary["status"],
        "variant": VARIANT,
        "routes": len(routes),
        "debug_flag": DEBUG_FORCE_SERIAL,
    }, indent=2, sort_keys=True), flush=True)


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
