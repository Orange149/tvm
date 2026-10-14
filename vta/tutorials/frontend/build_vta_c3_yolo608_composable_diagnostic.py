#!/usr/bin/env python3
"""Build and scope-audit the YOLO-608 two-route composable residency graph."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
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
DEFAULT_LOCAL_GATE = GROUPED / "20260924_c3_yolo608_composable_local_run02"
DEFAULT_REFERENCE = GROUPED / "20260924_c3_yolo608_graph_deployments_build_run02"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo608_composable_build_run01"
OBJDUMP = Path(
    "/home/orange/alinx_plsdk/install/sysroots/x86_64-petalinux-linux/usr/bin/"
    "aarch64-xilinx-linux/aarch64-xilinx-linux-objdump"
)
VARIANT = "selector_v3_composable"
MODE = "input_weight_resident_composable"
IMPLEMENTATION_MODE = 10
EXPECTED_DEBUG_FUNCTIONS = {
    "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_6_compute_",
    "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_10_compute_",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def composable_routes(args):
    summary = load(Path(args.local_gate) / "summary.json")
    if summary.get("status") != "LOCAL_COMPOSABLE_GATE_PASSED":
        raise ValueError("composable local gate did not pass")
    candidates = load(Path(args.local_gate) / "candidates.json")
    by_parent = {row["parent_mode5_candidate_id"]: row for row in candidates}
    routes = selected_routes(
        args.pools_dir, args.local_dir, args.analysis_dir, args.tasks_dir
    )["selector_v2"]
    converted = []
    for route in routes:
        parent = route["representative_candidate_id"]
        candidate = by_parent.get(parent)
        if candidate is None:
            raise ValueError("local mode-10 gate lacks parent route " + parent)
        row = json.loads(json.dumps(route))
        row["candidate_id"] = candidate["candidate_id"]
        row["representative_candidate_id"] = candidate["candidate_id"]
        row["parent_mode5_candidate_id"] = parent
        row["identity"]["public_mode"] = MODE
        row["identity"]["implementation_mode"] = IMPLEMENTATION_MODE
        converted.append(row)
    if len(converted) != 2:
        raise ValueError("expected exactly two composable routes")
    return converted


def debug_scope_audit(library):
    completed = subprocess.run(
        [str(OBJDUMP), "-dr", str(library)],
        check=True,
        capture_output=True,
        text=True,
    )
    current = None
    counts = {}
    header = re.compile(r"^[0-9a-f]+ <([^>]+)>:$")
    for raw in completed.stdout.splitlines():
        match = header.match(raw.strip())
        if match:
            current = match.group(1)
            continue
        if current and re.search(r"\bbl\s+[0-9a-f]+ <VTASetDebugMode@plt>", raw):
            counts[current] = counts.get(current, 0) + 1
    if set(counts) != EXPECTED_DEBUG_FUNCTIONS:
        raise RuntimeError("VTASetDebugMode escaped target functions: {}".format(counts))
    if any(count != 2 for count in counts.values()):
        raise RuntimeError("each target must have exact 32->0 calls: {}".format(counts))
    return {
        "status": "passed",
        "calls_by_function": dict(sorted(counts.items())),
        "expected_functions": sorted(EXPECTED_DEBUG_FUNCTIONS),
        "global_vta_debug_flag": 0,
        "interpretation": "force-serial is scoped to the two routed resident compute functions",
    }


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    routes = composable_routes(args)
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
        debug_flag=0,
    )
    reference = Path(args.reference_build).resolve()
    reference_summary = load(reference / "summary.json")
    if build["graph_sha256"] != sha256(reference / "stock_production/graph.json"):
        raise RuntimeError("composable diagnostic changed graph JSON")
    semantic = semantic_params_hash(output / VARIANT / "params.bin")[0]
    if semantic != reference_summary["parameter_semantic_sha256"]:
        raise RuntimeError("composable diagnostic changed semantic parameters")
    scope = debug_scope_audit(output / VARIANT / "graphlib.so")
    summary = {
        "schema": "c3_yolo608_composable_build_v1",
        "status": "COMPOSABLE_TWO_ROUTE_GRAPH_BUILT_AND_SCOPED",
        "input_shape": list(shape),
        "variant": build,
        "routes": routes,
        "vta_debug_flag": 0,
        "debug_scope_audit": scope,
        "local_gate": str(Path(args.local_gate).resolve()),
        "local_gate_summary_sha256": sha256(Path(args.local_gate) / "summary.json"),
        "reference_build": str(reference),
        "board_contacted": False,
        "timing_collected": False,
        "performance_claim": False,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "status": summary["status"],
        "routes": len(routes),
        "debug_scope": scope,
    }, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools-dir", default=str(DEFAULT_POOLS))
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL))
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS))
    parser.add_argument("--tasks-dir", default=str(DEFAULT_TASKS))
    parser.add_argument("--local-gate", default=str(DEFAULT_LOCAL_GATE))
    parser.add_argument("--reference-build", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--cfg", default=str(DEFAULT_PREPARE / "resolution_608/yolov3-tiny-608.cfg"))
    parser.add_argument("--weights", default="/tmp/tvm_test_data/darknet/yolov3-tiny.weights")
    parser.add_argument("--darknet-lib", default="/home/orange/.tvm_test_data/darknet/libdarknet2.0.so")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
