#!/usr/bin/env python3
"""Build ResNet18 with only the historically failing H3 route made composable."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import vta

import build_vta_c3_resnet18_fullgraph_programs as original
from build_vta_resnet50_relay_residency_dispatch_pair import (
    make_relay_testing_resnet_program,
    sha256,
    write_json,
)
from run_vta_resnet50_relay_residency_dispatch_pair_board import semantic_params_hash


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_PARENT = GROUPED / "20260914_p7r515_resnet18_selected_fullgraph_build_run01"
DEFAULT_LOCAL_GATE = GROUPED / "20260924_c3_resnet18_h3_composable_local_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_resnet18_h3_composable_build_run01"
PARENT_PROGRAM = "selected_918f902567c608ec"
VARIANT = "resnet18_h3_composable"
EXPECTED_FUNCTION = "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_7_compute_"
OBJDUMP = Path(
    "/home/orange/alinx_plsdk/install/sysroots/x86_64-petalinux-linux/usr/bin/"
    "aarch64-xilinx-linux/aarch64-xilinx-linux-objdump"
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scope_audit(library):
    text = subprocess.run([str(OBJDUMP), "-dr", str(library)], check=True,
                          capture_output=True, text=True).stdout
    current, counts = None, {}
    header = re.compile(r"^[0-9a-f]+ <([^>]+)>:$")
    for raw in text.splitlines():
        match = header.match(raw.strip())
        if match:
            current = match.group(1)
        elif current and re.search(r"\bbl\s+[0-9a-f]+ <VTASetDebugMode@plt>", raw):
            counts[current] = counts.get(current, 0) + 1
    if counts != {EXPECTED_FUNCTION: 2}:
        raise RuntimeError("unexpected ResNet18 debug scope {}".format(counts))
    return {"status": "passed", "calls_by_function": counts,
            "global_vta_debug_flag": 0}


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    parent_root = Path(args.parent_build).resolve()
    original.verify(parent_root)
    parent_program = read(parent_root / PARENT_PROGRAM / "program.json")
    gate_root = Path(args.local_gate).resolve()
    gate = read(gate_root / "summary.json")
    if gate["status"] != "LOCAL_COMPOSABLE_H3_GATE_PASSED":
        raise ValueError("ResNet18 H3 local gate did not pass")
    candidate = read(gate_root / "candidate.json")
    routes = json.loads(json.dumps(parent_program["routes"]))
    changed = 0
    for route in routes:
        if route["workload_id"] != "R18-H3":
            continue
        if route["candidate_id"] != candidate["parent_mode5_candidate_id"]:
            raise ValueError("parent program does not contain exact failed H3 route")
        route["parent_mode5_candidate_id"] = route["candidate_id"]
        route["candidate_id"] = candidate["candidate_id"]
        route["public_mode"] = "input_weight_resident_composable"
        route["implementation_mode"] = 10
        route["identity"]["public_mode"] = "input_weight_resident_composable"
        route["identity"]["implementation_mode"] = 10
        changed += 1
    if changed != 1:
        raise ValueError("expected exactly one H3 route replacement")
    write_json(output / "contract.json", {
        "schema": "c3_resnet18_h3_composable_build_v1",
        "status": "frozen_before_build",
        "parent_program": PARENT_PROGRAM,
        "parent_graphlib_sha256": parent_program["graphlib_sha256"],
        "local_gate_summary_sha256": sha256(gate_root / "summary.json"),
        "routes": routes,
        "global_vta_debug_flag": 0,
        "board_contacted": False,
        "timing_collected": False,
    })
    env = vta.get_env()
    relay_program, params = make_relay_testing_resnet_program(env, 18, device_annot=True)
    local = output / VARIANT
    built = original.build_routes(routes, relay_program, params, local, env)
    reference_graph = parent_root / "stock_reference/graph.json"
    if read(local / "graph.json") != read(reference_graph):
        raise RuntimeError("composable ResNet18 graph JSON differs from stock")
    if semantic_params_hash(local / "params.bin") != semantic_params_hash(
        parent_root / "stock_reference/params.bin"
    ):
        raise RuntimeError("composable ResNet18 semantic params differ from stock")
    scope = scope_audit(local / "graphlib.so")
    summary = {
        "schema": "c3_resnet18_h3_composable_build_v1",
        "status": "RESNET18_H3_COMPOSABLE_BUILT_AND_SCOPED",
        "variant": VARIANT, "routes": routes, "build": built,
        "debug_scope_audit": scope,
        "reference_stock_dir": str((parent_root / "stock_reference").resolve()),
        "board_contacted": False, "timing_collected": False,
    }
    write_json(output / "summary.json", summary)
    original.finish(output)
    print(json.dumps({"status": summary["status"], "scope": scope},
                     indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-build", default=str(DEFAULT_PARENT))
    parser.add_argument("--local-gate", default=str(DEFAULT_LOCAL_GATE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
