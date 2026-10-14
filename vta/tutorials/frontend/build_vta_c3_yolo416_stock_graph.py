#!/usr/bin/env python3
"""Build a sealed stock YOLOv3-tiny-416 graph for real-image latency tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tvm
from tvm import autotvm, relay
import vta

from build_vta_c3_evaluation_v5_graphs import save_build
from build_vta_yolov3_tiny_relay_residency_dispatch_pair import (
    cross_compiler,
    make_relay_program,
    sha256,
    write_json,
)
from run_vta_yolov3_tiny_relay_residency_dispatch_pair_board import semantic_params_hash


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_PREPARE = GROUPED / "20260924_c3_yolo_unseen_resolution_prepare_run05"


def build_stock(relay_program, params, output, env, compiler):
    local = output / "stock"
    local.mkdir()
    target = tvm.target.Target(env.target, host=env.target_host)
    relay.backend.te_compiler.get().clear()
    with autotvm.tophub.context(target):
        with vta.build_config(
            debug_flag=0,
            opt_level=3,
            disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
        ):
            graph, lib, lowered_params = relay.build(relay_program, target=target, params=params)
    return save_build(
        local,
        graph,
        lib,
        lowered_params,
        compiler,
        {
            "deployment_variant": "stock",
            "routes": [],
            "dispatch_audit": None,
            "vta_debug_flag": 0,
        },
    )


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    assets = [Path(args.cfg), Path(args.weights), Path(args.darknet_lib)]
    for asset in assets:
        if not asset.is_file():
            raise FileNotFoundError(asset)
    if sha256(args.weights) != "dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c":
        raise ValueError("YOLO weights do not match the frozen model")

    output.mkdir(parents=True)
    env = vta.get_env()
    relay_program, params, shape = make_relay_program(env, args)
    if tuple(shape) != (1, 3, 416, 416):
        raise ValueError("derived graph is not YOLO-416")
    build = build_stock(relay_program, params, output, env, cross_compiler())
    if not build["memory_contract"]["fits"]:
        raise RuntimeError("YOLO-416 stock graph exceeds the 192 MiB contract")
    semantic_hash, inventory = semantic_params_hash(output / "stock/params.bin")
    summary = {
        "schema": "c3_yolo416_stock_graph_build_v1",
        "status": "C3_YOLO416_STOCK_GRAPH_BUILT",
        "input_shape": list(shape),
        "variant": build,
        "parameter_semantic_sha256": semantic_hash,
        "parameter_inventory": inventory,
        "assets": {
            "cfg": {"path": str(assets[0].resolve()), "sha256": sha256(assets[0])},
            "weights": {"path": str(assets[1].resolve()), "sha256": sha256(assets[1])},
            "darknet_lib": {"path": str(assets[2].resolve()), "sha256": sha256(assets[2])},
        },
        "board_contacted": False,
        "claim_boundary": "stock YOLO-416 graph build; board correctness and latency pending",
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "sources": {str(Path(__file__).resolve()): sha256(__file__)},
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cfg",
        default=str(DEFAULT_PREPARE / "yolov3-tiny.base.cfg"),
    )
    parser.add_argument("--weights", default="/tmp/tvm_test_data/darknet/yolov3-tiny.weights")
    parser.add_argument("--darknet-lib", default="/home/orange/.tvm_test_data/darknet/libdarknet2.0.so")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
