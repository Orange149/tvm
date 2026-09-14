#!/usr/bin/env python3
"""Cross-build YOLOv3-tiny twice with one certified exact residency route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import time

import tvm
from tvm import autotvm, relay
from tvm.contrib import cc
from tvm.relay.testing.darknet import __darknetffi__
import vta
from vta.top import graph_pack
from vta.top.residency_dispatch import ExplicitResidencyDispatch


PACK_SPEC = ("nn.max_pool2d", "cast", 4, 186)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_candidate(contract_path, candidate_id):
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    for row in contract["candidates"]:
        if row["candidate_id"] == candidate_id:
            return row
    raise KeyError(candidate_id)


def make_relay_program(env, args):
    net = __darknetffi__.dlopen(args.darknet_lib).load_network(
        args.cfg.encode("utf-8"), args.weights.encode("utf-8"), 0
    )
    shape = (env.BATCH, net.c, net.h, net.w)
    mod, params = relay.frontend.from_darknet(net, dtype="float32", shape=shape)
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(
            global_scale=23.0,
            skip_conv_layers=[0],
            store_lowbit_output=True,
            round_for_shift=True,
        ):
            mod = relay.quantize.quantize(mod, params=params)
        packed = graph_pack(
            mod["main"], env.BATCH, env.BLOCK_OUT, env.WGT_WIDTH,
            start_name=PACK_SPEC[0], stop_name=PACK_SPEC[1],
            start_name_idx=PACK_SPEC[2], stop_name_idx=PACK_SPEC[3],
        )
    return packed, params, shape


def cross_compiler():
    tokens = shlex.split(os.environ.get("CXX", "aarch64-xilinx-linux-g++"))
    if not tokens:
        raise RuntimeError("CXX resolved to an empty command")
    return cc.cross_compiler(
        tokens[0], options=tokens[1:] + shlex.split(os.environ.get("LDFLAGS", ""))
    )


def build_one(row, relay_program, params, output, env):
    mode = row["public_mode"]
    local = output / mode
    local.mkdir()
    target = tvm.target.Target(env.target, host=env.target_host)
    relay.backend.te_compiler.get().clear()
    started = time.perf_counter()
    with autotvm.tophub.context(target):
        with ExplicitResidencyDispatch([row]) as dispatch:
            with vta.build_config(
                opt_level=3,
                disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
            ):
                graph, lib, lowered_params = relay.build(
                    relay_program, target=target, params=params
                )
    build_s = time.perf_counter() - started
    audit = dispatch.summary()
    if not audit["all_routes_scheduled"]:
        raise RuntimeError("exact YOLO residency route was not scheduled")
    (local / "graph.json").write_text(graph, encoding="utf-8")
    (local / "params.bin").write_bytes(relay.save_param_dict(lowered_params))
    lib.export_library(str(local / "graphlib.so"), cross_compiler())
    result = {
        "candidate_id": row["candidate_id"],
        "public_mode": mode,
        "implementation_mode": row["implementation_mode"],
        "build_seconds": build_s,
        "dispatch_audit": audit,
        "graph_sha256": sha256(local / "graph.json"),
        "params_sha256": sha256(local / "params.bin"),
        "graphlib_sha256": sha256(local / "graphlib.so"),
    }
    write_json(local / "build.json", result)
    return result


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    for asset in (args.cfg, args.weights, args.darknet_lib):
        if not Path(asset).is_file():
            raise FileNotFoundError(asset)
    rows = [
        load_candidate(args.contract, args.original_id),
        load_candidate(args.contract, args.residency_id),
    ]
    if [row["public_mode"] for row in rows] != ["original", "input_stationary"]:
        raise ValueError("expected original/input_stationary pair")
    if rows[0]["identity"]["workload"] != rows[1]["identity"]["workload"]:
        raise ValueError("workload mismatch")
    if rows[0]["identity"]["complete_config_entity"] != rows[1]["identity"]["complete_config_entity"]:
        raise ValueError("ConfigEntity mismatch")
    output.mkdir(parents=True)
    env = vta.get_env()
    relay_program, params, shape = make_relay_program(env, args)
    builds = [build_one(row, relay_program, params, output, env) for row in rows]
    summary = {
        "schema": "c3_vta_yolov3_tiny_exact_residency_dispatch_build_v1",
        "status": "yolov3_tiny_pair_cross_build_dispatch_verified",
        "model": "yolov3-tiny",
        "input_shape": list(shape),
        "same_workload_and_config": True,
        "pack_spec": list(PACK_SPEC),
        "assets": {
            "cfg": {"path": str(Path(args.cfg).resolve()), "sha256": sha256(args.cfg)},
            "weights": {"path": str(Path(args.weights).resolve()), "sha256": sha256(args.weights)},
            "darknet_lib": {
                "path": str(Path(args.darknet_lib).resolve()), "sha256": sha256(args.darknet_lib)
            },
        },
        "builds": builds,
        "board_contacted": False,
        "claim_boundary": "Whole YOLO graph compile only; board output and timing require a clean-start run",
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "status": summary["status"],
        "input_shape": summary["input_shape"],
        "builds": [{
            "candidate_id": row["candidate_id"], "public_mode": row["public_mode"],
            "build_seconds": row["build_seconds"],
            "schedule_hits": len(row["dispatch_audit"]["schedule_hits"]),
            "graph_sha256": row["graph_sha256"], "graphlib_sha256": row["graphlib_sha256"],
        } for row in builds],
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--original-id", required=True)
    parser.add_argument("--residency-id", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--darknet-lib", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
