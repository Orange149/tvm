#!/usr/bin/env python3
"""Cross-build YOLOv3-tiny with one versus two certified residency routes."""

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
    load_candidate,
    make_relay_program,
    sha256,
    write_json,
)


VARIANT_Y00 = "y00_input_only"
VARIANT_COMBINED = "y00_input_y02_weight"


def build_variant(name, routes, relay_program, params, output, env):
    local = output / name
    local.mkdir()
    target = tvm.target.Target(env.target, host=env.target_host)
    relay.backend.te_compiler.get().clear()
    started = time.perf_counter()
    with autotvm.tophub.context(target):
        with ExplicitResidencyDispatch(routes) as dispatch:
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
        raise RuntimeError("not every exact residency route was scheduled for " + name)
    (local / "graph.json").write_text(graph, encoding="utf-8")
    (local / "params.bin").write_bytes(relay.save_param_dict(lowered_params))
    lib.export_library(str(local / "graphlib.so"), cross_compiler())
    result = {
        "deployment_variant": name,
        "candidate_ids": [row["candidate_id"] for row in routes],
        "routes": [
            {
                "candidate_id": row["candidate_id"],
                "workload_id": row.get("workload_id"),
                "public_mode": row["public_mode"],
                "implementation_mode": row["implementation_mode"],
            }
            for row in routes
        ],
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

    y00 = load_candidate(args.y00_contract, args.y00_input_id)
    y02 = load_candidate(args.y02_contract, args.y02_weight_id)
    if y00["public_mode"] != "input_stationary" or int(y00["implementation_mode"]) != 1:
        raise ValueError("Y00 route must be input_stationary/mode1")
    if y02["public_mode"] != "weight_resident_barrier" or int(y02["implementation_mode"]) != 4:
        raise ValueError("Y02 route must be weight_resident_barrier/mode4")
    if y00["identity"]["workload"] == y02["identity"]["workload"]:
        raise ValueError("multi-route experiment requires two distinct workloads")

    output.mkdir(parents=True)
    env = vta.get_env()
    relay_program, params, shape = make_relay_program(env, args)
    builds = [
        build_variant(VARIANT_Y00, [y00], relay_program, params, output, env),
        build_variant(VARIANT_COMBINED, [y00, y02], relay_program, params, output, env),
    ]
    if builds[0]["graph_sha256"] != builds[1]["graph_sha256"]:
        raise RuntimeError("deployment variants produced different graph JSON")
    summary = {
        "schema": "c3_vta_yolov3_tiny_multi_residency_build_v1",
        "status": "yolov3_tiny_one_vs_two_residency_routes_cross_build_verified",
        "model": "yolov3-tiny",
        "input_shape": list(shape),
        "variants": builds,
        "incremental_route": {
            "candidate_id": y02["candidate_id"],
            "workload_id": y02.get("workload_id"),
            "public_mode": y02["public_mode"],
        },
        "assets": {
            "cfg": {"path": str(Path(args.cfg).resolve()), "sha256": sha256(args.cfg)},
            "weights": {"path": str(Path(args.weights).resolve()), "sha256": sha256(args.weights)},
            "darknet_lib": {
                "path": str(Path(args.darknet_lib).resolve()),
                "sha256": sha256(args.darknet_lib),
            },
        },
        "board_contacted": False,
        "claim_boundary": (
            "Same YOLO graph and parameters; the second variant adds only the exact "
            "Y02 weight-resident route on top of the already-qualified Y00 input route"
        ),
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
        "variants": [
            {
                "deployment_variant": row["deployment_variant"],
                "candidate_ids": row["candidate_ids"],
                "build_seconds": row["build_seconds"],
                "config_hits": len(row["dispatch_audit"]["config_hits"]),
                "schedule_hits": len(row["dispatch_audit"]["schedule_hits"]),
                "graph_sha256": row["graph_sha256"],
                "graphlib_sha256": row["graphlib_sha256"],
            }
            for row in builds
        ],
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--y00-contract", required=True)
    parser.add_argument("--y00-input-id", required=True)
    parser.add_argument("--y02-contract", required=True)
    parser.add_argument("--y02-weight-id", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--darknet-lib", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
