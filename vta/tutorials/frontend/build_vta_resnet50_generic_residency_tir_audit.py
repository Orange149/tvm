#!/usr/bin/env python3
"""Capture full fused VTA TIR for an exact ResNet50 residency pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import time

import tvm
from tvm import autotvm, relay, tir
from tvm.contrib import cc
import vta
from vta.top.residency_dispatch import ExplicitResidencyDispatch

from build_vta_resnet50_relay_residency_dispatch_pair import (
    load_candidate,
    make_relay_program,
    sha256,
    write_json,
)
from extract_static_vta_dma import extract_module_dma_compact


@tvm.instrument.pass_instrument
class FullTIRArchive:
    def __init__(self, env):
        self.env = env
        self.snapshots = []

    def run_after_pass(self, mod, info):
        if info.name != "tir.vta.CPUAccessRewrite":
            return
        functions = []
        for global_var, function in mod.functions.items():
            if not isinstance(function, tir.PrimFunc):
                continue
            symbol = None
            if function.attrs is not None and "global_symbol" in function.attrs:
                symbol = str(function.attrs["global_symbol"])
            single = tvm.IRModule({"main": function})
            dma = extract_module_dma_compact(single, self.env)
            functions.append({
                "global_var": str(global_var.name_hint),
                "global_symbol": symbol,
                "dma_totals": dma["totals"],
                "max_request_bytes_by_memory": dma["max_request_bytes_by_memory"],
            })
        text = mod.script()
        self.snapshots.append({
            "pass": str(info.name),
            "functions": functions,
            "json": tvm.ir.save_json(mod),
            "tir": text,
            "coproc_sync_occurrences": text.count("coproc_sync"),
        })


def cross_compiler():
    tokens = shlex.split(os.environ.get("CXX", "aarch64-xilinx-linux-g++"))
    return cc.cross_compiler(
        tokens[0], options=tokens[1:] + shlex.split(os.environ.get("LDFLAGS", ""))
    )


def build_one(row, relay_program, params, output, env, target_mode="heterogeneous"):
    mode = row["public_mode"]
    local = output / mode
    local.mkdir()
    archive = FullTIRArchive(env)
    target = tvm.target.Target(env.target, host=env.target_host)
    if target_mode == "heterogeneous":
        build_target = {
            "cpu": tvm.target.Target(env.target_vta_cpu, host=env.target_host),
            "ext_dev": target,
        }
    elif target_mode == "ext_dev_only":
        build_target = target
    else:
        raise ValueError("unsupported build target mode: " + repr(target_mode))
    relay.backend.te_compiler.get().clear()
    started = time.perf_counter()
    with autotvm.tophub.context(target):
        with ExplicitResidencyDispatch([row]) as dispatch:
            with vta.build_config(
                opt_level=3,
                disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
                instruments=[archive],
            ):
                graph, lib, lowered_params = relay.build(
                    relay_program, target=build_target, params=params
                )
    audit = dispatch.summary()
    if not audit["all_routes_scheduled"] or not archive.snapshots:
        raise RuntimeError("exact route or fused TIR archive is incomplete")
    (local / "graph.json").write_text(graph, encoding="utf-8")
    (local / "params.bin").write_bytes(relay.save_param_dict(lowered_params))
    lib.export_library(str(local / "graphlib.so"), cross_compiler())
    snapshots = []
    for index, item in enumerate(archive.snapshots):
        stem = "cpu_access_rewrite_{:03d}".format(index)
        (local / (stem + ".json")).write_text(item.pop("json"), encoding="utf-8")
        (local / (stem + ".tir")).write_text(item.pop("tir"), encoding="utf-8")
        snapshots.append({
            **item,
            "json_file": stem + ".json",
            "json_sha256": sha256(local / (stem + ".json")),
            "tir_file": stem + ".tir",
            "tir_sha256": sha256(local / (stem + ".tir")),
        })
    result = {
        "candidate_id": row["candidate_id"],
        "public_mode": mode,
        "implementation_mode": row["implementation_mode"],
        "build_seconds": time.perf_counter() - started,
        "dispatch_audit": audit,
        "snapshots": snapshots,
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
    output.mkdir(parents=True)
    rows = [
        load_candidate(args.candidates, args.original_id),
        load_candidate(args.candidates, args.residency_id),
    ]
    if [row["public_mode"] for row in rows] != ["original", args.residency_mode]:
        raise RuntimeError("unexpected candidate modes")
    if rows[0]["identity"]["workload"] != rows[1]["identity"]["workload"]:
        raise RuntimeError("workload mismatch")
    if (rows[0]["identity"]["complete_config_entity"]
            != rows[1]["identity"]["complete_config_entity"]):
        raise RuntimeError("ConfigEntity mismatch")
    env = vta.get_env()
    relay_program, params = make_relay_program(env, pretrained=True)
    builds = [build_one(row, relay_program, params, output, env) for row in rows]
    summary = {
        "schema": "c3_vta_resnet50_generic_residency_fused_tir_audit_v1",
        "status": "full_graph_fused_tir_pair_captured",
        "model": "resnet50_v2",
        "pretrained": True,
        "public_modes": [row["public_mode"] for row in rows],
        "builder_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_source_sha256": sha256(args.candidates),
        "builds": builds,
        "board_contacted": False,
        "claim_boundary": (
            "Compiler-only CPUAccessRewrite snapshots and logical DMA descriptors; "
            "not runtime latency, physical AXI, or a board result"
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
    print(json.dumps({
        "status": summary["status"],
        "modes": summary["public_modes"],
        "snapshots": {row["public_mode"]: len(row["snapshots"]) for row in builds},
        "functions": {
            row["public_mode"]: [
                function
                for snapshot in row["snapshots"]
                for function in snapshot["functions"]
            ]
            for row in builds
        },
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--original-id", required=True)
    parser.add_argument("--residency-id", required=True)
    parser.add_argument("--residency-mode", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
