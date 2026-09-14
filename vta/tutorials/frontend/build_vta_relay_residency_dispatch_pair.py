#!/usr/bin/env python3
"""Build a same-tile Relay VTA pair through explicit residency dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex

import tvm
from tvm import autotvm, relay
from tvm.contrib import cc
import vta
from vta.top.residency_dispatch import ExplicitResidencyDispatch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


@tvm.instrument.pass_instrument
class TIRArchive:
    def __init__(self):
        self.snapshots = []

    def run_after_pass(self, mod, info):
        if info.name != "tir.vta.CPUAccessRewrite":
            return
        text = mod.script()
        self.snapshots.append(
            {
                "pass": str(info.name),
                "json": tvm.ir.save_json(mod),
                "tir": text,
                "coproc_sync_occurrences": text.count("coproc_sync"),
            }
        )


def relay_module(workload):
    data_desc, weight_desc = workload[1], workload[2]
    strides, padding, dilation, layout, out_dtype = workload[3:]
    data = relay.var("data", shape=tuple(data_desc[1]), dtype=data_desc[2])
    weight = relay.var("weight", shape=tuple(weight_desc[1]), dtype=weight_desc[2])
    channels = int(weight_desc[1][0] * weight_desc[1][4])
    kernel_size = tuple(weight_desc[1][2:4])
    conv = relay.nn.conv2d(
        data,
        weight,
        strides=tuple(strides),
        padding=tuple(padding[:2]),
        dilation=tuple(dilation),
        groups=1,
        channels=channels,
        kernel_size=kernel_size,
        data_layout=layout,
        kernel_layout="OIHW16o16i",
        out_layout=layout,
        out_dtype=out_dtype,
    )
    shifted = relay.right_shift(conv, relay.const(8, "int32"))
    output = relay.cast(relay.clip(shifted, a_min=0, a_max=127), "int8")
    return tvm.IRModule.from_expr(relay.Function([data, weight], output))


def build_one(row, output, env):
    mode = row["public_mode"]
    local = output / mode
    local.mkdir()
    archive = TIRArchive()
    target = tvm.target.Target(env.target, host=env.target_host)
    relay.backend.te_compiler.get().clear()
    with autotvm.tophub.context(target):
        with ExplicitResidencyDispatch([row]) as dispatch:
            with vta.build_config(
                opt_level=3,
                disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
                instruments=[archive],
            ):
                graph, lib, params = relay.build(
                    relay_module(row["identity"]["workload"]), target=target
                )
    if not archive.snapshots:
        raise RuntimeError("no VTA TIR snapshot captured")
    audit = dispatch.summary()
    if not audit["all_routes_scheduled"]:
        raise RuntimeError("explicit residency route was queried but not scheduled")
    (local / "graph.json").write_text(graph, encoding="utf-8")
    (local / "params.bin").write_bytes(relay.save_param_dict(params))
    cxx = shlex.split(os.environ.get("CXX", "aarch64-xilinx-linux-g++"))
    if not cxx:
        raise RuntimeError("CXX resolved to an empty command")
    link_options = cxx[1:] + shlex.split(os.environ.get("LDFLAGS", ""))
    lib.export_library(
        str(local / "graphlib.so"), cc.cross_compiler(cxx[0], options=link_options)
    )
    snapshots = []
    for index, item in enumerate(archive.snapshots):
        stem = "lowered_{:03d}".format(index)
        (local / (stem + ".json")).write_text(item["json"], encoding="utf-8")
        (local / (stem + ".tir")).write_text(item["tir"], encoding="utf-8")
        snapshots.append(
            {
                "json": stem + ".json",
                "json_sha256": sha256(local / (stem + ".json")),
                "tir": stem + ".tir",
                "tir_sha256": sha256(local / (stem + ".tir")),
                "coproc_sync_occurrences": item["coproc_sync_occurrences"],
            }
        )
    result = {
        "candidate_id": row["candidate_id"],
        "public_mode": mode,
        "implementation_mode": row["implementation_mode"],
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
    rows = {row["candidate_id"]: row for row in load_rows(args.candidates)}
    selected = [rows[args.original_id], rows[args.residency_id]]
    if [row["public_mode"] for row in selected] != ["original", args.expected_mode]:
        raise ValueError("expected original and {} pair".format(args.expected_mode))
    if selected[0]["identity"]["workload"] != selected[1]["identity"]["workload"]:
        raise ValueError("pair workload mismatch")
    if selected[0]["identity"]["complete_config_entity"] != selected[1]["identity"][
        "complete_config_entity"
    ]:
        raise ValueError("pair ConfigEntity mismatch")
    output.mkdir(parents=True)
    env = vta.get_env()
    builds = [build_one(row, output, env) for row in selected]
    original_sync = sum(x["coproc_sync_occurrences"] for x in builds[0]["snapshots"])
    residency_sync = sum(x["coproc_sync_occurrences"] for x in builds[1]["snapshots"])
    summary = {
        "schema": "c3_vta_relay_residency_dispatch_pair_v1",
        "status": "relay_cross_build_dispatch_verified",
        "same_workload_and_config": True,
        "builds": builds,
        "original_coproc_sync_occurrences": original_sync,
        "residency_coproc_sync_occurrences": residency_sync,
        "residency_structure_differs": (
            builds[0]["snapshots"][0]["json_sha256"]
            != builds[1]["snapshots"][0]["json_sha256"]
        ),
        "board_contacted": False,
        "claim_boundary": (
            "Relay-level exact dispatch and cross-build only; board correctness and latency "
            "require a separate clean-start run"
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(" ".join([str(Path(__file__).resolve())] + __import__("sys").argv[1:]) + "\n")
    artifacts = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": artifacts,
            "source_sha256": {
                str(Path(__file__).resolve()): sha256(__file__),
                str(Path(vta.top.residency_dispatch.__file__).resolve()): sha256(
                    vta.top.residency_dispatch.__file__
                ),
            },
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--original-id", required=True)
    parser.add_argument("--residency-id", required=True)
    parser.add_argument(
        "--expected-mode", choices=("input_stationary", "weight_resident_barrier"), required=True
    )
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
