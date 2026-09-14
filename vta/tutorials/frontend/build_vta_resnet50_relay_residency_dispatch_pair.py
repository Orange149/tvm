#!/usr/bin/env python3
"""Cross-build ResNet50 twice while routing one exact VTA workload to two residency modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import time

import mxnet as mx
from mxnet.gluon.model_zoo import vision
import tvm
from tvm import autotvm, relay
from tvm.contrib import cc
import vta
from vta.top import graph_pack
from vta.top.residency_dispatch import ExplicitResidencyDispatch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_candidate(path, candidate_id):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["candidate_id"] == candidate_id:
            return row
    raise KeyError(candidate_id)


def make_relay_program(env, pretrained):
    mx.random.seed(20260912)
    model = vision.get_model("resnet50_v2", pretrained=pretrained)
    if not pretrained:
        model.initialize(mx.init.Uniform(0.05), ctx=mx.cpu())
        # Gluon defers several BatchNorm/dense shapes until the first forward pass.
        model(mx.nd.zeros((env.BATCH, 3, 224, 224), ctx=mx.cpu()))
    mod, params = relay.frontend.from_mxnet(model, {"data": (env.BATCH, 3, 224, 224)})
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params=params)
    packed = graph_pack(
        mod["main"], env.BATCH, env.BLOCK_OUT, env.WGT_WIDTH,
        start_name="nn.max_pool2d", stop_name="nn.global_avg_pool2d",
        device_annot=True,
    )
    return packed, params


def make_relay_testing_resnet50_program(env):
    """Build the graph from the same Relay implementation used for layer derivation."""
    from tvm.relay.testing import resnet

    mod, params = resnet.get_workload(
        num_layers=50,
        batch_size=env.BATCH,
        image_shape=(3, 224, 224),
        dtype="float32",
    )
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params=params)
    packed = graph_pack(
        mod["main"], env.BATCH, env.BLOCK_OUT, env.WGT_WIDTH,
        start_name="nn.max_pool2d", stop_name="nn.global_avg_pool2d",
        device_annot=True,
    )
    return packed, params


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
    build_target = {
        "cpu": tvm.target.Target(env.target_vta_cpu, host=env.target_host),
        "ext_dev": target,
    }
    relay.backend.te_compiler.get().clear()
    started = time.perf_counter()
    with autotvm.tophub.context(target):
        with ExplicitResidencyDispatch([row]) as dispatch:
            with vta.build_config(
                opt_level=3,
                disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
            ):
                graph, lib, lowered_params = relay.build(
                    relay_program, target=build_target, params=params
                )
    build_s = time.perf_counter() - started
    audit = dispatch.summary()
    if not audit["all_routes_scheduled"]:
        raise RuntimeError("exact R50B residency route was not scheduled in ResNet50")
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
    output.mkdir(parents=True)
    rows = [
        load_candidate(args.candidates, args.original_id),
        load_candidate(args.candidates, args.residency_id),
    ]
    if [row["public_mode"] for row in rows] != ["original", "weight_resident_barrier"]:
        raise ValueError("expected original/weight_resident_barrier pair")
    if rows[0]["identity"]["workload"] != rows[1]["identity"]["workload"]:
        raise ValueError("workload mismatch")
    if rows[0]["identity"]["complete_config_entity"] != rows[1]["identity"]["complete_config_entity"]:
        raise ValueError("ConfigEntity mismatch")
    env = vta.get_env()
    relay_program, params = make_relay_program(env, args.pretrained)
    builds = [build_one(row, relay_program, params, output, env) for row in rows]
    summary = {
        "schema": "c3_vta_resnet50_exact_residency_dispatch_build_v1",
        "status": "resnet50_pair_cross_build_dispatch_verified",
        "model": "resnet50_v2",
        "input_shape": [env.BATCH, 3, 224, 224],
        "pretrained": bool(args.pretrained),
        "parameter_seed": None if args.pretrained else 20260912,
        "same_workload_and_config": True,
        "builds": builds,
        "board_contacted": False,
        "claim_boundary": (
            "Whole-model compile integration with one exact workload route; board output and "
            "end-to-end latency require a separate clean-start run"
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--original-id", required=True)
    parser.add_argument("--residency-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrained", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
