#!/usr/bin/env python3
"""Minimal stage0_cpu -> stage1_vta -> stage2_cpu ResNet18 experiment.

This script is intentionally small and educational. It reuses the existing
split/build helpers, but keeps the control flow explicit so it is easy to map:

1. Graph-level split: ResNet18 -> three coarse stages
2. Relay graph-level optimization / quantization / pack
3. relay.build lowering to TIR / runtime modules
4. graph_executor creation
5. Sequential staged execution
"""

from __future__ import absolute_import, print_function

import argparse
import os
import sys

import numpy as np
import tvm
from tvm import rpc
from tvm.contrib import graph_executor

import vta

from mxnet.gluon.model_zoo import vision

# Allow direct execution from the grouped hp_hpc_quant directory.
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)

from split_resnet18_stages import (
    SCHEMES,
    build_resnet18_unit_blocks,
    build_resnet18_unit_metadata,
    lower_stage_to_relay,
    make_stage_block,
    relay_inputs_for_stage,
    stage_input_schema_for_stage,
    stage_output_schema_for_stage,
)
from profile_split_resnet18_stages import (
    build_cpu_stage,
    build_vta_stage,
    connect_remote,
    create_stage_module,
    export_and_upload,
    fetch_stage_output,
    graph_summary,
    preprocess_image,
    set_stage_inputs,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="three_stage_e", choices=sorted(SCHEMES.keys()))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--host", default=os.environ.get("VTA_RPC_HOST", "192.168.1.228"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VTA_RPC_PORT", "9090")))
    parser.add_argument(
        "--print-relay",
        action="store_true",
        help="Print Relay text before build for each stage",
    )
    parser.add_argument(
        "--print-packed-relay",
        action="store_true",
        help="Print packed/annotated Relay for stage1_vta",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Upload to remote and run one serial staged inference",
    )
    return parser.parse_args()


def summarize_stage(stage, input_schema, output_schema, relay_inputs):
    print("\n========== {} ==========".format(stage["name"]))
    print("[STAGE] device       =", stage["device"])
    print("[STAGE] units        =", stage["unit_names"])
    print("[STAGE] relay_inputs =", relay_inputs)
    print("[STAGE] input_schema =", input_schema)
    print("[STAGE] output_schema=", output_schema)


def build_stage(stage, feature_blocks, output_block, unit_blocks, unit_metadata, env, args):
    input_schema = stage_input_schema_for_stage(stage, unit_metadata)
    output_schema = stage_output_schema_for_stage(stage, unit_metadata)
    relay_inputs = relay_inputs_for_stage(stage, unit_metadata)
    stage_block = make_stage_block(feature_blocks, output_block, stage, unit_blocks=unit_blocks)
    mod, params = lower_stage_to_relay(stage_block, relay_inputs)

    summarize_stage(stage, input_schema, output_schema, relay_inputs)
    if args.print_relay:
        print(mod["main"].astext(show_meta_data=False))

    if stage["device"] == "cpu":
        cpu_target = tvm.target.Target(env.target_vta_cpu, host=env.target_host)
        graph, lib, lowered_params = build_cpu_stage(stage["name"], mod, params, cpu_target)
        packed_mod = None
    else:
        # build_vta_stage internally does:
        # Relay quantize -> explicit pack/unpack wrapping -> ExprPack ->
        # device annotation -> relay.build with {"cpu", "ext_dev"} targets
        if args.print_packed_relay:
            from profile_split_resnet18_stages import (
                annotate_all_ops_to_ext_dev,
                wrap_stage_with_explicit_pack,
            )
            from vta.top import graphpack as vta_graphpack
            from tvm.relay import transform

            with tvm.transform.PassContext(opt_level=3):
                with tvm.relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[]):
                    qmod = tvm.relay.quantize.quantize(tvm.IRModule.from_expr(mod["main"]), params=params)
            packed_mod = wrap_stage_with_explicit_pack(
                qmod["main"],
                input_dev=tvm.device("cpu"),
                body_dev=tvm.device("cpu"),
            )
            packer = vta_graphpack.ExprPack(env.BATCH, env.BLOCK_OUT, env.WGT_WIDTH)
            packed_mod = packer.visit(packed_mod)
            packed_mod = vta_graphpack.run_opt_pass(packed_mod, transform.InferType())
            packed_mod = annotate_all_ops_to_ext_dev(packed_mod)
            print("\n[PACKED RELAY] {}".format(stage["name"]))
            print(packed_mod.astext(show_meta_data=False))
        else:
            packed_mod = None

        graph, lib, lowered_params = build_vta_stage(
            stage["name"],
            mod["main"],
            params,
            env,
            use_graph_pack=False,
        )

    return {
        "name": stage["name"],
        "device": stage["device"],
        "graph": graph,
        "lib": lib,
        "lowered_params": lowered_params,
        "input_names": [name for name, _ in relay_inputs],
        "input_schema": input_schema,
        "output_schema": output_schema,
        "relay_mod": mod,
        "packed_mod": packed_mod,
        "graph_info": graph_summary(graph),
    }


def build_all_stages(args):
    env = vta.get_env()
    full_model = vision.get_model("resnet18_v1", pretrained=True)
    feature_blocks = list(full_model.features)
    output_block = full_model.output
    unit_blocks = build_resnet18_unit_blocks(feature_blocks, output_block)
    unit_metadata = build_resnet18_unit_metadata(unit_blocks, args.batch, args.image_size)
    scheme_cfg = SCHEMES[args.scheme]

    print("[INFO] scheme =", args.scheme)
    print("[INFO] image  =", (args.batch, 3, args.image_size, args.image_size))

    built = []
    for stage in scheme_cfg:
        built.append(
            build_stage(stage, feature_blocks, output_block, unit_blocks, unit_metadata, env, args)
        )
    return env, built


def run_serial_once(env, built, args):
    remote = connect_remote(env, args.host, args.port)
    modules = []
    for stage in built:
        remote_lib = export_and_upload(stage["lib"], remote, env, stage["name"])
        mod, ctx = create_stage_module(stage["name"], stage["device"], stage["graph"], remote_lib, remote)
        mod.set_input(**stage["lowered_params"])
        modules.append({"stage": stage, "module": mod, "ctx": ctx})

    current = preprocess_image(args.batch, args.image_size)
    print("\n========== SERIAL RUN ==========")
    print("[RUN] initial input =", current.shape, current.dtype)

    for idx, item in enumerate(modules):
        stage = item["stage"]
        mod = item["module"]
        next_stage = modules[idx + 1]["stage"] if idx + 1 < len(modules) else None
        next_stage_device = next_stage["device"] if next_stage is not None else "cpu"

        print("\n[RUN] {} input_location={}".format(stage["name"], type(current).__name__))
        set_stage_inputs(mod, stage["input_names"], current)
        mod.run()
        current = fetch_stage_output(
            mod,
            {"slots": stage["output_schema"]["slots"]},
            remote,
            stage["device"],
            next_stage_device,
        )
        if isinstance(current, tuple):
            desc = [tuple(x.shape) for x in current]
        else:
            desc = tuple(current.shape)
        print("[RUN] {} output_shape={}".format(stage["name"], desc))

    print("\n[RUN] done")


def main():
    args = parse_args()
    env, built = build_all_stages(args)
    if args.run:
        run_serial_once(env, built, args)


if __name__ == "__main__":
    main()
