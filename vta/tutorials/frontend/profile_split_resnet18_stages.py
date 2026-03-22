#!/usr/bin/env python3
"""Build and profile split ResNet18 stages for CPU/VTA research.

This script focuses on the validated workflow:
1. split ResNet18 into CPU/VTA/CPU stages
2. build each stage independently
3. run stages sequentially over RPC
4. simulate an ideal stage pipeline from measured stage service times
"""

from __future__ import absolute_import, print_function

import argparse
import json
import os
import time

import numpy as np
from PIL import Image
import tvm
from tvm import autotvm, relay, rpc
from tvm.contrib import cc, download, graph_executor, utils
from tvm.relay import op, transform

import vta
from vta.top import graphpack as vta_graphpack

from split_resnet18_stages import (  # pylint: disable=import-error
    SCHEMES,
    lower_stage_to_relay,
    make_stage_block,
    relay_shape_for_stage,
    summarize_feature_blocks,
    summarize_scheme,
    validate_scheme,
)

from mxnet.gluon.model_zoo import vision


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scheme",
        default="three_stage_a",
        choices=sorted(SCHEMES.keys()),
        help="Stage split scheme to build",
    )
    parser.add_argument("--model", default="resnet18_v1", choices=["resnet18_v1"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--vta-stage-mode",
        default="cpu_fallback",
        choices=["cpu_fallback", "vta_build"],
        help=(
            "How to build VTA-designated stages. "
            "'cpu_fallback' keeps them on CPU just to validate split/build structure. "
            "'vta_build' applies quantize+graph_pack and builds for VTA."
        ),
    )
    parser.add_argument(
        "--print-relay",
        action="store_true",
        help="Print Relay text of each stage before build",
    )
    parser.add_argument(
        "--run-stages",
        action="store_true",
        help="Build, upload, and run stages sequentially over RPC",
    )
    parser.add_argument("--host", default=os.environ.get("VTA_RPC_HOST", "192.168.1.228"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VTA_RPC_PORT", "9090")))
    parser.add_argument(
        "--pipeline-batches",
        type=int,
        default=0,
        help="If > 0, simulate an ideal stage pipeline for this many batches using measured stage service times",
    )
    return parser.parse_args()


def graph_summary(graph_json_str):
    g = json.loads(graph_json_str)
    summary = {
        "num_nodes": len(g["nodes"]),
        "arg_nodes": len(g.get("arg_nodes", [])),
        "heads": g.get("heads", []),
        "device_attrs": {},
    }
    for key, value in g.get("attrs", {}).items():
        if "device" in key.lower():
            summary["device_attrs"][key] = value
    return summary


def print_graph_summary(stage_name, graph_json_str):
    summary = graph_summary(graph_json_str)
    print("[GRAPH] {}".format(stage_name))
    print("  num_nodes =", summary["num_nodes"])
    print("  arg_nodes =", summary["arg_nodes"])
    print("  heads     =", summary["heads"])
    for key, value in summary["device_attrs"].items():
        print("  attr {} = {}".format(key, value))


def build_cpu_stage(stage_name, mod, params, target):
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, lowered_params = relay.build(mod, target=target, params=params)
    print("[BUILD] {} -> CPU build ok".format(stage_name))
    print_graph_summary(stage_name, graph)
    print("[BUILD] {} params={}".format(stage_name, len(lowered_params)))
    return graph, lib, lowered_params


def get_func_output_info(relay_func):
    relay_func = vta_graphpack.run_opt_pass(relay_func, transform.InferType())
    out_ty = relay_func.ret_type
    shape = tuple(int(x) for x in out_ty.shape)
    return shape, out_ty.dtype


def wrap_stage_with_explicit_pack(relay_func):
    """Treat pack/unpack as an internal detail of the VTA stage.

    This keeps the stage interface in ordinary 4D layout while still letting
    ExprPack convert the internal stage body into packed VTA form.
    """

    bitpack_start = op.op.get("annotation.bitpack_start")
    bitpack_end = op.op.get("annotation.bitpack_end")

    data_param = relay_func.params[0]
    packed_input = relay.Call(bitpack_start, [data_param])
    wrapped_body = relay.expr.bind(relay_func.body, {data_param: packed_input})
    wrapped_body = relay.Call(bitpack_end, [wrapped_body])
    wrapped_func = relay.Function(
        relay_func.params,
        wrapped_body,
        relay_func.ret_type,
        relay_func.type_params,
        relay_func.attrs,
    )
    return vta_graphpack.run_opt_pass(wrapped_func, transform.InferType())


def build_vta_stage(stage_name, relay_prog, params, env):
    # Unlike the official end-to-end VTA tutorial, this stage starts from an
    # ordinary 4D inter-stage tensor. We therefore make pack/unpack explicit
    # inside the stage instead of relying on graph_pack to discover a single
    # contiguous packed island from the first conv output.
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(
            global_scale=8.0,
            skip_conv_layers=[],
        ):
            qmod = relay.quantize.quantize(tvm.IRModule.from_expr(relay_prog), params=params)

    packed = wrap_stage_with_explicit_pack(qmod["main"])
    packer = vta_graphpack.ExprPack(env.BATCH, env.BLOCK_OUT, env.WGT_WIDTH)
    packed = packer.visit(packed)
    packed = vta_graphpack.run_opt_pass(packed, transform.InferType())

    target = env.target
    target_with_host = tvm.target.Target(target, host=env.target_host)
    with vta.build_config(
        opt_level=3,
        disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
    ):
        graph, lib, lowered_params = relay.build(
            packed,
            target=target_with_host,
            params=params,
        )
    print("[BUILD] {} -> VTA build ok".format(stage_name))
    print_graph_summary(stage_name, graph)
    print("[BUILD] {} params={}".format(stage_name, len(lowered_params)))
    return graph, lib, lowered_params


def preprocess_image(batch, image_size):
    categ_url = "https://github.com/uwsampl/web-data/raw/main/vta/models/"
    categ_fn = "synset.txt"
    download.download(os.path.join(categ_url, categ_fn), categ_fn)

    image_url = "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg"
    image_fn = "cat.png"
    download.download(image_url, image_fn)

    image = Image.open(image_fn).resize((image_size, image_size))
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    image = np.repeat(image, batch, axis=0)
    return image.astype("float32")


def export_and_upload(lib, remote, env, stage_name):
    temp = utils.tempdir()
    lib_name = "{}.so".format(stage_name)
    lib_path = temp.relpath(lib_name)

    if env.TARGET in ["sim", "tsim", "intelfocl"]:
        lib.export_library(lib_path, fcompile=cc.create_shared)
        return remote.load_module(lib_path)

    sysroot = os.environ["SDKTARGETSYSROOT"]
    fcompile = cc.cross_compiler("aarch64-xilinx-linux-g++")
    lib.export_library(
        lib_path,
        fcompile=fcompile,
        options=[
            "--sysroot={}".format(sysroot),
            "-Wl,-rpath-link,{}/lib".format(sysroot),
            "-Wl,-rpath-link,{}/usr/lib".format(sysroot),
            "-L{}/lib".format(sysroot),
            "-L{}/usr/lib".format(sysroot),
        ],
    )
    remote.upload(lib_path)
    return remote.load_module(lib_name)


def connect_remote(env, host, port):
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = os.environ.get("TVM_TRACKER_PORT", None)
    if not tracker_host or not tracker_port:
        return rpc.connect(host, int(port))
    return autotvm.measure.request_remote(
        env.TARGET, tracker_host, int(tracker_port), timeout=10000
    )


def create_stage_module(stage_name, stage_device, graph, lib, remote):
    if stage_device == "vta":
        ctx = remote.ext_dev(0)
    else:
        ctx = remote.cpu(0)
    mod = graph_executor.create(graph, lib, ctx)
    print("[RUN] {} graph executor created on {}".format(stage_name, stage_device))
    return mod, ctx


def print_serial_stage_timing(rows):
    print("\n[TIMING] serial staged run")
    total = 0.0
    for stage_name, value in rows:
        print("\t{:<12}: {:8.3f} ms".format(stage_name, value))
        total += value
    print("\t{:<12}: {:8.3f} ms".format("total", total))


def simulate_pipeline(stage_service_rows, num_batches):
    """Simple deterministic pipeline model using measured per-stage service times."""

    stage_names = [name for name, _ in stage_service_rows]
    stage_times = [float(val) for _, val in stage_service_rows]
    num_stages = len(stage_times)
    finish = np.zeros((num_batches, num_stages), dtype="float64")

    for batch_idx in range(num_batches):
        for stage_idx in range(num_stages):
            prev_stage_done = finish[batch_idx, stage_idx - 1] if stage_idx > 0 else 0.0
            same_stage_prev_batch_done = finish[batch_idx - 1, stage_idx] if batch_idx > 0 else 0.0
            start = max(prev_stage_done, same_stage_prev_batch_done)
            finish[batch_idx, stage_idx] = start + stage_times[stage_idx]

    serial_per_batch = float(sum(stage_times))
    serial_total = serial_per_batch * float(num_batches)
    pipeline_total = float(finish[-1, -1])
    steady_ms_per_batch = float(max(stage_times))
    throughput = 1000.0 / steady_ms_per_batch if steady_ms_per_batch > 0 else 0.0

    print("\n[PIPELINE] simulated ideal pipeline")
    print("        batches            = {}".format(num_batches))
    print("        stage_service_ms   = {}".format(
        ", ".join("{}={:.3f}".format(name, value) for name, value in zip(stage_names, stage_times))
    ))
    print("        serial_total_ms    = {:.3f}".format(serial_total))
    print("        pipeline_total_ms  = {:.3f}".format(pipeline_total))
    print("        steady_ms_per_item = {:.3f}".format(steady_ms_per_batch))
    print("        steady_throughput  = {:.3f} items/s".format(throughput))
    if pipeline_total > 0.0:
        print("        serial_speedup     = {:.3f}x".format(serial_total / pipeline_total))


def main():
    args = parse_args()
    env = vta.get_env()

    full_model = vision.get_model(args.model, pretrained=True)
    feature_blocks = list(full_model.features)
    output_block = full_model.output
    scheme_cfg = SCHEMES[args.scheme]

    summarize_scheme(args.scheme, scheme_cfg, args.batch, args.image_size)
    summarize_feature_blocks(feature_blocks, output_block)
    validate_scheme(feature_blocks, scheme_cfg)

    cpu_target = tvm.target.Target(env.target_vta_cpu, host=env.target_host)
    image = preprocess_image(args.batch, args.image_size) if args.run_stages else None
    remote = None
    built_stages = []

    if args.run_stages:
        print("[RPC] connecting directly to {}:{} ...".format(args.host, args.port))
        remote = connect_remote(env, args.host, args.port)

    for stage in scheme_cfg:
        stage_name = stage["name"]
        input_shape = relay_shape_for_stage(stage_name, args.scheme, args.batch, args.image_size)
        stage_block = make_stage_block(feature_blocks, output_block, stage)
        mod, params = lower_stage_to_relay(stage_block, input_shape)
        out_shape, out_dtype = get_func_output_info(mod["main"])

        print("\n========== {} ==========".format(stage_name))
        print("[STAGE] device      =", stage["device"])
        print("[STAGE] input_shape =", input_shape)
        print("[STAGE] output_shape=", out_shape)
        print("[STAGE] params      =", len(params))

        if args.print_relay:
            print(mod["main"].astext(show_meta_data=False))

        if stage["device"] == "cpu":
            graph, lib, lowered_params = build_cpu_stage(stage_name, mod, params, cpu_target)
        else:
            if args.vta_stage_mode == "cpu_fallback":
                print("[BUILD] {} uses cpu_fallback mode".format(stage_name))
                graph, lib, lowered_params = build_cpu_stage(stage_name, mod, params, cpu_target)
            else:
                graph, lib, lowered_params = build_vta_stage(stage_name, mod["main"], params, env)

        if args.run_stages:
            remote_lib = export_and_upload(lib, remote, env, stage_name)
            stage_module, _ = create_stage_module(stage_name, stage["device"], graph, remote_lib, remote)
            stage_module.set_input(**lowered_params)
            built_stages.append(
                {
                    "name": stage_name,
                    "device": stage["device"],
                    "module": stage_module,
                    "graph": graph,
                    "lib": lib,
                    "lowered_params": lowered_params,
                    "output_shape": out_shape,
                    "output_dtype": out_dtype,
                }
            )

    if args.run_stages:
        print("\n[RUN] serial stage execution ...")
        stage_rows = []
        stage_service_rows = []
        current = image
        for stage_info in built_stages:
            mod = stage_info["module"]
            stage_name = stage_info["name"]
            mod.set_input("data", current)
            t0 = time.time()
            mod.run()
            t1 = time.time()
            out = mod.get_output(
                0,
                tvm.nd.empty(stage_info["output_shape"], stage_info["output_dtype"], remote.cpu(0)),
            )
            t2 = time.time()
            run_ms = (t1 - t0) * 1000.0
            get_ms = (t2 - t1) * 1000.0
            stage_rows.append((stage_name + ".run", run_ms))
            stage_rows.append((stage_name + ".out", get_ms))
            stage_service_rows.append((stage_name, run_ms + get_ms))
            current = out.numpy()
        print_serial_stage_timing(stage_rows)
        if args.pipeline_batches > 0:
            simulate_pipeline(stage_service_rows, args.pipeline_batches)


if __name__ == "__main__":
    main()
