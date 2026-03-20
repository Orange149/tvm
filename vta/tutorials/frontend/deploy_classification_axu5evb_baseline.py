# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Clean AXU5EVB baseline for ResNet/ImageNet inference on VTA.

This file intentionally keeps only the stable single-target offload path:
quantize -> graph_pack -> relay.build(target=env.target) -> graph executor run.

Use this as the reference baseline before experimenting with explicit hetero.
"""

from __future__ import absolute_import, print_function

import argparse
import json
import os
import time
from collections import Counter
from os.path import join

from PIL import Image

from mxnet.gluon.model_zoo import vision
import numpy as np

import tvm
from tvm import autotvm, relay, rpc
from tvm.contrib import cc, download, graph_executor, utils

import vta
from vta.top import graph_pack

assert tvm.runtime.enabled("rpc")


PACK_DICT = {
    "resnet18_v1": ("nn.max_pool2d", "nn.global_avg_pool2d"),
    "resnet34_v1": ("nn.max_pool2d", "nn.global_avg_pool2d"),
    "resnet18_v2": ("nn.max_pool2d", "nn.global_avg_pool2d"),
    "resnet34_v2": ("nn.max_pool2d", "nn.global_avg_pool2d"),
    "resnet50_v2": ("nn.max_pool2d", "nn.global_avg_pool2d"),
    "resnet101_v2": ("nn.max_pool2d", "nn.global_avg_pool2d"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.1.228", help="RPC server host")
    parser.add_argument("--port", type=int, default=9090, help="RPC server port")
    parser.add_argument("--model", default="resnet18_v1", help="Gluon vision model name")
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=20,
        help="Params-once benchmark runs after warmup",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs before benchmark")
    parser.add_argument(
        "--debug-single-run",
        action="store_true",
        help="Run one plain inference before benchmark and print top1",
    )
    parser.add_argument(
        "--print-top5",
        action="store_true",
        help="Print top-5 predictions for the first sample",
    )
    parser.add_argument(
        "--print-device-stats",
        action="store_true",
        help="Print graph device distribution from graph JSON",
    )
    parser.add_argument(
        "--tune-log",
        default="",
        help="Optional AutoTVM tuning log file",
    )
    return parser.parse_args()


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def print_graph_device_stats(graph_json):
    device_attr = graph_json.get("attrs", {}).get("device_index")
    if not device_attr or len(device_attr) < 2:
        print("[GRAPH] device_index attr not found")
        return

    dev_idx = [int(v) for v in device_attr[1]]
    nodes = graph_json["nodes"]
    per_dev = Counter(dev_idx)
    print("\n[GRAPH] node device distribution")
    for dev_type, count in sorted(per_dev.items()):
        name = "cpu" if dev_type == 1 else "ext_dev" if dev_type == 12 else str(dev_type)
        print("  {}({}): {} nodes".format(name, dev_type, count))

    cross_edges = 0
    for idx, node in enumerate(nodes):
        dst = dev_idx[idx]
        for src_idx, _, _ in node.get("inputs", []):
            if dev_idx[int(src_idx)] != dst:
                cross_edges += 1
    print("[GRAPH] cross-device input edges =", cross_edges)


def print_top5(logits, synset):
    probs = softmax(logits.astype("float64"))
    top_idx = np.argsort(logits)[-5:][::-1]
    print("\nTop-5 prediction for sample 0")
    for rank, idx in enumerate(top_idx, start=1):
        print(
            "  #{}: {} (prob={:.4f}, logit={:.4f})".format(
                rank, synset[int(idx)], float(probs[idx]), float(logits[idx])
            )
        )


def connect_remote(env, host, port):
    if env.TARGET in ["sim", "tsim", "intelfocl"]:
        remote = rpc.LocalSession()
        if env.TARGET == "intelfocl":
            vta.program_fpga(remote, bitstream="vta.bitstream")
        return remote

    tracker_host = os.environ.get("TVM_TRACKER_HOST")
    tracker_port = os.environ.get("TVM_TRACKER_PORT")
    if tracker_host and tracker_port:
        print("[RPC] requesting remote via tracker {}:{} ...".format(tracker_host, tracker_port))
        return autotvm.measure.request_remote(
            env.TARGET, tracker_host, int(tracker_port), timeout=10000
        )

    print("[RPC] connecting directly to {}:{} ...".format(host, port))
    return rpc.connect(host, int(port))


def build_graph(env, model_name, tune_log):
    target = env.target
    start_name, stop_name = PACK_DICT[model_name]
    build_ctx = autotvm.tophub.context(target)
    if tune_log:
        if os.path.exists(tune_log):
            print("[BUILD] apply_history_best from", tune_log)
            build_ctx = autotvm.apply_history_best(tune_log)
        else:
            print("[BUILD] tune log not found, fallback to tophub:", tune_log)

    with build_ctx:
        shape_dict = {"data": (env.BATCH, 3, 224, 224)}
        print("[BUILD] loading Gluon model:", model_name)
        gluon_model = vision.get_model(model_name, pretrained=True)

        build_start = time.time()
        print("[BUILD] relay.frontend.from_mxnet ...")
        mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

        print("[BUILD] quantization for VTA ...")
        with tvm.transform.PassContext(opt_level=3):
            with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
                mod = relay.quantize.quantize(mod, params=params)

        print("[BUILD] graph_pack enabled for VTA ...")
        relay_prog = graph_pack(
            mod["main"],
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name=start_name,
            stop_name=stop_name,
        )

        print("[BUILD] relay.build ...")
        with vta.build_config(
            opt_level=3,
            disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
        ):
            graph, lib, params = relay.build(
                relay_prog,
                target=tvm.target.Target(target, host=env.target_host),
                params=params,
            )

        print("[BUILD] {} inference graph built in {:.2f}s!".format(model_name, time.time() - build_start))
        return graph, lib, params


def prepare_input(env):
    print("[DATA] download synset ...")
    categ_url = "https://github.com/uwsampl/web-data/raw/main/vta/models/"
    categ_fn = "synset.txt"
    download.download(join(categ_url, categ_fn), categ_fn)
    synset = eval(open(categ_fn).read())

    print("[DATA] download test image ...")
    image_url = "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg"
    image_fn = "cat.png"
    download.download(image_url, image_fn)

    print("[DATA] preprocess image ...")
    image = Image.open(image_fn).resize((224, 224))
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    image = np.repeat(image, env.BATCH, axis=0)
    return image, synset


def main():
    args = parse_args()
    env = vta.get_env()
    assert args.model in PACK_DICT, "Unsupported model: {}".format(args.model)

    print("========== Configuration ==========")
    print("env.TARGET         =", env.TARGET)
    print("target             =", env.target)
    print("model              =", args.model)
    print("benchmark_runs     =", args.benchmark_runs)
    print("warmup             =", args.warmup)
    print("debug_single_run   =", args.debug_single_run)
    print("print_device_stats =", args.print_device_stats)
    print("tune_log           =", args.tune_log if args.tune_log else "<none>")
    print("RPC host           =", args.host)
    print("RPC port           =", args.port)
    print("===================================")

    remote = connect_remote(env, args.host, args.port)
    if env.TARGET not in ["sim", "tsim", "intelfocl"]:
        print("Skip FPGA reconfiguration (AXU5EVB already configured)")

    graph, lib, params = build_graph(env, args.model, args.tune_log)
    graph_json = json.loads(graph)
    print("=== graph summary ===")
    print("num_nodes =", len(graph_json["nodes"]))
    print("heads =", graph_json.get("heads"))
    if args.print_device_stats:
        print_graph_device_stats(graph_json)

    temp = utils.tempdir()
    lib_path = temp.relpath("graphlib.so")
    sysroot = os.environ["SDKTARGETSYSROOT"]
    fcompile = cc.cross_compiler("aarch64-xilinx-linux-g++")
    print("[RPC] export_library ->", lib_path)
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
    print("[RPC] upload graphlib.so ...")
    remote.upload(lib_path)
    print("[RPC] load_module(graphlib.so) ...")
    lib = remote.load_module("graphlib.so")

    ctxes = [remote.ext_dev(0), remote.cpu(0)]
    print("[GRAPH] create graph executor ...")
    m = graph_executor.create(graph, lib, ctxes)

    image, synset = prepare_input(env)

    print("[RUN] set_input(params) ...")
    m.set_input(**params)
    print("[RUN] set_input(data) ...")
    m.set_input("data", image)

    if args.debug_single_run:
        print("[RUN] m.run() ...")
        t0 = time.time()
        m.run()
        t1 = time.time()
        out = m.get_output(0, tvm.nd.empty((env.BATCH, 1000), "float32", remote.cpu(0)))
        t2 = time.time()
        logits = out.numpy()[0]
        top1_idx = int(np.argmax(logits))
        top1_prob = float(softmax(logits.astype("float64"))[top1_idx])
        print("[RUN] run       : {:.3f} ms".format((t1 - t0) * 1000.0))
        print("[RUN] get_output: {:.3f} ms".format((t2 - t1) * 1000.0))
        print("[RUN] top1 index =", top1_idx)
        print("[RUN] top1 class =", synset[top1_idx])
        print("[RUN] top1 prob  = {:.4f}".format(top1_prob))
        if args.print_top5:
            print_top5(logits, synset)

    if args.benchmark_runs > 0:
        print("\n[BENCH] params-once benchmark")
        for _ in range(args.warmup):
            m.set_input("data", image)
            m.run()
            _ = m.get_output(0, tvm.nd.empty((env.BATCH, 1000), "float32", remote.cpu(0)))

        set_data_ms = []
        run_ms = []
        get_output_ms = []
        total_ms = []
        for _ in range(args.benchmark_runs):
            p0 = time.time()
            m.set_input("data", image)
            p1 = time.time()
            m.run()
            p2 = time.time()
            _ = m.get_output(0, tvm.nd.empty((env.BATCH, 1000), "float32", remote.cpu(0)))
            p3 = time.time()
            set_data_ms.append((p1 - p0) * 1000.0)
            run_ms.append((p2 - p1) * 1000.0)
            get_output_ms.append((p3 - p2) * 1000.0)
            total_ms.append((p3 - p0) * 1000.0)

        print("[TIMING] set_data avg  : {:.3f} ms".format(float(np.mean(set_data_ms))))
        print("[TIMING] run avg       : {:.3f} ms".format(float(np.mean(run_ms))))
        print("[TIMING] get_output avg: {:.3f} ms".format(float(np.mean(get_output_ms))))
        print("[TIMING] total avg     : {:.3f} ms".format(float(np.mean(total_ms))))


if __name__ == "__main__":
    main()
