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
Deploy Pretrained Vision Model from MxNet on VTA
================================================

Modified for AXU5EVB debugging:
1. keep original VTA compile / upload / graph executor flow
2. add graph summary / staged execution prints
3. use heterogenous build target for VTA:
      cpu     -> env.target_vta_cpu
      ext_dev -> env.target
4. use quantize + graph_pack for VTA
"""

from __future__ import absolute_import, print_function

import argparse
import json
import os
import time
import traceback
from os.path import join
from collections import Counter

from PIL import Image

from mxnet.gluon.model_zoo import vision
import numpy as np

import tvm
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_executor, utils, download, cc
from tvm.contrib.debugger import debug_executor

import vta
from vta.testing import simulator
from vta.top import graph_pack

# Make sure that TVM was compiled with RPC=1
assert tvm.runtime.enabled("rpc")


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def print_topk_with_scores(model_name, batch_idx, logits, synset, k=5):
    probs = softmax(logits.astype("float64"))
    top_idx = np.argsort(logits)[-k:][::-1]
    print("\n{} prediction for sample {}".format(model_name, batch_idx))
    for rank, idx in enumerate(top_idx, start=1):
        print(
            "\t#{}: {} (prob={:.4f}, logit={:.4f})".format(
                rank, synset[int(idx)], float(probs[idx]), float(logits[idx])
            )
        )


def print_stage_timing(title, timings_ms):
    print("\n[TIMING] {}".format(title))
    for key in ["set_params", "set_data", "run", "get_output", "total"]:
        if key in timings_ms:
            print("\t{:<10}: {:8.3f} ms".format(key, timings_ms[key]))


def _fmt_ms(val):
    return "{:.3f}".format(float(val))


def print_stage_table(title, rows):
    keys = ["set_params", "set_data", "run", "get_output", "total"]
    labels = {
        "set_params": "set_params",
        "set_data": "set_data",
        "run": "run",
        "get_output": "get_output",
        "total": "total",
    }
    available = [k for k in keys if any(k in values for _, values in rows)]
    if not available:
        return

    header = ["row"] + [labels[k] for k in available]
    widths = [max(len(header[0]), max(len(name) for name, _ in rows))]
    for key in available:
        max_width = len(labels[key])
        for _, values in rows:
            if key in values:
                max_width = max(max_width, len(_fmt_ms(values[key])))
        widths.append(max_width)

    def _print_row(cols):
        line = "  ".join(str(col).rjust(width) for col, width in zip(cols, widths))
        print(line)

    print("\n[TABLE] {}".format(title))
    _print_row(header)
    _print_row(["-" * width for width in widths])
    for name, values in rows:
        cols = [name]
        for key in available:
            cols.append(_fmt_ms(values[key]) if key in values else "-")
        _print_row(cols)

    # Highlight memory-related host/device transfer overhead as a quick summary.
    if len(rows) == 1:
        _, values = rows[0]
        if "total" in values:
            total = float(values["total"])
            mem_parts = [k for k in ["set_params", "set_data", "get_output"] if k in values]
            if total > 0 and mem_parts:
                mem_total = sum(float(values[k]) for k in mem_parts)
                print(
                    "[TABLE] memory-side share: {:.1f}% ({})".format(
                        100.0 * mem_total / total,
                        " + ".join("{}={:.3f}ms".format(k, float(values[k])) for k in mem_parts),
                    )
                )


def device_type_name(device_type):
    # TVM runtime device type ids commonly used in this script.
    # 1=cpu, 12=ext_dev.
    mapping = {
        1: "cpu",
        12: "ext_dev",
    }
    return mapping.get(int(device_type), "dev_type_%s" % int(device_type))


def print_graph_device_stats(graph_json):
    device_attr = graph_json.get("attrs", {}).get("device_index", None)
    if not device_attr or len(device_attr) < 2:
        print("[GRAPH] device_index attr not found, skip device stats")
        return

    dev_idx = device_attr[1]
    nodes = graph_json["nodes"]
    if len(dev_idx) != len(nodes):
        print(
            "[GRAPH] device_index size mismatch: len(dev_idx)=%d len(nodes)=%d"
            % (len(dev_idx), len(nodes))
        )
        return

    per_dev = Counter(dev_idx)
    print("\n[GRAPH] node device distribution")
    for dt, cnt in sorted(per_dev.items(), key=lambda x: x[0]):
        print("\t{}({}): {} nodes".format(device_type_name(dt), dt, cnt))

    per_dev_op = Counter()
    cross_edges = 0
    cross_pairs = Counter()
    for i, n in enumerate(nodes):
        dst = int(dev_idx[i])
        per_dev_op[(dst, n.get("op", "unknown"))] += 1
        for inp in n.get("inputs", []):
            src_idx = int(inp[0])
            src = int(dev_idx[src_idx])
            if src != dst:
                cross_edges += 1
                cross_pairs[(src, dst)] += 1

    print("[GRAPH] cross-device input edges =", cross_edges)
    if cross_edges:
        print("[GRAPH] cross-device edge pairs")
        for (src, dst), cnt in cross_pairs.most_common():
            print(
                "\t{}({}) -> {}({}): {}".format(
                    device_type_name(src), src, device_type_name(dst), dst, cnt
                )
            )

    print("[GRAPH] top op counts by device")
    for dt, _ in sorted(per_dev.items(), key=lambda x: x[0]):
        op_counts = Counter()
        for (dev_t, op), cnt in per_dev_op.items():
            if dev_t == dt:
                op_counts[op] += cnt
        top = op_counts.most_common(6)
        top_str = ", ".join(["{}:{}".format(op, c) for op, c in top])
        print("\t{}({}): {}".format(device_type_name(dt), dt, top_str))


def summarize_tir_usage(tir_text):
    markers = [
        "tir.vta",
        "VTAUopLoopBegin",
        "VTAUopPush",
        "VTALoadBuffer2D",
        "VTAStoreBuffer2D",
        "coproc_scope",
    ]
    counts = {m: tir_text.count(m) for m in markers}
    hit = any(v > 0 for v in counts.values())
    print("\n[TIR] marker summary")
    for k, v in counts.items():
        print("\t{:<16}: {}".format(k, v))
    print("[TIR] has_vta_markers =", hit)


def dump_relay_device_annotations(relay_func, max_lines=220):
    text = relay_func.astext(show_meta_data=False)
    lines = text.splitlines()
    print("\n[RELAY] annotated Relay (first %d lines)" % min(max_lines, len(lines)))
    for idx, line in enumerate(lines[:max_lines], start=1):
        print("%4d %s" % (idx, line))
    if len(lines) > max_lines:
        print("[RELAY] ... truncated %d more lines" % (len(lines) - max_lines))

    on_device_lines = [line for line in lines if "on_device" in line]
    print("[RELAY] on_device annotation count =", len(on_device_lines))
    for idx, line in enumerate(on_device_lines[:20], start=1):
        print("[RELAY] on_device[%d] %s" % (idx, line.strip()))
    if len(on_device_lines) > 20:
        print("[RELAY] ... truncated %d more on_device lines" % (len(on_device_lines) - 20))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="vta",
        choices=["vta", "arm_cpu"],
        help="Run on VTA or ARM CPU",
    )
    parser.add_argument(
        "--host",
        default="192.168.1.228",
        help="RPC server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="RPC server port",
    )
    parser.add_argument(
        "--model",
        default="resnet18_v1",
        help="Gluon vision model name",
    )
    parser.add_argument(
        "--debug-single-run",
        action="store_true",
        help="Do one plain m.run() first for debugging",
    )
    parser.add_argument(
        "--enable-timer",
        action="store_true",
        help="Enable time_evaluator after single-run succeeds",
    )
    parser.add_argument(
        "--number",
        type=int,
        default=4,
        help="time_evaluator number",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="time_evaluator repeat",
    )
    parser.add_argument(
        "--print-top5",
        action="store_true",
        help="Print top-5 predictions",
    )
    parser.add_argument(
        "--stage-profile-repeat",
        type=int,
        default=0,
        help="Repeat end-to-end set_input/run/get_output profiling N times (0 to disable)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup runs before benchmark loops",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=20,
        help="Benchmark runs with params already loaded",
    )
    parser.add_argument(
        "--community-bench",
        action="store_true",
        help="Run the common VTA/TVM benchmark flow: warmup, then time_evaluator('run') with params/data preloaded",
    )
    parser.add_argument(
        "--tune-log",
        default="",
        help="Optional AutoTVM tuning log file; if set and exists, use apply_history_best",
    )
    parser.add_argument(
        "--print-device-stats",
        action="store_true",
        help="Print per-device node distribution and cross-device edge stats",
    )
    parser.add_argument(
        "--op-profile-repeat",
        type=int,
        default=0,
        help="If > 0, run graph debug profiler and print per-op timing table",
    )
    parser.add_argument(
        "--dump-tir-path",
        default="",
        help="Optional file path to dump lowered TIR (phase-3) for operator placement inspection",
    )
    parser.add_argument(
        "--hetero",
        action="store_true",
        help="When device=vta, keep only the packed subgraph on VTA and leave the rest on CPU",
    )
    parser.add_argument(
        "--pack-start-op",
        default="",
        help="Override graph_pack start op name for VTA path",
    )
    parser.add_argument(
        "--pack-stop-op",
        default="",
        help="Override graph_pack stop op name for VTA path",
    )
    parser.add_argument(
        "--hetero-annot-start",
        default="nn.conv2d",
        help="graph_pack device annotation start op in hetero mode",
    )
    parser.add_argument(
        "--hetero-annot-end",
        default="annotation.stop_fusion",
        help="graph_pack device annotation end op in hetero mode",
    )
    parser.add_argument(
        "--fusion-mode",
        default="default",
        choices=["default", "limited", "off"],
        help="Control Relay FuseOps during build",
    )
    parser.add_argument(
        "--fuse-max-depth",
        type=int,
        default=1,
        help="Used when --fusion-mode=limited",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square input resolution for build and preprocessing",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=0,
        help="Override input image height; 0 means use --image-size",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=0,
        help="Override input image width; 0 means use --image-size",
    )
    parser.add_argument(
        "--dump-relay-annot",
        action="store_true",
        help="Print Relay text after graph_pack to inspect on_device annotations",
    )
    return parser.parse_args()


args = parse_args()
input_height = args.image_height if args.image_height > 0 else args.image_size
input_width = args.image_width if args.image_width > 0 else args.image_size

# Load VTA parameters from vta_config.json
env = vta.get_env()

# Select logical device
device = args.device
target = env.target if device == "vta" else env.target_vta_cpu

# Dictionary lookup for when to start/end bit packing
pack_dict = {
    "resnet18_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet18_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet50_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet101_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
}

model = args.model
assert model in pack_dict, "Unsupported model: %s" % model
pack_start_op = args.pack_start_op if args.pack_start_op else pack_dict[model][0]
pack_stop_op = args.pack_stop_op if args.pack_stop_op else pack_dict[model][1]

print("========== Configuration ==========")
print("env.TARGET           =", env.TARGET)
print("device               =", device)
print("target               =", target)
print("model                =", model)
print("debug_single_run     =", args.debug_single_run)
print("enable_timer         =", args.enable_timer)
print("stage_profile_repeat =", args.stage_profile_repeat)
print("warmup               =", args.warmup)
print("benchmark_runs       =", args.benchmark_runs)
print("community_bench      =", args.community_bench)
print("tune_log             =", args.tune_log if args.tune_log else "<none>")
print("print_device_stats   =", args.print_device_stats)
print("op_profile_repeat    =", args.op_profile_repeat)
print("dump_tir_path        =", args.dump_tir_path if args.dump_tir_path else "<none>")
print("hetero               =", args.hetero)
print("pack_start_op        =", pack_start_op)
print("pack_stop_op         =", pack_stop_op)
print("fusion_mode          =", args.fusion_mode)
print("fuse_max_depth       =", args.fuse_max_depth)
print("input_resolution     = {}x{}".format(input_height, input_width))
print("dump_relay_annot     =", args.dump_relay_annot)
print("RPC host             =", args.host)
print("RPC port             =", args.port)
print("===================================")

######################################################################
# Obtain an execution remote
######################################################################

if env.TARGET not in ["sim", "tsim", "intelfocl"]:
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = os.environ.get("TVM_TRACKER_PORT", None)

    device_host = args.host
    device_port = args.port

    if not tracker_host or not tracker_port:
        print("[RPC] connecting directly to %s:%d ..." % (device_host, device_port))
        remote = rpc.connect(device_host, int(device_port))
    else:
        print(
            "[RPC] requesting remote via tracker %s:%s ..."
            % (tracker_host, tracker_port)
        )
        remote = autotvm.measure.request_remote(
            env.TARGET, tracker_host, int(tracker_port), timeout=10000
        )

    print("Skip FPGA reconfiguration (AXU5EVB already configured)")
else:
    remote = rpc.LocalSession()

    if env.TARGET in ["intelfocl"]:
        vta.program_fpga(remote, bitstream="vta.bitstream")

# Contexts
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)
ctxes = [remote.ext_dev(0), remote.cpu(0)] if device == "vta" else [remote.cpu(0)]

######################################################################
# Build the inference graph executor
######################################################################

build_ctx = autotvm.tophub.context(target)
if args.tune_log:
    if os.path.exists(args.tune_log):
        print("[BUILD] apply_history_best from", args.tune_log)
        build_ctx = autotvm.apply_history_best(args.tune_log)
    else:
        print("[BUILD] tune log not found, fallback to tophub:", args.tune_log)

with build_ctx:
    dtype_dict = {"data": "float32"}
    shape_dict = {"data": (env.BATCH, 3, input_height, input_width)}

    print("[BUILD] loading Gluon model:", model)
    gluon_model = vision.get_model(model, pretrained=True)

    build_start = time.time()

    print("[BUILD] relay.frontend.from_mxnet ...")
    mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    # Quantize + graph_pack only for explicit VTA execution mode.
    # Do not infer this from target.device_name because ARM targets can
    # still carry VTA-related metadata and cause packed layouts to leak.
    if device == "vta":
        print("[BUILD] quantization for VTA ...")
        with tvm.transform.PassContext(opt_level=3):
            with relay.quantize.qconfig(
                global_scale=8.0,
                skip_conv_layers=[0],
            ):
                mod = relay.quantize.quantize(mod, params=params)

        print("[BUILD] graph_pack enabled for VTA ...")
        relay_prog = graph_pack(
            mod["main"],
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name=pack_start_op,
            stop_name=pack_stop_op,
            device_annot=args.hetero,
            annot_start_name=args.hetero_annot_start,
            annot_end_name=args.hetero_annot_end,
        )
        if args.dump_relay_annot:
            dump_relay_device_annotations(relay_prog)
    else:
        print("[BUILD] graph_pack not needed for CPU path")
        relay_prog = mod["main"]

    if args.fusion_mode == "off":
        print("[BUILD] defuse Relay primitive functions before build ...")
        relay_mod = tvm.IRModule.from_expr(relay_prog)
        relay_mod = relay.transform.InferType()(relay_mod)
        relay_mod = relay.transform.DefuseOps()(relay_mod)
        relay_prog = relay_mod["main"]

    print("[BUILD] relay.build ...")
    tir_dumps = []
    pass_config = {}
    enable_tir_dump = bool(args.dump_tir_path and device != "vta")
    if args.dump_tir_path and device == "vta":
        print("[TIR] dump_tir_path is disabled in VTA mode to avoid interfering with vta.build_config")
    disabled_passes = {"AlterOpLayout"}
    if args.fusion_mode == "off":
        disabled_passes.add("FuseOps")
    elif args.fusion_mode == "limited":
        pass_config["relay.FuseOps.max_depth"] = args.fuse_max_depth
    if enable_tir_dump:
        @tvm.tir.transform.prim_func_pass(opt_level=0)
        def _dump_tir_pass(tir_func, _, __):
            tir_dumps.append(str(tir_func))
            return tir_func

        pass_config["tir.add_lower_pass"] = [(3, _dump_tir_pass)]

    if device != "vta":
        with tvm.transform.PassContext(
            opt_level=3, disabled_pass=disabled_passes, config=pass_config
        ):
            graph, lib, params = relay.build(
                relay_prog,
                target=tvm.target.Target(target, host=env.target_host),
                params=params,
            )
    else:
        if args.hetero:
            build_target = {
                "cpu": env.target_vta_cpu,
                "ext_dev": target,
            }
        else:
            build_target = tvm.target.Target(target, host=env.target_host)
        with vta.build_config(
            opt_level=3,
            disabled_pass=disabled_passes | {"tir.CommonSubexprElimTIR"},
        ):
            graph, lib, params = relay.build(
                relay_prog,
                target=build_target,
                target_host=env.target_host if args.hetero else None,
                params=params,
            )

    if enable_tir_dump:
        tir_text = "\n\n".join(tir_dumps)
        with open(args.dump_tir_path, "w") as f:
            f.write(tir_text)
        print("[TIR] dumped to", args.dump_tir_path)
        summarize_tir_usage(tir_text)

    build_time = time.time() - build_start
    print("[BUILD] %s inference graph built in %.2fs!" % (model, build_time))

    g = json.loads(graph)
    print("=== graph summary ===")
    print("num_nodes =", len(g["nodes"]))
    print("arg_nodes =", g.get("arg_nodes"))
    print("heads =", g.get("heads"))

    if "attrs" in g:
        for k, v in g["attrs"].items():
            if "device" in k.lower():
                print("attr", k, "=", v)

    if args.print_device_stats:
        print_graph_device_stats(g)

    for i, n in enumerate(g["nodes"][:40]):
        print(i, n["op"], n["name"])

    ##################################################################
    # Export and upload
    ##################################################################

    temp = utils.tempdir()
    lib_path = temp.relpath("graphlib.so")

    sysroot = os.environ["SDKTARGETSYSROOT"]
    fcompile = cc.cross_compiler("aarch64-xilinx-linux-g++")

    print("[RPC] export_library ->", lib_path)
    lib.export_library(
        lib_path,
        fcompile=fcompile,
        options=[
            f"--sysroot={sysroot}",
            f"-Wl,-rpath-link,{sysroot}/lib",
            f"-Wl,-rpath-link,{sysroot}/usr/lib",
            f"-L{sysroot}/lib",
            f"-L{sysroot}/usr/lib",
        ],
    )

    print("[RPC] upload graphlib.so ...")
    remote.upload(lib_path)

    print("[RPC] load_module(graphlib.so) ...")
    lib = remote.load_module("graphlib.so")

    print("[GRAPH] create graph executor ...")
    if device == "vta":
        m = graph_executor.create(graph, lib, ctxes)
    else:
        m = graph_executor.create(graph, lib, ctx)

    md = None
    if args.op_profile_repeat > 0:
        print("[GRAPH] create debug executor for per-op profiling ...")
        if device == "vta":
            md = debug_executor.create(graph, lib, ctxes)
        else:
            md = debug_executor.create(graph, lib, ctx)

######################################################################
# Perform image classification inference
######################################################################

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
image = Image.open(image_fn).resize((input_width, input_height))
image = np.array(image) - np.array([123.0, 117.0, 104.0])
image /= np.array([58.395, 57.12, 57.375])
image = image.transpose((2, 0, 1))
image = image[np.newaxis, :]
image = np.repeat(image, env.BATCH, axis=0)

######################################################################
# Debug-first execution flow
######################################################################

try:
    # Load params once (real deployment path). We still time this one-time cost.
    print("[RUN] 1/6 set_input(params) ...")
    t0 = time.time()
    m.set_input(**params)
    t1 = time.time()

    print("[RUN] 2/6 set_input(data) ...")
    m.set_input("data", image)
    t2 = time.time()

    if args.debug_single_run or not args.enable_timer:
        print("[RUN] 3/6 before m.run() ...")
        m.run()
        t3 = time.time()
        print("[RUN] 4/6 after m.run()")

        print("[RUN] 5/6 before get_output() ...")
        tvm_output = m.get_output(
            0, tvm.nd.empty((env.BATCH, 1000), "float32", remote.cpu(0))
        )
        t4 = time.time()
        print("[RUN] 6/6 after get_output()")

        output_np = tvm_output.numpy()
        timings_ms = {
            "set_params": (t1 - t0) * 1000.0,
            "set_data": (t2 - t1) * 1000.0,
            "run": (t3 - t2) * 1000.0,
            "get_output": (t4 - t3) * 1000.0,
            "total": (t4 - t0) * 1000.0,
        }
        print_stage_timing("single run", timings_ms)
        print_stage_table("single run", [("single", timings_ms)])

        if args.print_top5:
            for b in range(env.BATCH):
                print_topk_with_scores(model, b, output_np[b], synset, k=5)
        else:
            print("[RUN] single-run finished successfully.")
            top1_idx = int(np.argmax(output_np[0]))
            print("[RUN] top1 index of sample0 =", top1_idx)
            top1_prob = float(softmax(output_np[0].astype("float64"))[top1_idx])
            print("[RUN] top1 class of sample0 =", synset[top1_idx])
            print("[RUN] top1 prob of sample0  = {:.4f}".format(top1_prob))

        if args.stage_profile_repeat > 0:
            print("\n[TIMING] stage profiling repeat =", args.stage_profile_repeat)
            set_params_ms = []
            set_data_ms = []
            run_ms = []
            get_output_ms = []
            total_ms = []
            for i in range(args.stage_profile_repeat):
                p0 = time.time()
                m.set_input(**params)
                p1 = time.time()
                m.set_input("data", image)
                p2 = time.time()
                m.run()
                p3 = time.time()
                _ = m.get_output(0, tvm.nd.empty((env.BATCH, 1000), "float32", remote.cpu(0)))
                p4 = time.time()
                set_params_ms.append((p1 - p0) * 1000.0)
                set_data_ms.append((p2 - p1) * 1000.0)
                run_ms.append((p3 - p2) * 1000.0)
                get_output_ms.append((p4 - p3) * 1000.0)
                total_ms.append((p4 - p0) * 1000.0)

            print_stage_timing(
                "repeat avg",
                {
                    "set_params": float(np.mean(set_params_ms)),
                    "set_data": float(np.mean(set_data_ms)),
                    "run": float(np.mean(run_ms)),
                    "get_output": float(np.mean(get_output_ms)),
                    "total": float(np.mean(total_ms)),
                },
            )
            repeat_avg = {
                "set_params": float(np.mean(set_params_ms)),
                "set_data": float(np.mean(set_data_ms)),
                "run": float(np.mean(run_ms)),
                "get_output": float(np.mean(get_output_ms)),
                "total": float(np.mean(total_ms)),
            }
            print_stage_timing(
                "repeat std",
                {
                    "set_params": float(np.std(set_params_ms)),
                    "set_data": float(np.std(set_data_ms)),
                    "run": float(np.std(run_ms)),
                    "get_output": float(np.std(get_output_ms)),
                    "total": float(np.std(total_ms)),
                },
            )
            repeat_std = {
                "set_params": float(np.std(set_params_ms)),
                "set_data": float(np.std(set_data_ms)),
                "run": float(np.std(run_ms)),
                "get_output": float(np.std(get_output_ms)),
                "total": float(np.std(total_ms)),
            }
            print_stage_table("stage repeat summary", [("avg", repeat_avg), ("std", repeat_std)])

        if args.benchmark_runs > 0:
            print("\n[BENCH] params-once benchmark")
            # Warmup without timing.
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

            print_stage_timing(
                "params-once avg",
                {
                    "set_data": float(np.mean(set_data_ms)),
                    "run": float(np.mean(run_ms)),
                    "get_output": float(np.mean(get_output_ms)),
                    "total": float(np.mean(total_ms)),
                },
            )
            params_once_avg = {
                "set_data": float(np.mean(set_data_ms)),
                "run": float(np.mean(run_ms)),
                "get_output": float(np.mean(get_output_ms)),
                "total": float(np.mean(total_ms)),
            }
            print_stage_timing(
                "params-once std",
                {
                    "set_data": float(np.std(set_data_ms)),
                    "run": float(np.std(run_ms)),
                    "get_output": float(np.std(get_output_ms)),
                    "total": float(np.std(total_ms)),
                },
            )
            params_once_std = {
                "set_data": float(np.std(set_data_ms)),
                "run": float(np.std(run_ms)),
                "get_output": float(np.std(get_output_ms)),
                "total": float(np.std(total_ms)),
            }
            print_stage_table("params-once summary", [("avg", params_once_avg), ("std", params_once_std)])

        if args.community_bench:
            print("\n[BENCH] community benchmark (warmup + time_evaluator on run)")
            m.set_input(**params)
            m.set_input("data", image)
            for _ in range(args.warmup):
                m.run()
            num = args.number
            rep = args.repeat
            timer_ctx = ctx if device == "vta" else ctx
            timer = m.module.time_evaluator("run", timer_ctx, number=num, repeat=rep)
            tcost = timer()
            results_ms = np.array(tcost.results, dtype="float64") * 1000.0
            print(
                "[BENCH] run-only mean={:.3f} ms  std={:.3f} ms  min={:.3f} ms  median={:.3f} ms  number={}  repeat={}".format(
                    float(np.mean(results_ms)),
                    float(np.std(results_ms)),
                    float(np.min(results_ms)),
                    float(np.median(results_ms)),
                    num,
                    rep,
                )
            )
            print(
                "[BENCH] note: this excludes local Python/RPC roundtrip timing, but still includes remote runtime/device execution for GraphModule.run()"
            )
            print_stage_table(
                "community benchmark run-only",
                [
                    (
                        "run",
                        {
                            "run": float(np.mean(results_ms)),
                        },
                    ),
                    (
                        "run_std",
                        {
                            "run": float(np.std(results_ms)),
                        },
                    ),
                ],
            )

        if args.op_profile_repeat > 0:
            print("\n[PROFILE] per-op graph profiling repeat =", args.op_profile_repeat)
            md.set_input(**params)
            md.set_input("data", image)
            report = md.profile()
            print("\n[PROFILE] per-op table")
            print(report.table(sort=True, aggregate=True, col_sums=True))

    if args.enable_timer:
        print(
            "[TIMER] enable time_evaluator(number=%d, repeat=%d) ..."
            % (args.number, args.repeat)
        )

        num = args.number
        rep = args.repeat

        # Keep using ext_dev context for run timing in VTA mode
        timer_ctx = ctx if device == "vta" else ctxes[0]
        timer = m.module.time_evaluator("run", timer_ctx, number=num, repeat=rep)

        if env.TARGET in ["sim", "tsim"]:
            simulator.clear_stats()
            timer()
            sim_stats = simulator.stats()
            print("\nExecution statistics:")
            for k, v in sim_stats.items():
                print("\t{:<16}: {:>16}".format(k, v // (num * rep + 1)))
        else:
            tcost = timer()
            std = np.std(tcost.results) * 1000
            mean = tcost.mean * 1000
            print(
                "\nPerformed inference in %.2fms (std = %.2f) for %d samples"
                % (mean, std, env.BATCH)
            )
            print("Average per sample inference time: %.2fms" % (mean / env.BATCH))

            print("[TIMER] fetch output after timing run ...")
            tvm_output = m.get_output(
                0, tvm.nd.empty((env.BATCH, 1000), "float32", remote.cpu(0))
            )
            output_np = tvm_output.numpy()

            if args.print_top5:
                for b in range(env.BATCH):
                    print_topk_with_scores(model, b, output_np[b], synset, k=5)

except Exception as e:
    print("\n[ERROR] Exception during inference flow:")
    print(type(e).__name__, str(e))
    print("\n[ERROR] Full traceback:")
    traceback.print_exc()
    raise
