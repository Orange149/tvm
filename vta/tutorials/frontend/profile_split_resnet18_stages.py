#!/usr/bin/env python3
"""Build and profile split ResNet18 stages for CPU/VTA research."""

from __future__ import absolute_import, print_function

import argparse
import json
import os
import time
import csv

import numpy as np
from PIL import Image
import tvm
from tvm import autotvm, relay, rpc
from tvm.contrib import cc, download, graph_executor, utils
from tvm.relay import op, transform

import vta
from vta.top import graphpack as vta_graphpack
from vta.testing import simulator

from hp_hpc_quant.vta_runtime_profile_utils import (
    dump_runtime_snapshot,
    fetch_runtime_profiler_hooks,
    hooks_available,
)

from split_resnet18_stages import (  # pylint: disable=import-error
    AUTO_RESOURCE_AWARE_SCHEME,
    SCHEMES,
    build_resource_aware_calibration_set,
    build_resnet18_unit_blocks,
    build_resnet18_unit_metadata,
    classify_downsample_window_relation,
    classify_vta_window_category,
    lower_stage_to_relay,
    make_stage_block,
    relay_inputs_for_stage,
    relay_shape_for_stage,
    resolve_scheme_config,
    save_partition_visualization,
    stage_input_schema_for_stage,
    stage_output_schema_for_stage,
    summarize_feature_blocks,
    summarize_scheme,
    summarize_units,
    validate_scheme,
)

from mxnet.gluon.model_zoo import vision


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scheme",
        default="three_stage_a",
        choices=sorted(list(SCHEMES.keys()) + [AUTO_RESOURCE_AWARE_SCHEME]),
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
        "--final-output-to-host",
        action="store_true",
        help="Fetch the final stage output back to host numpy for inspection",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the serial staged run this many times",
    )
    parser.add_argument(
        "--warmup-repeat",
        type=int,
        default=0,
        help="Run this many warmup iterations before the steady-state window",
    )
    parser.add_argument(
        "--print-vta-runtime-profile",
        action="store_true",
        help="Print VTA runtime profiler status after the steady-state window",
    )
    parser.add_argument(
        "--vta-runtime-profile-dir",
        default="",
        help="Optional directory to dump VTA runtime profiler snapshots as JSON",
    )
    parser.add_argument(
        "--vta-runtime-profile-events-limit",
        type=int,
        default=200,
        help="Maximum number of VTA runtime profiler events to dump per snapshot",
    )
    parser.add_argument(
        "--resource-aware-candidate-name",
        default="",
        help="When scheme=auto_resource_aware, force-run a specific enumerated candidate window",
    )
    parser.add_argument(
        "--resource-aware-top-k",
        type=int,
        default=10,
        help="When scheme=auto_resource_aware, print this many ranked candidates",
    )
    parser.add_argument(
        "--resource-aware-calibration-report",
        action="store_true",
        help="When scheme=auto_resource_aware, print the representative fsim calibration set",
    )
    parser.add_argument(
        "--resource-aware-run-calibration-set",
        action="store_true",
        help="When scheme=auto_resource_aware, run the representative calibration set and print measured vs predicted totals",
    )
    parser.add_argument(
        "--resource-aware-calibration-max-items",
        type=int,
        default=14,
        help="Maximum number of calibration candidates to run when --resource-aware-run-calibration-set is enabled",
    )
    parser.add_argument(
        "--resource-aware-buildability-sweep",
        action="store_true",
        help="When scheme=auto_resource_aware, probe the top static candidates and report which ones build under the current VTA stack",
    )
    parser.add_argument(
        "--resource-aware-buildability-max-items",
        type=int,
        default=40,
        help="Maximum number of top-ranked candidates to probe in the buildability sweep",
    )
    parser.add_argument(
        "--resource-aware-run-buildable-main-table",
        action="store_true",
        help="When scheme=auto_resource_aware, run fsim only for the buildable representative candidates and print the total.service main table",
    )
    parser.add_argument(
        "--resource-aware-main-table-max-items",
        type=int,
        default=0,
        help="Maximum number of buildable candidates to include in the fsim main table; 0 means all buildable candidates from the sweep",
    )
    parser.add_argument(
        "--save-partition-viz",
        default="",
        help="If set, save an SVG visualization of the resolved partition",
    )
    parser.add_argument(
        "--single-op-threshold-csv",
        default="",
        help="Optional threshold decision CSV from analyze_resnet18_single_op_csv.py; if set, print a single-op threshold comparison summary after the buildable main table",
    )
    parser.add_argument(
        "--runtime-num-threads",
        type=int,
        default=0,
        help="If > 0, reconfigure the remote TVM runtime thread pool to use this many CPU threads",
    )
    parser.add_argument(
        "--stage0-num-threads",
        type=int,
        default=4,
        help="Remote TVM threadpool size to use before each stage0_cpu execution",
    )
    parser.add_argument(
        "--stage2-num-threads",
        type=int,
        default=4,
        help="Remote TVM threadpool size to use before each stage2_cpu execution",
    )
    parser.add_argument(
        "--timer-number",
        type=int,
        default=10,
        help="time_evaluator number when collecting per-stage pure run metrics",
    )
    parser.add_argument(
        "--timer-repeat",
        type=int,
        default=10,
        help="time_evaluator repeat when collecting per-stage pure run metrics",
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


def _relay_type_to_schema(relay_type, prefix="out"):
    if isinstance(relay_type, tvm.ir.tensor_type.TensorType):
        return {
            "kind": "tensor",
            "arity": 1,
            "slots": [
                {
                    "slot_index": 0,
                    "role": prefix,
                    "shape": tuple(int(x) for x in relay_type.shape),
                    "dtype": relay_type.dtype,
                }
            ],
        }
    if isinstance(relay_type, tvm.ir.type.TupleType):
        slots = []
        for idx, field in enumerate(relay_type.fields):
            if not isinstance(field, tvm.ir.tensor_type.TensorType):
                raise TypeError("Only flat tuple tensor outputs are supported, got {}".format(type(field)))
            slots.append(
                {
                    "slot_index": idx,
                    "role": "{}{}".format(prefix, idx),
                    "shape": tuple(int(x) for x in field.shape),
                    "dtype": field.dtype,
                }
            )
        return {"kind": "tuple", "arity": len(slots), "slots": slots}
    raise TypeError("Unsupported Relay output type {}".format(type(relay_type)))


def get_func_output_info(relay_func):
    relay_func = vta_graphpack.run_opt_pass(relay_func, transform.InferType())
    return _relay_type_to_schema(relay_func.ret_type)


def _pack_stage_inputs(relay_func, cpu_dev, bitpack_start):
    bind_map = {}
    packed_params = []
    for param in relay_func.params:
        cpu_input = relay.annotation.on_device(param, cpu_dev)
        packed_input = relay.Call(bitpack_start, [cpu_input])
        bind_map[param] = packed_input
        packed_params.append(param)
    wrapped_body = relay.expr.bind(relay_func.body, bind_map)
    return packed_params, wrapped_body


def _unpack_stage_outputs(expr, bitpack_end):
    if isinstance(expr, relay.expr.Tuple):
        return relay.Tuple([relay.Call(bitpack_end, [field]) for field in expr.fields])
    return relay.Call(bitpack_end, [expr])


def wrap_stage_with_explicit_pack(relay_func):
    """Treat pack/unpack as an internal detail of the VTA stage.

    This keeps the stage interface in ordinary 4D layout while still letting
    ExprPack convert the internal stage body into packed VTA form.
    """

    bitpack_start = op.op.get("annotation.bitpack_start")
    bitpack_end = op.op.get("annotation.bitpack_end")
    cpu_dev = tvm.device("cpu")

    packed_params, wrapped_body = _pack_stage_inputs(relay_func, cpu_dev, bitpack_start)
    wrapped_body = _unpack_stage_outputs(wrapped_body, bitpack_end)
    wrapped_body = relay.annotation.on_device(wrapped_body, cpu_dev)
    wrapped_func = relay.Function(
        packed_params,
        wrapped_body,
        relay_func.ret_type,
        relay_func.type_params,
        relay_func.attrs,
    )
    return vta_graphpack.run_opt_pass(wrapped_func, transform.InferType())


def annotate_all_ops_to_ext_dev(relay_func):
    """Build stage1 as a hetero graph with CPU-visible inputs and ext_dev body.

    Keep the stage input and the explicit pack path on CPU, then switch to
    ext_dev from the first packed int32 conv2d onward. This lets the stage
    accept the previous CPU stage's output directly on-board.
    """

    relay_func = vta_graphpack.run_opt_pass(relay_func, transform.InferType())
    locator = vta_graphpack.ExprLocator()
    locator.visit(relay_func)
    conv2d = op.op.get("nn.conv2d")
    start_candidates = locator.op2nodes.get((conv2d, "int32"), [])
    start = start_candidates[0] if start_candidates else 0
    end = locator.counter + 2
    annotator = vta_graphpack.ExprDeviceAnnot(start=start, end=end)
    annotated = annotator.visit(relay_func)
    return vta_graphpack.run_opt_pass(annotated, transform.InferType())


def build_vta_stage(stage_name, relay_prog, params, env, use_graph_pack=False):
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

    if use_graph_pack:
        packed = vta_graphpack.graph_pack(
            qmod["main"],
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name="nn.max_pool2d",
            stop_name="nn.global_avg_pool2d",
            device_annot=True,
            annot_start_name="nn.conv2d",
            annot_end_name="annotation.stop_fusion",
        )
        packed = vta_graphpack.run_opt_pass(packed, transform.InferType())
    else:
        packed = wrap_stage_with_explicit_pack(qmod["main"])
        packer = vta_graphpack.ExprPack(env.BATCH, env.BLOCK_OUT, env.WGT_WIDTH)
        packed = packer.visit(packed)
        packed = vta_graphpack.run_opt_pass(packed, transform.InferType())
        packed = annotate_all_ops_to_ext_dev(packed)

    target = env.target
    target_with_host = tvm.target.Target(target, host=env.target_host)
    build_target = {
        "cpu": tvm.target.Target(env.target_vta_cpu, host=env.target_host),
        "ext_dev": target_with_host,
    }
    with vta.build_config(
        opt_level=3,
        disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
    ):
        graph, lib, lowered_params = relay.build(
            packed,
            target=build_target,
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
    if env.TARGET in ["sim", "tsim", "intelfocl"]:
        return rpc.LocalSession()
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = os.environ.get("TVM_TRACKER_PORT", None)
    if not tracker_host or not tracker_port:
        return rpc.connect(host, int(port))
    return autotvm.measure.request_remote(
        env.TARGET, tracker_host, int(tracker_port), timeout=10000
    )


def create_stage_module(stage_name, stage_device, graph, lib, remote):
    if stage_device == "vta":
        ctx = [remote.ext_dev(0), remote.cpu(0)]
    else:
        ctx = remote.cpu(0)
    mod = graph_executor.create(graph, lib, ctx)
    print("[RUN] {} graph executor created on {}".format(stage_name, stage_device))
    return mod, ctx


def timer_device_from_ctx(ctx):
    return ctx[0] if isinstance(ctx, (list, tuple)) else ctx


def get_stage_thread_override(stage_name, args):
    if stage_name == "stage0_cpu" and args.stage0_num_threads > 0:
        return int(args.stage0_num_threads)
    if stage_name == "stage2_cpu" and args.stage2_num_threads > 0:
        return int(args.stage2_num_threads)
    return 0


def maybe_configure_stage_threads(remote, stage_name, args, debug=False, frame_id=None):
    requested = get_stage_thread_override(stage_name, args)
    info = configure_remote_threadpool(remote, num_threads=requested)
    if debug and requested > 0:
        print(
            "[THREADPOOL] frame_id={} stage={} requested={} before={} after={}".format(
                frame_id if frame_id is not None else "-",
                stage_name,
                requested,
                info["runtime_num_threads_before"] if info["runtime_num_threads_before"] is not None else "-",
                info["runtime_num_threads_after"] if info["runtime_num_threads_after"] is not None else "-",
            )
        )
    return info


def prepare_stage_sample_inputs(built_stages, image, remote, args):
    current = image
    sample_inputs = {}
    for stage_info in built_stages:
        stage_name = stage_info["name"]
        sample_inputs[stage_name] = current
        maybe_configure_stage_threads(remote, stage_name, args, debug=False)
        set_stage_inputs(stage_info["module"], stage_info["input_names"], current)
        stage_info["module"].run()
        current = fetch_stage_output(stage_info["module"], stage_info["output_schema"], remote)
    return sample_inputs


def measure_stage_run_timers(built_stages, sample_inputs, args):
    timer_rows = {}
    for stage_info in built_stages:
        stage_name = stage_info["name"]
        remote = stage_info["remote"]
        maybe_configure_stage_threads(remote, stage_name, args, debug=False)
        set_stage_inputs(stage_info["module"], stage_info["input_names"], sample_inputs[stage_name])
        timer = stage_info["module"].module.time_evaluator(
            "run",
            timer_device_from_ctx(stage_info["ctx"]),
            number=max(1, int(args.timer_number)),
            repeat=max(1, int(args.timer_repeat)),
        )
        timer_result = timer()
        run_ms = [float(item) * 1000.0 for item in timer_result.results]
        timer_rows[stage_name] = {
            "mean_ms": float(np.mean(run_ms)),
            "std_ms": float(np.std(run_ms)),
            "min_ms": float(np.min(run_ms)),
            "max_ms": float(np.max(run_ms)),
            "number": int(args.timer_number),
            "repeat": int(args.timer_repeat),
        }
    return timer_rows




def print_serial_stage_timing(rows):
    print("\n[TIMING] serial staged run")
    service_rows = [value for stage_name, value in rows if stage_name.endswith(".service")]
    total = sum(service_rows) if service_rows else 0.0
    for stage_name, value in rows:
        print("\t{:<12}: {:8.3f} ms".format(stage_name, value))
    print("\t{:<12}: {:8.3f} ms".format("total_service", total))


def print_repeat_summary(run_records):
    if not run_records:
        return

    metric_names = []
    for record in run_records:
        for name in record:
            if name not in metric_names:
                metric_names.append(name)

    print("\n[TIMING] repeated serial staged run")
    for name in metric_names:
        values = [record[name] for record in run_records if name in record]
        print(
            "\t{:<16} avg={:8.3f} ms  std={:8.3f} ms".format(
                name, float(np.mean(values)), float(np.std(values))
            )
        )


def fetch_optional_runtime_func(remote, name):
    try:
        return remote.get_function(name)
    except Exception:  # pylint: disable=broad-except
        return None


def configure_remote_threadpool(remote, num_threads=0, affinity_mode=1):
    config_fn = fetch_optional_runtime_func(remote, "runtime.config_threadpool")
    num_threads_fn = fetch_optional_runtime_func(remote, "runtime.NumThreads")
    before = int(num_threads_fn()) if num_threads_fn is not None else None
    if num_threads > 0 and config_fn is not None:
        config_fn(int(affinity_mode), int(num_threads))
    after = int(num_threads_fn()) if num_threads_fn is not None else before
    return {
        "requested_num_threads": int(num_threads) if num_threads > 0 else None,
        "affinity_mode": int(affinity_mode),
        "runtime_num_threads_before": before,
        "runtime_num_threads_after": after,
        "config_available": config_fn is not None,
        "num_threads_available": num_threads_fn is not None,
    }


def _fmt_bytes(value):
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-6:
            value = int(round(value))
        else:
            return "{:.1f}".format(value)
    return str(int(value))


def dtype_nbytes(dtype):
    return int(np.dtype(dtype).itemsize)


def shape_nbytes(shape, dtype):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return int(numel) * dtype_nbytes(dtype)


def schema_nbytes(schema):
    return sum(shape_nbytes(slot["shape"], slot["dtype"]) for slot in schema["slots"])


def build_stage_boundary_info(built_stages):
    boundaries = []
    for idx in range(len(built_stages) - 1):
        src = built_stages[idx]
        dst = built_stages[idx + 1]
        boundaries.append(
            {
                "src_stage": src["name"],
                "dst_stage": dst["name"],
                "schema": src["output_schema"],
                "bytes": schema_nbytes(src["output_schema"]),
            }
        )
    return boundaries


def print_stage_boundary_summary(boundaries):
    print("\n[BOUNDARY] staged tensor interfaces")
    total = 0
    for info in boundaries:
        total += info["bytes"]
        slot_summary = ", ".join(
            "{}:{} {}".format(slot["role"], slot["shape"], slot["dtype"])
            for slot in info["schema"]["slots"]
        )
        print(
            "        {} -> {}  slots=[{}] bytes={}".format(
                info["src_stage"],
                info["dst_stage"],
                slot_summary,
                info["bytes"],
            )
        )
    print("        boundary_count     = {}".format(len(boundaries)))
    print("        total_boundary_bytes = {}".format(total))


def print_vta_runtime_profile(title, stats, divisor=None):
    if not stats:
        return

    print("\n[VTA-RUNTIME] {}".format(title))
    fields = [
        ("input_copy", "input_copy_calls", "input_copy_bytes", "input_copy_us"),
        ("output_copy", "output_copy_calls", "output_copy_bytes", "output_copy_us"),
        ("device_copy", "device_copy_calls", "device_copy_bytes", "device_copy_us"),
        ("flush_cache", "flush_cache_calls", "flush_cache_bytes", "flush_cache_us"),
        (
            "invalidate_cache",
            "invalidate_cache_calls",
            "invalidate_cache_bytes",
            "invalidate_cache_us",
        ),
    ]
    for label, calls_key, bytes_key, us_key in fields:
        calls = int(stats.get(calls_key, 0))
        num_bytes = int(stats.get(bytes_key, 0))
        total_us = float(stats.get(us_key, 0.0))
        avg_calls = float(calls) / divisor if divisor else float(calls)
        avg_bytes = float(num_bytes) / divisor if divisor else float(num_bytes)
        avg_ms = (total_us / divisor) / 1000.0 if divisor else total_us / 1000.0
        print(
            "{:<18} calls={:<6} bytes={:<10} avg_bytes={:<10} time={:8.3f} ms".format(
                label,
                calls,
                _fmt_bytes(num_bytes),
                _fmt_bytes(avg_bytes),
                avg_ms,
            )
        )

    load_calls = int(stats.get("load_buffer_2d_calls", 0))
    load_bytes = int(stats.get("load_buffer_2d_bytes", 0))
    store_calls = int(stats.get("store_buffer_2d_calls", 0))
    store_bytes = int(stats.get("store_buffer_2d_bytes", 0))
    print(
        "{:<18} calls={:<6} bytes={:<10} small={:<6} strided={:<6} padded={:<6}".format(
            "load_buffer_2d",
            load_calls,
            _fmt_bytes(load_bytes),
            int(stats.get("load_buffer_2d_small_calls", 0)),
            int(stats.get("load_buffer_2d_strided_calls", 0)),
            int(stats.get("load_buffer_2d_padded_calls", 0)),
        )
    )
    print(
        "{:<18} x1={:<6} y1={:<6}".format(
            "load_shape",
            int(stats.get("load_buffer_2d_xsize1_calls", 0)),
            int(stats.get("load_buffer_2d_ysize1_calls", 0)),
        )
    )
    print(
        "{:<18} calls={:<6} bytes={:<10} small={:<6} strided={:<6}".format(
            "store_buffer_2d",
            store_calls,
            _fmt_bytes(store_bytes),
            int(stats.get("store_buffer_2d_small_calls", 0)),
            int(stats.get("store_buffer_2d_strided_calls", 0)),
        )
    )
    print(
        "{:<18} x1={:<6} y1={:<6}".format(
            "store_shape",
            int(stats.get("store_buffer_2d_xsize1_calls", 0)),
            int(stats.get("store_buffer_2d_ysize1_calls", 0)),
        )
    )
    avg_load_bytes = (float(load_bytes) / float(load_calls)) if load_calls else 0.0
    avg_store_bytes = (float(store_bytes) / float(store_calls)) if store_calls else 0.0
    print(
        "{:<18} load_avg_bytes={} store_avg_bytes={} load_enqueue_ms={:8.3f} store_enqueue_ms={:8.3f}".format(
            "dma_summary",
            _fmt_bytes(avg_load_bytes),
            _fmt_bytes(avg_store_bytes),
            (float(stats.get("load_buffer_2d_enqueue_us", 0.0)) / (1000.0 * divisor)) if divisor else float(stats.get("load_buffer_2d_enqueue_us", 0.0)) / 1000.0,
            (float(stats.get("store_buffer_2d_enqueue_us", 0.0)) / (1000.0 * divisor)) if divisor else float(stats.get("store_buffer_2d_enqueue_us", 0.0)) / 1000.0,
        )
    )
    gemm_calls = int(stats.get("push_gemm_op_calls", 0))
    alu_calls = int(stats.get("push_alu_op_calls", 0))
    gemm_ms = (float(stats.get("push_gemm_op_us", 0.0)) / (1000.0 * divisor)) if divisor else float(stats.get("push_gemm_op_us", 0.0)) / 1000.0
    alu_ms = (float(stats.get("push_alu_op_us", 0.0)) / (1000.0 * divisor)) if divisor else float(stats.get("push_alu_op_us", 0.0)) / 1000.0
    print(
        "{:<18} gemm_calls={:<8} alu_calls={:<8} gemm_enqueue_ms={:8.3f} alu_enqueue_ms={:8.3f}".format(
            "uop_summary",
            _fmt_bytes(float(gemm_calls) / divisor if divisor else gemm_calls),
            _fmt_bytes(float(alu_calls) / divisor if divisor else alu_calls),
            gemm_ms,
            alu_ms,
        )
    )
    wait_calls = int(stats.get("synchronize_calls", 0))
    wait_insns = int(stats.get("synchronize_insns", 0))
    wait_ms = float(stats.get("device_run_wait_us", 0.0)) / 1000.0
    print(
        "{:<18} calls={:<6} insns={:<8} load_bytes={:<10} store_bytes={:<10} time={:8.3f} ms".format(
            "device_run_wait",
            wait_calls,
            wait_insns,
            _fmt_bytes(float(stats.get("synchronize_load_bytes", 0)) / divisor if divisor else float(stats.get("synchronize_load_bytes", 0))),
            _fmt_bytes(float(stats.get("synchronize_store_bytes", 0)) / divisor if divisor else float(stats.get("synchronize_store_bytes", 0))),
            wait_ms / divisor if divisor else wait_ms,
        )
    )


def extract_vta_runtime_metrics(stats, divisor=None):
    if not stats:
        return {}

    def _scaled_int(key):
        value = float(stats.get(key, 0))
        return float(value) / divisor if divisor else float(value)

    return {
        "input_copy_bytes": _scaled_int("input_copy_bytes"),
        "output_copy_bytes": _scaled_int("output_copy_bytes"),
        "device_copy_bytes": _scaled_int("device_copy_bytes"),
        "load_buffer_2d_bytes": _scaled_int("load_buffer_2d_bytes"),
        "load_buffer_2d_calls": _scaled_int("load_buffer_2d_calls"),
        "store_buffer_2d_bytes": _scaled_int("store_buffer_2d_bytes"),
        "store_buffer_2d_calls": _scaled_int("store_buffer_2d_calls"),
        "load_buffer_2d_small_calls": _scaled_int("load_buffer_2d_small_calls"),
        "store_buffer_2d_small_calls": _scaled_int("store_buffer_2d_small_calls"),
        "load_buffer_2d_xsize1_calls": _scaled_int("load_buffer_2d_xsize1_calls"),
        "load_buffer_2d_ysize1_calls": _scaled_int("load_buffer_2d_ysize1_calls"),
        "store_buffer_2d_xsize1_calls": _scaled_int("store_buffer_2d_xsize1_calls"),
        "store_buffer_2d_ysize1_calls": _scaled_int("store_buffer_2d_ysize1_calls"),
        "load_buffer_2d_enqueue_ms": (_scaled_int("load_buffer_2d_enqueue_us") / 1000.0),
        "store_buffer_2d_enqueue_ms": (_scaled_int("store_buffer_2d_enqueue_us") / 1000.0),
        "push_gemm_op_calls": _scaled_int("push_gemm_op_calls"),
        "push_alu_op_calls": _scaled_int("push_alu_op_calls"),
        "push_gemm_op_ms": (_scaled_int("push_gemm_op_us") / 1000.0),
        "push_alu_op_ms": (_scaled_int("push_alu_op_us") / 1000.0),
        "synchronize_insns": _scaled_int("synchronize_insns"),
        "device_run_wait_ms": (_scaled_int("device_run_wait_us") / 1000.0),
    }


def print_vta_internal_diagnosis(metrics):
    if not metrics:
        return
    load_calls = float(metrics.get("load_buffer_2d_calls", 0.0))
    store_calls = float(metrics.get("store_buffer_2d_calls", 0.0))
    load_bytes = float(metrics.get("load_buffer_2d_bytes", 0.0))
    store_bytes = float(metrics.get("store_buffer_2d_bytes", 0.0))
    load_small = float(metrics.get("load_buffer_2d_small_calls", 0.0))
    store_small = float(metrics.get("store_buffer_2d_small_calls", 0.0))
    avg_load = load_bytes / load_calls if load_calls else 0.0
    avg_store = store_bytes / store_calls if store_calls else 0.0
    load_small_ratio = load_small / load_calls if load_calls else 0.0
    store_small_ratio = store_small / store_calls if store_calls else 0.0
    print("\n[VTA-INTERNAL] single-run diagnosis")
    print(
        "        dma_load_calls={} dma_load_bytes={} dma_load_avg_bytes={}".format(
            _fmt_bytes(load_calls),
            _fmt_bytes(load_bytes),
            _fmt_bytes(avg_load),
        )
    )
    print(
        "        dma_store_calls={} dma_store_bytes={} dma_store_avg_bytes={}".format(
            _fmt_bytes(store_calls),
            _fmt_bytes(store_bytes),
            _fmt_bytes(avg_store),
        )
    )
    print(
        "        small_dma_ratio load={:.3f} store={:.3f} x1/y1 load={}/{} store={}/{}".format(
            load_small_ratio,
            store_small_ratio,
            _fmt_bytes(metrics.get("load_buffer_2d_xsize1_calls", 0.0)),
            _fmt_bytes(metrics.get("load_buffer_2d_ysize1_calls", 0.0)),
            _fmt_bytes(metrics.get("store_buffer_2d_xsize1_calls", 0.0)),
            _fmt_bytes(metrics.get("store_buffer_2d_ysize1_calls", 0.0)),
        )
    )
    print(
        "        enqueue_ms load={:.3f} store={:.3f} gemm={:.3f} alu={:.3f} synchronize_insns={} device_wait_ms={:.3f}".format(
            float(metrics.get("load_buffer_2d_enqueue_ms", 0.0)),
            float(metrics.get("store_buffer_2d_enqueue_ms", 0.0)),
            float(metrics.get("push_gemm_op_ms", 0.0)),
            float(metrics.get("push_alu_op_ms", 0.0)),
            _fmt_bytes(metrics.get("synchronize_insns", 0.0)),
            float(metrics.get("device_run_wait_ms", 0.0)),
        )
    )


def describe_value_location(value):
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(describe_value_location(item) for item in value) + "]"
    if isinstance(value, np.ndarray):
        return "host.numpy"
    if hasattr(value, "device"):
        dev = value.device
        return "{}:{}".format(dev.device_type, dev.device_id)
    return type(value).__name__


def set_stage_inputs(mod, input_names, current):
    if isinstance(current, (tuple, list)):
        if len(current) != len(input_names):
            raise ValueError("Expected {} stage inputs, got {}".format(len(input_names), len(current)))
        for name, value in zip(input_names, current):
            if hasattr(value, "device"):
                try:
                    mod.set_input_zero_copy(name, value)
                except Exception:
                    mod.set_input(name, value)
            else:
                mod.set_input(name, value)
        return
    if len(input_names) != 1:
        raise ValueError("Expected tuple stage input for names {}".format(input_names))
    name = input_names[0]
    if hasattr(current, "device"):
        try:
            mod.set_input_zero_copy(name, current)
        except Exception:
            mod.set_input(name, current)
    else:
        mod.set_input(name, current)


def fetch_stage_output(mod, output_schema, remote):
    outputs = []
    for idx, slot in enumerate(output_schema["slots"]):
        outputs.append(
            mod.get_output(
                idx,
                tvm.nd.empty(slot["shape"], slot["dtype"], remote.cpu(0)),
            )
        )
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def execute_scheme_run(
    args,
    env,
    feature_blocks,
    output_block,
    unit_blocks,
    scheme_cfg,
    resolved_scheme_name,
    remote,
    vta_runtime_profiler_clear,
    vta_runtime_profiler_status,
    profiler_hooks=None,
):
    cpu_target = tvm.target.Target(env.target_vta_cpu, host=env.target_host)
    image = preprocess_image(args.batch, args.image_size) if args.run_stages else None
    built_stages = []
    unit_metadata = build_resnet18_unit_metadata(unit_blocks, args.batch, args.image_size)

    for stage in scheme_cfg:
        stage_name = stage["name"]
        input_schema = stage_input_schema_for_stage(stage, unit_metadata)
        output_schema_static = stage_output_schema_for_stage(stage, unit_metadata)
        stage_inputs = relay_inputs_for_stage(stage, unit_metadata)
        stage_block = make_stage_block(feature_blocks, output_block, stage, unit_blocks=unit_blocks)
        mod, params = lower_stage_to_relay(stage_block, stage_inputs)
        out_schema = get_func_output_info(mod["main"])

        print("\n========== {} ==========".format(stage_name))
        print("[STAGE] device      =", stage["device"])
        print("[STAGE] input_schema =", input_schema)
        print("[STAGE] output_schema=", out_schema)
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
                graph, lib, lowered_params = build_vta_stage(
                    stage_name,
                    mod["main"],
                    params,
                    env,
                    use_graph_pack=(resolved_scheme_name == "all_vta"),
                )

        if args.run_stages:
            remote_lib = export_and_upload(lib, remote, env, stage_name)
            stage_module, stage_ctx = create_stage_module(stage_name, stage["device"], graph, remote_lib, remote)
            stage_module.set_input(**lowered_params)
            built_stages.append(
                {
                    "name": stage_name,
                    "device": stage["device"],
                    "module": stage_module,
                    "ctx": stage_ctx,
                    "remote": remote,
                    "graph": graph,
                    "lib": lib,
                    "lowered_params": lowered_params,
                    "input_names": [name for name, _ in stage_inputs],
                    "input_schema": input_schema,
                    "output_schema": out_schema if out_schema else output_schema_static,
                }
            )

    result = {
        "built_stages": built_stages,
        "boundary_info": [],
        "all_run_records": [],
        "final_rows": [],
        "final_stage_service_rows": [],
        "profile_stats": {},
        "runtime_metrics": {},
        "timer_metrics": {},
        "total_service_avg": None,
        "stage_service_avg": {},
        "boundary_bytes": 0,
    }

    if not args.run_stages:
        return result

    boundary_info = build_stage_boundary_info(built_stages)
    result["boundary_info"] = boundary_info
    result["boundary_bytes"] = sum(item["bytes"] for item in boundary_info)
    print_stage_boundary_summary(boundary_info)
    if env.TARGET in ["sim", "tsim"]:
        simulator.clear_stats()
    if vta_runtime_profiler_clear is not None:
        vta_runtime_profiler_clear()
    print("[RUN] clear profiler after param loading / stage setup")

    def run_stage_iteration(log_inputs=False, fetch_final_host=False):
        rows = []
        record = {}
        stage_service_rows = []
        current = image
        total_service_ms = 0.0
        for stage_info in built_stages:
            mod = stage_info["module"]
            stage_name = stage_info["name"]
            maybe_configure_stage_threads(remote, stage_name, args, debug=False)
            if log_inputs:
                print(
                    "[RUN] {} set_input({}) from {}".format(
                        stage_name,
                        ",".join(stage_info["input_names"]),
                        describe_value_location(current),
                    )
                )
            t_set0 = time.perf_counter()
            set_stage_inputs(mod, stage_info["input_names"], current)
            t0 = time.perf_counter()
            mod.run()
            t1 = time.perf_counter()
            out = fetch_stage_output(mod, stage_info["output_schema"], remote)
            t2 = time.perf_counter()
            set_ms = (t0 - t_set0) * 1000.0
            run_ms = (t1 - t0) * 1000.0
            get_ms = (t2 - t1) * 1000.0
            service_ms = set_ms + run_ms + get_ms
            rows.append((stage_name + ".in", set_ms))
            rows.append((stage_name + ".run", run_ms))
            rows.append((stage_name + ".out", get_ms))
            rows.append((stage_name + ".service", service_ms))
            record[stage_name + ".in"] = set_ms
            record[stage_name + ".run"] = run_ms
            record[stage_name + ".out"] = get_ms
            record[stage_name + ".service"] = service_ms
            record[stage_name] = service_ms
            stage_service_rows.append((stage_name, service_ms))
            total_service_ms += service_ms
            current = out
        if fetch_final_host:
            h0 = time.perf_counter()
            if isinstance(current, (tuple, list)):
                final_output = [item.numpy() for item in current]
                shape_summary = [item.shape for item in final_output]
            else:
                final_output = current.numpy()
                shape_summary = final_output.shape
            h1 = time.perf_counter()
            rows.append(("final_host.copy", (h1 - h0) * 1000.0))
            record["final_host.copy"] = (h1 - h0) * 1000.0
            print("[RUN] final output shape =", shape_summary)
        record["total.service"] = total_service_ms
        return rows, record, stage_service_rows

    if args.warmup_repeat > 0:
        print("[RUN] warmup {} iteration(s) before steady-state profiling ...".format(args.warmup_repeat))
        for warmup_idx in range(args.warmup_repeat):
            run_stage_iteration(log_inputs=(warmup_idx == 0), fetch_final_host=False)
        if env.TARGET in ["sim", "tsim"]:
            simulator.clear_stats()
        if vta_runtime_profiler_clear is not None:
            vta_runtime_profiler_clear()
        print("[RUN] clear profiler after warmup; starting steady-state window")

    print("\n[RUN] serial stage execution (steady-state window) ...")
    all_run_records = []
    final_rows = []
    final_stage_service_rows = []
    for repeat_idx in range(args.repeat):
        if args.repeat > 1:
            print("[RUN] repeat {}/{}".format(repeat_idx + 1, args.repeat))
        rows, record, stage_service_rows = run_stage_iteration(
            log_inputs=(repeat_idx == 0),
            fetch_final_host=args.final_output_to_host and repeat_idx == args.repeat - 1,
        )
        all_run_records.append(record)
        final_rows = rows
        final_stage_service_rows = stage_service_rows

    print_serial_stage_timing(final_rows)
    if args.repeat > 1:
        print_repeat_summary(all_run_records)

    profile_stats = {}
    if vta_runtime_profiler_status is not None:
        profile_stats = json.loads(vta_runtime_profiler_status())
        if args.print_vta_runtime_profile:
            print_vta_runtime_profile("steady-state staged run", profile_stats, divisor=float(args.repeat))
    if args.vta_runtime_profile_dir and hooks_available(profiler_hooks):
        dump_runtime_snapshot(
            args.vta_runtime_profile_dir,
            "{}_steady_state_staged_run".format(resolved_scheme_name),
            profiler_hooks,
            events_limit=args.vta_runtime_profile_events_limit,
            extra={
                "phase": "steady_state_staged_run",
                "scheme": resolved_scheme_name,
                "repeat": int(args.repeat),
                "warmup_repeat": int(args.warmup_repeat),
                "image_size": int(args.image_size),
            },
        )

    result["all_run_records"] = all_run_records
    result["final_rows"] = final_rows
    result["final_stage_service_rows"] = final_stage_service_rows
    result["profile_stats"] = profile_stats
    result["runtime_metrics"] = extract_vta_runtime_metrics(profile_stats, divisor=float(args.repeat))
    if result["runtime_metrics"]:
        print_vta_internal_diagnosis(result["runtime_metrics"])
    sample_inputs = prepare_stage_sample_inputs(built_stages, image, remote, args)
    result["timer_metrics"] = measure_stage_run_timers(built_stages, sample_inputs, args)
    print("\n[TIMER] per-stage pure run via time_evaluator")
    for stage_name, metrics in result["timer_metrics"].items():
        print(
            "        {} run_mean_ms={:.3f} run_std_ms={:.3f} number={} repeat={}".format(
                stage_name,
                metrics["mean_ms"],
                metrics["std_ms"],
                metrics["number"],
                metrics["repeat"],
            )
        )
    result["total_service_avg"] = float(
        np.mean([record["total.service"] for record in all_run_records])
    ) if all_run_records else None
    result["stage_service_avg"] = {
        name: float(np.mean([record[name] for record in all_run_records]))
        for name, _ in final_stage_service_rows
    }
    return result


def probe_scheme_buildability(args, env, feature_blocks, output_block, unit_blocks, scheme_cfg, resolved_scheme_name):
    cpu_target = tvm.target.Target(env.target_vta_cpu, host=env.target_host)
    unit_metadata = build_resnet18_unit_metadata(unit_blocks, args.batch, args.image_size)
    for stage in scheme_cfg:
        stage_inputs = relay_inputs_for_stage(stage, unit_metadata)
        stage_block = make_stage_block(feature_blocks, output_block, stage, unit_blocks=unit_blocks)
        mod, params = lower_stage_to_relay(stage_block, stage_inputs)
        if stage["device"] == "cpu":
            build_cpu_stage(stage["name"], mod, params, cpu_target)
        else:
            build_vta_stage(
                stage["name"],
                mod["main"],
                params,
                env,
                use_graph_pack=(resolved_scheme_name == "all_vta"),
            )
    return True


def classify_buildability_error(err):
    text = str(err)
    if "Tuple" in text and "does not match" in text and "6 dimensions" in text:
        return "tuple_output_type_mismatch"
    if "inject_copy_intrin.cc" in text or "Cannot match copy pattern" in text:
        return "inject_copy_intrin_tail_failure"
    return "other_compile_error"


def build_buildability_probe_set(candidates, max_items=12):
    selected = []
    seen = set()

    def add(candidate):
        if candidate["scheme_name"] in seen:
            return
        seen.add(candidate["scheme_name"])
        selected.append(candidate)

    selectable = [item for item in candidates if not item["hard_reject"]]
    for candidate in selectable[:max_items]:
        add(candidate)

    for special_name in [
        "three_stage_a",
        "three_stage_b",
        "block_stage_c",
        "window_layer2_block0_main_preadd__layer2_block0_add_relu_tail",
        "window_layer2_block1_main_preadd__layer2_block1_add_relu_tail",
        "window_layer3_block0_main_preadd__layer3_block0_add_relu_tail",
        "window_layer3_block1_main_preadd__layer3_block1_add_relu_tail",
        "window_layer4_block0_main_preadd__layer4_block0_add_relu_tail",
        "window_layer4_block1_main_preadd__layer4_block1_add_relu_tail",
    ]:
        special = next((item for item in candidates if item["scheme_name"] == special_name), None)
        if special is not None:
            add(special)
    return selected


def run_buildability_sweep(args, env, feature_blocks, output_block, unit_blocks, candidates, max_items=12):
    rows = []
    probe_set = build_buildability_probe_set(candidates, max_items=max_items)
    print("\n========== Resource-Aware Buildability Sweep ==========")
    print("count =", len(probe_set))
    for idx, candidate in enumerate(probe_set, start=1):
        row = {
            "summary_rank": idx,
            "static_rank": next(
                i + 1 for i, item in enumerate(candidates) if item["scheme_name"] == candidate["scheme_name"]
            ),
            "scheme_name": candidate["scheme_name"],
            "alias_scheme_name": candidate["alias_scheme_name"],
            "static_score": candidate["score"],
            "vta_unit_names": candidate["vta_unit_names"],
            "window_category": classify_vta_window_category(
                candidate["vta_unit_names"], candidate["scheme_name"]
            ),
            "downsample_relation": classify_downsample_window_relation(candidate["vta_unit_names"]),
            "buildable": False,
            "failure_class": "",
            "error": "",
        }
        try:
            probe_scheme_buildability(
                args,
                env,
                feature_blocks,
                output_block,
                unit_blocks,
                candidate["scheme_cfg"],
                candidate["scheme_name"],
            )
            row["buildable"] = True
        except Exception as err:  # pylint: disable=broad-except
            row["failure_class"] = classify_buildability_error(err)
            row["error"] = repr(err)
        rows.append(row)
        print(
            "#{:02d} static_rank={} scheme={} alias={} category={} downsample={} buildable={} failure_class={} vta_units={}".format(
                row["summary_rank"],
                row["static_rank"],
                row["scheme_name"],
                row["alias_scheme_name"] or "-",
                row["window_category"],
                row["downsample_relation"],
                "yes" if row["buildable"] else "no",
                row["failure_class"] or "-",
                row["vta_unit_names"],
            )
        )
        if row["error"]:
            print("    error={}".format(row["error"]))
    print("======================================================")
    print_buildability_category_summary(rows)
    return rows


def select_buildable_main_table_candidates(candidates, sweep_rows, max_items=0):
    by_name = {row["scheme_name"]: row for row in sweep_rows if row["buildable"]}
    selected = []
    seen = set()

    def add(candidate):
        if candidate is None or candidate["scheme_name"] in seen:
            return
        if candidate["scheme_name"] not in by_name:
            return
        seen.add(candidate["scheme_name"])
        selected.append(candidate)

    for candidate in candidates:
        if candidate["scheme_name"] == "all_vta":
            continue
        add(candidate)
        if max_items > 0 and len(selected) >= max_items:
            break
    return selected[:max_items] if max_items > 0 else selected


def print_buildable_main_table(rows):
    if not rows:
        return
    ranked = sorted(
        rows,
        key=lambda item: (
            item.get("pipeline_cycle_run_ms", float("inf")),
            item["total_service_avg"],
        ),
    )
    print("\n========== Buildable Main Table ==========")
    for idx, row in enumerate(ranked, start=1):
        print(
            "#{:02d} scheme={} alias={} category={} downsample={} total_service_avg={:.3f} total_service_std={:.3f} stage0_ms={:.3f} stage1_ms={:.3f} stage2_ms={:.3f} stage0_run_ms={:.3f} stage1_run_ms={:.3f} stage2_run_ms={:.3f} cpu_shared_run_ms={:.3f} pipeline_cycle_run_ms={:.3f} pipeline_throughput_run_fps={:.3f} pipeline_bottleneck_run={} all_vta_run_ms={} vs_all_vta_serial_ratio={} vs_all_vta_pipeline_ratio={} passes_all_vta_gate={} boundary_bytes={} load_bytes={} load_calls={} store_bytes={} store_calls={} load_avg_bytes={} store_avg_bytes={} load_small_calls={} store_small_calls={} vta_units={}".format(
                idx,
                row["scheme_name"],
                row["alias_scheme_name"] or "-",
                row["window_category"],
                row["downsample_relation"],
                row["total_service_avg"],
                row["total_service_std"],
                row["stage0_service_ms"],
                row["stage1_service_ms"],
                row["stage2_service_ms"],
                row.get("stage0_run_ms", 0.0),
                row.get("stage1_run_ms", 0.0),
                row.get("stage2_run_ms", 0.0),
                row.get("cpu_shared_run_ms", 0.0),
                row.get("pipeline_cycle_run_ms", 0.0),
                row.get("pipeline_throughput_run_fps", 0.0),
                row.get("pipeline_bottleneck_run", "-"),
                row.get("all_vta_run_ms", "-"),
                row.get("vs_all_vta_serial_ratio", "-"),
                row.get("vs_all_vta_pipeline_ratio", "-"),
                "yes" if row.get("passes_all_vta_gate") else "no",
                row["boundary_bytes"],
                int(row["load_buffer_2d_bytes"]),
                int(row["load_buffer_2d_calls"]),
                int(row["store_buffer_2d_bytes"]),
                int(row["store_buffer_2d_calls"]),
                _fmt_bytes(row["load_avg_bytes"]),
                _fmt_bytes(row["store_avg_bytes"]),
                int(row["load_small_calls"]),
                int(row["store_small_calls"]),
                row["vta_unit_names"],
            )
        )
    print("==============================================")


def print_buildability_category_summary(rows):
    if not rows:
        return
    grouped = {}
    for row in rows:
        key = (row["window_category"], row["downsample_relation"])
        bucket = grouped.setdefault(key, {"buildable": 0, "unbuildable": 0, "failure_classes": {}})
        if row["buildable"]:
            bucket["buildable"] += 1
        else:
            bucket["unbuildable"] += 1
            if row["failure_class"]:
                bucket["failure_classes"][row["failure_class"]] = (
                    bucket["failure_classes"].get(row["failure_class"], 0) + 1
                )
    print("\n========== Buildability Category Summary ==========")
    for (category, downsample_relation), bucket in sorted(grouped.items()):
        failure_text = ",".join(
            "{}={}".format(name, count)
            for name, count in sorted(bucket["failure_classes"].items())
        ) or "-"
        print(
            "category={} downsample={} buildable={} unbuildable={} failures={}".format(
                category,
                downsample_relation,
                bucket["buildable"],
                bucket["unbuildable"],
                failure_text,
            )
        )
    print("===================================================")


def print_downsample_suitability_summary(buildable_rows, main_table_rows):
    print("\n========== Downsample Suitability Summary ==========")
    buildable_by_group = {}
    for row in buildable_rows:
        rel = row["downsample_relation"]
        buildable_by_group.setdefault(rel, {"buildable": 0, "unbuildable": 0})
        if row["buildable"]:
            buildable_by_group[rel]["buildable"] += 1
        else:
            buildable_by_group[rel]["unbuildable"] += 1
    for rel in sorted(buildable_by_group):
        bucket = buildable_by_group[rel]
        print(
            "buildability downsample={} buildable={} unbuildable={}".format(
                rel, bucket["buildable"], bucket["unbuildable"]
            )
        )

    ranked_groups = {}
    for row in main_table_rows:
        rel = row["downsample_relation"]
        ranked_groups.setdefault(rel, []).append(row)
    for rel in sorted(ranked_groups):
        best = sorted(ranked_groups[rel], key=lambda item: item["total_service_avg"])[0]
        print(
            "best_main_table downsample={} scheme={} category={} total_service_avg={:.3f}".format(
                rel,
                best["scheme_name"],
                best["window_category"],
                best["total_service_avg"],
            )
        )
    print("===================================================")


def _load_single_op_threshold_csv(path):
    rows = []
    with open(path) as f:
        for raw in csv.DictReader(f):
            rows.append(
                {
                    "case_id": raw["case"],
                    "conv_type": raw["conv_type"],
                    "spatial_bucket": raw["spatial_bucket"],
                    "cpu_over_vta_total": float(raw["cpu_over_vta_total"]),
                    "vta_copy_share_pct": float(raw["vta_copy_share_pct"]),
                    "offload_decision": raw["offload_decision"],
                }
            )
    return rows


def _group_single_op_threshold_rows(rows):
    grouped = {}
    for row in rows:
        bucket = grouped.setdefault(
            row["spatial_bucket"],
            {
                "count": 0,
                "avg_cpu_over_vta_total": 0.0,
                "avg_vta_copy_share_pct": 0.0,
                "worth_offloading": 0,
                "borderline": 0,
                "not_worth_offloading": 0,
            },
        )
        bucket["count"] += 1
        bucket["avg_cpu_over_vta_total"] += row["cpu_over_vta_total"]
        bucket["avg_vta_copy_share_pct"] += row["vta_copy_share_pct"]
        bucket[row["offload_decision"]] += 1
    for bucket in grouped.values():
        count = float(bucket["count"])
        bucket["avg_cpu_over_vta_total"] /= count
        bucket["avg_vta_copy_share_pct"] /= count
    return grouped


def _candidate_spatial_range(vta_unit_names, unit_blocks, batch, image_size):
    metadata = build_resnet18_unit_metadata(unit_blocks, batch, image_size)
    first_meta = metadata[vta_unit_names[0]]
    last_meta = metadata[vta_unit_names[-1]]
    in_shape = first_meta["input_shape"]
    out_shape = last_meta["output_shape"]
    return "{}x{} -> {}x{}".format(
        int(in_shape[2]),
        int(in_shape[3]),
        int(out_shape[2]),
        int(out_shape[3]),
    )


def print_single_op_threshold_summary(csv_path, main_table_rows, unit_blocks, batch, image_size):
    if not csv_path or not os.path.exists(csv_path):
        return
    rows = _load_single_op_threshold_csv(csv_path)
    grouped = _group_single_op_threshold_rows(rows)
    ranked_buckets = sorted(
        grouped.items(), key=lambda item: item[1]["avg_cpu_over_vta_total"], reverse=True
    )
    print("\n========== Single-Op Threshold Comparison ==========")
    print("source_csv={}".format(csv_path))
    print("Top spatial buckets from single-op experiment:")
    for spatial_bucket, bucket in ranked_buckets[:5]:
        print(
            "  spatial={} avg_cpu_over_vta_total={:.2f} avg_vta_copy_share_pct={:.1f} worth={} borderline={} not_worth={}".format(
                spatial_bucket,
                bucket["avg_cpu_over_vta_total"],
                bucket["avg_vta_copy_share_pct"],
                bucket["worth_offloading"],
                bucket["borderline"],
                bucket["not_worth_offloading"],
            )
        )
    ranked_rows = sorted(main_table_rows, key=lambda item: item["total_service_avg"])
    print("Top buildable windows vs single-op spatial ranges:")
    for row in ranked_rows[:5]:
        print(
            "  scheme={} category={} spatial_range={} total_service_avg={:.3f}".format(
                row["scheme_name"],
                row["window_category"],
                _candidate_spatial_range(row["vta_unit_names"], unit_blocks, batch, image_size),
                row["total_service_avg"],
            )
        )
    print("====================================================")


def print_resource_aware_calibration_results(rows):
    if not rows:
        return
    print("\n========== Resource-Aware Calibration Results ==========")
    measured_ranked = sorted(
        [row for row in rows if row["status"] == "ok"],
        key=lambda item: item["measured_total_service"],
    )
    measured_rank = {row["scheme_name"]: idx + 1 for idx, row in enumerate(measured_ranked)}
    static_rank = {
        row["scheme_name"]: idx + 1
        for idx, row in enumerate(sorted(rows, key=lambda item: item["static_score"]))
    }
    for row in rows:
        measured = "{:.3f}".format(row["measured_total_service"]) if row["measured_total_service"] is not None else "-"
        wait_ms = "{:.3f}".format(row["device_run_wait_ms"]) if row["device_run_wait_ms"] is not None else "-"
        print(
            "#{:02d} scheme={} alias={} status={} static_rank={} measured_rank={} static_score={} total_service_ms={} stage0_ms={} stage1_ms={} stage2_ms={} device_wait_ms={} load_bytes={} store_bytes={} boundary_bytes={} vta_units={}".format(
                row["summary_rank"],
                row["scheme_name"],
                row["alias_scheme_name"] or "-",
                row["status"],
                static_rank.get(row["scheme_name"], "-"),
                measured_rank.get(row["scheme_name"], "-"),
                row["static_score"],
                measured,
                row["stage0_service_ms"] if row["stage0_service_ms"] is not None else "-",
                row["stage1_service_ms"] if row["stage1_service_ms"] is not None else "-",
                row["stage2_service_ms"] if row["stage2_service_ms"] is not None else "-",
                wait_ms,
                int(row["load_buffer_2d_bytes"]) if row["load_buffer_2d_bytes"] is not None else "-",
                int(row["store_buffer_2d_bytes"]) if row["store_buffer_2d_bytes"] is not None else "-",
                row["boundary_bytes"] if row["boundary_bytes"] is not None else "-",
                row["vta_unit_names"],
            )
        )
        if row["status"] != "ok":
            print("    error={}".format(row["error"]))
    print("=======================================================")


def build_three_stage_run_pipeline_metrics(stage0_run_ms, stage1_run_ms, stage2_run_ms):
    stage0_run_ms = float(stage0_run_ms)
    stage1_run_ms = float(stage1_run_ms)
    stage2_run_ms = float(stage2_run_ms)
    cpu_shared_run_ms = stage0_run_ms + stage2_run_ms
    stage_costs = {
        "cpu_shared": cpu_shared_run_ms,
        "stage1_vta": stage1_run_ms,
    }
    pipeline_cycle_ms = max(stage_costs.values())
    pipeline_fill_drain_ms = stage0_run_ms + stage1_run_ms + stage2_run_ms
    pipeline_throughput_fps = 1000.0 / pipeline_cycle_ms if pipeline_cycle_ms > 0.0 else 0.0
    pipeline_bottleneck = max(stage_costs, key=stage_costs.get)
    return {
        "stage0_run_ms": stage0_run_ms,
        "stage1_run_ms": stage1_run_ms,
        "stage2_run_ms": stage2_run_ms,
        "cpu_shared_run_ms": float(cpu_shared_run_ms),
        "pipeline_cycle_run_ms": float(pipeline_cycle_ms),
        "pipeline_fill_drain_run_ms": float(pipeline_fill_drain_ms),
        "pipeline_throughput_run_fps": float(pipeline_throughput_fps),
        "pipeline_bottleneck_run": pipeline_bottleneck,
    }


def main():
    args = parse_args()
    env = vta.get_env()
    if args.resource_aware_run_calibration_set and args.scheme != AUTO_RESOURCE_AWARE_SCHEME:
        raise ValueError("--resource-aware-run-calibration-set requires --scheme auto_resource_aware")
    if args.resource_aware_run_calibration_set and not args.run_stages:
        raise ValueError("--resource-aware-run-calibration-set requires --run-stages")
    if args.resource_aware_buildability_sweep and args.scheme != AUTO_RESOURCE_AWARE_SCHEME:
        raise ValueError("--resource-aware-buildability-sweep requires --scheme auto_resource_aware")
    if args.resource_aware_run_buildable_main_table and args.scheme != AUTO_RESOURCE_AWARE_SCHEME:
        raise ValueError("--resource-aware-run-buildable-main-table requires --scheme auto_resource_aware")
    if args.resource_aware_run_buildable_main_table and not args.run_stages:
        raise ValueError("--resource-aware-run-buildable-main-table requires --run-stages")

    full_model = vision.get_model(args.model, pretrained=True)
    feature_blocks = list(full_model.features)
    output_block = full_model.output
    unit_blocks = build_resnet18_unit_blocks(feature_blocks, output_block)
    scheme_cfg, selected, candidates = resolve_scheme_config(
        args.scheme,
        feature_blocks,
        output_block,
        args.batch,
        args.image_size,
        print_resource_aware=(args.scheme == AUTO_RESOURCE_AWARE_SCHEME),
        selected_candidate_name=args.resource_aware_candidate_name or None,
        calibration_report=args.resource_aware_calibration_report,
        candidate_top_k=args.resource_aware_top_k,
    )
    if (
        args.scheme == AUTO_RESOURCE_AWARE_SCHEME
        and args.vta_stage_mode == "vta_build"
        and not args.resource_aware_candidate_name
        and not args.resource_aware_buildability_sweep
        and not args.resource_aware_run_buildable_main_table
        and not args.resource_aware_run_calibration_set
    ):
        buildable_selected = None
        buildable_error = None
        for candidate in [item for item in candidates if not item["hard_reject"]]:
            try:
                probe_scheme_buildability(
                    args,
                    env,
                    feature_blocks,
                    output_block,
                    unit_blocks,
                    candidate["scheme_cfg"],
                    candidate["scheme_name"],
                )
                buildable_selected = candidate
                break
            except Exception as err:  # pylint: disable=broad-except
                buildable_error = repr(err)
                print(
                    "[AUTO-RESOURCE] buildability probe rejected {}: {}".format(
                        candidate["scheme_name"], buildable_error
                    )
                )
        if buildable_selected is None:
            raise RuntimeError(
                "auto_resource_aware found no buildable candidate; last_error={}".format(
                    buildable_error or "-"
                )
            )
        if selected is None or buildable_selected["scheme_name"] != selected["scheme_name"]:
            print(
                "[AUTO-RESOURCE] switching from static_top1={} to first_buildable={} under current stack constraints".format(
                    selected["scheme_name"] if selected is not None else "-",
                    buildable_selected["scheme_name"],
                )
            )
        selected = buildable_selected
        scheme_cfg = selected["scheme_cfg"]
    resolved_scheme_name = selected["scheme_name"] if selected is not None else args.scheme
    remote = None
    vta_runtime_profiler_clear = None
    vta_runtime_profiler_status = None
    profiler_hooks = None

    if args.run_stages:
        print("[RPC] connecting directly to {}:{} ...".format(args.host, args.port))
        remote = connect_remote(env, args.host, args.port)
        threadpool_info = configure_remote_threadpool(remote, num_threads=args.runtime_num_threads)
        print(
            "[RPC] runtime thread pool: requested={} before={} after={} config_available={} num_threads_available={}".format(
                threadpool_info["requested_num_threads"] if threadpool_info["requested_num_threads"] is not None else "-",
                threadpool_info["runtime_num_threads_before"] if threadpool_info["runtime_num_threads_before"] is not None else "-",
                threadpool_info["runtime_num_threads_after"] if threadpool_info["runtime_num_threads_after"] is not None else "-",
                "yes" if threadpool_info["config_available"] else "no",
                "yes" if threadpool_info["num_threads_available"] else "no",
            )
        )
        profiler_hooks = fetch_runtime_profiler_hooks(remote)
        vta_runtime_profiler_clear = profiler_hooks.get("clear")
        vta_runtime_profiler_status = profiler_hooks.get("status")

    if args.resource_aware_run_calibration_set:
        calibration_candidates = build_resource_aware_calibration_set(
            candidates, max_items=args.resource_aware_calibration_max_items
        )
        calibration_rows = []
        for idx, candidate in enumerate(calibration_candidates, start=1):
            print("\n\n========== Calibration Candidate {}/{} ==========".format(idx, len(calibration_candidates)))
            print("[CALIB] scheme={} alias={} static_score={} vta_units={}".format(
                candidate["scheme_name"],
                candidate["alias_scheme_name"] or "-",
                candidate["score"],
                candidate["vta_unit_names"],
            ))
            summarize_scheme(candidate["scheme_name"], candidate["scheme_cfg"], args.batch, args.image_size)
            summarize_feature_blocks(feature_blocks, output_block)
            summarize_units(unit_blocks)
            validate_scheme(feature_blocks, candidate["scheme_cfg"], unit_blocks)
            row = {
                "summary_rank": idx,
                "scheme_name": candidate["scheme_name"],
                "alias_scheme_name": candidate["alias_scheme_name"],
                "vta_unit_names": candidate["vta_unit_names"],
                "static_score": candidate["score"],
                "status": "ok",
                "measured_total_service": None,
                "stage0_service_ms": None,
                "stage1_service_ms": None,
                "stage2_service_ms": None,
                "device_run_wait_ms": None,
                "load_buffer_2d_bytes": None,
                "store_buffer_2d_bytes": None,
                "boundary_bytes": None,
                "error": "",
            }
            try:
                run_result = execute_scheme_run(
                    args,
                    env,
                    feature_blocks,
                    output_block,
                    unit_blocks,
                    candidate["scheme_cfg"],
                    candidate["scheme_name"],
                    remote,
                    vta_runtime_profiler_clear,
                    vta_runtime_profiler_status,
                    profiler_hooks,
                )
                row["measured_total_service"] = run_result["total_service_avg"]
                row["stage0_service_ms"] = run_result["stage_service_avg"].get("stage0_cpu")
                row["stage1_service_ms"] = (
                    run_result["stage_service_avg"].get("stage1_vta")
                    or run_result["stage_service_avg"].get("stage0_vta")
                )
                row["stage2_service_ms"] = (
                    run_result["stage_service_avg"].get("stage2_cpu")
                    or run_result["stage_service_avg"].get("stage1_cpu")
                )
                row["device_run_wait_ms"] = run_result["runtime_metrics"].get("device_run_wait_ms")
                row["load_buffer_2d_bytes"] = run_result["runtime_metrics"].get("load_buffer_2d_bytes")
                row["store_buffer_2d_bytes"] = run_result["runtime_metrics"].get("store_buffer_2d_bytes")
                row["boundary_bytes"] = run_result["boundary_bytes"]
            except Exception as err:  # pylint: disable=broad-except
                row["status"] = "error"
                row["error"] = repr(err)
                print("[CALIB] build/run failed:", repr(err))
            calibration_rows.append(row)
        print_resource_aware_calibration_results(calibration_rows)
        return

    if args.resource_aware_buildability_sweep:
        run_buildability_sweep(
            args,
            env,
            feature_blocks,
            output_block,
            unit_blocks,
            candidates,
            max_items=args.resource_aware_buildability_max_items,
        )
        return

    if args.resource_aware_run_buildable_main_table:
        sweep_rows = run_buildability_sweep(
            args,
            env,
            feature_blocks,
            output_block,
            unit_blocks,
            candidates,
            max_items=args.resource_aware_buildability_max_items,
        )
        main_candidates = select_buildable_main_table_candidates(
            candidates,
            sweep_rows,
            max_items=args.resource_aware_main_table_max_items,
        )
        all_vta_candidate = next((item for item in candidates if item["scheme_name"] == "all_vta"), None)
        all_vta_buildable = next(
            (
                row
                for row in sweep_rows
                if row["scheme_name"] == "all_vta" and row["buildable"]
            ),
            None,
        )
        all_vta_baseline = None
        if all_vta_candidate is not None and all_vta_buildable is not None:
            print("\n\n========== all_vta Baseline ==========")
            print("[BASELINE] scheme=all_vta static_score={} vta_units={}".format(
                all_vta_candidate["score"],
                all_vta_candidate["vta_unit_names"],
            ))
            summarize_scheme(all_vta_candidate["scheme_name"], all_vta_candidate["scheme_cfg"], args.batch, args.image_size)
            summarize_feature_blocks(feature_blocks, output_block)
            summarize_units(unit_blocks)
            validate_scheme(feature_blocks, all_vta_candidate["scheme_cfg"], unit_blocks)
            baseline_result = execute_scheme_run(
                args,
                env,
                feature_blocks,
                output_block,
                unit_blocks,
                all_vta_candidate["scheme_cfg"],
                all_vta_candidate["scheme_name"],
                remote,
                vta_runtime_profiler_clear,
                vta_runtime_profiler_status,
                profiler_hooks,
            )
            baseline_stage1_ms = float(
                baseline_result["stage_service_avg"].get(
                    "stage1_vta",
                    baseline_result["stage_service_avg"].get("stage0_vta", 0.0),
                )
            )
            baseline_run_ms = float(
                baseline_result["timer_metrics"].get("stage0_vta", {}).get("mean_ms", baseline_stage1_ms)
            )
            all_vta_baseline = {
                "serial_total_ms": float(baseline_result["total_service_avg"]),
                "run_ms": float(baseline_run_ms),
            }
            print(
                "[BASELINE] all_vta serial_total_ms={:.3f} run_ms={:.3f}".format(
                    all_vta_baseline["serial_total_ms"],
                    all_vta_baseline["run_ms"],
                )
            )
        table_rows = []
        for idx, candidate in enumerate(main_candidates, start=1):
            print("\n\n========== Buildable Main Table Candidate {}/{} ==========".format(idx, len(main_candidates)))
            print("[MAIN-TABLE] scheme={} alias={} static_score={} vta_units={}".format(
                candidate["scheme_name"],
                candidate["alias_scheme_name"] or "-",
                candidate["score"],
                candidate["vta_unit_names"],
            ))
            summarize_scheme(candidate["scheme_name"], candidate["scheme_cfg"], args.batch, args.image_size)
            summarize_feature_blocks(feature_blocks, output_block)
            summarize_units(unit_blocks)
            validate_scheme(feature_blocks, candidate["scheme_cfg"], unit_blocks)
            run_result = execute_scheme_run(
                args,
                env,
                feature_blocks,
                output_block,
                unit_blocks,
                candidate["scheme_cfg"],
                candidate["scheme_name"],
                remote,
                vta_runtime_profiler_clear,
                vta_runtime_profiler_status,
                profiler_hooks,
            )
            metrics = run_result["runtime_metrics"]
            load_calls = float(metrics.get("load_buffer_2d_calls", 0.0))
            store_calls = float(metrics.get("store_buffer_2d_calls", 0.0))
            load_bytes = float(metrics.get("load_buffer_2d_bytes", 0.0))
            store_bytes = float(metrics.get("store_buffer_2d_bytes", 0.0))
            stage0_service_ms = float(run_result["stage_service_avg"].get("stage0_cpu", 0.0))
            stage1_service_ms = float(
                run_result["stage_service_avg"].get("stage1_vta", run_result["stage_service_avg"].get("stage0_vta", 0.0))
            )
            stage2_service_ms = float(
                run_result["stage_service_avg"].get("stage2_cpu", run_result["stage_service_avg"].get("stage1_cpu", 0.0))
            )
            timer_metrics = run_result.get("timer_metrics", {})
            stage0_run_ms = float(timer_metrics.get("stage0_cpu", {}).get("mean_ms", 0.0))
            stage1_run_ms = float(
                timer_metrics.get("stage1_vta", timer_metrics.get("stage0_vta", {})).get("mean_ms", 0.0)
            )
            stage2_run_ms = float(timer_metrics.get("stage2_cpu", timer_metrics.get("stage1_cpu", {})).get("mean_ms", 0.0))
            run_pipeline_metrics = build_three_stage_run_pipeline_metrics(
                stage0_run_ms,
                stage1_run_ms,
                stage2_run_ms,
            )
            vs_all_vta_serial_ratio = None
            vs_all_vta_pipeline_ratio = None
            passes_all_vta_gate = False
            if all_vta_baseline is not None and all_vta_baseline["serial_total_ms"] > 0.0:
                vs_all_vta_serial_ratio = float(run_result["total_service_avg"]) / all_vta_baseline["serial_total_ms"]
                if all_vta_baseline["run_ms"] > 0.0:
                    vs_all_vta_pipeline_ratio = (
                        run_pipeline_metrics["pipeline_cycle_run_ms"] / all_vta_baseline["run_ms"]
                    )
                    passes_all_vta_gate = bool(run_pipeline_metrics["pipeline_cycle_run_ms"] <= all_vta_baseline["run_ms"])
            table_rows.append(
                {
                    "scheme_name": candidate["scheme_name"],
                    "alias_scheme_name": candidate["alias_scheme_name"],
                    "window_category": classify_vta_window_category(
                        candidate["vta_unit_names"], candidate["scheme_name"]
                    ),
                    "downsample_relation": classify_downsample_window_relation(
                        candidate["vta_unit_names"]
                    ),
                    "vta_unit_names": candidate["vta_unit_names"],
                    "start_idx": candidate.get("start_idx", -1),
                    "end_idx": candidate.get("end_idx", -1),
                    "total_service_avg": float(run_result["total_service_avg"]),
                    "total_service_std": float(
                        np.std([record["total.service"] for record in run_result["all_run_records"]])
                    ),
                    "stage0_service_ms": stage0_service_ms,
                    "stage1_service_ms": stage1_service_ms,
                    "stage2_service_ms": stage2_service_ms,
                    "stage0_run_ms": float(stage0_run_ms),
                    "stage1_run_ms": float(stage1_run_ms),
                    "stage2_run_ms": float(stage2_run_ms),
                    "cpu_shared_run_ms": float(run_pipeline_metrics["cpu_shared_run_ms"]),
                    "pipeline_cycle_run_ms": float(run_pipeline_metrics["pipeline_cycle_run_ms"]),
                    "pipeline_fill_drain_run_ms": float(run_pipeline_metrics["pipeline_fill_drain_run_ms"]),
                    "pipeline_throughput_run_fps": float(run_pipeline_metrics["pipeline_throughput_run_fps"]),
                    "pipeline_bottleneck_run": run_pipeline_metrics["pipeline_bottleneck_run"],
                    "all_vta_serial_total_ms": all_vta_baseline["serial_total_ms"] if all_vta_baseline is not None else None,
                    "all_vta_run_ms": all_vta_baseline["run_ms"] if all_vta_baseline is not None else None,
                    "vs_all_vta_serial_ratio": "{:.3f}".format(vs_all_vta_serial_ratio) if vs_all_vta_serial_ratio is not None else "-",
                    "vs_all_vta_pipeline_ratio": "{:.3f}".format(vs_all_vta_pipeline_ratio) if vs_all_vta_pipeline_ratio is not None else "-",
                    "passes_all_vta_gate": bool(passes_all_vta_gate),
                    "boundary_bytes": int(run_result["boundary_bytes"]),
                    "load_buffer_2d_bytes": load_bytes,
                    "load_buffer_2d_calls": load_calls,
                    "store_buffer_2d_bytes": store_bytes,
                    "store_buffer_2d_calls": store_calls,
                    "load_avg_bytes": load_bytes / load_calls if load_calls else 0.0,
                    "store_avg_bytes": store_bytes / store_calls if store_calls else 0.0,
                    "load_small_calls": float(metrics.get("load_buffer_2d_small_calls", 0.0)),
                    "store_small_calls": float(metrics.get("store_buffer_2d_small_calls", 0.0)),
                }
            )
        print_buildable_main_table(table_rows)
        print_downsample_suitability_summary(sweep_rows, table_rows)
        print_single_op_threshold_summary(
            args.single_op_threshold_csv,
            table_rows,
            unit_blocks,
            args.batch,
            args.image_size,
        )
        return

    summarize_scheme(args.scheme, scheme_cfg, args.batch, args.image_size)
    summarize_feature_blocks(feature_blocks, output_block)
    summarize_units(unit_blocks)
    validate_scheme(feature_blocks, scheme_cfg, unit_blocks)
    if args.save_partition_viz:
        unit_metadata = build_resnet18_unit_metadata(unit_blocks, args.batch, args.image_size)
        save_partition_visualization(
            args.save_partition_viz,
            resolved_scheme_name,
            scheme_cfg,
            unit_metadata,
            selected=selected,
        )
        print("[VIZ] saved partition visualization ->", args.save_partition_viz)
    execute_scheme_run(
        args,
        env,
        feature_blocks,
        output_block,
        unit_blocks,
        scheme_cfg,
        resolved_scheme_name,
        remote,
        vta_runtime_profiler_clear,
        vta_runtime_profiler_status,
        profiler_hooks,
    )


if __name__ == "__main__":
    main()
