#!/usr/bin/env python3
"""Theory-guided YOLOv3-tiny native CPU/VTA/CPU split search.

This script is intentionally separate from deploy_detection.py.  It uses the
ResNet18 measured split results as a cross-model heuristic prior, then tries a
small set of coarse YOLOv3-tiny Relay splits through the native stage runner.
"""

from __future__ import absolute_import, print_function

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import tarfile
import time

import numpy as np
import tvm
import vta
from tvm import autotvm, relay
from tvm.relay import transform
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.testing.darknet import __darknetffi__

from vta.top import graphpack as vta_graphpack

from test_yolov3_tiny_pipeline_baseline import (  # pylint: disable=import-error
    MODEL_NAME,
    board_host,
    compare_raw_outputs,
    compile_runner,
    copy_runtime_libs,
    detection_summary,
    download_yolo_assets,
    export_shared_lib,
    graph_factory_parts,
    prepare_input_data,
    run_cmd,
    ssh_options,
    ssh_target,
    write_csv,
    write_json,
)
from profile_split_resnet18_stages import wrap_stage_with_explicit_pack  # pylint: disable=import-error


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


DEFAULT_OUTPUT_ROOT = (
    "vta/tutorials/frontend/report_out/yolov3_tiny_heuristic_search/"
    "20260512_resnetfit20"
)
DEFAULT_HEURISTIC_JSON = (
    "vta/tutorials/frontend/report_out/yolov3_tiny_heuristic_search/"
    "heuristic_params_resnet18_v23_200.json"
)
DEFAULT_RPC_BASELINE = (
    "vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/"
    "20260512_rpc_all_vta_runs20/rpc_all_vta_baseline.json"
)
DEFAULT_PACKED_GRAPH_PACKAGE = (
    "vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/"
    "20260512_native_serial_smoke/native_package"
)
RESNET_TRAIN_ROOTS = [
    "vta/tutorials/frontend/report_out/native_stage_pipeline_searches/20260507_v23_warm100",
    "vta/tutorials/frontend/report_out/native_stage_pipeline_searches/20260512_theory_build100_board",
]
YOLO_CONVS = [
    # idx, name, cin, cout, h, w, kernel
    (0, "conv0", 3, 16, 416, 416, 3),
    (1, "conv2", 16, 32, 208, 208, 3),
    (2, "conv4", 32, 64, 104, 104, 3),
    (3, "conv6", 64, 128, 52, 52, 3),
    (4, "conv8", 128, 256, 26, 26, 3),
    (5, "conv10", 256, 512, 13, 13, 3),
    (6, "conv12", 512, 1024, 13, 13, 3),
    (7, "conv13", 1024, 256, 13, 13, 1),
    (8, "conv18", 256, 128, 13, 13, 1),
    (9, "conv21", 384, 256, 26, 26, 3),
    (10, "conv22", 256, 255, 26, 26, 1),
    (11, "conv14", 256, 512, 13, 13, 3),
    (12, "conv15", 512, 255, 13, 13, 1),
]
START_SPECS = [
    ("data", -1, []),
    ("pool0", 0, [0]),
    ("pool1", 1, [0, 1]),
    ("pool2", 2, [0, 1, 2]),
    ("pool3", 3, [0, 1, 2, 3]),
    ("pool4_route", 4, [0, 1, 2, 3, 4]),
]
END_SPECS = [
    ("trunk12", [0, 1, 2, 3, 4, 5, 6], ["route23", "trunk34"]),
    ("shared13", [0, 1, 2, 3, 4, 5, 6, 7], ["route23", "shared38"]),
    ("small_pre18", [0, 1, 2, 3, 4, 5, 6, 7, 8], ["route23", "shared38", "small42"]),
    ("dual_pre18_14", [0, 1, 2, 3, 4, 5, 6, 7, 8, 11], ["route23", "small42", "big64"]),
    (
        "dual_pre_logits",
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11],
        ["small_pre26", "big64"],
    ),
    (
        "logits",
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ["small_logits51", "big_logits66"],
    ),
]


def parse_modes(text):
    modes = []
    valid = {"fit", "build", "measure"}
    for item in text.split(","):
        item = item.strip()
        if item:
            if item not in valid:
                raise argparse.ArgumentTypeError("unknown mode {}".format(item))
            modes.append(item)
    return modes or ["fit", "build", "measure"]


def parse_int_csv(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=parse_modes, default=parse_modes("fit,build,measure"))
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--heuristic-params-json", default=DEFAULT_HEURISTIC_JSON)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument(
        "--candidate-id",
        default="",
        help="Optional exact candidate id to build/measure from the enumerated candidate set.",
    )
    parser.add_argument(
        "--measure-count",
        type=int,
        default=20,
        help="Measure at most this many buildable candidates; candidate-count may be larger.",
    )
    parser.add_argument("--board", default="root@192.168.1.185")
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--rpc-baseline-json", default=DEFAULT_RPC_BASELINE)
    parser.add_argument("--build-cache-dir", default="")
    parser.add_argument(
        "--split-quantization-mode",
        default="full_graph",
        choices=["full_graph", "per_stage"],
        help=(
            "full_graph quantizes YOLO once before extracting stages; per_stage keeps "
            "the old behavior and quantizes each VTA stage independently."
        ),
    )
    parser.add_argument(
        "--tail-device",
        default="cpu",
        choices=["cpu", "vta"],
        help="Device used for the YOLO tail stage. Use vta to preserve all-VTA quantized tail semantics.",
    )
    parser.add_argument("--remote-dir", default="/var/volatile/yolov3_tiny_pipeline_search")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--serial-runs", type=int, default=2)
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--runtime-num-threads", type=int, default=4)
    parser.add_argument(
        "--runtime-config-search-count",
        type=int,
        default=0,
        help="Expand Relay split candidates into this many split+runtime configs.",
    )
    parser.add_argument("--max-runtime-configs-per-split", type=int, default=6)
    parser.add_argument("--stage0-thread-options", default="2,3,4")
    parser.add_argument("--stage2-thread-options", default="2,3,4")
    parser.add_argument("--queue-depth-options", default="1,2")
    parser.add_argument("--poll-sleep-ns-options", default="1000,5000")
    parser.add_argument("--post-start-sleep-ns-options", default="1000")
    parser.add_argument("--threshold", type=float, default=0.560)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--topk-detections", type=int, default=20)
    parser.add_argument(
        "--correctness-policy",
        default="detection_gate",
        choices=["raw_gate", "detection_gate"],
        help="Gate measured YOLO candidates by raw tensor summaries or decoded detection results.",
    )
    parser.add_argument(
        "--runner-output-mode",
        default="raw",
        choices=["raw", "raw_all_stages"],
        help="Use raw_all_stages for generic split-boundary diagnostics.",
    )
    parser.add_argument(
        "--split-backend",
        default="relay",
        choices=["relay", "packed_graph", "packed_hetero"],
        help=(
            "relay builds each stage from Relay expressions. packed_graph splits an "
            "already-correct all-VTA graph JSON, preserving packed layout, dtype, "
            "padding, quantization and concat/upsample semantics by construction."
        ),
    )
    parser.add_argument(
        "--packed-graph-package",
        default=DEFAULT_PACKED_GRAPH_PACKAGE,
        help="Native all-VTA package used by --split-backend packed_graph.",
    )
    parser.add_argument(
        "--packed-graph-preset",
        default="route_shared_tail",
        choices=["route_shared_tail"],
        help="Named physical tensor boundary preset for packed_graph splitting.",
    )
    parser.add_argument("--serial-timeout-s", type=int, default=240)
    parser.add_argument("--pipeline-timeout-s", type=int, default=600)
    parser.add_argument("--fetch-timeout-s", type=int, default=120)
    parser.add_argument("--ssh-option", action="append", default=[])
    parser.add_argument("--cfg-path", default="")
    parser.add_argument("--weights-path", default="/tmp/tvm_test_data/darknet/yolov3-tiny.weights")
    parser.add_argument("--darknet-lib-path", default="")
    parser.add_argument("--coco-path", default="")
    parser.add_argument("--font-path", default="")
    parser.add_argument("--image-path", default="")
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def safe_name(text):
    return "".join(c if c.isalnum() or c in "_.-" else "_" for c in str(text)).strip("_")


def conv_ops(conv_indices):
    total = 0.0
    for idx in conv_indices:
        _, _, cin, cout, h, w, kernel = YOLO_CONVS[idx]
        total += float(2 * h * w * cin * cout * kernel * kernel)
    return total


def conv_output_bytes(conv_idx):
    _, _, _, cout, h, w, _ = YOLO_CONVS[conv_idx]
    return int(1 * cout * h * w * 4)


def resnet_rows():
    rows = []
    for root_text in RESNET_TRAIN_ROOTS:
        root = Path(root_text)
        for path in sorted(root.glob("batch*/summary.json")):
            payload = json.load(open(path, encoding="utf-8"))
            for row in payload.get("rows", []) or []:
                fps = float(row.get("pipeline_throughput_fps") or 0.0)
                if row.get("status") == "ok" and fps > 0:
                    rows.append(row)
    return rows


def parse_json_field(value, default):
    if isinstance(value, str) and value.strip().startswith(("{", "[")):
        try:
            return json.loads(value)
        except Exception:
            return default
    return value if value else default


def features_from_resnet_row(row):
    comps = parse_json_field(row.get("score_components_json"), {}) or {}
    stages = comps.get("stages", []) or []
    cpu_ms = [float(s.get("static_ms_est", 0.0)) for s in stages if s.get("device") == "cpu"]
    vta_ms = [float(s.get("static_ms_est", 0.0)) for s in stages if s.get("device") == "vta"]
    vta_dma = [
        float(s.get("static_vta_dma_ms_est", 0.0)) for s in stages if s.get("device") == "vta"
    ]
    devs = [s.get("device") for s in stages]
    eff = []
    for dev in devs:
        if not eff or eff[-1] != dev:
            eff.append(dev)
    islands = parse_json_field(row.get("vta_islands"), []) or []
    lengths = [
        int(item.get("end_idx", 0)) - int(item.get("start_idx", 0)) + 1 for item in islands
    ]
    pattern = "/".join(eff)
    return {
        "intercept": 1.0,
        "resource_cycle_ms": max(max(cpu_ms) if cpu_ms else 0.0, sum(vta_ms)),
        "vta_dma_ms": sum(vta_dma),
        "boundary_ms": float(comps.get("boundary_penalty_ms", row.get("boundary_penalty_ms_est", 0))),
        "dma_fragmentation_ms": float(
            comps.get("dma_fragmentation_penalty_ms", row.get("dma_fragmentation_penalty_ms_est", 0))
        ),
        "sram_risk_ms": float(comps.get("sram_risk_penalty_ms", row.get("sram_risk_penalty_ms_est", 0))),
        "tail_cpu_ms": float(stages[-1].get("static_ms_est", 0.0))
        if stages and stages[-1].get("device") == "cpu"
        else 0.0,
        "stage_count": float(row.get("stage_count") or len(stages)),
        "effective_segment_count": float(len(eff)),
        "small_island_count": float(sum(1 for length in lengths if length <= 3)),
        "pattern_cpu_vta_cpu": 1.0 if pattern == "cpu/vta/cpu" else 0.0,
        "pattern_cpu_vta_cpu_vta_cpu": 1.0 if pattern == "cpu/vta/cpu/vta/cpu" else 0.0,
    }


def fit_heuristic(output_root, heuristic_json):
    rows = resnet_rows()
    if len(rows) < 20:
        raise RuntimeError("expected ResNet18 measured rows, found {}".format(len(rows)))
    feature_names = [
        "intercept",
        "resource_cycle_ms",
        "vta_dma_ms",
        "boundary_ms",
        "dma_fragmentation_ms",
        "sram_risk_ms",
        "tail_cpu_ms",
        "stage_count",
        "effective_segment_count",
        "small_island_count",
        "pattern_cpu_vta_cpu",
        "pattern_cpu_vta_cpu_vta_cpu",
    ]
    data = []
    for row in rows:
        fps = float(row.get("pipeline_throughput_fps"))
        data.append((1000.0 / fps, features_from_resnet_row(row), row))
    x = np.array([[feat[name] for name in feature_names] for _, feat, _ in data], dtype="float64")
    y = np.array([target for target, _, _ in data], dtype="float64")
    means = np.zeros(x.shape[1])
    scales = np.ones(x.shape[1])
    xs = x.copy()
    for idx, name in enumerate(feature_names):
        if name == "intercept":
            continue
        means[idx] = xs[:, idx].mean()
        scales[idx] = xs[:, idx].std() or 1.0
        xs[:, idx] = (xs[:, idx] - means[idx]) / scales[idx]
    # Robust ridge: fit, trim largest residuals once, refit.
    reg = 1.0
    coef = np.linalg.solve(xs.T.dot(xs) + reg * np.eye(xs.shape[1]), xs.T.dot(y))
    residual = np.abs(xs.dot(coef) - y)
    keep = residual <= np.percentile(residual, 85.0)
    coef = np.linalg.solve(xs[keep].T.dot(xs[keep]) + reg * np.eye(xs.shape[1]), xs[keep].T.dot(y[keep]))
    pred = xs.dot(coef)
    original = {}
    intercept = float(coef[0])
    for idx, name in enumerate(feature_names):
        if name == "intercept":
            continue
        weight = float(coef[idx] / scales[idx])
        original[name] = max(0.0, weight) if name not in ("pattern_cpu_vta_cpu", "pattern_cpu_vta_cpu_vta_cpu") else weight
        intercept -= weight * means[idx]
    original["intercept"] = float(intercept)
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(math.sqrt(np.mean((pred - y) ** 2)))
    actual_top20 = set(np.argsort(y)[:20].tolist())
    pred_order = np.argsort(pred)
    static_order = np.argsort([float(row.get("static_score_ms") or 1e9) for _, _, row in data])
    metrics = {
        "sample_count": len(rows),
        "mae_ms": mae,
        "rmse_ms": rmse,
        "pred_top20_recall_at_50": len(set(pred_order[:50].tolist()) & actual_top20) / 20.0,
        "static_top20_recall_at_50": len(set(static_order[:50].tolist()) & actual_top20) / 20.0,
    }
    payload = {
        "kind": "resnet18_v23_200_fitted_heuristic",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "target": "measured_cycle_ms=1000/pipeline_throughput_fps",
        "feature_names": feature_names,
        "weights": original,
        "metrics": metrics,
        "training_roots": RESNET_TRAIN_ROOTS,
    }
    write_json(heuristic_json, payload)
    readme = [
        "# ResNet18 V2.3 200-Result Heuristic Fit",
        "",
        "- Samples: {}".format(metrics["sample_count"]),
        "- MAE ms: {:.3f}".format(metrics["mae_ms"]),
        "- RMSE ms: {:.3f}".format(metrics["rmse_ms"]),
        "- Pred Top20 recall@50: {:.3f}".format(metrics["pred_top20_recall_at_50"]),
        "- Static Top20 recall@50: {:.3f}".format(metrics["static_top20_recall_at_50"]),
        "",
        "The JSON weights are used as a cross-model prior for YOLOv3-tiny coarse splits.",
        "",
    ]
    Path(output_root).mkdir(parents=True, exist_ok=True)
    Path(output_root, "heuristic_fit_README.md").write_text("\n".join(readme), encoding="utf-8")
    return payload


class YoloExprCollector(ExprVisitor):
    def __init__(self):
        super(YoloExprCollector, self).__init__()
        self.by_op = {}
        self.calls = []

    def visit_call(self, call):
        super(YoloExprCollector, self).visit_call(call)
        op_name = getattr(call.op, "name", str(call.op))
        self.by_op.setdefault(op_name, []).append(call)
        self.calls.append(call)


class ReplaceExprMutator(ExprMutator):
    def __init__(self, replacements):
        super(ReplaceExprMutator, self).__init__()
        self.replacements = list(replacements)

    def visit(self, expr):
        for old, new in self.replacements:
            if expr.same_as(old):
                return new
        return super(ReplaceExprMutator, self).visit(expr)


def infer_func(func):
    mod = tvm.IRModule.from_expr(func)
    mod = transform.InferType()(mod)
    return mod["main"]


def expr_var_like(expr, name):
    typ = expr.checked_type
    return relay.var(name, shape=[int(dim) for dim in typ.shape], dtype=typ.dtype)


def load_darknet_net(assets):
    return __darknetffi__.dlopen(assets["darknet_lib_path"]).load_network(
        assets["cfg_path"].encode("utf-8"),
        assets["weights_path"].encode("utf-8"),
        0,
    )


def collect_yolo_named(func):
    collector = YoloExprCollector()
    collector.visit(func.body)
    convs = collector.by_op.get("nn.conv2d", [])
    named = {
        "data": func.params[0],
        "pool0": collector.by_op["nn.max_pool2d"][0],
        "pool1": collector.by_op["nn.max_pool2d"][1],
        "pool2": collector.by_op["nn.max_pool2d"][2],
        "pool3": collector.by_op["nn.max_pool2d"][3],
        "pool4_route": collector.by_op["nn.max_pool2d"][4],
        "route23": collector.by_op["nn.leaky_relu"][4],
        "trunk34": collector.by_op["nn.leaky_relu"][6],
        "shared38": collector.by_op["nn.leaky_relu"][7],
        "small42": collector.by_op["nn.leaky_relu"][8],
        "small_pre26": collector.by_op["nn.leaky_relu"][9],
        "big64": collector.by_op["nn.leaky_relu"][10],
    }
    if collector.by_op.get("nn.bias_add"):
        named["small_logits51"] = collector.by_op["nn.bias_add"][0]
        named["big_logits66"] = collector.by_op["nn.bias_add"][1]
    else:
        named["small_logits51"] = convs[10]
        named["big_logits66"] = convs[12]
    return named


def yolo_relay(args):
    assets = download_yolo_assets(args)
    net = load_darknet_net(assets)
    env = vta.get_env()
    dshape = (env.BATCH, net.c, net.h, net.w)
    mod, params = relay.frontend.from_darknet(net, dtype="float32", shape=dshape)
    mod = transform.InferType()(mod)
    if args.split_quantization_mode == "full_graph":
        with tvm.transform.PassContext(opt_level=3):
            with relay.quantize.qconfig(
                global_scale=23.0,
                skip_conv_layers=[0],
                store_lowbit_output=True,
                round_for_shift=True,
            ):
                mod = relay.quantize.quantize(mod, params=params)
        mod = transform.InferType()(mod)
        params = {}
    func = mod["main"]
    data, data_hwc = prepare_input_data(env, net, assets["image_path"])
    named = collect_yolo_named(func)
    return env, assets, func, params, named, data, data_hwc


def make_stage_funcs(func, named, candidate, tail_device="cpu"):
    start_name = candidate["start_name"]
    end_outputs = candidate["end_output_names"]
    start_expr = named[start_name]
    route_expr = named["route23"]

    stage0_outputs = [start_expr]
    if candidate["needs_route_input"]:
        stage0_outputs.append(route_expr)
    stage0_body = relay.Tuple(stage0_outputs) if len(stage0_outputs) > 1 else stage0_outputs[0]
    stage0_func = infer_func(relay.Function(relay.analysis.free_vars(stage0_body), stage0_body))

    stage1_replacements = []
    stage1_input_names = []
    stage1_input_vars = []
    for idx, expr in enumerate(stage0_outputs):
        var = expr_var_like(expr, "stage0_out{}".format(idx))
        stage1_replacements.append((expr, var))
        stage1_input_names.append(var.name_hint)
        stage1_input_vars.append(var)
    stage1_body_exprs = [named[name] for name in end_outputs]
    stage1_body = relay.Tuple(stage1_body_exprs) if len(stage1_body_exprs) > 1 else stage1_body_exprs[0]
    stage1_body = ReplaceExprMutator(stage1_replacements).visit(stage1_body)
    stage1_func = infer_func(relay.Function(relay.analysis.free_vars(stage1_body), stage1_body))

    stage2_replacements = []
    stage2_input_names = []
    for idx, name in enumerate(end_outputs):
        expr = named[name]
        var = expr_var_like(expr, "stage1_out{}".format(idx))
        stage2_replacements.append((expr, var))
        stage2_input_names.append(var.name_hint)
    stage2_body = ReplaceExprMutator(stage2_replacements).visit(func.body)
    stage2_func = infer_func(relay.Function(relay.analysis.free_vars(stage2_body), stage2_body))

    return [
        {"name": "stage0_cpu", "device": "cpu", "func": stage0_func, "input_names": ["data"]},
        {"name": "stage1_vta", "device": "vta", "func": stage1_func, "input_names": stage1_input_names},
        {
            "name": "stage2_{}".format(tail_device),
            "device": tail_device,
            "func": stage2_func,
            "input_names": stage2_input_names,
        },
    ]


def build_cpu_stage(stage_name, relay_func, params, env):
    target = tvm.target.Target(env.target_vta_cpu, host=env.target_host)
    with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}):
        graph, lib, lowered_params = relay.build(
            tvm.IRModule.from_expr(relay_func), target=target, params=params
        )
    return graph, lib, lowered_params


def build_vta_stage(stage_name, relay_func, params, env, skip_first_conv=False, already_quantized=False):
    if already_quantized:
        qfunc = infer_func(relay_func)
    else:
        with tvm.transform.PassContext(opt_level=3):
            with relay.quantize.qconfig(
                global_scale=23.0,
                skip_conv_layers=[0] if skip_first_conv else [],
                store_lowbit_output=True,
                round_for_shift=True,
            ):
                qmod = relay.quantize.quantize(tvm.IRModule.from_expr(relay_func), params=params)
        qfunc = qmod["main"]
    packed = wrap_stage_with_explicit_pack(qfunc)
    packed = vta_graphpack.graph_pack(
        packed,
        env.BATCH,
        env.BLOCK_OUT,
        env.WGT_WIDTH,
        start_name=None,
        stop_name=None,
        boundary_bridge=True,
    )
    packed = vta_graphpack.run_opt_pass(packed, transform.InferType())
    target = tvm.target.Target(env.target, host=env.target_host)
    with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}):
        graph, lib, lowered_params = relay.build(packed, target=target, params=params)
    return graph, lib, lowered_params


def _is_4d_tensor_type(checked_type):
    return isinstance(checked_type, tvm.ir.TensorType) and len(checked_type.shape) == 4


def _selective_unpack_stage_outputs(expr, bitpack_end):
    checked_type = expr.checked_type
    if isinstance(expr, relay.expr.Tuple):
        fields = []
        for idx, field in enumerate(expr.fields):
            field_type = checked_type.fields[idx]
            fields.append(
                relay.Call(bitpack_end, [field]) if _is_4d_tensor_type(field_type) else field
            )
        return relay.Tuple(fields)
    return relay.Call(bitpack_end, [expr]) if _is_4d_tensor_type(checked_type) else expr


def wrap_stage_with_selective_explicit_pack(relay_func):
    bitpack_start = tvm.ir.Op.get("annotation.bitpack_start")
    bitpack_end = tvm.ir.Op.get("annotation.bitpack_end")
    relay_func = infer_func(relay_func)
    bind_map = {}
    for param in relay_func.params:
        bind_map[param] = relay.Call(bitpack_start, [param])
    wrapped_body = relay.expr.bind(relay_func.body, bind_map)
    wrapped_body = _selective_unpack_stage_outputs(wrapped_body, bitpack_end)
    wrapped_func = relay.Function(
        relay_func.params,
        wrapped_body,
        relay_func.ret_type,
        relay_func.type_params,
        relay_func.attrs,
    )
    return vta_graphpack.run_opt_pass(wrapped_func, transform.InferType())


def build_vta_tail_stage(stage_name, relay_func, params, env, already_quantized=False):
    if not already_quantized:
        raise RuntimeError("VTA YOLO tail currently requires full_graph quantization mode")
    packed = infer_func(relay_func)
    packed = vta_graphpack.graph_pack(
        packed,
        env.BATCH,
        env.BLOCK_OUT,
        env.WGT_WIDTH,
        start_name=None,
        stop_name=None,
        boundary_bridge=True,
    )
    packed = vta_graphpack.run_opt_pass(packed, transform.InferType())
    target = tvm.target.Target(env.target, host=env.target_host)
    with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}):
        graph, lib, lowered_params = relay.build(packed, target=target, params=params)
    return graph, lib, lowered_params


def stage_output_schema(relay_func):
    relay_func = infer_func(relay_func)
    typ = relay_func.ret_type
    fields = list(typ.fields) if isinstance(typ, tvm.ir.type.TupleType) else [typ]
    return [
        {"shape": [int(dim) for dim in field.shape], "dtype": field.dtype}
        for field in fields
    ]


def _round_up(value, factor):
    value = int(value)
    factor = int(factor)
    return value if value % factor == 0 else value + (factor - value % factor)


def _expr_shape_dtype(expr):
    typ = expr.checked_type
    return [int(dim) for dim in typ.shape], str(typ.dtype)


def boundary_contract_for_expr(name, expr, producer_device, consumer_device, env):
    logical_shape, dtype = _expr_shape_dtype(expr)
    channel_padding = 0
    physical_shape = list(logical_shape)
    layout = "NCHW"
    adapters = []
    if len(logical_shape) == 4:
        padded_channel = _round_up(logical_shape[1], env.BLOCK_OUT)
        channel_padding = padded_channel - int(logical_shape[1])
        if producer_device == "vta" or consumer_device == "vta":
            layout = "packed6d" if producer_device == "vta" and consumer_device == "vta" else "NCHW"
            physical_shape = [logical_shape[0], padded_channel, logical_shape[2], logical_shape[3]]
        if channel_padding:
            adapters.append("pad_channel_to_{}".format(padded_channel))
            if consumer_device == "cpu":
                adapters.append("slice_channel_to_{}".format(logical_shape[1]))
    if producer_device != consumer_device:
        if producer_device == "cpu" and consumer_device == "vta":
            adapters.append("pack_quantize")
        elif producer_device == "vta" and consumer_device == "cpu":
            adapters.append("unpack_dequantize")
    return {
        "name": name,
        "logical_shape": logical_shape,
        "logical_dtype": dtype,
        "physical_shape": physical_shape,
        "physical_dtype": dtype,
        "layout": layout,
        "channel_padding": channel_padding,
        "producer_device": producer_device,
        "consumer_device": consumer_device,
        "adapters": adapters,
        "quant_domain": "full_graph_quantized" if producer_device == "vta" or consumer_device == "vta" else "cpu_float",
    }


def boundary_contracts_for_candidate(candidate, named, env, tail_device):
    contracts = []
    stage0_outputs = [candidate["start_name"]]
    if candidate.get("needs_route_input"):
        stage0_outputs.append("route23")
    for name in stage0_outputs:
        contracts.append(
            boundary_contract_for_expr(
                name,
                named[name],
                producer_device="cpu",
                consumer_device="vta",
                env=env,
            )
        )
    for name in candidate["end_output_names"]:
        contracts.append(
            boundary_contract_for_expr(
                name,
                named[name],
                producer_device="vta",
                consumer_device=tail_device,
                env=env,
            )
        )
    return contracts


def validate_boundary_contracts(candidate, contracts, args):
    reasons = []
    warnings = []
    if args.split_quantization_mode != "full_graph":
        warnings.append("per_stage_quantization_experimental")
    if candidate.get("start_name") != "pool4_route":
        warnings.append("non_pool4_start_boundary_experimental")
    if args.tail_device == "cpu":
        warnings.append("cpu_tail_requires_serial_raw_gate")
    if candidate.get("needs_route_input"):
        warnings.append("route_branch_cpu_to_vta_boundary")
    if candidate.get("end_name") == "logits":
        warnings.append("logits_boundary_experimental")
    end_outputs = set(candidate.get("end_output_names") or [])
    if "route23" in end_outputs and args.tail_device == "cpu":
        warnings.append("concat_route_cpu_tail_boundary")
    for contract in contracts:
        if contract["channel_padding"] and not any(
            adapter.startswith("slice_channel_to_") for adapter in contract["adapters"]
        ) and contract["consumer_device"] == "cpu":
            reasons.append("padded_channel_without_slice_adapter:{}".format(contract["name"]))
        if contract["producer_device"] == "vta" and contract["consumer_device"] == "cpu":
            warnings.append("vta_to_cpu_boundary_requires_unpack_dequant:{}".format(contract["name"]))
        if contract["producer_device"] == "cpu" and contract["consumer_device"] == "vta":
            warnings.append("cpu_to_vta_boundary_requires_pack_quant:{}".format(contract["name"]))
    status = "safe" if not reasons else "boundary_invalid"
    return {
        "candidate_id": candidate["candidate_id"],
        "status": status,
        "failure_type": "" if status == "safe" else "boundary_invalid",
        "boundary_valid": status == "safe",
        "boundary_reasons": reasons,
        "boundary_warnings": warnings,
        "boundary_contracts": contracts,
    }


def write_boundary_artifacts(output_root, candidates, validations):
    by_id = {item["candidate_id"]: item for item in validations}
    enriched = []
    safe = []
    rejected = []
    for candidate in candidates:
        validation = by_id[candidate["candidate_id"]]
        row = dict(candidate)
        row.update(
            {
                "boundary_valid": validation["boundary_valid"],
                "boundary_status": validation["status"],
                "failure_type": validation["failure_type"],
                "boundary_reasons": ";".join(validation["boundary_reasons"]),
                "boundary_warnings": ";".join(validation["boundary_warnings"]),
            }
        )
        enriched.append(row)
        if validation["boundary_valid"]:
            safe.append(row)
        else:
            rejected.append(row)
    write_json(
        output_root / "boundary_contracts.json",
        {
            "rows": [
                {
                    "candidate_id": item["candidate_id"],
                    "contracts": item["boundary_contracts"],
                }
                for item in validations
            ]
        },
    )
    write_json(output_root / "boundary_validation.json", {"rows": validations})
    write_csv(output_root / "safe_candidates.csv", safe)
    write_csv(output_root / "rejected_candidates.csv", rejected)
    return enriched, safe, rejected


def candidate_features(candidate):
    stage0_ops = conv_ops(candidate["cpu_prefix_convs"])
    vta_ops = conv_ops(candidate["vta_convs"])
    cpu_tail_ops = conv_ops(candidate["cpu_tail_convs"])
    cpu_gops = 6.6
    vta_gops = 32.0
    stage0_ms = stage0_ops / 1e9 / cpu_gops * 1000.0
    vta_compute_ms = vta_ops / 1e9 / vta_gops * 1000.0
    tail_ms = cpu_tail_ops / 1e9 / cpu_gops * 1000.0
    output_bytes = sum(conv_output_bytes(idx) for idx in candidate["boundary_conv_outputs"])
    dma_ms = output_bytes / (2.2 * 1e9) * 1000.0
    boundary_ms = output_bytes / (1.5 * 1e9) * 1000.0
    return {
        "resource_cycle_ms": max(stage0_ms, vta_compute_ms, tail_ms),
        "vta_dma_ms": dma_ms,
        "boundary_ms": boundary_ms,
        "dma_fragmentation_ms": 0.02 * len(candidate["end_output_names"]),
        "sram_risk_ms": 0.0,
        "tail_cpu_ms": tail_ms,
        "stage_count": 3.0,
        "effective_segment_count": 3.0,
        "small_island_count": 1.0 if len(candidate["vta_convs"]) <= 2 else 0.0,
        "pattern_cpu_vta_cpu": 1.0,
        "pattern_cpu_vta_cpu_vta_cpu": 0.0,
        "stage0_ms_est": stage0_ms,
        "stage1_vta_ms_est": vta_compute_ms + dma_ms,
        "stage2_ms_est": tail_ms,
        "boundary_bytes_est": output_bytes,
    }


def score_candidate(candidate, params):
    weights = params["weights"]
    feats = candidate_features(candidate)
    fitted_score = float(weights.get("intercept", 0.0))
    for key, value in feats.items():
        fitted_score += float(weights.get(key, 0.0)) * float(value)

    stage_times = [
        float(feats["stage0_ms_est"]),
        float(feats["stage1_vta_ms_est"]),
        float(feats["stage2_ms_est"]),
    ]
    resource_cycle = max(stage_times)
    imbalance = resource_cycle - statistics.median(stage_times)
    balanced_score = (
        resource_cycle
        + float(feats["boundary_ms"])
        + float(feats["vta_dma_ms"])
        + 0.70 * imbalance
        + 4.0 * float(feats["effective_segment_count"])
    )

    # Conservative clamps: the ResNet fit is only a cross-model prior.  YOLO
    # selection is dominated by throughput-oriented balance and by keeping the
    # CPU tail real; a logits cut leaves no CPU tail and has already proven
    # fragile for VTA scheduling.
    score = 0.35 * fitted_score + 0.65 * balanced_score
    if len(candidate["vta_convs"]) <= 2:
        score += 40.0
    if candidate["end_name"] == "logits":
        score += 160.0
    if not candidate["cpu_tail_convs"]:
        score += 120.0
    if candidate["start_name"] == "data":
        score += 8.0
    candidate["balance_imbalance_ms_est"] = imbalance
    candidate["balanced_cycle_ms_est"] = balanced_score
    candidate["fitted_cycle_ms_est"] = fitted_score
    candidate.update(feats)
    candidate["predicted_cycle_ms"] = max(1.0, score)
    candidate["predicted_fps"] = 1000.0 / candidate["predicted_cycle_ms"]
    return candidate["predicted_cycle_ms"]


def enumerate_candidates(params, count):
    candidates = []
    for start_name, start_idx, prefix_convs in START_SPECS:
        for end_name, end_convs, output_names in END_SPECS:
            vta_convs = [idx for idx in end_convs if idx not in prefix_convs]
            if not vta_convs:
                continue
            if max(vta_convs) < min(idx for idx in end_convs):
                continue
            all_convs = set(range(len(YOLO_CONVS)))
            cpu_tail = sorted(all_convs - set(prefix_convs) - set(vta_convs))
            needs_route_input = start_idx >= 4
            boundary_outputs = []
            for name in output_names:
                if name == "trunk34":
                    boundary_outputs.append(6)
                elif name == "shared38":
                    boundary_outputs.append(7)
                elif name == "small42":
                    boundary_outputs.append(8)
                elif name == "small_pre26":
                    boundary_outputs.append(9)
                elif name == "big64":
                    boundary_outputs.append(11)
                elif name == "small_logits51":
                    boundary_outputs.append(10)
                elif name == "big_logits66":
                    boundary_outputs.append(12)
                elif name == "route23":
                    boundary_outputs.append(4)
            candidate_id = "yolo_{}_to_{}".format(start_name, end_name)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "start_name": start_name,
                    "end_name": end_name,
                    "end_output_names": output_names,
                    "needs_route_input": needs_route_input,
                    "cpu_prefix_convs": prefix_convs,
                    "vta_convs": vta_convs,
                    "cpu_tail_convs": cpu_tail,
                    "boundary_conv_outputs": sorted(set(boundary_outputs)),
                    "stage_devices": "cpu/vta/cpu",
                }
            )
    for candidate in candidates:
        score_candidate(candidate, params)
    return sorted(candidates, key=lambda item: (item["predicted_cycle_ms"], item["candidate_id"]))[
        :count
    ]


def runtime_config_grid(args):
    configs = []
    for stage0_threads in parse_int_csv(args.stage0_thread_options):
        for stage2_threads in parse_int_csv(args.stage2_thread_options):
            for queue_depth in parse_int_csv(args.queue_depth_options):
                for poll_sleep_ns in parse_int_csv(args.poll_sleep_ns_options):
                    for post_start_sleep_ns in parse_int_csv(args.post_start_sleep_ns_options):
                        configs.append(
                            {
                                "stage0_threads": stage0_threads,
                                "stage2_threads": stage2_threads,
                                "queue_depth": queue_depth,
                                "poll_sleep_ns": poll_sleep_ns,
                                "post_start_sleep_ns": post_start_sleep_ns,
                            }
                        )
    return configs


def runtime_config_label(config):
    return "s0{}_s2{}_q{}_poll{}_post{}".format(
        int(config["stage0_threads"]),
        int(config["stage2_threads"]),
        int(config["queue_depth"]),
        int(config["poll_sleep_ns"]),
        int(config["post_start_sleep_ns"]),
    )


def runtime_config_penalty(candidate, config):
    stage0_cost = 0.0
    for idx in candidate.get("cpu_prefix_convs", []):
        stage0_cost += conv_ops([idx]) / 1e9
    tail_cost = 0.0
    for idx in candidate.get("cpu_tail_convs", []):
        tail_cost += conv_ops([idx]) / 1e9
    stage0_threads = max(int(config["stage0_threads"]), 1)
    stage2_threads = max(int(config["stage2_threads"]), 1)
    cpu_balance = abs((stage0_cost / stage0_threads) - (tail_cost / stage2_threads))
    queue_bonus = -2.0 if int(config["queue_depth"]) == 2 else 0.0
    poll_penalty = 0.0 if int(config["poll_sleep_ns"]) <= 1000 else 1.5
    return cpu_balance * 0.01 + queue_bonus + poll_penalty


def expand_runtime_configs(args, candidates):
    target = int(args.runtime_config_search_count or 0)
    if target <= 0:
        return candidates
    configs = runtime_config_grid(args)
    expanded = []
    for candidate in candidates:
        for config in configs:
            row = dict(candidate)
            split_id = candidate["candidate_id"]
            label = runtime_config_label(config)
            row["split_candidate_id"] = split_id
            row["runtime_config"] = config
            row["runtime_config_label"] = label
            row["candidate_id"] = "{}__rt_{}".format(split_id, label)
            row["predicted_cycle_ms"] = float(candidate["predicted_cycle_ms"]) + runtime_config_penalty(
                candidate, config
            )
            row["predicted_fps"] = 1000.0 / max(float(row["predicted_cycle_ms"]), 1e-9)
            expanded.append(row)
    expanded.sort(key=lambda item: (float(item["predicted_cycle_ms"]), item["candidate_id"]))
    selected = []
    per_split = {}
    max_per_split = int(args.max_runtime_configs_per_split or 0)
    for item in expanded:
        split_id = item.get("split_candidate_id") or item["candidate_id"]
        if max_per_split > 0 and per_split.get(split_id, 0) >= max_per_split:
            continue
        selected.append(item)
        per_split[split_id] = per_split.get(split_id, 0) + 1
        if len(selected) >= target:
            return selected
    for item in expanded:
        if item["candidate_id"] in {row["candidate_id"] for row in selected}:
            continue
        selected.append(item)
        if len(selected) >= target:
            break
    return selected


def package_candidate(args, out_dir, candidate, env, func, params, named, data):
    split_candidate_id = candidate.get("split_candidate_id")
    if split_candidate_id and split_candidate_id != candidate["candidate_id"]:
        package_dir = Path(out_dir) / "packages" / candidate["candidate_id"]
        base_failure_path = Path(out_dir) / "packages" / (split_candidate_id + ".build_failed.json")
        if base_failure_path.exists():
            failure = read_json(base_failure_path, default={})
            raise RuntimeError(
                "base split package failed for {}: {}".format(
                    split_candidate_id, failure.get("error_summary", "unknown")
                )
            )
        if (
            package_dir.exists()
            and (package_dir / "manifest.json").exists()
            and (package_dir / "vta_stage_pipeline_runner").exists()
        ):
            try:
                manifest = json.load(open(package_dir / "manifest.json", encoding="utf-8"))
                if manifest.get("runtime_config_label") == candidate.get("runtime_config_label"):
                    return package_dir
            except Exception:
                pass
        base_candidate = dict(candidate)
        base_candidate["candidate_id"] = split_candidate_id
        base_candidate.pop("split_candidate_id", None)
        base_candidate.pop("runtime_config", None)
        base_candidate.pop("runtime_config_label", None)
        try:
            base_dir = package_candidate(args, out_dir, base_candidate, env, func, params, named, data)
        except Exception as err:
            write_json(
                base_failure_path,
                {
                    "candidate_id": split_candidate_id,
                    "failure_type": "build_failed",
                    "error_summary": str(err)[-1000:],
                },
            )
            raise
        if package_dir.exists():
            shutil.rmtree(str(package_dir))
        shutil.copytree(str(base_dir), str(package_dir))
        manifest = json.load(open(package_dir / "manifest.json", encoding="utf-8"))
        manifest.update(
            {
                "candidate_id": candidate["candidate_id"],
                "split_candidate_id": split_candidate_id,
                "runtime_config": candidate.get("runtime_config", {}),
                "runtime_config_label": candidate.get("runtime_config_label", ""),
            }
        )
        write_json(package_dir / "manifest.json", manifest)
        write_run_script(args, package_dir, manifest["stages"], serial=True, runtime_config=candidate.get("runtime_config"))
        write_run_script(args, package_dir, manifest["stages"], serial=False, runtime_config=candidate.get("runtime_config"))
        return package_dir

    package_dir = Path(out_dir) / "packages" / candidate["candidate_id"]
    if (
        package_dir.exists()
        and (package_dir / "manifest.json").exists()
        and (package_dir / "vta_stage_pipeline_runner").exists()
    ):
        try:
            manifest = json.load(open(package_dir / "manifest.json", encoding="utf-8"))
            if (
                manifest.get("split_quantization_mode") == args.split_quantization_mode
                and manifest.get("tail_device", "cpu") == args.tail_device
                and manifest.get("cpu_build_config") == "vta_disabled_alter_layout"
                and manifest.get("runner_output_mode", "raw") == args.runner_output_mode
            ):
                return package_dir
        except Exception:
            pass
    if package_dir.exists():
        shutil.rmtree(str(package_dir))
    package_dir.mkdir(parents=True)
    stages = make_stage_funcs(func, named, candidate, tail_device=args.tail_device)
    def stage_input_sources(stage_index, stage):
        if stage_index == 0:
            return ["input:0"]
        if stage_index == 1:
            return ["stage0:{}".format(index) for index, _ in enumerate(stage["input_names"])]
        sources = []
        for index, _ in enumerate(stage["input_names"]):
            output_name = candidate["end_output_names"][index]
            if (
                output_name == "route23"
                and candidate.get("needs_route_input")
                and candidate.get("start_name") == "pool4_route"
            ):
                sources.append("stage0:1")
            else:
                sources.append("stage1:{}".format(index))
        return sources

    stage_records = []
    for idx, stage in enumerate(stages):
        stage_dir = package_dir / "stages" / stage["name"]
        stage_dir.mkdir(parents=True, exist_ok=True)
        print("[BUILD] {} {}".format(candidate["candidate_id"], stage["name"]))
        if stage["device"] == "cpu":
            graph, lib, lowered_params = build_cpu_stage(stage["name"], stage["func"], params, env)
        elif idx == 2:
            graph, lib, lowered_params = build_vta_tail_stage(
                stage["name"],
                stage["func"],
                params,
                env,
                already_quantized=(args.split_quantization_mode == "full_graph"),
            )
        else:
            graph, lib, lowered_params = build_vta_stage(
                stage["name"],
                stage["func"],
                params,
                env,
                skip_first_conv=(candidate["start_name"] == "data"),
                already_quantized=(args.split_quantization_mode == "full_graph"),
            )
        (stage_dir / "graph.json").write_text(graph, encoding="utf-8")
        (stage_dir / "params.params").write_bytes(tvm.runtime.save_param_dict(lowered_params))
        export_shared_lib(lib, stage_dir / "graphlib.so")
        stage_records.append(
            {
                "index": idx,
                "name": stage["name"],
                "device": stage["device"],
                "input_names": stage["input_names"],
                "input_sources": stage_input_sources(idx, stage),
                "output_schema": stage_output_schema(stage["func"]),
                "graph": "stages/{}/graph.json".format(stage["name"]),
                "lib": "stages/{}/graphlib.so".format(stage["name"]),
                "params": "stages/{}/params.params".format(stage["name"]),
            }
        )
    (package_dir / "input.bin").write_bytes(data.tobytes(order="C"))
    compile_runner(package_dir)
    copy_runtime_libs(package_dir)
    write_run_script(args, package_dir, stage_records, serial=True, runtime_config=candidate.get("runtime_config"))
    write_run_script(args, package_dir, stage_records, serial=False, runtime_config=candidate.get("runtime_config"))
    manifest = dict(candidate)
    manifest.update(
        {
            "kind": "yolov3_tiny_native_cpu_vta_cpu_candidate",
            "stage_count": 3,
            "split_quantization_mode": args.split_quantization_mode,
            "tail_device": args.tail_device,
            "cpu_build_config": "vta_disabled_alter_layout",
            "runner_output_mode": args.runner_output_mode,
            "input_shape": [int(x) for x in data.shape],
            "stages": stage_records,
        }
    )
    write_json(package_dir / "manifest.json", manifest)
    return package_dir


def graph_node_inputs(node):
    return [int(item[0]) for item in node.get("inputs", []) or []]


def graph_dependency_closure(graph, head_node_ids, stop_node_ids=None):
    stop = set(int(x) for x in (stop_node_ids or []))
    closure = set()

    def visit(node_id):
        node_id = int(node_id)
        if node_id in stop or node_id in closure:
            return
        closure.add(node_id)
        for dep in graph_node_inputs(graph["nodes"][node_id]):
            visit(dep)

    for head in head_node_ids:
        visit(head)
    return closure


def graph_used_null_nodes(graph, selected_node_ids, boundary_input_node_ids=None):
    used = set(int(x) for x in (boundary_input_node_ids or []))
    arg_nodes = set(int(x) for x in graph.get("arg_nodes", []))
    for node_id in selected_node_ids:
        if node_id in arg_nodes:
            used.add(node_id)
        for dep in graph_node_inputs(graph["nodes"][node_id]):
            if dep in arg_nodes or dep in used:
                used.add(dep)
    return used


def graph_attr_list(graph, key):
    value = graph["attrs"][key]
    return value[1] if isinstance(value, list) and len(value) == 2 else value


def graph_attr_kind(graph, key, default):
    value = graph["attrs"].get(key)
    if isinstance(value, list) and len(value) == 2:
        return value[0]
    return default


def dtype_size_bytes(dtype):
    if dtype in ("int8", "uint8", "bool"):
        return 1
    if dtype in ("int16", "uint16", "float16"):
        return 2
    if dtype in ("int32", "uint32", "float32"):
        return 4
    if dtype in ("int64", "uint64", "float64"):
        return 8
    return 4


def shape_element_count(shape):
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


def graph_node_bytes(graph, node_id):
    shapes = graph_attr_list(graph, "shape")
    dtypes = graph_attr_list(graph, "dltype")
    return shape_element_count(shapes[int(node_id)]) * dtype_size_bytes(dtypes[int(node_id)])


def graph_node_ops_proxy(graph, node_id):
    node_id = int(node_id)
    node = graph["nodes"][node_id]
    if node.get("op") != "tvm_op":
        return 0.0
    shapes = graph_attr_list(graph, "shape")
    output_shape = shapes[node_id]
    inputs = graph_node_inputs(node)
    conv_weight = None
    for dep in inputs:
        dep_shape = shapes[dep]
        if len(dep_shape) == 6:
            conv_weight = dep_shape
            break
    if conv_weight and len(output_shape) == 6:
        out_blocks = int(output_shape[1])
        height = int(output_shape[2])
        width = int(output_shape[3])
        in_blocks = int(conv_weight[1])
        kernel_h = int(conv_weight[2])
        kernel_w = int(conv_weight[3])
        return float(2 * height * width * out_blocks * 16 * in_blocks * 16 * kernel_h * kernel_w)
    return float(shape_element_count(output_shape) * 8)


def make_subgraph_json(full_graph, output_node_ids, boundary_input_node_ids=None):
    """Build a TVM graph-executor JSON subgraph while preserving physical tensors.

    `boundary_input_node_ids` are producer nodes from the full graph that become
    null input nodes in the subgraph. Their shape/dtype are copied from the
    producer output slot, so the next stage consumes exactly the physical tensor
    the previous stage emitted.
    """

    boundary_inputs = set(int(x) for x in (boundary_input_node_ids or []))
    selected = graph_dependency_closure(full_graph, output_node_ids, boundary_inputs)
    used_null = graph_used_null_nodes(full_graph, selected, boundary_inputs)
    ordered_old_ids = sorted(used_null) + sorted(n for n in selected if n not in used_null)
    old_to_new = {old: new for new, old in enumerate(ordered_old_ids)}
    nodes = []
    for old_id in ordered_old_ids:
        old_node = full_graph["nodes"][old_id]
        if old_id in used_null:
            nodes.append(
                {
                    "op": "null",
                    "name": old_node["name"],
                    "inputs": [],
                }
            )
            continue
        node = dict(old_node)
        node["inputs"] = [
            [old_to_new[int(inp[0])], int(inp[1]), int(inp[2])]
            for inp in old_node.get("inputs", []) or []
        ]
        nodes.append(node)

    attrs = {}
    for key in ("dltype", "device_index", "shape"):
        values = graph_attr_list(full_graph, key)
        attrs[key] = [graph_attr_kind(full_graph, key, "list_str"), [values[old] for old in ordered_old_ids]]
    storage_kind = graph_attr_kind(full_graph, "storage_id", "list_int")
    attrs["storage_id"] = [storage_kind, list(range(len(ordered_old_ids)))]

    return {
        "nodes": nodes,
        "arg_nodes": [old_to_new[old] for old in sorted(used_null)],
        "heads": [[old_to_new[int(head)], 0, 0] for head in output_node_ids],
        "attrs": attrs,
        "node_row_ptr": list(range(len(nodes) + 1)),
    }


def stage_selected_nodes(full_graph, output_node_ids, boundary_input_node_ids=None):
    stop = set(int(x) for x in (boundary_input_node_ids or []))
    selected = graph_dependency_closure(full_graph, output_node_ids, stop)
    for node_id in output_node_ids:
        if int(node_id) in stop:
            selected.add(int(node_id))
    return selected


def packed_graph_stage_metrics(full_graph, output_node_ids, boundary_input_node_ids=None):
    selected = stage_selected_nodes(full_graph, output_node_ids, boundary_input_node_ids)
    tvm_nodes = [node_id for node_id in selected if full_graph["nodes"][node_id].get("op") == "tvm_op"]
    ops = sum(graph_node_ops_proxy(full_graph, node_id) for node_id in tvm_nodes)
    output_bytes = sum(graph_node_bytes(full_graph, node_id) for node_id in output_node_ids)
    compute_ms = ops / 32.0e9 * 1000.0
    memory_ms = sum(graph_node_bytes(full_graph, node_id) for node_id in tvm_nodes) / 6.0e9 * 1000.0
    return {
        "output_node_ids": [int(x) for x in output_node_ids],
        "boundary_input_node_ids": [int(x) for x in (boundary_input_node_ids or [])],
        "tvm_op_count": len(tvm_nodes),
        "ops_proxy": ops,
        "output_bytes": output_bytes,
        "runtime_proxy_ms": compute_ms + memory_ms + 0.35 * len(tvm_nodes),
        "node_ids": sorted(int(x) for x in selected),
    }


def packed_graph_candidate_score(full_graph, stage_output_nodes):
    stage_metrics = []
    prev_outputs = []
    for outputs in stage_output_nodes:
        stage_metrics.append(packed_graph_stage_metrics(full_graph, outputs, prev_outputs))
        prev_outputs = list(outputs)
    boundary_bytes = sum(item["output_bytes"] for item in stage_metrics[:-1])
    boundary_ms = boundary_bytes / 2.2e9 * 1000.0
    launch_penalty_ms = max(0, len(stage_metrics) - 1) * 0.6
    stage_times = [item["runtime_proxy_ms"] for item in stage_metrics]
    imbalance_ms = max(stage_times) - (sum(stage_times) / len(stage_times))
    small_stage_penalty_ms = sum(4.0 for item in stage_metrics[:-1] if item["tvm_op_count"] < 2)
    predicted_cycle_ms = max(stage_times) + boundary_ms + launch_penalty_ms + 0.25 * imbalance_ms + small_stage_penalty_ms
    return {
        "stage_metrics": stage_metrics,
        "boundary_bytes_est": boundary_bytes,
        "boundary_ms_est": boundary_ms,
        "stage_runtime_proxy_ms": stage_times,
        "stage_balance_ms_est": imbalance_ms,
        "stage_launch_penalty_ms": launch_penalty_ms,
        "small_stage_penalty_ms": small_stage_penalty_ms,
        "predicted_cycle_ms": predicted_cycle_ms,
        "predicted_fps": 1000.0 / max(predicted_cycle_ms, 1e-9),
    }


def packed_hetero_candidate_score(full_graph, stage_output_nodes, stage_devices):
    """Resource-aware score for native CPU/VTA pipeline candidates.

    VTA stages are serialized by the native runner's global VTA mutex, so two
    adjacent VTA stages do not create two independent pipeline resources.  The
    steady-state cycle is better approximated by max(CPU work, total VTA
    occupancy) plus boundary and launch overheads.
    """
    base = packed_graph_candidate_score(full_graph, stage_output_nodes)
    stage_metrics = base["stage_metrics"]
    devices = list(stage_devices)

    # Calibrated from the first true YOLO CPU-prefix/VTA Top20 run.  These
    # constants are intentionally conservative priors, not correctness gates.
    cpu_prefix_node5_ms = 52.0
    vta_full_body_ms = 260.0
    cpu_tail_default_ms = 80.0
    ps_pl_bw_bytes_s = 2.2e9

    vta_proxy_total = sum(
        float(item["runtime_proxy_ms"])
        for item, device in zip(stage_metrics, devices)
        if device == "vta"
    )
    if vta_proxy_total <= 0.0:
        vta_proxy_total = 1.0

    resource_ms = []
    total_vta_occupied_ms = 0.0
    cpu_total_ms = 0.0
    for index, (item, device) in enumerate(zip(stage_metrics, devices)):
        if device == "cpu":
            if index == 0 and item["output_node_ids"] == [5]:
                ms = cpu_prefix_node5_ms
            else:
                ms = max(cpu_tail_default_ms, float(item["runtime_proxy_ms"]) * 0.20)
            cpu_total_ms += ms
        else:
            ms = float(item["runtime_proxy_ms"]) / vta_proxy_total * vta_full_body_ms
            total_vta_occupied_ms += ms
        resource_ms.append(ms)

    boundary_bytes = sum(item["output_bytes"] for item in stage_metrics[:-1])
    boundary_ms = boundary_bytes / ps_pl_bw_bytes_s * 1000.0
    launch_penalty_ms = max(0, len(stage_metrics) - 1) * 0.8
    vta_split_penalty_ms = max(0, sum(1 for item in devices if item == "vta") - 1) * 3.0
    small_stage_penalty_ms = sum(
        6.0
        for item, device in zip(stage_metrics[:-1], devices[:-1])
        if device == "vta" and item["tvm_op_count"] < 3
    )

    resource_bottlenecks = [cpu_total_ms, total_vta_occupied_ms]
    if devices[-1] == "cpu":
        resource_bottlenecks.append(resource_ms[-1])
    resource_cycle_ms = max(resource_bottlenecks)
    nonzero_resources = [value for value in resource_bottlenecks if value > 0.0]
    median_resource_ms = statistics.median(nonzero_resources) if nonzero_resources else 0.0
    imbalance_ms = max(0.0, resource_cycle_ms - median_resource_ms)
    predicted_cycle_ms = (
        resource_cycle_ms
        + boundary_ms
        + launch_penalty_ms
        + vta_split_penalty_ms
        + small_stage_penalty_ms
        + 0.35 * imbalance_ms
    )

    base.update(
        {
            "boundary_bytes_est": boundary_bytes,
            "boundary_ms_est": boundary_ms,
            "stage_runtime_proxy_ms": resource_ms,
            "stage_balance_ms_est": imbalance_ms,
            "resource_cycle_ms_est": resource_cycle_ms,
            "cpu_resource_ms_est": cpu_total_ms,
            "vta_occupied_ms_est": total_vta_occupied_ms,
            "stage_launch_penalty_ms": launch_penalty_ms,
            "vta_split_penalty_ms": vta_split_penalty_ms,
            "small_stage_penalty_ms": small_stage_penalty_ms,
            "predicted_cycle_ms": predicted_cycle_ms,
            "predicted_fps": 1000.0 / max(predicted_cycle_ms, 1e-9),
        }
    )
    return base


def packed_graph_split_validation(full_graph, stage_output_nodes):
    arg_nodes = set(int(x) for x in full_graph.get("arg_nodes", []))
    data_nodes = {
        int(node_id)
        for node_id in arg_nodes
        if full_graph["nodes"][int(node_id)].get("name") == "data"
    }
    reasons = []
    prev_outputs = []
    stage_validations = []
    for idx, outputs in enumerate(stage_output_nodes):
        selected = graph_dependency_closure(full_graph, outputs, prev_outputs)
        used_null = graph_used_null_nodes(full_graph, selected, prev_outputs)
        allowed = set(prev_outputs) | arg_nodes
        if idx > 0:
            allowed -= data_nodes
        extra = sorted(int(x) for x in used_null if int(x) not in allowed)
        hidden_data = sorted(int(x) for x in used_null if int(x) in data_nodes and idx > 0)
        if extra:
            reasons.append("stage{}_hidden_boundary_inputs:{}".format(idx, ",".join(map(str, extra))))
        if hidden_data:
            reasons.append("stage{}_reintroduces_data_input".format(idx))
        tvm_count = sum(1 for node_id in selected if full_graph["nodes"][node_id].get("op") == "tvm_op")
        if idx < len(stage_output_nodes) - 1 and tvm_count < 2:
            reasons.append("stage{}_too_small:{}_tvm_ops".format(idx, tvm_count))
        stage_validations.append(
            {
                "stage_index": idx,
                "output_node_ids": [int(x) for x in outputs],
                "boundary_input_node_ids": [int(x) for x in prev_outputs],
                "used_null_node_ids": sorted(int(x) for x in used_null),
                "hidden_data_node_ids": hidden_data,
                "extra_hidden_boundary_node_ids": extra,
                "tvm_op_count": tvm_count,
            }
        )
        prev_outputs = list(outputs)
    return {
        "boundary_valid": not reasons,
        "boundary_reasons": reasons,
        "stage_validations": stage_validations,
    }


def packed_graph_candidate(candidate_id, description, stage_output_nodes, full_graph):
    score = packed_graph_candidate_score(full_graph, stage_output_nodes)
    validation = packed_graph_split_validation(full_graph, stage_output_nodes)
    return {
        "candidate_id": candidate_id,
        "description": description,
        "split_backend": "packed_graph",
        "stage_devices": "/".join(["vta"] * len(stage_output_nodes)),
        "stage_count": len(stage_output_nodes),
        "stage_output_nodes_json": json.dumps(stage_output_nodes, separators=(",", ":")),
        "boundary_node_ids": ",".join(str(x) for stage in stage_output_nodes[:-1] for x in stage),
        "boundary_bytes_est": score["boundary_bytes_est"],
        "boundary_ms_est": score["boundary_ms_est"],
        "stage_runtime_proxy_ms_json": json.dumps(score["stage_runtime_proxy_ms"]),
        "stage_balance_ms_est": score["stage_balance_ms_est"],
        "predicted_cycle_ms": score["predicted_cycle_ms"],
        "predicted_fps": score["predicted_fps"],
        "stage_metrics": score["stage_metrics"],
        "boundary_valid": validation["boundary_valid"],
        "boundary_reasons": ";".join(validation["boundary_reasons"]),
        "stage_validations": validation["stage_validations"],
    }


def packed_hetero_candidate(candidate_id, description, stage_output_nodes, stage_devices, full_graph):
    candidate = packed_graph_candidate(candidate_id, description, stage_output_nodes, full_graph)
    score = packed_hetero_candidate_score(full_graph, stage_output_nodes, stage_devices)
    candidate["split_backend"] = "packed_hetero"
    candidate["stage_devices"] = "/".join(stage_devices)
    candidate["stage_devices_json"] = json.dumps(stage_devices, separators=(",", ":"))
    candidate["stage_metrics"] = score["stage_metrics"]
    candidate["boundary_bytes_est"] = score["boundary_bytes_est"]
    candidate["boundary_ms_est"] = score["boundary_ms_est"]
    candidate["stage_runtime_proxy_ms_json"] = json.dumps(score["stage_runtime_proxy_ms"])
    candidate["stage_balance_ms_est"] = score["stage_balance_ms_est"]
    candidate["resource_cycle_ms_est"] = score["resource_cycle_ms_est"]
    candidate["cpu_resource_ms_est"] = score["cpu_resource_ms_est"]
    candidate["vta_occupied_ms_est"] = score["vta_occupied_ms_est"]
    candidate["vta_split_penalty_ms"] = score["vta_split_penalty_ms"]
    candidate["predicted_cycle_ms"] = score["predicted_cycle_ms"]
    candidate["predicted_fps"] = score["predicted_fps"]
    return candidate


def enumerate_packed_hetero_candidates(full_graph, count):
    final_outputs = [int(head[0]) for head in full_graph["heads"]]
    specs = [
        (
            "cpu5_vta_heads",
            "CPU prefix through float maxpool node5, VTA body to final heads.",
            [[5], final_outputs],
            ["cpu", "vta"],
        ),
        (
            "cpu5_vta25_heads",
            "CPU prefix node5, VTA body split after route node25.",
            [[5], [25], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_35_heads",
            "CPU prefix node5, VTA body split after route node25 and trunk node35.",
            [[5], [25, 35], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_40_heads",
            "CPU prefix node5, VTA body split after route node25 and shared node40.",
            [[5], [25, 40], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_43_heads",
            "CPU prefix node5, VTA body split after route node25 and small pre-concat node43.",
            [[5], [25, 43], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_44_heads",
            "CPU prefix node5, VTA body split after concat node44.",
            [[5], [25, 44], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_48_heads",
            "CPU prefix node5, VTA body split after small branch node48.",
            [[5], [25, 48], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_51_heads",
            "CPU prefix node5, VTA body split after small logits node51.",
            [[5], [25, 51], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_56_heads",
            "CPU prefix node5, VTA body split after big branch cast node56.",
            [[5], [25, 56], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_60_heads",
            "CPU prefix node5, VTA body split after big branch node60.",
            [[5], [25, 60], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_63_heads",
            "CPU prefix node5, VTA body split after big logits node63.",
            [[5], [25, 63], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_40_60_heads",
            "CPU prefix node5, VTA body split after shared node40 and big branch node60.",
            [[5], [25, 40, 60], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_40_63_heads",
            "CPU prefix node5, VTA body split after shared node40 and big logits node63.",
            [[5], [25, 40, 63], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_40_43_heads",
            "CPU prefix node5, VTA body split after shared node40 and small pre-concat node43.",
            [[5], [25, 40, 43], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_40_48_heads",
            "CPU prefix node5, VTA body split after shared node40 and small branch node48.",
            [[5], [25, 40, 48], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_40_51_heads",
            "CPU prefix node5, VTA body split after shared node40 and small logits node51.",
            [[5], [25, 40, 51], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_43_56_heads",
            "CPU prefix node5, VTA body split after small pre-concat node43 and big cast node56.",
            [[5], [25, 43, 56], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_44_56_heads",
            "CPU prefix node5, VTA body split after concat node44 and big cast node56.",
            [[5], [25, 44, 56], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_48_60_heads",
            "CPU prefix node5, VTA body split after small node48 and big node60.",
            [[5], [25, 48, 60], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_51_60_heads",
            "CPU prefix node5, VTA body split after small logits node51 and big node60.",
            [[5], [25, 51, 60], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "cpu5_vta25_51_63_heads",
            "CPU prefix node5, VTA body split after small/big logits nodes.",
            [[5], [25, 51, 63], final_outputs],
            ["cpu", "vta", "vta"],
        ),
        (
            "vta25_35_cpu_heads",
            "VTA body through route node25 and trunk node35, CPU final heads.",
            [[25, 35], final_outputs],
            ["vta", "cpu"],
        ),
        (
            "vta25_40_cpu_heads",
            "VTA body through route node25 and shared node40, CPU final heads.",
            [[25, 40], final_outputs],
            ["vta", "cpu"],
        ),
        (
            "vta25_48_cpu_heads",
            "VTA body through small branch node48, CPU final heads.",
            [[25, 48], final_outputs],
            ["vta", "cpu"],
        ),
        (
            "vta25_51_cpu_heads",
            "VTA body through small logits node51, CPU final heads.",
            [[25, 51], final_outputs],
            ["vta", "cpu"],
        ),
        (
            "vta25_51_63_cpu_heads",
            "VTA body through both logits tensors, CPU final decode heads.",
            [[25, 51, 63], final_outputs],
            ["vta", "cpu"],
        ),
        (
            "cpu5_vta25_35_cpu_heads",
            "CPU prefix, VTA body through route/trunk, CPU heads.",
            [[5], [25, 35], final_outputs],
            ["cpu", "vta", "cpu"],
        ),
        (
            "cpu5_vta25_40_cpu_heads",
            "CPU prefix, VTA body through route/shared, CPU heads.",
            [[5], [25, 40], final_outputs],
            ["cpu", "vta", "cpu"],
        ),
        (
            "cpu5_vta25_48_cpu_heads",
            "CPU prefix, VTA body through small branch node48, CPU heads.",
            [[5], [25, 48], final_outputs],
            ["cpu", "vta", "cpu"],
        ),
        (
            "cpu5_vta25_51_cpu_heads",
            "CPU prefix, VTA body through small logits node51, CPU heads.",
            [[5], [25, 51], final_outputs],
            ["cpu", "vta", "cpu"],
        ),
        (
            "cpu5_vta25_51_63_cpu_heads",
            "CPU prefix, VTA body through both logits tensors, CPU final decode heads.",
            [[5], [25, 51, 63], final_outputs],
            ["cpu", "vta", "cpu"],
        ),
        (
            "cpu5_vta25_40_63_cpu_heads",
            "CPU prefix, VTA body through shared/big logits, CPU heads.",
            [[5], [25, 40, 63], final_outputs],
            ["cpu", "vta", "cpu"],
        ),
    ]
    candidates = []
    for suffix, description, stages, devices in specs:
        if devices[-1] == "cpu" and "51_63_cpu_heads" not in suffix:
            continue
        candidate = packed_hetero_candidate(
            "yolo_hetero_{}".format(suffix),
            description,
            stages,
            devices,
            full_graph,
        )
        if candidate["boundary_valid"]:
            candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            float(item["predicted_cycle_ms"]),
            int(item["stage_count"]),
            item["candidate_id"],
        )
    )
    return candidates[: int(count)]


def enumerate_packed_graph_candidates(full_graph, count):
    final_outputs = [int(head[0]) for head in full_graph["heads"]]
    specs = [
        ("n25_to_heads", "2-stage split after route tensor node25.", [[25], final_outputs]),
        ("n25_35_to_heads", "2-stage split after route node25 and trunk node35.", [[25, 35], final_outputs]),
        ("n25_40_to_heads", "2-stage split after route node25 and shared node40.", [[25, 40], final_outputs]),
        ("n25_43_to_heads", "2-stage split after small branch pre-concat node43, forwarding node25.", [[25, 43], final_outputs]),
        ("n25_44_to_heads", "2-stage split after concat tensor node44, forwarding node25.", [[25, 44], final_outputs]),
        ("n25_48_to_heads", "2-stage split after small post-concat node48, forwarding node25.", [[25, 48], final_outputs]),
        ("n25_51_to_heads", "2-stage split after small logits node51, forwarding node25.", [[25, 51], final_outputs]),
        ("n25_56_to_heads", "2-stage split exposing big branch cast node56, forwarding node25.", [[25, 56], final_outputs]),
        ("n25_60_to_heads", "2-stage split exposing big branch node60, forwarding node25.", [[25, 60], final_outputs]),
        ("n25_63_to_heads", "2-stage split exposing big logits node63, forwarding node25.", [[25, 63], final_outputs]),
        ("n25_40_43_to_heads", "2-stage split after shared node40 and small node43.", [[25, 40, 43], final_outputs]),
        ("n25_40_48_to_heads", "2-stage split after shared node40 and small node48.", [[25, 40, 48], final_outputs]),
        ("n25_40_51_to_heads", "2-stage split after shared node40 and small logits node51.", [[25, 40, 51], final_outputs]),
        ("n25_40_60_to_heads", "2-stage split after shared node40 and big node60.", [[25, 40, 60], final_outputs]),
        ("n25_40_63_to_heads", "2-stage split after shared node40 and big logits node63.", [[25, 40, 63], final_outputs]),
        ("n25__n25_40__heads", "3-stage trunk split: route node25 then shared node40.", [[25], [25, 40], final_outputs]),
        ("n25__n25_43__heads", "3-stage split with small pre-concat node43.", [[25], [25, 43], final_outputs]),
        ("n25__n25_44__heads", "3-stage split with concat node44.", [[25], [25, 44], final_outputs]),
        ("n25__n25_48__heads", "3-stage split with small branch node48.", [[25], [25, 48], final_outputs]),
        ("n25__n25_51__heads", "3-stage split with small logits node51.", [[25], [25, 51], final_outputs]),
        ("n25__n25_60__heads", "3-stage split with big branch node60.", [[25], [25, 60], final_outputs]),
        ("n25__n25_63__heads", "3-stage split with big logits node63.", [[25], [25, 63], final_outputs]),
        ("n25_40__n25_44__heads", "3-stage split with shared node40 then concat node44.", [[25, 40], [25, 44], final_outputs]),
        ("n25_40__n25_48__heads", "3-stage split with shared node40 then small node48.", [[25, 40], [25, 48], final_outputs]),
        ("n25_40__n25_51__heads", "3-stage split with shared node40 then small logits node51.", [[25, 40], [25, 51], final_outputs]),
        ("n25_40__n25_60__heads", "3-stage split with shared node40 then big node60.", [[25, 40], [25, 60], final_outputs]),
        ("n25_40__n25_63__heads", "3-stage split with shared node40 then big logits node63.", [[25, 40], [25, 63], final_outputs]),
        ("n25_40__n25_51_63__heads", "3-stage head-only final split after logits nodes51 and63.", [[25, 40], [25, 51, 63], final_outputs]),
        ("n25_35__n25_40__heads", "3-stage split from trunk node35 to shared node40.", [[25, 35], [25, 40], final_outputs]),
        ("n25_40__n25_43_56__heads", "3-stage branch split after pre-head nodes43 and56.", [[25, 40], [25, 43, 56], final_outputs]),
        ("n25_40__n25_44_56__heads", "3-stage branch split after concat node44 and big cast node56.", [[25, 40], [25, 44, 56], final_outputs]),
    ]
    candidates = []
    for suffix, description, stages in specs:
        candidate = packed_graph_candidate(
            "yolo_packed_{}".format(suffix),
            description,
            stages,
            full_graph,
        )
        if candidate["boundary_valid"]:
            candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            float(item["predicted_cycle_ms"]),
            int(item["stage_count"]),
            item["candidate_id"],
        )
    )
    # Always keep the previously verified smoke split in the set if possible.
    smoke_id = "yolo_packed_n25_40_to_heads"
    selected = []
    by_id = {item["candidate_id"]: item for item in candidates}
    if smoke_id in by_id:
        selected.append(by_id[smoke_id])
    for item in candidates:
        if item["candidate_id"] not in {row["candidate_id"] for row in selected}:
            selected.append(item)
        if len(selected) >= int(count):
            break
    return selected


def packed_graph_presets():
    return {
        "route_shared_tail": {
            "candidate_id": "yolo_packed_route25_shared40_to_heads",
            "description": (
                "Split the already-packed all-VTA graph at physical tensors node25 "
                "(route branch) and node40 (shared head input)."
            ),
            "stage0_outputs": [25, 40],
            "stage1_boundary_inputs": [25, 40],
            "final_outputs": [52, 53, 54, 55, 64, 65, 66, 67],
        }
    }


def copy_packed_graph_runtime(src_package, package_dir):
    for name in ("input.bin", "libtvm_runtime.so", "libvta.so"):
        src = src_package / name
        if src.exists():
            shutil.copy2(str(src), str(package_dir / name))


def packed_graph_source_paths(args):
    src_package = Path(args.packed_graph_package)
    src_stage = src_package / "stages" / "stage0_all_vta"
    graph_path = src_stage / "graph.json"
    lib_path = src_stage / "graphlib.so"
    params_path = src_stage / "params.params"
    if not graph_path.exists():
        raise RuntimeError("missing packed graph JSON: {}".format(graph_path))
    if not lib_path.exists() or not params_path.exists():
        raise RuntimeError("missing packed graph lib/params under {}".format(src_stage))
    return src_package, graph_path, lib_path, params_path


def build_cpu_prefix_node5_stage(stage_dir, cpu_context):
    if cpu_context is None:
        raise RuntimeError("CPU prefix stage requires Relay CPU context")
    env, _assets, _func, params, named, _data, _data_hwc = cpu_context
    body = named["pool0"]
    stage_func = infer_func(relay.Function(relay.analysis.free_vars(body), body))
    graph, lib, lowered_params = build_cpu_stage("stage0_cpu_prefix", stage_func, params, env)
    (stage_dir / "graph.json").write_text(graph, encoding="utf-8")
    (stage_dir / "params.params").write_bytes(tvm.runtime.save_param_dict(lowered_params))
    export_shared_lib(lib, stage_dir / "graphlib.so")
    return stage_output_schema(stage_func)


def package_packed_graph_candidate(args, output_root, candidate=None, full_graph=None, cpu_context=None):
    src_package, graph_path, lib_path, params_path = packed_graph_source_paths(args)
    if full_graph is None:
        full_graph = json.load(open(graph_path, encoding="utf-8"))
    if candidate is None:
        preset = packed_graph_presets()[args.packed_graph_preset]
        candidate = packed_graph_candidate(
            preset["candidate_id"],
            preset["description"],
            [preset["stage0_outputs"], preset["final_outputs"]],
            full_graph,
        )
        candidate["packed_graph_preset"] = args.packed_graph_preset
    stage_output_nodes = json.loads(candidate["stage_output_nodes_json"])
    stage_devices = json.loads(
        candidate.get("stage_devices_json")
        or json.dumps(["vta"] * len(stage_output_nodes), separators=(",", ":"))
    )
    if len(stage_devices) != len(stage_output_nodes):
        raise RuntimeError(
            "stage device count {} does not match stage count {}".format(
                len(stage_devices), len(stage_output_nodes)
            )
        )
    package_dir = Path(output_root) / "packages" / candidate["candidate_id"]
    if package_dir.exists():
        shutil.rmtree(str(package_dir))
    package_dir.mkdir(parents=True, exist_ok=True)
    stage_records = []
    prev_outputs = []
    for idx, output_nodes in enumerate(stage_output_nodes):
        stage_name = "stage{}_packed".format(idx)
        if idx == 0:
            stage_name += "_prefix"
            input_names = ["data"]
        elif idx == len(stage_output_nodes) - 1:
            stage_name += "_heads"
            input_names = [full_graph["nodes"][node_id]["name"] for node_id in prev_outputs]
        else:
            stage_name += "_middle"
            input_names = [full_graph["nodes"][node_id]["name"] for node_id in prev_outputs]
        device = stage_devices[idx]
        stage_dir = package_dir / "stages" / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        if device == "cpu" and idx == 0 and output_nodes == [5]:
            output_schema = build_cpu_prefix_node5_stage(stage_dir, cpu_context)
        elif device == "cpu" and idx == len(stage_output_nodes) - 1:
            graph = make_subgraph_json(full_graph, output_nodes, boundary_input_node_ids=prev_outputs)
            write_json(stage_dir / "graph.json", graph)
            shutil.copy2(str(lib_path), str(stage_dir / "graphlib.so"))
            shutil.copy2(str(params_path), str(stage_dir / "params.params"))
            output_schema = [
                {
                    "shape": graph_attr_list(graph, "shape")[int(head[0])],
                    "dtype": graph_attr_list(graph, "dltype")[int(head[0])],
                }
                for head in graph["heads"]
            ]
        elif device == "cpu":
            raise RuntimeError("packed_hetero CPU middle stages are not supported")
        else:
            graph = make_subgraph_json(full_graph, output_nodes, boundary_input_node_ids=prev_outputs)
            write_json(stage_dir / "graph.json", graph)
            shutil.copy2(str(lib_path), str(stage_dir / "graphlib.so"))
            shutil.copy2(str(params_path), str(stage_dir / "params.params"))
            output_schema = [
                {
                    "shape": graph_attr_list(graph, "shape")[int(head[0])],
                    "dtype": graph_attr_list(graph, "dltype")[int(head[0])],
                }
                for head in graph["heads"]
            ]
        stage_records.append(
            {
                "index": idx,
                "name": stage_name,
                "device": device,
                "input_names": input_names,
                "output_schema": output_schema,
                "graph": "stages/{}/graph.json".format(stage_name),
                "lib": "stages/{}/graphlib.so".format(stage_name),
                "params": "stages/{}/params.params".format(stage_name),
            }
        )
        prev_outputs = list(output_nodes)
    copy_packed_graph_runtime(src_package, package_dir)
    compile_runner(package_dir)
    copy_runtime_libs(package_dir)
    write_run_script(args, package_dir, stage_records, serial=True)
    write_run_script(args, package_dir, stage_records, serial=False)
    manifest = dict(candidate)
    manifest.update(
        {
            "kind": "yolov3_tiny_packed_graph_split_candidate",
            "stage_count": len(stage_records),
            "runner_output_mode": args.runner_output_mode,
            "source_package": str(src_package),
            "source_graph": str(graph_path),
            "stages": stage_records,
            "stage_output_nodes": stage_output_nodes,
            "stage_devices": stage_devices,
        }
    )
    write_json(package_dir / "manifest.json", manifest)
    write_json(
        package_dir / "packed_graph_split_validation.json",
        {
            "candidate_id": candidate["candidate_id"],
            "status": "safe",
            "reason": "subgraphs are cut from the already-correct packed all-VTA graph",
            "stage_output_nodes": stage_output_nodes,
            "stage_records": stage_records,
            "stage_metrics": candidate.get("stage_metrics", []),
            "stage_validations": candidate.get("stage_validations", []),
            "boundary_bytes_est": candidate.get("boundary_bytes_est"),
            "predicted_cycle_ms": candidate.get("predicted_cycle_ms"),
            "predicted_fps": candidate.get("predicted_fps"),
        },
    )
    return candidate, package_dir


def write_run_script(args, package_dir, stage_records, serial, runtime_config=None):
    output_jsonl = "native_serial_result.jsonl" if serial else "native_pipeline_result.jsonl"
    script_name = "run_native_serial.sh" if serial else "run_native_pipeline.sh"
    serial_arg = " \\\n  --serial" if serial else ""
    runtime_config = runtime_config or {}
    stage0_threads = int(runtime_config.get("stage0_threads", 3))
    stage2_threads = int(runtime_config.get("stage2_threads", 4))
    queue_depth = int(runtime_config.get("queue_depth", args.queue_depth))
    poll_sleep_ns = int(runtime_config.get("poll_sleep_ns", 1000))
    post_start_sleep_ns = int(runtime_config.get("post_start_sleep_ns", 1000))
    stage_parts = []
    for idx, stage in enumerate(stage_records):
        prefix = "stage{}".format(idx)
        source_line = ""
        if stage.get("input_sources"):
            source_line = "  --{p}-input-sources {sources} \\\n".format(
                p=prefix,
                sources=shlex.quote(",".join(stage.get("input_sources") or [])),
            )
        stage_parts.append(
            "  --{p}-graph {graph} \\\n"
            "  --{p}-lib {lib} \\\n"
            "  --{p}-params {params} \\\n"
            "  --{p}-input-names {inputs} \\\n"
            "{source_line}"
            "  --{p}-name {name} \\\n"
            "  --{p}-device {device} \\\n"
            "  --{p}-runtime-num-threads {threads}".format(
                p=prefix,
                graph=shlex.quote(stage["graph"]),
                lib=shlex.quote(stage["lib"]),
                params=shlex.quote(stage["params"]),
                inputs=shlex.quote(",".join(stage["input_names"])),
                source_line=source_line,
                name=shlex.quote(stage["name"]),
                device=stage["device"],
                threads=stage0_threads
                if stage["device"] == "cpu" and idx == 0
                else (stage2_threads if stage["device"] == "cpu" else 1),
            )
        )
    script = """#!/bin/sh
set -eu
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
export LD_PRELOAD="$PWD/libtvm_runtime.so:$PWD/libvta.so"
export TVM_NUM_THREADS={threads}
: "${{TVM_THREAD_POOL_SPIN_COUNT:=0}}"
: "${{AXU5EVB_DRIVER_POST_START_SLEEP_NS:={post_start_sleep_ns}}}"
: "${{AXU5EVB_DRIVER_POLL_SLEEP_NS:={poll_sleep_ns}}}"
export TVM_THREAD_POOL_SPIN_COUNT
export AXU5EVB_DRIVER_POST_START_SLEEP_NS
export AXU5EVB_DRIVER_POLL_SLEEP_NS

exec ./vta_stage_pipeline_runner \\
{stage_cli} \\
  --input input.bin \\
  --runs {runs} \\
  --queue-depth {queue_depth} \\
  --runtime-num-threads {runtime_threads} \\
  --output-mode {output_mode} \\
  --output-dump-dir {dump_dir} \\
  --output-jsonl {output_jsonl}{serial_arg}
""".format(
        stage_cli=" \\\n".join(stage_parts),
        runs=int(args.serial_runs if serial else args.runs),
        queue_depth=queue_depth,
        runtime_threads=int(args.runtime_num_threads),
        threads=int(args.runtime_num_threads),
        post_start_sleep_ns=post_start_sleep_ns,
        poll_sleep_ns=poll_sleep_ns,
        output_mode=shlex.quote(args.runner_output_mode),
        dump_dir=shlex.quote(("native_serial" if serial else "native_pipeline").replace("native_", "") + "_output_dumps"),
        output_jsonl=output_jsonl,
        serial_arg=serial_arg,
    )
    path = package_dir / script_name
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def package_tar(package_dir):
    tar_path = package_dir.parent / (package_dir.name + ".tar.gz")
    with tarfile.open(str(tar_path), "w:gz") as tar:
        for item in sorted(package_dir.iterdir()):
            tar.add(str(item), arcname=item.name)
    return tar_path


def split_boundary_diagnostics(candidate, row):
    diagnostics = {
        "candidate_id": candidate.get("candidate_id"),
        "start_name": candidate.get("start_name"),
        "end_name": candidate.get("end_name"),
        "end_output_names": candidate.get("end_output_names"),
        "stage_count": row.get("stage_count"),
        "stage_raw_outputs_available": bool(row.get("stage_raw_outputs")),
    }
    if row.get("stage_raw_outputs"):
        diagnostics["stage_raw_outputs"] = row.get("stage_raw_outputs")
    return diagnostics


def _numpy_dtype(dtype):
    mapping = {
        "float32": np.float32,
        "float64": np.float64,
        "int8": np.int8,
        "uint8": np.uint8,
        "int16": np.int16,
        "uint16": np.uint16,
        "int32": np.int32,
        "uint32": np.uint32,
        "int64": np.int64,
        "uint64": np.uint64,
    }
    if dtype not in mapping:
        raise RuntimeError("unsupported dumped tensor dtype: {}".format(dtype))
    return mapping[dtype]


def load_dumped_output_arrays(row, candidate_output_dir):
    files = sorted(row.get("raw_output_files") or [], key=lambda item: int(item.get("index", 0)))
    arrays = []
    for item in files:
        rel_path = Path(str(item["path"]))
        if rel_path.is_absolute():
            rel_path = Path(*rel_path.parts[1:])
        path = Path(candidate_output_dir) / rel_path
        if not path.exists():
            # Older packages wrote only tensor summaries; keep this optional.
            return []
        shape = [int(dim) for dim in item.get("shape", [])]
        arr = np.fromfile(str(path), dtype=_numpy_dtype(item.get("dtype", "float32")))
        if shape:
            arr = arr.reshape(shape)
        arrays.append(arr)
    return arrays


def _box_iou(lhs, rhs):
    left = max(float(lhs.get("left", 0)), float(rhs.get("left", 0)))
    top = max(float(lhs.get("top", 0)), float(rhs.get("top", 0)))
    right = min(float(lhs.get("right", 0)), float(rhs.get("right", 0)))
    bottom = min(float(lhs.get("bottom", 0)), float(rhs.get("bottom", 0)))
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    lhs_area = max(0.0, float(lhs.get("right", 0)) - float(lhs.get("left", 0))) * max(
        0.0, float(lhs.get("bottom", 0)) - float(lhs.get("top", 0))
    )
    rhs_area = max(0.0, float(rhs.get("right", 0)) - float(rhs.get("left", 0))) * max(
        0.0, float(rhs.get("bottom", 0)) - float(rhs.get("top", 0))
    )
    return inter / max(lhs_area + rhs_area - inter, 1.0)


def compare_detection_outputs(reference, candidate, topk=3):
    ref = (reference.get("detection_summary") or {}).get("top_detections") or []
    cand = (candidate.get("detection_summary") or {}).get("top_detections") or []
    count = min(int(topk), len(ref))
    rows = []
    passes = bool(count) and len(cand) >= count
    used = set()
    for idx in range(count):
        ref_item = ref[idx]
        best_index = None
        best_item = {}
        best_iou = -1.0
        for cand_index, cand_item in enumerate(cand):
            if cand_index in used:
                continue
            if int(ref_item.get("class_id", -1)) != int(cand_item.get("class_id", -2)):
                continue
            iou = _box_iou(ref_item, cand_item)
            if iou > best_iou:
                best_iou = iou
                best_index = cand_index
                best_item = cand_item
        score_delta = abs(float(ref_item.get("score", 0.0)) - float(best_item.get("score", 0.0)))
        class_match = bool(best_item)
        row_passes = bool(class_match and best_iou >= 0.45 and score_delta <= 0.35)
        if row_passes and best_index is not None:
            used.add(best_index)
        passes = passes and row_passes
        rows.append(
            {
                "rank": idx,
                "ref_class_id": ref_item.get("class_id"),
                "candidate_rank": best_index,
                "candidate_class_id": best_item.get("class_id"),
                "class_match": class_match,
                "iou": max(best_iou, 0.0),
                "score_delta": score_delta,
                "passes": row_passes,
            }
        )
    return {
        "passes_detection_gate": bool(passes),
        "reference_detection_count": len(ref),
        "candidate_detection_count": len(cand),
        "compared_topk": count,
        "rows": rows,
    }


def apply_yolo_correctness(args, baseline, result, detection_context, candidate_output_dir):
    raw_gate = compare_raw_outputs(baseline, result)
    result["raw_gate_passed"] = bool(raw_gate.get("passes_raw_output_gate"))
    result["raw_mismatch_summary"] = raw_gate
    result["correctness"] = {"raw_output_gate": raw_gate}
    if detection_context is not None:
        net, assets, data_hwc = detection_context
        arrays = load_dumped_output_arrays(result.get("_last_row", {}), candidate_output_dir)
        if arrays:
            result["detection_summary"] = detection_summary(args, net, assets, data_hwc, arrays)
            detection_gate = compare_detection_outputs(baseline, result)
        else:
            detection_gate = {
                "passes_detection_gate": False,
                "error_summary": "native output dumps are unavailable",
            }
        result["detection_gate_passed"] = bool(detection_gate.get("passes_detection_gate"))
        result["correctness"]["detection_gate"] = detection_gate
    else:
        result["detection_gate_passed"] = False
    if args.correctness_policy == "raw_gate":
        result["passes_correctness_gate"] = bool(result["raw_gate_passed"])
    else:
        result["passes_correctness_gate"] = bool(result["detection_gate_passed"])
    result["correctness_policy"] = args.correctness_policy
    result.pop("_last_row", None)
    return result


def run_package(args, candidate, package_dir, serial, output_root, baseline, detection_context=None):
    mode = "serial" if serial else "pipeline"
    result_jsonl = "native_{}_result.jsonl".format(mode)
    target = ssh_target(args)
    opts = ssh_options(args)
    remote = args.remote_dir.rstrip("/") + "/" + safe_name(candidate["candidate_id"])
    cand_out = Path(output_root) / candidate["candidate_id"]
    cand_out.mkdir(parents=True, exist_ok=True)
    log_path = cand_out / "{}.log".format(mode)
    tar_path = package_tar(package_dir)
    timeout = args.serial_timeout_s if serial else args.pipeline_timeout_s
    try:
        run_cmd(["ssh"] + opts + [target, "rm -rf {d} && mkdir -p {d}".format(d=shlex.quote(remote))])
        run_cmd(["scp"] + opts + [str(tar_path), "{}:{}/package.tar.gz".format(target, remote)])
        run_cmd(["ssh"] + opts + [target, "cd {d} && tar xzf package.tar.gz".format(d=shlex.quote(remote))])
        script = "./run_native_serial.sh" if serial else "./run_native_pipeline.sh"
        run_cmd(
            ["ssh"] + opts + [target, "cd {d} && {s}".format(d=shlex.quote(remote), s=script)],
            timeout=int(timeout),
            stdout_path=log_path,
        )
        local_jsonl = cand_out / result_jsonl
        run_cmd(
            ["scp"] + opts + ["{}:{}/{}".format(target, remote, result_jsonl), str(local_jsonl)],
            timeout=int(args.fetch_timeout_s),
        )
        dump_dir_name = "{}_output_dumps".format(mode)
        try:
            run_cmd(
                ["scp", "-r"]
                + opts
                + ["{}:{}/{}".format(target, remote, dump_dir_name), str(cand_out / dump_dir_name)],
                timeout=int(args.fetch_timeout_s),
            )
        except Exception:
            pass
        run_cmd(["ssh"] + opts + [target, "rm -rf {}".format(shlex.quote(remote))])
        rows = [json.loads(line) for line in local_jsonl.read_text(encoding="utf-8").splitlines() if line]
        if not rows:
            raise RuntimeError("no result rows")
        completion_key = "stage{}_end_ms".format(int(rows[0].get("stage_count", 3)) - 1)
        completion = [float(row.get(completion_key, 0.0)) for row in rows]
        latencies = [float(row.get("total_latency_ms", 0.0)) for row in rows]
        if (not serial) and len(completion) > 1 and completion[-1] > completion[0]:
            fps = (len(completion) - 1) * 1000.0 / (completion[-1] - completion[0])
        else:
            fps = 1000.0 / statistics.mean(latencies)
        stage_count = int(rows[0].get("stage_count", candidate.get("stage_count", 0)) or 0)
        stage_mean_ms = []
        stage_set_mean_ms = []
        stage_run_mean_ms = []
        stage_get_mean_ms = []
        for stage_index in range(stage_count):
            stage_mean_ms.append(
                statistics.mean(
                    float(row.get("stage{}_ms".format(stage_index), 0.0)) for row in rows
                )
            )
            stage_set_mean_ms.append(
                statistics.mean(
                    float(row.get("stage{}_set_ms".format(stage_index), 0.0)) for row in rows
                )
            )
            stage_run_mean_ms.append(
                statistics.mean(
                    float(row.get("stage{}_run_ms".format(stage_index), 0.0)) for row in rows
                )
            )
            stage_get_mean_ms.append(
                statistics.mean(
                    float(row.get("stage{}_get_ms".format(stage_index), 0.0)) for row in rows
                )
            )
        bottleneck_stage = None
        stage_imbalance_ms = None
        if stage_mean_ms:
            bottleneck_stage = max(range(len(stage_mean_ms)), key=lambda item: stage_mean_ms[item])
            stage_imbalance_ms = max(stage_mean_ms) - statistics.median(stage_mean_ms)
        result = {
            "candidate_id": candidate["candidate_id"],
            "mode": mode,
            "status": "ok",
            "throughput_fps": fps,
            "pipeline_throughput_fps": fps if mode == "pipeline" else None,
            "mean_total_latency_ms": statistics.mean(latencies),
            "measured_cycle_ms": 1000.0 / max(fps, 1e-9),
            "stage_mean_ms": stage_mean_ms,
            "stage_times_ms": stage_mean_ms,
            "stage_set_mean_ms": stage_set_mean_ms,
            "stage_run_mean_ms": stage_run_mean_ms,
            "stage_get_mean_ms": stage_get_mean_ms,
            "bottleneck_stage": bottleneck_stage,
            "stage_imbalance_ms": stage_imbalance_ms,
            "raw_outputs": rows[-1].get("raw_outputs", []),
            "stage_raw_outputs": rows[-1].get("stage_raw_outputs", []),
            "_last_row": rows[-1],
            "split_boundary_diagnostics": split_boundary_diagnostics(candidate, rows[-1]),
            "result_jsonl": str(local_jsonl),
            "log": str(log_path),
        }
        result = apply_yolo_correctness(args, baseline, result, detection_context, cand_out)
        write_json(cand_out / "{}_result.json".format(mode), result)
        return result
    except Exception as err:  # pylint: disable=broad-except
        try:
            run_cmd(["ssh"] + opts + [target, "rm -rf {}".format(shlex.quote(remote))])
        except Exception:
            pass
        return {
            "candidate_id": candidate["candidate_id"],
            "mode": mode,
            "status": "timeout" if "timed out" in str(err).lower() else "failed",
            "failure_type": "timeout" if "timed out" in str(err).lower() else "{}_run_failed".format(mode),
            "error_summary": str(err)[-1000:],
            "passes_correctness_gate": False,
        }


def preflight(args):
    target = ssh_target(args)
    opts = ssh_options(args)
    cmd = (
        "cat /sys/class/fpga_manager/fpga0/state; "
        "cat /sys/class/u-dma-buf/udmabuf0/size; "
        "df -h /var/volatile"
    )
    return run_cmd(["ssh"] + opts + [target, cmd])


def run_packed_graph_flow(args, output_root):
    _, graph_path, _, _ = packed_graph_source_paths(args)
    full_graph = json.load(open(graph_path, encoding="utf-8"))
    cpu_context = None
    if args.split_backend == "packed_hetero":
        candidates = enumerate_packed_hetero_candidates(full_graph, int(args.candidate_count))
        candidate_csv_name = "packed_hetero_candidates.csv"
        cpu_context = yolo_relay(args)
    else:
        candidates = enumerate_packed_graph_candidates(full_graph, int(args.candidate_count))
        candidate_csv_name = "packed_graph_candidates.csv"
    if args.candidate_id:
        candidates = [item for item in candidates if item["candidate_id"] == args.candidate_id]
        if not candidates:
            raise RuntimeError("candidate id not found: {}".format(args.candidate_id))
    csv_candidates = []
    for candidate in candidates:
        row = dict(candidate)
        row["stage_metrics_json"] = json.dumps(row.pop("stage_metrics", []), separators=(",", ":"))
        csv_candidates.append(row)
    write_csv(output_root / candidate_csv_name, csv_candidates)
    write_csv(output_root / "yolo_heuristic_candidates.csv", csv_candidates)
    write_json(output_root / "yolo_heuristic_candidates.json", {"candidates": candidates})

    packages = {}
    build_rows = []
    if "build" in args.mode:
        for index, candidate in enumerate(candidates, 1):
            print("[PACKED BUILD] {}/{} {}".format(index, len(candidates), candidate["candidate_id"]))
            try:
                _, package_dir = package_packed_graph_candidate(
                    args,
                    output_root,
                    candidate=candidate,
                    full_graph=full_graph,
                    cpu_context=cpu_context,
                )
                packages[candidate["candidate_id"]] = package_dir
                build_rows.append({**candidate, "status": "buildable", "package_dir": str(package_dir)})
            except Exception as err:  # pylint: disable=broad-except
                build_rows.append(
                    {
                        **candidate,
                        "status": "build_failed",
                        "failure_type": "build_failed",
                        "error_summary": str(err)[-1000:],
                    }
                )
            write_csv(output_root / "buildability.csv", build_rows)
    else:
        buildability_path = output_root / "buildability.csv"
        with buildability_path.open(encoding="utf-8") as inp:
            build_rows = list(csv.DictReader(inp))
        for row in build_rows:
            if row.get("status") == "buildable":
                package_dir = Path(row.get("package_dir") or output_root / "packages" / row["candidate_id"])
                if package_dir.exists():
                    packages[row["candidate_id"]] = package_dir

    write_json(
        output_root / "buildability_summary.json",
        {
            "rows": build_rows,
            "candidate_count": len(candidates),
            "buildable_count": sum(1 for row in build_rows if row.get("status") == "buildable"),
            "safe_candidate_count": len(candidates),
            "rejected_candidate_count": 0,
            "split_backend": args.split_backend,
        },
    )
    if args.build_only or "measure" not in args.mode:
        return

    if not Path(args.rpc_baseline_json).exists():
        raise RuntimeError("missing RPC baseline JSON: {}".format(args.rpc_baseline_json))
    baseline = json.load(open(args.rpc_baseline_json, encoding="utf-8"))
    preflight_text = preflight(args)
    (output_root / "board_preflight.txt").write_text(preflight_text, encoding="utf-8")
    measure_rows = []
    buildable = [row for row in build_rows if row.get("status") == "buildable"][: int(args.measure_count)]
    for index, build_row in enumerate(buildable, 1):
        candidate = next(item for item in candidates if item["candidate_id"] == build_row["candidate_id"])
        package_dir = packages.get(candidate["candidate_id"]) or Path(build_row["package_dir"])
        print("[PACKED SERIAL] {}/{} {}".format(index, len(buildable), candidate["candidate_id"]))
        serial_result = run_package(
            args,
            candidate,
            package_dir,
            serial=True,
            output_root=output_root,
            baseline=baseline,
        )
        serial_row = {**build_row, **serial_result, "run_kind": "serial_smoke"}
        if serial_result.get("status") == "ok" and not serial_result.get("passes_correctness_gate"):
            serial_row["status"] = "serial_correctness_failed"
            serial_row["failure_type"] = "serial_correctness_failed"
            serial_row["error_summary"] = "serial raw-output correctness gate failed"
        measure_rows.append(serial_row)
        write_csv(output_root / "summary.csv", measure_rows)
        write_json(output_root / "summary.json", {"rows": measure_rows, "phase": "serial_smoke"})
        if not serial_result.get("passes_correctness_gate"):
            continue
        print("[PACKED PIPELINE] {}/{} {}".format(index, len(buildable), candidate["candidate_id"]))
        pipeline_result = run_package(
            args,
            candidate,
            package_dir,
            serial=False,
            output_root=output_root,
            baseline=baseline,
        )
        measure_rows.append({**build_row, **pipeline_result, "run_kind": "pipeline"})
        write_csv(output_root / "summary.csv", measure_rows)
        write_json(output_root / "summary.json", {"rows": measure_rows, "phase": "measure"})

    ok = [
        row
        for row in measure_rows
        if row.get("run_kind") == "pipeline" and row.get("status") == "ok"
        and row.get("passes_correctness_gate")
    ]
    ok.sort(key=lambda row: float(row.get("throughput_fps") or 0.0), reverse=True)
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split_backend": args.split_backend,
        "candidate_count": len(candidates),
        "buildable_count": len(buildable),
        "measure_count": int(args.measure_count),
        "pipeline_ok_count": len(ok),
        "best": ok[0] if ok else None,
        "top10": ok[:10],
        "rows": measure_rows,
        "baselines": {
            "rpc_all_vta_fps": 3.724346058177234,
            "packed_graph_smoke_fps": 6.52152984603568,
        },
    }
    write_csv(output_root / "summary.csv", measure_rows)
    write_json(output_root / "summary.json", summary)
    readme = [
        "# YOLOv3-tiny Packed Graph Top20 Search",
        "",
        "This run splits the already-correct all-VTA graph JSON instead of rebuilding stages from Relay.",
        "Boundary tensors keep physical shape, dtype, layout, padding, and quantization domain.",
        "",
        "- Candidates: {}".format(len(candidates)),
        "- Buildable: {}".format(len(buildable)),
        "- Pipeline OK: {}".format(len(ok)),
        "- Baseline RPC all-VTA FPS: 3.7243",
        "- Baseline packed smoke FPS: 6.5215",
    ]
    if ok:
        readme.append("- Best: {} FPS {:.4f}".format(ok[0]["candidate_id"], float(ok[0].get("throughput_fps") or 0.0)))
        readme.append("")
        readme.append("## Top 10")
        for rank, row in enumerate(ok[:10], 1):
            readme.append(
                "{}. `{}` {:.4f} fps correctness={}".format(
                    rank,
                    row["candidate_id"],
                    float(row.get("throughput_fps") or 0.0),
                    bool(row.get("passes_correctness_gate")),
                )
            )
    (output_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.split_backend in ("packed_graph", "packed_hetero"):
        run_packed_graph_flow(args, output_root)
        return

    heuristic_path = Path(args.heuristic_params_json)

    if "fit" in args.mode or not heuristic_path.exists():
        heuristic = fit_heuristic(output_root, heuristic_path)
    else:
        heuristic = json.load(open(heuristic_path, encoding="utf-8"))

    base_candidate_count = int(args.candidate_count)
    if int(args.runtime_config_search_count or 0) > 0:
        base_candidate_count = max(base_candidate_count, len(START_SPECS) * len(END_SPECS))
    candidates = enumerate_candidates(heuristic, base_candidate_count)
    candidates = expand_runtime_configs(args, candidates)
    if args.candidate_id:
        candidates = [
            item
            for item in candidates
            if item["candidate_id"] == args.candidate_id
            or item.get("split_candidate_id") == args.candidate_id
        ]
        if not candidates:
            raise RuntimeError("candidate id not found: {}".format(args.candidate_id))

    if "build" not in args.mode and "measure" not in args.mode:
        write_csv(output_root / "yolo_heuristic_candidates.csv", candidates)
        write_json(output_root / "yolo_heuristic_candidates.json", {"candidates": candidates})
        return

    env, assets, func, params, named, data, data_hwc = yolo_relay(args)
    detection_context = (load_darknet_net(assets), assets, data_hwc)
    validations = [
        validate_boundary_contracts(
            candidate,
            boundary_contracts_for_candidate(candidate, named, env, args.tail_device),
            args,
        )
        for candidate in candidates
    ]
    candidates, safe_candidates, rejected_candidates = write_boundary_artifacts(
        output_root, candidates, validations
    )
    write_csv(output_root / "yolo_heuristic_candidates.csv", candidates)
    write_json(output_root / "yolo_heuristic_candidates.json", {"candidates": candidates})

    buildability_path = output_root / "buildability.csv"
    packages = {}
    if "build" not in args.mode and buildability_path.exists():
        with buildability_path.open(encoding="utf-8") as inp:
            build_rows = list(csv.DictReader(inp))
        for row in build_rows:
            if row.get("status") == "buildable":
                package_dir = Path(row.get("package_dir") or output_root / "packages" / row["candidate_id"])
                if package_dir.exists():
                    packages[row["candidate_id"]] = package_dir
    else:
        build_rows = [
            {
                **candidate,
                "status": "boundary_invalid",
                "failure_type": "boundary_invalid",
                "error_summary": candidate.get("boundary_reasons", ""),
            }
            for candidate in rejected_candidates
        ]
        for candidate in safe_candidates:
            try:
                package_dir = package_candidate(args, output_root, candidate, env, func, params, named, data)
                packages[candidate["candidate_id"]] = package_dir
                build_rows.append({**candidate, "status": "buildable", "package_dir": str(package_dir)})
            except Exception as err:  # pylint: disable=broad-except
                build_rows.append(
                    {**candidate, "status": "build_failed", "failure_type": "build_failed", "error_summary": str(err)[-1000:]}
                )
                write_csv(output_root / "buildability.csv", build_rows)
        write_csv(output_root / "buildability.csv", build_rows)
        write_json(
            output_root / "buildability_summary.json",
            {
                "rows": build_rows,
                "candidate_count": len(candidates),
                "safe_candidate_count": len(safe_candidates),
                "rejected_candidate_count": len(rejected_candidates),
            },
        )

    if args.build_only or "measure" not in args.mode:
        return

    if not Path(args.rpc_baseline_json).exists():
        raise RuntimeError("missing RPC baseline JSON: {}".format(args.rpc_baseline_json))
    baseline = json.load(open(args.rpc_baseline_json, encoding="utf-8"))
    preflight_text = preflight(args)
    (output_root / "board_preflight.txt").write_text(preflight_text, encoding="utf-8")

    measure_rows = []
    buildable = [row for row in build_rows if row.get("status") == "buildable"][
        : int(args.measure_count)
    ]
    if not buildable:
        raise RuntimeError("no buildable YOLO candidates")

    serial_pass = []
    for first in buildable:
        serial_result = run_package(
            args,
            first,
            packages[first["candidate_id"]],
            serial=True,
            output_root=output_root,
            baseline=baseline,
            detection_context=detection_context,
        )
        serial_row = {**first, **serial_result, "run_kind": "serial_smoke"}
        if serial_result.get("status") == "ok" and not serial_result.get("passes_correctness_gate"):
            serial_row["status"] = "serial_correctness_failed"
            serial_row["failure_type"] = "serial_correctness_failed"
            serial_row["error_summary"] = "serial {} correctness gate failed".format(
                args.correctness_policy
            )
        measure_rows.append(serial_row)
        write_csv(output_root / "summary.csv", measure_rows)
        write_json(output_root / "summary.json", {"rows": measure_rows, "phase": "serial_smoke"})
        if serial_result.get("passes_correctness_gate"):
            serial_pass.append(first)

    if not serial_pass:
        write_json(
            output_root / "summary.json",
            {"rows": measure_rows, "aborted": "all_serial_smoke_failed"},
        )
        return

    for row in serial_pass:
        candidate = next(item for item in safe_candidates if item["candidate_id"] == row["candidate_id"])
        result = run_package(
            args,
            candidate,
            packages[candidate["candidate_id"]],
            serial=False,
            output_root=output_root,
            baseline=baseline,
            detection_context=detection_context,
        )
        measure_rows.append({**candidate, **result, "run_kind": "pipeline"})
        write_csv(output_root / "summary.csv", measure_rows)
        write_json(output_root / "summary.json", {"rows": measure_rows})

    ok = [row for row in measure_rows if row.get("run_kind") == "pipeline" and row.get("status") == "ok"]
    ok.sort(key=lambda row: float(row.get("throughput_fps") or 0.0), reverse=True)
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "heuristic_params_json": str(heuristic_path),
        "candidate_count": len(candidates),
        "buildable_count": len(buildable),
        "measure_count": int(args.measure_count),
        "pipeline_ok_count": len(ok),
        "best": ok[0] if ok else None,
        "top10": ok[:10],
        "rows": measure_rows,
        "baselines": {
            "rpc_all_vta_fps": 3.724346058177234,
            "native_single_stage_pipeline_fps": 2.9619880760327932,
            "packed_graph_smoke_fps": 6.52152984603568,
            "packed_hetero_best_fps": 4.2922,
        },
    }
    write_json(output_root / "summary.json", summary)
    readme = [
        "# YOLOv3-tiny ResNet-Style CPU/VTA/CPU Smoke",
        "",
        "- Candidates: {}".format(len(candidates)),
        "- Buildable: {}".format(len(buildable)),
        "- Pipeline OK: {}".format(len(ok)),
        "- Best: {} fps={:.4f}".format(
            ok[0]["candidate_id"], float(ok[0].get("throughput_fps", 0.0))
        )
        if ok
        else "- Best: n/a",
        "- Correctness policy: {}".format(args.correctness_policy),
        "",
        "Baselines: RPC all-VTA 3.724 fps; native single-stage pipeline 2.962 fps; "
        "packed graph smoke 6.5215 fps; packed hetero best 4.2922 fps.",
        "",
    ]
    if ok:
        best = ok[0]
        readme.extend(
            [
                "## Best Details",
                "",
                "- Candidate: `{}`".format(best.get("candidate_id")),
                "- Split: `stage0_cpu -> stage1_vta -> stage2_cpu`",
                "- Stage mean ms: {}".format(best.get("stage_mean_ms")),
                "- Raw gate passed: {}".format(best.get("raw_gate_passed")),
                "- Detection gate passed: {}".format(best.get("detection_gate_passed")),
                "- Raw mismatch max mean delta: {}".format(
                    (best.get("raw_mismatch_summary") or {}).get("max_mean_abs_delta")
                ),
                "",
            ]
        )
    (output_root / "README.md").write_text("\n".join(readme), encoding="utf-8")


if __name__ == "__main__":
    main()
