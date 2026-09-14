#!/usr/bin/env python3
"""Benchmark representative ResNet18 single operators on CPU and VTA.

This entrypoint is intentionally pragmatic:
1. Reuse the existing standalone conv2d VTA/CPU benchmark implementation.
2. Keep the ResNet18 operator list in one place.
3. Emit one CSV that is easy to sort/filter for heterogeneous partitioning.

At the moment, only conv2d has a stable standalone VTA benchmark path in this
repo. Pooling and dense cases are still included in the output so the user can
track coverage, but they are marked as skipped by default.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tvm
from tvm import autotvm, relay, rpc
from tvm.contrib import cc, graph_executor, utils

import vta
from hp_hpc_quant.vta_runtime_profile_utils import (
    clear_runtime_profiler,
    dump_runtime_snapshot,
    ensure_dir,
    fetch_runtime_profiler_hooks,
    hooks_available,
)


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    op_name: str
    batch: int
    channels_in: int
    height: int
    width: int
    channels_out: Optional[int]
    kernel_h: Optional[int]
    kernel_w: Optional[int]
    stride_h: Optional[int]
    stride_w: Optional[int]
    pad_h: Optional[int]
    pad_w: Optional[int]
    note: str


class CaptureWriter:
    """Capture a single row from DictWriter-style callbacks."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, object]] = []

    def writerow(self, row: Dict[str, object]) -> None:
        self.rows.append(dict(row))


env = vta.get_env()


def load_conv_benchmark_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "python"
        / "integration"
        / "test_benchmark_topi_conv2d.py"
    )
    spec = importlib.util.spec_from_file_location("vta_conv2d_bench", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load conv2d benchmark helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_conv_bench = load_conv_benchmark_module()
Workload = _conv_bench.Workload
run_conv2d_decomposed = _conv_bench.run_conv2d_decomposed


RESNET18_CASES: List[BenchmarkCase] = [
    BenchmarkCase("stem_conv7x7_s2", "conv2d", env.BATCH, 3, 224, 224, 64, 7, 7, 2, 2, 3, 3, "Stem conv"),
    BenchmarkCase("stem_maxpool3x3_s2", "max_pool2d", env.BATCH, 64, 112, 112, 64, 3, 3, 2, 2, 1, 1, "Stem max pool"),
    BenchmarkCase("s1_conv3x3_64_64", "conv2d", env.BATCH, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, "Stage1 main path"),
    BenchmarkCase("s2_conv3x3_64_128_s2", "conv2d", env.BATCH, 64, 56, 56, 128, 3, 3, 2, 2, 1, 1, "Stage2 downsample"),
    BenchmarkCase("s2_proj1x1_64_128_s2", "conv2d", env.BATCH, 64, 56, 56, 128, 1, 1, 2, 2, 0, 0, "Stage2 shortcut"),
    BenchmarkCase("s2_conv3x3_128_128", "conv2d", env.BATCH, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, "Stage2 main path"),
    BenchmarkCase("s3_conv3x3_128_256_s2", "conv2d", env.BATCH, 128, 28, 28, 256, 3, 3, 2, 2, 1, 1, "Stage3 downsample"),
    BenchmarkCase("s3_proj1x1_128_256_s2", "conv2d", env.BATCH, 128, 28, 28, 256, 1, 1, 2, 2, 0, 0, "Stage3 shortcut"),
    BenchmarkCase("s3_conv3x3_256_256", "conv2d", env.BATCH, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, "Stage3 main path"),
    BenchmarkCase("s4_conv3x3_256_512_s2", "conv2d", env.BATCH, 256, 14, 14, 512, 3, 3, 2, 2, 1, 1, "Stage4 downsample"),
    BenchmarkCase("s4_proj1x1_256_512_s2", "conv2d", env.BATCH, 256, 14, 14, 512, 1, 1, 2, 2, 0, 0, "Stage4 shortcut"),
    BenchmarkCase("s4_conv3x3_512_512", "conv2d", env.BATCH, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, "Stage4 main path"),
    BenchmarkCase("residual_add_relu_256_14", "add_relu", env.BATCH, 256, 14, 14, 256, None, None, None, None, None, None, "Residual add and ReLU tail"),
    BenchmarkCase("tail_global_avg_pool", "global_avg_pool2d", env.BATCH, 512, 7, 7, 512, None, None, None, None, None, None, "Tail pooling"),
    BenchmarkCase("tail_dense_512_1000", "dense", env.BATCH, 512, 1, 1, 1000, None, None, None, None, None, None, "Classifier"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.1.248", help="RPC server host")
    parser.add_argument("--port", type=int, default=9090, help="RPC server port")
    parser.add_argument(
        "--devices",
        default="arm_cpu,vta",
        help="Comma-separated devices to benchmark: arm_cpu,vta",
    )
    parser.add_argument("--number", type=int, default=10, help="Repeat count for steady-state timing")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs per case")
    parser.add_argument(
        "--cpu-num-threads",
        type=int,
        default=0,
        help="Configure the remote TVM CPU thread pool before each CPU case; 0 keeps runtime default.",
    )
    parser.add_argument("--check-correctness", action="store_true", help="Run reference checks")
    parser.add_argument("--tune-log", default="", help="Optional AutoTVM log file")
    parser.add_argument("--output", default="", help="Optional CSV path")
    parser.add_argument(
        "--output-dir",
        default="",
        help="If set, write the benchmark CSV as raw_benchmark.csv under this directory unless --output is explicitly given",
    )
    parser.add_argument(
        "--case-file",
        default="",
        help="Optional CSV or JSON file that defines custom benchmark cases; when set, it replaces the built-in RESNET18_CASES list",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="Comma-separated case ids to run; empty means all cases",
    )
    parser.add_argument(
        "--vta-runtime-profile-dir",
        default="",
        help="Optional directory to dump per-case VTA runtime profiler status/events JSON",
    )
    parser.add_argument(
        "--vta-runtime-profile-events-limit",
        type=int,
        default=200,
        help="Maximum number of VTA runtime profiler events to dump per snapshot",
    )
    return parser.parse_args()


def _parse_optional_int(value):
    if value in ["", None]:
        return None
    return int(value)


def _case_from_mapping(raw: Dict[str, object]) -> BenchmarkCase:
    def get_required(key):
        if key not in raw or raw[key] in ["", None]:
            raise ValueError("Missing required benchmark case field: {}".format(key))
        return raw[key]

    return BenchmarkCase(
        case_id=str(get_required("case_id")),
        op_name=str(raw.get("op_name", "conv2d")),
        batch=int(raw.get("batch", env.BATCH)),
        channels_in=int(get_required("channels_in")),
        height=int(get_required("height")),
        width=int(get_required("width")),
        channels_out=_parse_optional_int(raw.get("channels_out")),
        kernel_h=_parse_optional_int(raw.get("kernel_h")),
        kernel_w=_parse_optional_int(raw.get("kernel_w")),
        stride_h=_parse_optional_int(raw.get("stride_h")),
        stride_w=_parse_optional_int(raw.get("stride_w")),
        pad_h=_parse_optional_int(raw.get("pad_h")),
        pad_w=_parse_optional_int(raw.get("pad_w")),
        note=str(raw.get("note", "")),
    )


def load_case_file(case_file: str) -> List[BenchmarkCase]:
    path = Path(case_file)
    if not path.exists():
        raise FileNotFoundError("Case file not found: {}".format(case_file))
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON case file must contain a list of case objects")
        return [_case_from_mapping(item) for item in payload]
    if path.suffix.lower() == ".csv":
        with path.open() as f:
            return [_case_from_mapping(row) for row in csv.DictReader(f)]
    raise ValueError("Unsupported case file format: {}".format(path.suffix))


def selected_cases(case_filter: str, case_file: str = "") -> List[BenchmarkCase]:
    source_cases = load_case_file(case_file) if case_file else RESNET18_CASES
    if not case_filter.strip():
        return source_cases
    wanted = {item.strip() for item in case_filter.split(",") if item.strip()}
    return [case for case in source_cases if case.case_id in wanted]


def make_conv_workload(case: BenchmarkCase) -> Workload:
    assert case.channels_out is not None
    assert case.kernel_h is not None and case.kernel_w is not None
    assert case.pad_h is not None and case.pad_w is not None
    assert case.stride_h is not None and case.stride_w is not None
    return Workload(
        case.batch,
        case.height,
        case.width,
        case.channels_in,
        case.channels_out,
        case.kernel_h,
        case.kernel_w,
        case.pad_h,
        case.pad_w,
        case.stride_h,
        case.stride_w,
    )


def device_target(device: str):
    if device == "vta":
        return env.target
    if device == "arm_cpu":
        return env.target_vta_cpu
    raise ValueError(f"Unsupported device: {device}")


def connect_remote(host: str, port: int):
    if env.TARGET in ["sim", "tsim", "intelfocl"]:
        return rpc.LocalSession()
    return rpc.connect(host, port)


def fetch_optional_runtime_func(remote, name: str):
    try:
        return remote.get_function(name)
    except Exception:  # pylint: disable=broad-except
        return None


def configure_remote_threadpool(remote, num_threads: int) -> Dict[str, object]:
    config_fn = fetch_optional_runtime_func(remote, "runtime.config_threadpool")
    num_threads_fn = fetch_optional_runtime_func(remote, "runtime.NumThreads")
    before = int(num_threads_fn()) if num_threads_fn is not None else None
    if int(num_threads) > 0:
        if config_fn is None:
            raise RuntimeError("remote runtime.config_threadpool is unavailable")
        config_fn(1, int(num_threads))
    after = int(num_threads_fn()) if num_threads_fn is not None else before
    if int(num_threads) > 0 and after is not None and after != int(num_threads):
        raise RuntimeError(
            "remote CPU thread-pool configuration failed: requested={} actual={}".format(
                num_threads, after
            )
        )
    return {
        "cpu_num_threads_requested": int(num_threads) if int(num_threads) > 0 else "",
        "cpu_num_threads_actual": after if after is not None else "",
    }


def skipped_row(case: BenchmarkCase, device: str, reason: str) -> Dict[str, object]:
    return {
        "case_id": case.case_id,
        "op_name": case.op_name,
        "device": device,
        "implemented": False,
        "status": "skipped",
        "note": f"{case.note}; {reason}",
        "batch": case.batch,
        "channels_in": case.channels_in,
        "height": case.height,
        "width": case.width,
        "channels_out": case.channels_out,
        "kernel_h": case.kernel_h,
        "kernel_w": case.kernel_w,
        "stride_h": case.stride_h,
        "stride_w": case.stride_w,
        "pad_h": case.pad_h,
        "pad_w": case.pad_w,
        "kernel_ms": "",
        "total_ms": "",
        "submit_ms": "",
        "sync_ms": "",
        "h2d_ms": "",
        "d2h_ms": "",
        "pack_ms": "",
        "gops": "",
        "ok": "",
        "cpu_num_threads_requested": "",
        "cpu_num_threads_actual": "",
    }


def is_vta_compatible_conv(case: BenchmarkCase) -> bool:
    if case.op_name != "conv2d":
        return False
    if case.channels_out is None:
        return False
    if case.channels_in % env.BLOCK_IN != 0:
        return False
    if case.channels_out % env.BLOCK_OUT != 0:
        return False
    if case.batch % env.BATCH != 0:
        return False
    return True


def case_memory_bytes(case: BenchmarkCase) -> int:
    """Conservative compulsory tensor traffic for one operator invocation."""

    input_elements = case.batch * case.channels_in * case.height * case.width
    if case.op_name == "conv2d":
        out_h = (
            case.height + 2 * int(case.pad_h or 0) - int(case.kernel_h or 1)
        ) // int(case.stride_h or 1) + 1
        out_w = (
            case.width + 2 * int(case.pad_w or 0) - int(case.kernel_w or 1)
        ) // int(case.stride_w or 1) + 1
        output_elements = case.batch * int(case.channels_out or 0) * out_h * out_w
        weight_elements = (
            int(case.channels_out or 0)
            * case.channels_in
            * int(case.kernel_h or 1)
            * int(case.kernel_w or 1)
        )
        return int(4 * (input_elements + weight_elements + output_elements))
    if case.op_name == "add_relu":
        return int(4 * 3 * input_elements)
    if case.op_name == "global_avg_pool2d":
        return int(4 * (input_elements + case.batch * case.channels_in))
    if case.op_name == "dense":
        output_elements = case.batch * int(case.channels_out or 0)
        weight_elements = case.channels_in * int(case.channels_out or 0)
        bias_elements = int(case.channels_out or 0)
        return int(4 * (case.batch * case.channels_in + weight_elements + bias_elements + output_elements))
    output_elements = case.batch * int(case.channels_out or case.channels_in)
    return int(4 * (input_elements + output_elements))


def benchmark_conv_case(
    remote,
    case: BenchmarkCase,
    device: str,
    number: int,
    warmup: int,
    check_correctness: bool,
    runtime_profile_dir: str = "",
    runtime_profile_events_limit: int = 200,
) -> Dict[str, object]:
    workload = make_conv_workload(case)
    target = device_target(device)
    capture = CaptureWriter()
    records: List[Dict[str, object]] = []
    profiler_hooks = None
    if device == "vta" and runtime_profile_dir:
        profiler_hooks = fetch_runtime_profiler_hooks(remote)
        if not hooks_available(profiler_hooks):
            raise RuntimeError(
                "VTA runtime profiler hooks are required when --vta-runtime-profile-dir is set"
            )
        clear_runtime_profiler(profiler_hooks)
    _, _, row = run_conv2d_decomposed(
        env,
        remote,
        workload,
        target,
        csv_writer=capture,
        records=records,
        check_correctness=check_correctness,
        print_ir=False,
        number=number,
        warmup=warmup,
    )
    if profiler_hooks is not None:
        dump_runtime_snapshot(
            runtime_profile_dir,
            "benchmark_totals",
            profiler_hooks,
            events_limit=runtime_profile_events_limit,
            extra={
                "case_id": case.case_id,
                "device": device,
                "warmup": int(warmup),
                "manual_repeats": int(number),
                "timer_repeats": int(number),
                "profiled_run_count": int(warmup) + 2 * int(number),
            },
        )
    memory_bytes = case_memory_bytes(case)
    kernel_ms = row["T_kernel_steady_s"] * 1e3
    return {
        "case_id": case.case_id,
        "op_name": case.op_name,
        "device": device,
        "implemented": True,
        "status": "ok" if row["ok"] else "failed",
        "note": case.note,
        "batch": case.batch,
        "channels_in": case.channels_in,
        "height": case.height,
        "width": case.width,
        "channels_out": case.channels_out,
        "kernel_h": case.kernel_h,
        "kernel_w": case.kernel_w,
        "stride_h": case.stride_h,
        "stride_w": case.stride_w,
        "pad_h": case.pad_h,
        "pad_w": case.pad_w,
        "kernel_ms": kernel_ms,
        "total_ms": row["total_s"] * 1e3,
        "submit_ms": row["T_submit_s"] * 1e3,
        "sync_ms": row["T_sync_s"] * 1e3,
        "h2d_ms": row["T_h2d_s"] * 1e3,
        "d2h_ms": row["T_d2h_s"] * 1e3,
        "pack_ms": row["T_pack_s"] * 1e3,
        "gops": row["gops_kernel_steady"],
        "memory_bytes_est": memory_bytes,
        "effective_memory_bandwidth_GBps": memory_bytes / max(kernel_ms, 1.0e-12) / 1.0e6,
        "ok": row["ok"],
    }


def export_and_upload_cpu_lib(lib, remote, case_id: str):
    temp = utils.tempdir()
    lib_name = "ramps_{}.so".format(case_id)
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


def make_cpu_relay_microbenchmark(case: BenchmarkCase):
    rng = np.random.RandomState(0)
    params = {}
    if case.op_name == "add_relu":
        shape = (case.batch, case.channels_in, case.height, case.width)
        lhs = relay.var("lhs", shape=shape, dtype="float32")
        rhs = relay.var("rhs", shape=shape, dtype="float32")
        body = relay.nn.relu(relay.add(lhs, rhs))
        inputs = {
            "lhs": rng.uniform(-1.0, 1.0, size=shape).astype("float32"),
            "rhs": rng.uniform(-1.0, 1.0, size=shape).astype("float32"),
        }
        num_ops = 2 * int(np.prod(shape))
        func = relay.Function([lhs, rhs], body)
    elif case.op_name == "global_avg_pool2d":
        shape = (case.batch, case.channels_in, case.height, case.width)
        data = relay.var("data", shape=shape, dtype="float32")
        body = relay.nn.global_avg_pool2d(data, layout="NCHW")
        inputs = {"data": rng.uniform(-1.0, 1.0, size=shape).astype("float32")}
        num_ops = int(np.prod(shape))
        func = relay.Function([data], body)
    elif case.op_name == "dense":
        units = int(case.channels_out or 0)
        data_shape = (case.batch, case.channels_in)
        weight_shape = (units, case.channels_in)
        data = relay.var("data", shape=data_shape, dtype="float32")
        weight = relay.var("weight", shape=weight_shape, dtype="float32")
        bias = relay.var("bias", shape=(units,), dtype="float32")
        body = relay.nn.bias_add(relay.nn.dense(data, weight, units=units), bias)
        inputs = {"data": rng.uniform(-1.0, 1.0, size=data_shape).astype("float32")}
        params = {
            "weight": rng.uniform(-0.1, 0.1, size=weight_shape).astype("float32"),
            "bias": rng.uniform(-0.1, 0.1, size=(units,)).astype("float32"),
        }
        num_ops = case.batch * units * (2 * case.channels_in + 1)
        func = relay.Function([data, weight, bias], body)
    else:
        raise ValueError("unsupported CPU Relay microbenchmark op: {}".format(case.op_name))
    return func, params, inputs, int(num_ops)


def benchmark_cpu_relay_case(
    remote,
    case: BenchmarkCase,
    number: int,
    warmup: int,
) -> Dict[str, object]:
    func, params, inputs, num_ops = make_cpu_relay_microbenchmark(case)
    target = tvm.target.Target(env.target_vta_cpu, host=env.target_host)
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, lowered_params = relay.build(
            tvm.IRModule.from_expr(func), target=target, params=params
        )
    loaded = export_and_upload_cpu_lib(lib, remote, case.case_id)
    module = graph_executor.create(graph, loaded, remote.cpu(0))
    module.set_input(**lowered_params)

    set_samples = []
    for _ in range(max(1, int(number))):
        start = time.perf_counter()
        module.set_input(**inputs)
        set_samples.append((time.perf_counter() - start) * 1000.0)
    for _ in range(max(0, int(warmup))):
        module.run()
    timer = module.module.time_evaluator(
        "run", remote.cpu(0), number=1, repeat=max(1, int(number))
    )
    run_samples = [float(item) * 1000.0 for item in timer().results]
    get_samples = []
    for _ in range(max(1, int(number))):
        start = time.perf_counter()
        module.get_output(0).numpy()
        get_samples.append((time.perf_counter() - start) * 1000.0)

    set_ms = float(np.median(set_samples))
    run_ms = float(np.median(run_samples))
    get_ms = float(np.median(get_samples))
    gops = (float(num_ops) / 1.0e9) / max(run_ms / 1000.0, 1.0e-12)
    memory_bytes = case_memory_bytes(case)
    return {
        "case_id": case.case_id,
        "op_name": case.op_name,
        "device": "arm_cpu",
        "implemented": True,
        "status": "ok",
        "note": case.note,
        "batch": case.batch,
        "channels_in": case.channels_in,
        "height": case.height,
        "width": case.width,
        "channels_out": case.channels_out,
        "kernel_h": case.kernel_h,
        "kernel_w": case.kernel_w,
        "stride_h": case.stride_h,
        "stride_w": case.stride_w,
        "pad_h": case.pad_h,
        "pad_w": case.pad_w,
        "kernel_ms": run_ms,
        "total_ms": set_ms + run_ms + get_ms,
        "submit_ms": 0.0,
        "sync_ms": 0.0,
        "h2d_ms": set_ms,
        "d2h_ms": get_ms,
        "pack_ms": 0.0,
        "gops": gops,
        "memory_bytes_est": memory_bytes,
        "effective_memory_bandwidth_GBps": memory_bytes / max(run_ms, 1.0e-12) / 1.0e6,
        "ok": True,
    }


def output_path(arg_output: str) -> str:
    if arg_output:
        return arg_output
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"resnet18_single_op_bench_{ts}.csv"


def resolve_output_path(arg_output: str, output_dir: str) -> str:
    if arg_output:
        return arg_output
    if output_dir:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = Path("/home/orange/code/tvm") / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / "raw_benchmark.csv")
    return output_path(arg_output)


def main() -> None:
    args = parse_args()
    cases = selected_cases(args.cases, args.case_file)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    csv_path = resolve_output_path(args.output, args.output_dir)
    remote = connect_remote(args.host, args.port)

    print("========== Configuration ==========")
    print("env.TARGET   =", env.TARGET)
    print("host         =", args.host)
    print("port         =", args.port)
    print("devices      =", devices)
    print("case_file    =", args.case_file if args.case_file else "<builtin>")
    print("cases        =", [case.case_id for case in cases])
    print("number       =", args.number)
    print("warmup       =", args.warmup)
    print("cpu threads  =", args.cpu_num_threads if args.cpu_num_threads else "<runtime default>")
    print("tune_log     =", args.tune_log if args.tune_log else "<none>")
    print("output       =", csv_path)
    print("===================================")

    fieldnames = [
        "case_id",
        "op_name",
        "device",
        "implemented",
        "status",
        "note",
        "batch",
        "channels_in",
        "height",
        "width",
        "channels_out",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        "pad_h",
        "pad_w",
        "kernel_ms",
        "total_ms",
        "submit_ms",
        "sync_ms",
        "h2d_ms",
        "d2h_ms",
        "pack_ms",
        "gops",
        "memory_bytes_est",
        "effective_memory_bandwidth_GBps",
        "ok",
        "cpu_num_threads_requested",
        "cpu_num_threads_actual",
    ]

    if args.tune_log and os.path.exists(args.tune_log):
        build_ctx = autotvm.apply_history_best(args.tune_log)
    else:
        build_ctx = nullcontext()

    rows: List[Dict[str, object]] = []
    with build_ctx:
        for device in devices:
            for case in cases:
                print(f"\n=== {case.case_id} op={case.op_name} device={device} ===")
                thread_info = {
                    "cpu_num_threads_requested": "",
                    "cpu_num_threads_actual": "",
                }
                if device == "arm_cpu":
                    thread_info = configure_remote_threadpool(remote, args.cpu_num_threads)
                if case.op_name != "conv2d" and device == "vta":
                    row = skipped_row(
                        case,
                        device,
                        "standalone VTA benchmark path is implemented only for conv2d",
                    )
                elif case.op_name != "conv2d":
                    row = benchmark_cpu_relay_case(
                        remote,
                        case,
                        number=args.number,
                        warmup=args.warmup,
                    )
                elif device == "vta" and not is_vta_compatible_conv(case):
                    row = skipped_row(
                        case,
                        device,
                        "incompatible with VTA packed conv requirements; channels/batch must align with VTA block sizes",
                    )
                else:
                    case_profile_dir = ""
                    if device == "vta" and args.vta_runtime_profile_dir:
                        case_profile_dir = str(
                            Path(ensure_dir(args.vta_runtime_profile_dir)) / "single_op" / case.case_id
                        )
                    row = benchmark_conv_case(
                        remote,
                        case,
                        device,
                        number=args.number,
                        warmup=args.warmup,
                        check_correctness=args.check_correctness,
                        runtime_profile_dir=case_profile_dir,
                        runtime_profile_events_limit=args.vta_runtime_profile_events_limit,
                    )
                    print(
                        "kernel={:.4f} ms total={:.4f} ms gops={:.2f} ok={}".format(
                            row["kernel_ms"],
                            row["total_ms"],
                            row["gops"],
                            row["ok"],
                        )
                    )
                row.update(thread_info)
                rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[CSV] wrote {len(rows)} rows to {csv_path}")
    conv_rows = [row for row in rows if row["implemented"]]
    if conv_rows:
        print("\n=== Fastest kernels by device ===")
        for device in devices:
            device_rows = [row for row in conv_rows if row["device"] == device]
            if not device_rows:
                continue
            fastest = sorted(device_rows, key=lambda row: row["kernel_ms"])[:5]
            for row in fastest:
                print(
                    "{:<7} {:<28} kernel={:>8.4f} ms total={:>8.4f} ms".format(
                        device,
                        row["case_id"],
                        row["kernel_ms"],
                        row["total_ms"],
                    )
                )


if __name__ == "__main__":
    main()
