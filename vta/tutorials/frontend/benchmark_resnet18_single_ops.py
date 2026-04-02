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

from tvm import autotvm, rpc

import vta


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


def benchmark_conv_case(
    remote,
    case: BenchmarkCase,
    device: str,
    number: int,
    warmup: int,
    check_correctness: bool,
) -> Dict[str, object]:
    workload = make_conv_workload(case)
    target = device_target(device)
    capture = CaptureWriter()
    records: List[Dict[str, object]] = []
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
        "kernel_ms": row["T_kernel_steady_s"] * 1e3,
        "total_ms": row["total_s"] * 1e3,
        "submit_ms": row["T_submit_s"] * 1e3,
        "sync_ms": row["T_sync_s"] * 1e3,
        "h2d_ms": row["T_h2d_s"] * 1e3,
        "d2h_ms": row["T_d2h_s"] * 1e3,
        "pack_ms": row["T_pack_s"] * 1e3,
        "gops": row["gops_kernel_steady"],
        "ok": row["ok"],
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
        "ok",
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
                if case.op_name != "conv2d":
                    row = skipped_row(
                        case,
                        device,
                        "standalone VTA/CPU benchmark path is implemented only for conv2d in this script",
                    )
                elif device == "vta" and not is_vta_compatible_conv(case):
                    row = skipped_row(
                        case,
                        device,
                        "incompatible with VTA packed conv requirements; channels/batch must align with VTA block sizes",
                    )
                else:
                    row = benchmark_conv_case(
                        remote,
                        case,
                        device,
                        number=args.number,
                        warmup=args.warmup,
                        check_correctness=args.check_correctness,
                    )
                    print(
                        "kernel={:.4f} ms total={:.4f} ms gops={:.2f} ok={}".format(
                            row["kernel_ms"],
                            row["total_ms"],
                            row["gops"],
                            row["ok"],
                        )
                    )
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
