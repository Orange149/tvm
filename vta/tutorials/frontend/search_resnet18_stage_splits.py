#!/usr/bin/env python3
"""Search ResNet18 N-stage CPU/VTA cuts for native pipeline throughput.

This version enumerates up to three non-overlapping VTA islands over the
fine-grained residual units exposed by split_resnet18_stages.py. Static metrics
are used for pruning and explanation; measured board throughput remains the
final ranking signal.
"""

from __future__ import absolute_import, print_function

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from mxnet.gluon.model_zoo import vision

import vta

from split_resnet18_stages import (  # pylint: disable=import-error
    SCHEMES,
    UNIT_ORDER,
    _estimate_cpu_stage_cost,
    _estimate_unit_tile_cost,
    _unit_kind,
    build_resnet18_unit_blocks,
    build_resnet18_unit_metadata,
    build_vta_capacity_spec,
    stage_unit_names,
    validate_scheme,
)


DEFAULT_KNOWN_SCHEMES = (
    "three_stage_a,three_stage_b,three_stage_e,three_stage_f,"
    "block_stage_c,fine_conv_vta_a"
)
DEFAULT_POLL_VALUES = [("poll_1us", 1000)]
CPU_BUCKETS = [
    "stem_large_input_conv",
    "residual_3x3_conv",
    "skip_proj_1x1_conv",
    "add_relu_tail_elementwise",
    "head_pool_dense",
]
VTA_BUCKETS = [
    "conv3x3_c_small",
    "conv3x3_c_large",
    "conv1x1",
    "stride2_downsample",
    "skip_proj",
]
DMA_BUCKETS = [
    "large_contiguous_load_store",
    "small_tensor_load_store",
    "strided_load",
    "padded_load",
]
CALIBRATION_CASES = {
    "stem_large_input_conv": "stem_conv7x7_s2",
    "residual_3x3_conv": "s2_conv3x3_128_128",
    "skip_proj_1x1_conv": "s3_proj1x1_128_256_s2",
    "conv3x3_c_small": "s1_conv3x3_64_64",
    "conv3x3_c_large": "s4_conv3x3_512_512",
    "conv1x1": "s3_proj1x1_128_256_s2",
    "stride2_downsample": "s3_conv3x3_128_256_s2",
    "skip_proj": "s3_proj1x1_128_256_s2",
}
FAILURE_TYPES = {
    "static_invalid",
    "build_failed",
    "package_failed",
    "upload_failed",
    "remote_space_low",
    "serial_run_failed",
    "pipeline_run_failed",
    "correctness_failed",
    "profile_missing",
    "timeout",
}
CAT_EQUIVALENT_TOP1 = {281, 282, 283, 284, 285}


def repo_root():
    return Path(__file__).resolve().parents[3]


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def parse_positive_csv(text):
    if not text:
        return []
    parts = [item.strip() for item in text.split(",")]
    if any(not item for item in parts):
        raise argparse.ArgumentTypeError("CSV contains an empty field")
    try:
        values = [int(item) for item in parts]
    except ValueError as err:
        raise argparse.ArgumentTypeError("CSV values must be integers") from err
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("CSV values must be positive")
    return values


def parse_poll_value(text):
    if ":" not in text:
        raise argparse.ArgumentTypeError("expected LABEL:NS, for example poll_1us:1000")
    label, ns_text = text.split(":", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("poll label must be non-empty")
    try:
        ns = int(ns_text)
    except ValueError as err:
        raise argparse.ArgumentTypeError("poll ns must be an integer") from err
    if ns < 0:
        raise argparse.ArgumentTypeError("poll ns must be non-negative")
    return label, ns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search fine-grained ResNet18 CPU/VTA native stage-pipeline cuts."
    )
    parser.add_argument("--board", default="", help="SSH target, for example root@192.168.1.247")
    parser.add_argument("--remote-dir", default="/var/volatile/vta_stage_pipeline_search")
    parser.add_argument("--remote-min-free-mb", type=int, default=128)
    parser.add_argument(
        "--no-cleanup-remote",
        action="store_true",
        help="Keep board-side candidate directories after each measured run.",
    )
    parser.add_argument("--ssh-option", action="append", default=[])
    parser.add_argument("--model", default="resnet18_v1", choices=["resnet18_v1"])
    parser.add_argument("--image", default="")
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--runtime-num-threads", type=int, default=4)
    parser.add_argument("--skip-first", type=int, default=3)
    parser.add_argument("--max-vta-islands", type=int, default=3)
    parser.add_argument("--measure-top-n", type=int, default=64)
    parser.add_argument("--refine-top-n", type=int, default=8)
    parser.add_argument("--static-shortlist-n", type=int, default=256)
    parser.add_argument("--buildability-top-n", type=int, default=128)
    parser.add_argument("--max-board-test-configs", type=int, default=100)
    parser.add_argument(
        "--shortlist-policy",
        default="stratified",
        choices=["stratified", "theory_residual"],
        help="Static shortlist policy.",
    )
    parser.add_argument(
        "--stage-runtime-thread-policy",
        default="positional",
        choices=["positional"],
        help="Thread policy for generated N-stage candidates.",
    )
    parser.add_argument(
        "--stage-runtime-num-threads",
        default="",
        help="Optional explicit per-stage CSV. If set, it is used for every candidate "
        "and must match that candidate stage count.",
    )
    parser.add_argument(
        "--poll-value",
        action="append",
        type=parse_poll_value,
        default=[],
        help="Driver poll/post-start sleep as LABEL:NS. Repeat to sweep.",
    )
    parser.add_argument("--include-known", default=DEFAULT_KNOWN_SCHEMES)
    parser.add_argument("--candidate-name", action="append", default=[])
    parser.add_argument(
        "--candidate-prior-file",
        default="",
        help=(
            "Optional ordered candidate_id/scheme_name list. When present, shortlist "
            "and buildability follow this order before applying other fill policies."
        ),
    )
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--allow-hard-reject", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--buildability-only", action="store_true")
    parser.add_argument("--skip-buildability", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--exclude-candidate-id-file",
        action="append",
        default=[],
        help=(
            "File containing candidate_id or scheme_name values to skip. "
            "May be repeated; blank lines and # comments are ignored."
        ),
    )
    parser.add_argument(
        "--measured-candidate-id-output",
        default="",
        help="Write attempted measured candidate ids to this file as rows are recorded.",
    )
    parser.add_argument(
        "--file-cache-policy",
        default="prewarm",
        choices=["none", "prewarm", "cold_then_warm"],
        help=(
            "Remote _file_cache handling before board measurements. "
            "'prewarm' deploys with --skip-run before the measured run; "
            "'cold_then_warm' records a non-ranked cold probe before the ranked warm run."
        ),
    )
    parser.add_argument("--output-root", default="/tmp/vta_stage_split_searches")
    parser.add_argument("--search-label", default="")
    parser.add_argument(
        "--progress-batch-index",
        type=int,
        default=1,
        help="1-based batch index for progress.json/progress.txt.",
    )
    parser.add_argument(
        "--progress-batch-count",
        type=int,
        default=1,
        help="Total batch count for progress.json/progress.txt.",
    )
    parser.add_argument(
        "--progress-total-offset",
        type=int,
        default=0,
        help="Number of measured configs completed before this batch.",
    )
    parser.add_argument(
        "--progress-total-count",
        type=int,
        default=0,
        help="Total planned measured configs across batches. 0 means use this batch total.",
    )
    parser.add_argument(
        "--archive-root",
        default="",
        help="Optional aggregate-report archive root; relative paths are under repo root.",
    )
    parser.add_argument(
        "--build-cache-dir",
        default=str(
            repo_root()
            / "vta"
            / "tutorials"
            / "frontend"
            / "report_out"
            / "native_stage_pipeline_build_cache"
            / "v23_20260506"
        ),
    )
    parser.add_argument(
        "--reuse-buildability-cache",
        action="store_true",
        help="Reuse existing candidate_status and buildability package dirs under --build-cache-dir.",
    )
    parser.add_argument("--ssh-command-timeout-s", type=int, default=60)
    parser.add_argument("--scp-timeout-s", type=int, default=600)
    parser.add_argument("--serial-timeout-s", type=int, default=180)
    parser.add_argument("--pipeline-timeout-s", type=int, default=180)
    parser.add_argument("--fetch-timeout-s", type=int, default=120)
    parser.add_argument("--vta-runtime-profile-events-limit", type=int, default=200)
    parser.add_argument("--rpc-baseline-result", default="")
    parser.add_argument(
        "--correctness-policy",
        default="exact",
        choices=["exact", "cat_equivalent"],
        help="Measured correctness policy against RPC all_vta baseline.",
    )
    parser.add_argument(
        "--allow-missing-rpc-baseline",
        action="store_true",
        help="Allow board measurement without RPC all_vta baseline gate.",
    )
    parser.add_argument("--no-compare-serial-pipeline", action="store_true")
    parser.add_argument("--cpu-peak-gops", type=float, default=0.0)
    parser.add_argument("--vta-peak-gops", type=float, default=0.0)
    parser.add_argument("--ps-pl-bandwidth-gbps", type=float, default=0.0)
    parser.add_argument("--calibrate-cost-model", action="store_true")
    parser.add_argument(
        "--calibration-output",
        default="/tmp/vta_stage_search_cost_model.json",
        help="Path for the calibrated static cost model JSON.",
    )
    parser.add_argument("--static-cost-model-json", default="")
    parser.add_argument("--calibration-budget", type=int, default=20)
    parser.add_argument("--reuse-calibration-if-exists", action="store_true")
    parser.add_argument(
        "--calibration-host",
        default="",
        help="RPC host for single-op calibration; defaults to host parsed from --board.",
    )
    parser.add_argument("--calibration-port", type=int, default=9090)
    parser.add_argument("--calibration-number", type=int, default=1)
    parser.add_argument("--calibration-warmup", type=int, default=0)
    return parser.parse_args()


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    with path.open(encoding="utf-8") as inp:
        return json.load(inp)


def read_jsonl(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as inp:
        for line in inp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2, sort_keys=True)
        out.write("\n")


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as inp:
        return [dict(row) for row in csv.DictReader(inp)]


def as_float(value, default=0.0):
    try:
        if value in ["", None]:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def median_value(values):
    values = [float(item) for item in values if float(item) > 0.0]
    return statistics.median(values) if values else 0.0


def board_host(args):
    if args.calibration_host:
        return args.calibration_host
    if args.board and "@" in args.board:
        return args.board.split("@", 1)[1].split(":", 1)[0]
    return args.board


def mean_value(rows, key, skip_first=0):
    values = [float(row[key]) for row in rows[int(skip_first) :] if key in row]
    return statistics.mean(values) if values else 0.0


def std_value(rows, key, skip_first=0):
    values = [float(row[key]) for row in rows[int(skip_first) :] if key in row]
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def stage_count_from_rows(rows):
    if not rows:
        return 0
    if "stage_count" in rows[0]:
        return int(rows[0]["stage_count"])
    count = 0
    while "stage{}_ms".format(count) in rows[0]:
        count += 1
    return count


def throughput_from_span(rows, skip_first):
    keep = rows[int(skip_first) :]
    if len(keep) < 2:
        return 0.0
    stage_count = stage_count_from_rows(keep)
    if stage_count <= 0:
        return 0.0
    first_start = float(keep[0].get("stage0_start_ms", 0.0))
    last_key = "stage{}_end_ms".format(stage_count - 1)
    last_end = max(float(row.get(last_key, 0.0)) for row in keep)
    span_ms = last_end - first_start
    return float(len(keep)) * 1000.0 / span_ms if span_ms > 0.0 else 0.0


def first_stage_interval_ms(rows, skip_first):
    starts = [float(row["stage0_start_ms"]) for row in rows if "stage0_start_ms" in row]
    intervals = [starts[idx] - starts[idx - 1] for idx in range(1, len(starts))]
    if skip_first:
        intervals = intervals[max(0, int(skip_first) - 1) :]
    return statistics.mean(intervals) if intervals else 0.0


def top1_match(serial_rows, pipeline_rows):
    if not serial_rows or not pipeline_rows or len(serial_rows) != len(pipeline_rows):
        return False
    for serial, pipeline in zip(serial_rows, pipeline_rows):
        for key in ["input_index", "input_file", "top1"]:
            if serial.get(key) != pipeline.get(key):
                return False
    return True


def read_baseline_top1_by_input(path):
    path = Path(path)
    rows = {}
    if not path:
        return rows
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as inp:
            reader = csv.DictReader(inp)
            for row_idx, row in enumerate(reader):
                if "top1" in row:
                    rows[int(row.get("input_index", row_idx))] = int(row["top1"])
        return rows
    with path.open("r", encoding="utf-8") as inp:
        for row_idx, line in enumerate(inp):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "top1" in row:
                rows[int(row.get("input_index", row_idx))] = int(row["top1"])
    return rows


def rpc_correctness_summary(args, native_rows):
    result = {
        "rpc_baseline_checked": bool(args.rpc_baseline_result),
        "passes_rpc_baseline": bool(args.allow_missing_rpc_baseline and not args.rpc_baseline_result),
        "correctness_policy": args.correctness_policy,
        "correctness_relaxed": False,
        "correctness_gate_reason": "missing_baseline" if not args.rpc_baseline_result else "",
        "baseline_top1": "",
        "native_top1": "",
    }
    if not args.rpc_baseline_result:
        return result
    baseline = read_baseline_top1_by_input(args.rpc_baseline_result)
    if not baseline or not native_rows:
        result["correctness_gate_reason"] = "missing_rows"
        return result
    relaxed = False
    for native_row in native_rows:
        input_index = int(native_row.get("input_index", 0))
        native_top1 = int(native_row.get("top1", -1))
        baseline_top1 = baseline.get(input_index)
        result["baseline_top1"] = baseline_top1 if baseline_top1 is not None else ""
        result["native_top1"] = native_top1
        if baseline_top1 is None:
            result["correctness_gate_reason"] = "missing_baseline_input_index"
            return result
        if native_top1 == int(baseline_top1):
            continue
        if (
            args.correctness_policy == "cat_equivalent"
            and native_top1 in CAT_EQUIVALENT_TOP1
            and int(baseline_top1) in CAT_EQUIVALENT_TOP1
        ):
            relaxed = True
            continue
        result["correctness_gate_reason"] = "top1_mismatch"
        return result
    result["passes_rpc_baseline"] = True
    result["correctness_relaxed"] = relaxed
    result["correctness_gate_reason"] = "cat_equivalent" if relaxed else "exact_match"
    return result


def bytes_per_us_to_gbps(byte_count, duration_us):
    duration_us = float(duration_us)
    if duration_us <= 0.0:
        return 0.0
    return float(byte_count) * 8.0 / duration_us / 1000.0


def profile_status(output_dir, mode):
    return read_json(Path(output_dir) / "profile" / mode / "benchmark_totals_status.json", default={})


def profile_bandwidth_metrics(status):
    load_bytes = float(status.get("load_buffer_2d_bytes", 0.0))
    store_bytes = float(status.get("store_buffer_2d_bytes", 0.0))
    wait_us = float(status.get("device_run_wait_us", 0.0))
    total_bytes = load_bytes + store_bytes
    return {
        "ps_pl_load_bytes": load_bytes,
        "ps_pl_store_bytes": store_bytes,
        "ps_pl_total_dma_bytes": total_bytes,
        "vta_device_wait_ms": wait_us / 1000.0,
        "ps_pl_load_bw_gbps": bytes_per_us_to_gbps(load_bytes, wait_us),
        "ps_pl_store_bw_gbps": bytes_per_us_to_gbps(store_bytes, wait_us),
        "ps_pl_total_bw_gbps": bytes_per_us_to_gbps(total_bytes, wait_us),
        "flush_cache_ms": float(status.get("flush_cache_us", 0.0)) / 1000.0,
        "invalidate_cache_ms": float(status.get("invalidate_cache_us", 0.0)) / 1000.0,
    }


def profile_dma_fragmentation_metrics(status):
    load_calls = float(status.get("load_buffer_2d_calls", 0.0))
    store_calls = float(status.get("store_buffer_2d_calls", 0.0))
    load_bytes = float(status.get("load_buffer_2d_bytes", 0.0))
    store_bytes = float(status.get("store_buffer_2d_bytes", 0.0))
    total_calls = load_calls + store_calls
    total_bytes = load_bytes + store_bytes
    small_calls = float(status.get("load_buffer_2d_small_calls", 0.0)) + float(
        status.get("store_buffer_2d_small_calls", 0.0)
    )
    strided_calls = float(status.get("load_buffer_2d_strided_calls", 0.0)) + float(
        status.get("store_buffer_2d_strided_calls", 0.0)
    )
    padded_calls = float(status.get("load_buffer_2d_padded_calls", 0.0))
    thin_shape_calls = (
        float(status.get("load_buffer_2d_xsize1_calls", 0.0))
        + float(status.get("load_buffer_2d_ysize1_calls", 0.0))
        + float(status.get("store_buffer_2d_xsize1_calls", 0.0))
        + float(status.get("store_buffer_2d_ysize1_calls", 0.0))
    )
    total_calls = max(0.0, total_calls)
    avg_bytes = total_bytes / total_calls if total_calls else 0.0
    small_ratio = small_calls / total_calls if total_calls else 0.0
    strided_ratio = strided_calls / total_calls if total_calls else 0.0
    padded_ratio = padded_calls / total_calls if total_calls else 0.0
    thin_shape_ratio = thin_shape_calls / total_calls if total_calls else 0.0
    fragmentation_score = small_ratio + strided_ratio + padded_ratio + 0.5 * thin_shape_ratio
    return {
        "dma_load_calls": load_calls,
        "dma_store_calls": store_calls,
        "dma_total_calls": total_calls,
        "dma_avg_bytes_per_call": avg_bytes,
        "dma_small_call_ratio": small_ratio,
        "dma_strided_call_ratio": strided_ratio,
        "dma_padded_call_ratio": padded_ratio,
        "dma_thin_shape_call_ratio": thin_shape_ratio,
        "dma_fragmentation_score": fragmentation_score,
    }


def empty_overhead(estimated_from):
    return {
        "set_input_ms": 0.05,
        "run_ms": 0.0,
        "get_output_ms": 0.05,
        "bridge_pack_ms": 0.0,
        "bridge_unpack_ms": 0.0,
        "runner_submit_ms": 0.05,
        "sync_wait_ms": 0.05,
        "estimated_from": estimated_from,
    }


def default_cost_model(args):
    cpu_peak = float(args.cpu_peak_gops) if float(args.cpu_peak_gops) > 0.0 else 8.0
    vta_peak = float(args.vta_peak_gops) if float(args.vta_peak_gops) > 0.0 else 16.0
    pspl_gbps = float(args.ps_pl_bandwidth_gbps) if float(args.ps_pl_bandwidth_gbps) > 0.0 else 12.0
    pspl_gBps = pspl_gbps / 8.0
    model = {
        "version": 1,
        "units": {
            "ops": "OP",
            "gops": "GOP/s",
            "time": "ms",
            "dma_bytes": "B",
            "ps_pl_bw": "decimal GB/s",
        },
        "source": "default_static_estimate",
        "cpu_eff_gops": {},
        "vta_eff_gops": {},
        "dma": {},
        "stage_overheads": {
            "cpu": empty_overhead("default_static_estimate"),
            "vta": empty_overhead("default_static_estimate"),
        },
        "boundary_bw_GBps": max(0.1, pspl_gBps),
        "weights": {
            "imbalance": 0.25,
            "sram_risk": 0.15,
            "dma_fragmentation": 1.0,
            "boundary": 1.0,
        },
    }
    for bucket in CPU_BUCKETS:
        model["cpu_eff_gops"][bucket] = {
            "1": max(0.05, cpu_peak * 0.35),
            "2": max(0.05, cpu_peak * 0.60),
            "3": max(0.05, cpu_peak * 0.80),
            "4": max(0.05, cpu_peak),
            "estimated_from": "default_static_estimate",
        }
    for bucket in VTA_BUCKETS:
        scale = 0.75 if bucket in ["conv1x1", "skip_proj"] else 1.0
        if bucket == "conv3x3_c_small":
            scale = 0.70
        model["vta_eff_gops"][bucket] = {
            "gops": max(0.05, vta_peak * scale),
            "estimated_from": "default_static_estimate",
        }
    for bucket in DMA_BUCKETS:
        penalty = 0.01
        if bucket == "small_tensor_load_store":
            penalty = 0.04
        elif bucket in ["strided_load", "padded_load"]:
            penalty = 0.08
        model["dma"][bucket] = {
            "ps_pl_load_bw_GBps": max(0.1, pspl_gBps),
            "ps_pl_store_bw_GBps": max(0.1, pspl_gBps),
            "avg_bytes_per_call": 4096.0,
            "small_call_ratio": 0.0,
            "strided_call_ratio": 0.0,
            "padded_call_ratio": 0.0,
            "fragmentation_penalty_ms_per_call": penalty,
            "estimated_from": "default_static_estimate",
        }
    return model


def validate_cost_model(model):
    for bucket in CPU_BUCKETS:
        if bucket not in model.get("cpu_eff_gops", {}):
            raise RuntimeError("cost model missing CPU bucket {}".format(bucket))
    for bucket in VTA_BUCKETS:
        if bucket not in model.get("vta_eff_gops", {}):
            raise RuntimeError("cost model missing VTA bucket {}".format(bucket))
    for bucket in DMA_BUCKETS:
        if bucket not in model.get("dma", {}):
            raise RuntimeError("cost model missing DMA bucket {}".format(bucket))
    return model


def load_cost_model(args):
    if args.static_cost_model_json:
        return validate_cost_model(read_json(args.static_cost_model_json, default={}))
    return default_cost_model(args)


def calibration_case_ids(budget):
    ordered = [
        "stem_conv7x7_s2",
        "s1_conv3x3_64_64",
        "s2_conv3x3_128_128",
        "s4_conv3x3_512_512",
        "s2_conv3x3_64_128_s2",
        "s3_conv3x3_128_256_s2",
        "s2_proj1x1_64_128_s2",
        "s3_proj1x1_128_256_s2",
        "s4_proj1x1_256_512_s2",
        "tail_global_avg_pool",
        "tail_dense_512_1000",
    ]
    return ordered[: max(1, min(len(ordered), int(budget)))]


def profile_status_for_case(profile_root, case_id):
    return read_json(
        Path(profile_root) / "single_op" / case_id / "benchmark_totals_status.json",
        default={},
    )


def update_cost_model_from_benchmark(model, rows, profile_root):
    case_to_cpu_bucket = {
        "stem_conv7x7_s2": "stem_large_input_conv",
        "s1_conv3x3_64_64": "residual_3x3_conv",
        "s2_conv3x3_128_128": "residual_3x3_conv",
        "s2_proj1x1_64_128_s2": "skip_proj_1x1_conv",
        "s3_proj1x1_128_256_s2": "skip_proj_1x1_conv",
        "tail_global_avg_pool": "head_pool_dense",
        "tail_dense_512_1000": "head_pool_dense",
    }
    case_to_vta_bucket = {
        "s1_conv3x3_64_64": "conv3x3_c_small",
        "s4_conv3x3_512_512": "conv3x3_c_large",
        "s2_proj1x1_64_128_s2": "conv1x1",
        "s3_proj1x1_128_256_s2": "skip_proj",
        "s2_conv3x3_64_128_s2": "stride2_downsample",
        "s3_conv3x3_128_256_s2": "stride2_downsample",
    }
    for row in rows:
        if row.get("status") != "ok":
            continue
        case_id = row.get("case_id", "")
        device = row.get("device", "")
        gops = as_float(row.get("gops"), 0.0)
        if gops <= 0.0:
            continue
        if device == "arm_cpu" and case_id in case_to_cpu_bucket:
            bucket = case_to_cpu_bucket[case_id]
            # Existing benchmark does not sweep thread count; map the measured value
            # to 4 threads and retain conservative scaled estimates for lower counts.
            model["cpu_eff_gops"][bucket] = {
                "1": max(0.05, gops * 0.35),
                "2": max(0.05, gops * 0.60),
                "3": max(0.05, gops * 0.80),
                "4": max(0.05, gops),
                "set_input_ms": as_float(row.get("h2d_ms"), 0.0),
                "run_ms": as_float(row.get("kernel_ms"), 0.0),
                "get_output_ms": as_float(row.get("d2h_ms"), 0.0),
                "estimated_from": case_id,
            }
        if device == "vta" and case_id in case_to_vta_bucket:
            bucket = case_to_vta_bucket[case_id]
            model["vta_eff_gops"][bucket] = {
                "gops": max(0.05, gops),
                "runner_submit_ms": as_float(row.get("submit_ms"), 0.0),
                "sync_wait_ms": as_float(row.get("sync_ms"), 0.0),
                "bridge_pack_ms": as_float(row.get("pack_ms"), 0.0),
                "estimated_from": case_id,
            }
            status = profile_status_for_case(profile_root, case_id)
            if status:
                dma_metrics = profile_bandwidth_metrics(status)
                frag_metrics = profile_dma_fragmentation_metrics(status)
                load_bw = max(0.1, dma_metrics["ps_pl_load_bw_gbps"] / 8.0)
                store_bw = max(0.1, dma_metrics["ps_pl_store_bw_gbps"] / 8.0)
                avg_bytes = max(1.0, frag_metrics["dma_avg_bytes_per_call"])
                if avg_bytes >= 16384:
                    dma_bucket = "large_contiguous_load_store"
                elif frag_metrics["dma_padded_call_ratio"] > 0.0:
                    dma_bucket = "padded_load"
                elif frag_metrics["dma_strided_call_ratio"] > 0.0:
                    dma_bucket = "strided_load"
                else:
                    dma_bucket = "small_tensor_load_store"
                model["dma"][dma_bucket] = {
                    "ps_pl_load_bw_GBps": load_bw,
                    "ps_pl_store_bw_GBps": store_bw,
                    "avg_bytes_per_call": avg_bytes,
                    "small_call_ratio": frag_metrics["dma_small_call_ratio"],
                    "strided_call_ratio": frag_metrics["dma_strided_call_ratio"],
                    "padded_call_ratio": frag_metrics["dma_padded_call_ratio"],
                    "fragmentation_penalty_ms_per_call": max(
                        0.001, frag_metrics["dma_fragmentation_score"] * 0.05
                    ),
                    "estimated_from": case_id,
                }
    return validate_cost_model(model)


def run_calibration(args, output_dir):
    output_path = Path(args.calibration_output)
    if args.reuse_calibration_if_exists and output_path.exists():
        print("[CALIBRATION] reuse", output_path)
        return validate_cost_model(read_json(output_path, default={}))
    model = default_cost_model(args)
    host = board_host(args)
    if host:
        calibration_dir = Path(output_dir) / "calibration"
        profile_dir = calibration_dir / "profile"
        cases = calibration_case_ids(args.calibration_budget)
        cmd = [
            sys.executable,
            str(repo_root() / "vta" / "tutorials" / "frontend" / "benchmark_resnet18_single_ops.py"),
            "--host",
            host,
            "--port",
            str(int(args.calibration_port)),
            "--devices",
            "arm_cpu,vta",
            "--number",
            str(int(args.calibration_number)),
            "--warmup",
            str(int(args.calibration_warmup)),
            "--cases",
            ",".join(cases),
            "--output-dir",
            str(calibration_dir),
            "--vta-runtime-profile-dir",
            str(profile_dir),
            "--vta-runtime-profile-events-limit",
            str(int(args.vta_runtime_profile_events_limit)),
        ]
        try:
            run_command(cmd, DEFAULT_POLL_VALUES[0][1])
            rows = read_csv_rows(calibration_dir / "raw_benchmark.csv")
            model = update_cost_model_from_benchmark(model, rows, profile_dir)
            model["source"] = "benchmark_resnet18_single_ops"
            model["calibration_cases"] = cases
        except Exception as err:  # pylint: disable=broad-except
            model["source"] = "default_static_estimate_after_calibration_failure"
            model["calibration_error"] = repr(err)[:800]
            print("[CALIBRATION] failed, wrote fallback model:", repr(err))
    else:
        model["source"] = "default_static_estimate_no_calibration_host"
    write_json(output_path, validate_cost_model(model))
    return model


def window_is_valid(unit_names):
    if not any(_unit_kind(name) == "main_preadd" for name in unit_names):
        return False
    if unit_names == ["head"] or unit_names == ["stem"]:
        return False
    if _unit_kind(unit_names[0]) not in ["main_preadd", "stem"]:
        return False
    if _unit_kind(unit_names[-1]) not in ["main_preadd", "skip_proj", "add_relu_tail", "head"]:
        return False
    return True


def enumerate_vta_windows():
    windows = []
    total_units = len(UNIT_ORDER)
    for start_idx in range(total_units):
        for end_idx in range(start_idx, total_units):
            unit_names = UNIT_ORDER[start_idx : end_idx + 1]
            if window_is_valid(unit_names):
                windows.append((start_idx, end_idx))
    return windows


def enumerate_island_sets(max_islands):
    windows = enumerate_vta_windows()
    by_start = {}
    for start_idx, end_idx in windows:
        by_start.setdefault(start_idx, []).append((start_idx, end_idx))

    results = []

    def rec(next_start, remaining, chosen):
        if chosen:
            results.append(tuple(chosen))
        if remaining <= 0:
            return
        for start_idx in range(next_start, len(UNIT_ORDER)):
            for window in by_start.get(start_idx, []):
                rec(window[1] + 1, remaining - 1, chosen + [window])

    rec(0, int(max_islands), [])
    return results


def make_stage(name, device, unit_names):
    return {"name": name, "device": device, "kind": "units", "unit_names": list(unit_names)}


def scheme_from_islands(islands):
    scheme_cfg = []
    cursor = 0
    for start_idx, end_idx in islands:
        if cursor < start_idx:
            scheme_cfg.append(make_stage("stage{}_cpu".format(len(scheme_cfg)), "cpu", UNIT_ORDER[cursor:start_idx]))
        scheme_cfg.append(
            make_stage(
                "stage{}_vta".format(len(scheme_cfg)),
                "vta",
                UNIT_ORDER[start_idx : end_idx + 1],
            )
        )
        cursor = end_idx + 1
    if cursor < len(UNIT_ORDER):
        scheme_cfg.append(make_stage("stage{}_cpu".format(len(scheme_cfg)), "cpu", UNIT_ORDER[cursor:]))
    return scheme_cfg


def scheme_devices(scheme_cfg):
    return [stage["device"] for stage in scheme_cfg]


def native_pipeline_scheme(scheme_cfg):
    devices = scheme_devices(scheme_cfg)
    return len(devices) >= 3 and devices[0] == "cpu" and devices[-1] == "cpu" and "vta" in devices


def islands_from_scheme(scheme_cfg):
    islands = []
    for stage in scheme_cfg:
        if stage["device"] != "vta":
            continue
        units = stage_unit_names(stage)
        islands.append(
            {
                "start_idx": UNIT_ORDER.index(units[0]),
                "end_idx": UNIT_ORDER.index(units[-1]),
                "start_unit": units[0],
                "end_unit": units[-1],
                "unit_names": list(units),
            }
        )
    return islands


def unit_assignment_from_scheme(scheme_cfg):
    assignment = []
    for stage in scheme_cfg:
        for unit_name in stage_unit_names(stage):
            assignment.append(
                {
                    "unit_name": unit_name,
                    "device": stage["device"],
                    "stage": stage["name"],
                }
            )
    return assignment


def assignment_signature(scheme_cfg):
    unit_to_device = {}
    for stage in scheme_cfg:
        for unit_name in stage_unit_names(stage):
            unit_to_device[unit_name] = stage["device"]
    return tuple(unit_to_device.get(name, "missing") for name in UNIT_ORDER)


def default_threads_for_scheme(scheme_cfg):
    threads = []
    cpu_indices = [idx for idx, stage in enumerate(scheme_cfg) if stage["device"] == "cpu"]
    first_cpu = cpu_indices[0] if cpu_indices else -1
    last_cpu = cpu_indices[-1] if cpu_indices else -1
    for idx, stage in enumerate(scheme_cfg):
        if stage["device"] == "vta":
            threads.append(1)
        elif idx == first_cpu:
            threads.append(3)
        elif idx == last_cpu:
            threads.append(4)
        else:
            threads.append(1)
    return threads


def threads_for_candidate(args, candidate):
    explicit = parse_positive_csv(args.stage_runtime_num_threads)
    stage_count = len(candidate["scheme_cfg"])
    if explicit:
        defaults = default_threads_for_scheme(candidate["scheme_cfg"])
        if len(explicit) >= stage_count:
            return explicit[:stage_count]
        return explicit + defaults[len(explicit) :]
    return default_threads_for_scheme(candidate["scheme_cfg"])


def refine_cpu4_threads(candidate):
    return [4 if stage["device"] == "cpu" else 1 for stage in candidate["scheme_cfg"]]


def stage_ops(unit_names, unit_metadata):
    return sum(int(unit_metadata[name].get("compute_ops_est", 0)) for name in unit_names)


def scaled_cpu_peak(args, threads):
    cpu_peak = float(args.cpu_peak_gops)
    if cpu_peak <= 0.0:
        return 0.0
    denom = max(1, int(args.runtime_num_threads))
    return cpu_peak * (float(max(1, int(threads))) / float(denom))


def compute_ms(ops, peak_gops):
    if peak_gops <= 0.0:
        return 0.0
    return float(ops) / (float(peak_gops) * 1e9) * 1000.0


def transfer_ms(byte_count, bw_gbps):
    if bw_gbps <= 0.0:
        return 0.0
    return float(byte_count) * 8.0 / (float(bw_gbps) * 1e9) * 1000.0


def transfer_ms_GBps(byte_count, bw_GBps):
    if bw_GBps <= 0.0:
        return 0.0
    return float(byte_count) / (float(bw_GBps) * 1e9) * 1000.0


def cpu_bucket_for_unit(unit_name):
    kind = _unit_kind(unit_name)
    if unit_name == "stem":
        return "stem_large_input_conv"
    if unit_name == "head":
        return "head_pool_dense"
    if kind == "skip_proj":
        return "skip_proj_1x1_conv"
    if kind == "add_relu_tail":
        return "add_relu_tail_elementwise"
    return "residual_3x3_conv"


def vta_bucket_for_unit(unit_name):
    kind = _unit_kind(unit_name)
    if kind == "skip_proj":
        return "skip_proj"
    if "skip_proj" in unit_name:
        return "conv1x1"
    if "block0_main_preadd" in unit_name and (
        unit_name.startswith("layer2_")
        or unit_name.startswith("layer3_")
        or unit_name.startswith("layer4_")
    ):
        return "stride2_downsample"
    if unit_name.startswith("layer1_") or unit_name.startswith("layer2_"):
        return "conv3x3_c_small"
    if unit_name == "stem":
        return "conv3x3_c_small"
    return "conv3x3_c_large"


def dma_bucket_for_unit(unit_name, meta, tile_info):
    avg = float(
        tile_info["tile_input_load_bytes_est"]
        + tile_info["tile_weight_load_bytes_est"]
        + tile_info["tile_output_store_bytes_est"]
    ) / float(max(1, tile_info["tile_count_est"]))
    if meta.get("shape_change", False):
        return "padded_load"
    if _unit_kind(unit_name) == "skip_proj":
        return "strided_load"
    if avg < 4096.0:
        return "small_tensor_load_store"
    return "large_contiguous_load_store"


def cpu_eff_gops(model, bucket, threads):
    entry = model["cpu_eff_gops"].get(bucket, {})
    key = str(int(max(1, threads)))
    if key in entry:
        return max(0.001, float(entry[key]))
    numeric = [float(value) for key, value in entry.items() if str(key).isdigit()]
    return max(0.001, max(numeric) if numeric else 1.0)


def vta_eff_gops(model, bucket):
    entry = model["vta_eff_gops"].get(bucket, {})
    return max(0.001, float(entry.get("gops", 1.0)))


def stage_overhead(model, device, key):
    return float(model.get("stage_overheads", {}).get(device, {}).get(key, 0.0))


def score_scheme(scheme_cfg, unit_metadata, capacity_spec, args, cost_model):
    stage_threads = default_threads_for_scheme(scheme_cfg)
    inp_cap = float(capacity_spec["inp_cap_bytes"])
    wgt_cap = float(capacity_spec["wgt_cap_bytes"])
    acc_cap = float(capacity_spec["acc_cap_bytes"])
    out_cap = float(capacity_spec["out_cap_bytes"])

    stage_infos = []
    hard_reject = False
    reject_reasons = set()
    total_boundary_bytes = 0
    boundary_count = 0
    total_dma_bytes = 0
    total_tile_count = 0
    total_tile_spill = 0
    total_internal_boundary_bytes = 0
    total_residual_live_bytes = 0
    max_inp_ratio = 0.0
    max_wgt_ratio = 0.0
    max_acc_ratio = 0.0
    max_out_ratio = 0.0
    vta_ops = 0
    cpu_ops = 0
    static_stage_ms = []
    proxy_costs = []

    for idx, stage in enumerate(scheme_cfg):
        units = stage_unit_names(stage)
        ops = stage_ops(units, unit_metadata)
        if stage["device"] == "cpu":
            raw_cost, tail_penalty = _estimate_cpu_stage_cost(
                units,
                unit_metadata,
                tail_weight=(idx == len(scheme_cfg) - 1),
            )
            proxy = int(math.ceil(float(raw_cost) / float(max(1, stage_threads[idx]))))
            kind_ops = {}
            compute_parts = []
            for name in units:
                bucket = cpu_bucket_for_unit(name)
                unit_ops = int(unit_metadata[name].get("compute_ops_est", 0))
                kind_ops[bucket] = kind_ops.get(bucket, 0) + unit_ops
            compute_ms_total = 0.0
            for bucket, bucket_ops in sorted(kind_ops.items()):
                gops = cpu_eff_gops(cost_model, bucket, stage_threads[idx])
                part_ms = compute_ms(bucket_ops, gops)
                compute_ms_total += part_ms
                compute_parts.append(
                    {
                        "bucket": bucket,
                        "ops": int(bucket_ops),
                        "gops": float(gops),
                        "ms": part_ms,
                    }
                )
            set_ms = stage_overhead(cost_model, "cpu", "set_input_ms")
            get_ms = stage_overhead(cost_model, "cpu", "get_output_ms")
            ms = set_ms + compute_ms_total + get_ms
            cpu_ops += ops
            stage_infos.append(
                {
                    "index": idx,
                    "name": stage["name"],
                    "device": "cpu",
                    "unit_names": list(units),
                    "threads": int(stage_threads[idx]),
                    "compute_ops_est": int(ops),
                    "compute_gops_est": float(ops) / 1e9,
                    "cpu_raw_cost_est": int(raw_cost),
                    "cpu_tail_penalty": int(tail_penalty),
                    "proxy_cost_est": int(proxy),
                    "static_ms_est": ms,
                    "static_set_input_ms_est": set_ms,
                    "static_run_ms_est": compute_ms_total,
                    "static_get_output_ms_est": get_ms,
                    "static_compute_parts": compute_parts,
                }
            )
            proxy_costs.append(proxy)
            static_stage_ms.append(ms)
            continue

        tile_input = 0
        tile_weight = 0
        tile_output = 0
        tile_count = 0
        tile_spill = 0
        dma_call_est = 0
        dma_ms = 0.0
        vta_compute_ms = 0.0
        compute_parts = []
        dma_parts = []
        stage_inp_ratio = 0.0
        stage_wgt_ratio = 0.0
        stage_acc_ratio = 0.0
        stage_out_ratio = 0.0
        residual_live = 0
        internal_boundary = 0
        if len(units) == 1 and _unit_kind(units[0]) in ["skip_proj", "add_relu_tail"]:
            hard_reject = True
            reject_reasons.add("partial_layer_single_block")
        for unit_idx, name in enumerate(units):
            meta = unit_metadata[name]
            tile = _estimate_unit_tile_cost(meta, capacity_spec)
            compute_bucket = vta_bucket_for_unit(name)
            unit_ops = int(meta.get("compute_ops_est", 0))
            unit_gops = vta_eff_gops(cost_model, compute_bucket)
            unit_compute_ms = compute_ms(unit_ops, unit_gops)
            vta_compute_ms += unit_compute_ms
            compute_parts.append(
                {
                    "bucket": compute_bucket,
                    "unit_name": name,
                    "ops": int(unit_ops),
                    "gops": float(unit_gops),
                    "ms": unit_compute_ms,
                }
            )
            dma_bucket = dma_bucket_for_unit(name, meta, tile)
            dma_entry = cost_model["dma"][dma_bucket]
            unit_dma_bytes = int(
                tile["tile_input_load_bytes_est"]
                + tile["tile_weight_load_bytes_est"]
                + tile["tile_output_store_bytes_est"]
            )
            load_bw = float(dma_entry.get("ps_pl_load_bw_GBps", 1.0))
            store_bw = float(dma_entry.get("ps_pl_store_bw_GBps", load_bw))
            load_bytes = int(tile["tile_input_load_bytes_est"] + tile["tile_weight_load_bytes_est"])
            store_bytes = int(tile["tile_output_store_bytes_est"])
            unit_dma_calls = max(1, int(tile["tile_count_est"]))
            unit_dma_ms = transfer_ms_GBps(load_bytes, load_bw) + transfer_ms_GBps(
                store_bytes, store_bw
            )
            unit_frag_ms = unit_dma_calls * float(
                dma_entry.get("fragmentation_penalty_ms_per_call", 0.0)
            )
            dma_ms += unit_dma_ms + unit_frag_ms
            dma_call_est += unit_dma_calls
            dma_parts.append(
                {
                    "bucket": dma_bucket,
                    "unit_name": name,
                    "bytes": int(unit_dma_bytes),
                    "calls_est": int(unit_dma_calls),
                    "dma_ms": unit_dma_ms,
                    "fragmentation_ms": unit_frag_ms,
                }
            )
            tile_input += int(tile["tile_input_load_bytes_est"])
            tile_weight += int(tile["tile_weight_load_bytes_est"])
            tile_output += int(tile["tile_output_store_bytes_est"])
            tile_count += int(tile["tile_count_est"])
            tile_spill += int(tile["tile_spill_penalty"])
            stage_inp_ratio = max(stage_inp_ratio, float(meta["vta_inp_bytes_est"]) / inp_cap)
            stage_wgt_ratio = max(stage_wgt_ratio, float(meta["vta_wgt_bytes_est"]) / wgt_cap)
            stage_acc_ratio = max(stage_acc_ratio, float(meta["vta_acc_bytes_est"]) / acc_cap)
            stage_out_ratio = max(stage_out_ratio, float(meta["vta_out_bytes_est"]) / out_cap)
            residual_live += int(meta["residual_live_bytes_est"])
            if unit_idx < len(units) - 1:
                internal_boundary += int(meta["vta_out_bytes_est"])
            for tile_key, reason in [
                ("inp_tiles", "inp_tile_excess"),
                ("wgt_tiles", "wgt_tile_excess"),
                ("acc_tiles", "acc_tile_excess"),
                ("out_tiles", "out_tile_excess"),
            ]:
                if int(tile[tile_key]) > 8:
                    hard_reject = True
                    reject_reasons.add(reason)
        dma_bytes = tile_input + tile_weight + tile_output
        bridge_pack_ms = stage_overhead(cost_model, "vta", "bridge_pack_ms")
        bridge_unpack_ms = stage_overhead(cost_model, "vta", "bridge_unpack_ms")
        runner_submit_ms = stage_overhead(cost_model, "vta", "runner_submit_ms")
        sync_wait_ms = stage_overhead(cost_model, "vta", "sync_wait_ms")
        ms = bridge_pack_ms + runner_submit_ms + vta_compute_ms + dma_ms + sync_wait_ms + bridge_unpack_ms
        proxy = int(dma_bytes + tile_spill + tile_count * 25000)
        vta_ops += ops
        total_dma_bytes += dma_bytes
        total_tile_count += tile_count
        total_tile_spill += tile_spill
        total_internal_boundary_bytes += internal_boundary
        total_residual_live_bytes += residual_live
        max_inp_ratio = max(max_inp_ratio, stage_inp_ratio)
        max_wgt_ratio = max(max_wgt_ratio, stage_wgt_ratio)
        max_acc_ratio = max(max_acc_ratio, stage_acc_ratio)
        max_out_ratio = max(max_out_ratio, stage_out_ratio)
        stage_infos.append(
            {
                "index": idx,
                "name": stage["name"],
                "device": "vta",
                "unit_names": list(units),
                "threads": int(stage_threads[idx]),
                "compute_ops_est": int(ops),
                "compute_gops_est": float(ops) / 1e9,
                "tile_input_load_bytes_est": int(tile_input),
                "tile_weight_load_bytes_est": int(tile_weight),
                "tile_output_store_bytes_est": int(tile_output),
                "static_dma_bytes_est": int(dma_bytes),
                "static_tile_count_est": int(tile_count),
                "static_tile_spill_penalty": int(tile_spill),
                "onchip_inp_util_pct": 100.0 * stage_inp_ratio,
                "onchip_wgt_util_pct": 100.0 * stage_wgt_ratio,
                "onchip_acc_util_pct": 100.0 * stage_acc_ratio,
                "onchip_out_util_pct": 100.0 * stage_out_ratio,
                "proxy_cost_est": int(proxy),
                "static_ms_est": ms,
                "static_vta_compute_ms_est": vta_compute_ms,
                "static_vta_dma_ms_est": dma_ms,
                "static_bridge_pack_ms_est": bridge_pack_ms,
                "static_bridge_unpack_ms_est": bridge_unpack_ms,
                "static_runner_submit_ms_est": runner_submit_ms,
                "static_sync_wait_ms_est": sync_wait_ms,
                "static_dma_call_count_est": int(dma_call_est),
                "static_compute_parts": compute_parts,
                "static_dma_parts": dma_parts,
            }
        )
        proxy_costs.append(proxy)
        static_stage_ms.append(ms)

    for idx in range(len(scheme_cfg) - 1):
        units = stage_unit_names(scheme_cfg[idx])
        if not units:
            continue
        boundary_count += 1
        total_boundary_bytes += int(unit_metadata[units[-1]]["output_bytes"])

    max_proxy = max(proxy_costs) if proxy_costs else 0
    min_proxy = min([cost for cost in proxy_costs if cost > 0], default=0)
    proxy_imbalance = (max_proxy - min_proxy) / float(max_proxy) if max_proxy else 0.0
    max_static_ms = max(static_stage_ms) if static_stage_ms else 0.0
    min_static_ms = min([value for value in static_stage_ms if value > 0.0], default=0.0)
    static_imbalance = (
        (max_static_ms - min_static_ms) / float(max_static_ms) if max_static_ms > 0.0 else 0.0
    )
    tail_cpu_proxy = 0
    if scheme_cfg[-1]["device"] == "cpu":
        tail_cpu_proxy = next(
            item["proxy_cost_est"] for item in stage_infos if item["index"] == len(scheme_cfg) - 1
        )
    dma_avg_bytes = float(total_dma_bytes) / float(total_tile_count) if total_tile_count else 0.0
    static_dma_fragmentation_score = (
        float(total_tile_count) / max(1.0, float(total_dma_bytes) / 4096.0)
        if total_dma_bytes
        else 0.0
    )
    boundary_bw = float(cost_model.get("boundary_bw_GBps", 1.0))
    boundary_penalty_ms = transfer_ms_GBps(total_boundary_bytes, boundary_bw)
    median_stage_ms = median_value(static_stage_ms)
    imbalance_penalty_ms = max(0.0, max_static_ms - median_stage_ms) * float(
        cost_model.get("weights", {}).get("imbalance", 0.25)
    )
    sram_peak_ratio = max(max_inp_ratio, max_wgt_ratio, max_acc_ratio, max_out_ratio)
    sram_risk_penalty_ms = max(0.0, sram_peak_ratio - 0.80) * max_static_ms * float(
        cost_model.get("weights", {}).get("sram_risk", 0.15)
    )
    dma_fragmentation_penalty_ms = static_dma_fragmentation_score * float(
        cost_model.get("weights", {}).get("dma_fragmentation", 1.0)
    )
    if max_static_ms > 0.0:
        score_ms = (
            max_static_ms
            + boundary_penalty_ms
            + imbalance_penalty_ms
            + sram_risk_penalty_ms
            + dma_fragmentation_penalty_ms
        )
        static_rank_score = score_ms
    else:
        static_rank_score = (
            float(max_proxy)
            + float(total_boundary_bytes) * 0.35
            + float(total_dma_bytes) * 0.10
            + float(total_tile_spill) * 0.25
            + float(tail_cpu_proxy) * 0.15
            + static_dma_fragmentation_score * 100000.0
            + max(proxy_imbalance, static_imbalance) * 1000000.0
        )
        score_ms = 0.0
    score_components = {
        "unit_system": {
            "ops": "OP",
            "gops": "GOP/s",
            "time": "ms",
            "dma_bytes": "B",
            "ps_pl_bw": "decimal GB/s",
        },
        "score_ms": score_ms,
        "pipeline_cycle_ms": max_static_ms,
        "boundary_penalty_ms": boundary_penalty_ms,
        "imbalance_penalty_ms": imbalance_penalty_ms,
        "sram_risk_penalty_ms": sram_risk_penalty_ms,
        "dma_fragmentation_penalty_ms": dma_fragmentation_penalty_ms,
        "boundary_bytes": int(total_boundary_bytes),
        "dma_bytes": int(total_dma_bytes),
        "dma_call_count_est": int(
            sum(item.get("static_dma_call_count_est", 0) for item in stage_infos)
        ),
        "sram_utilization_pct": {
            "inp": 100.0 * max_inp_ratio,
            "wgt": 100.0 * max_wgt_ratio,
            "acc": 100.0 * max_acc_ratio,
            "out": 100.0 * max_out_ratio,
            "peak": 100.0 * sram_peak_ratio,
        },
        "bottleneck_stage": max(stage_infos, key=lambda item: item["static_ms_est"])["name"]
        if stage_infos
        else "",
        "stages": stage_infos,
    }

    return {
        "stage_infos": stage_infos,
        "hard_reject": bool(hard_reject),
        "reject_reasons": sorted(reject_reasons),
        "static_score": int(static_rank_score * 1000000.0 if score_ms > 0.0 else static_rank_score),
        "static_rank_score": float(static_rank_score),
        "static_score_ms": float(score_ms),
        "score_components_json": json.dumps(score_components, sort_keys=True),
        "boundary_penalty_ms_est": boundary_penalty_ms,
        "imbalance_penalty_ms_est": imbalance_penalty_ms,
        "sram_risk_penalty_ms_est": sram_risk_penalty_ms,
        "dma_fragmentation_penalty_ms_est": dma_fragmentation_penalty_ms,
        "boundary_bytes": int(total_boundary_bytes),
        "boundary_count": int(boundary_count),
        "cpu_total_compute_ops_est": int(cpu_ops),
        "cpu_total_gops_est": float(cpu_ops) / 1e9,
        "vta_compute_ops_est": int(vta_ops),
        "vta_stage_gops_est": float(vta_ops) / 1e9,
        "total_compute_ops_est": int(cpu_ops + vta_ops),
        "static_dma_bytes_est": int(total_dma_bytes),
        "static_dma_avg_bytes_per_tile_est": dma_avg_bytes,
        "static_dma_fragmentation_score_est": static_dma_fragmentation_score,
        "static_tile_count_est": int(total_tile_count),
        "static_tile_spill_penalty": int(total_tile_spill),
        "static_internal_boundary_bytes": int(total_internal_boundary_bytes),
        "static_residual_live_bytes": int(total_residual_live_bytes),
        "onchip_inp_util_pct": 100.0 * max_inp_ratio,
        "onchip_wgt_util_pct": 100.0 * max_wgt_ratio,
        "onchip_acc_util_pct": 100.0 * max_acc_ratio,
        "onchip_out_util_pct": 100.0 * max_out_ratio,
        "onchip_peak_util_pct": 100.0
        * max(max_inp_ratio, max_wgt_ratio, max_acc_ratio, max_out_ratio),
        "static_pipeline_cycle_ms_est": max_static_ms,
        "static_pipeline_fps_est": 1000.0 / max_static_ms if max_static_ms > 0.0 else 0.0,
        "static_stage_imbalance_ratio_est": max(proxy_imbalance, static_imbalance),
        "static_bottleneck_stage": max(stage_infos, key=lambda item: item["static_ms_est"])["name"]
        if stage_infos
        else "",
        "static_stage_ms_json": json.dumps(
            [
                {
                    "name": item["name"],
                    "device": item["device"],
                    "threads": item["threads"],
                    "ms_est": item["static_ms_est"],
                    "proxy_cost_est": item["proxy_cost_est"],
                    "gops_est": item["compute_gops_est"],
                }
                for item in stage_infos
            ],
            sort_keys=True,
        ),
    }


def candidate_from_scheme(
    scheme_name, alias_scheme_name, scheme_cfg, unit_metadata, capacity_spec, args, cost_model
):
    metrics = score_scheme(scheme_cfg, unit_metadata, capacity_spec, args, cost_model)
    islands = islands_from_scheme(scheme_cfg)
    candidate_id = safe_name(scheme_name)
    if not alias_scheme_name and scheme_name.startswith("islands_"):
        candidate_id = scheme_name
    return {
        "scheme_name": scheme_name,
        "alias_scheme_name": alias_scheme_name or "",
        "candidate_id": candidate_id,
        "scheme_cfg": scheme_cfg,
        "stage_count": len(scheme_cfg),
        "vta_islands": islands,
        "unit_assignment": unit_assignment_from_scheme(scheme_cfg),
        "vta_unit_names": [
            unit_name
            for stage in scheme_cfg
            if stage["device"] == "vta"
            for unit_name in stage_unit_names(stage)
        ],
        **metrics,
    }


def generated_scheme_name(islands):
    parts = []
    for start_idx, end_idx in islands:
        parts.append("{}_{}".format(start_idx, end_idx))
    return "islands_" + "__".join(parts)


def load_candidates(args, cost_model):
    env = vta.get_env()
    full_model = vision.get_model(args.model, pretrained=True)
    feature_blocks = list(full_model.features)
    output_block = full_model.output
    unit_blocks = build_resnet18_unit_blocks(feature_blocks, output_block)
    unit_metadata = build_resnet18_unit_metadata(unit_blocks, env.BATCH, args.image_size)
    capacity_spec = build_vta_capacity_spec()
    all_candidates = []

    def add_candidate(candidate):
        if not native_pipeline_scheme(candidate["scheme_cfg"]):
            return
        all_candidates.append(candidate)

    for islands in enumerate_island_sets(args.max_vta_islands):
        scheme_cfg = scheme_from_islands(islands)
        if not native_pipeline_scheme(scheme_cfg):
            continue
        scheme_name = generated_scheme_name(islands)
        add_candidate(
            candidate_from_scheme(
                scheme_name, "", scheme_cfg, unit_metadata, capacity_spec, args, cost_model
            )
        )

    for name, scheme_cfg in sorted(SCHEMES.items()):
        if name == "all_vta" or not native_pipeline_scheme(scheme_cfg):
            continue
        add_candidate(
            candidate_from_scheme(
                name, name, scheme_cfg, unit_metadata, capacity_spec, args, cost_model
            )
        )

    all_candidates = sorted(
        all_candidates,
        key=lambda item: (
            1 if item["hard_reject"] else 0,
            item["static_rank_score"],
            item["stage_count"],
            item["scheme_name"],
        ),
    )
    by_name = {candidate["scheme_name"]: candidate for candidate in all_candidates}
    by_alias = {
        candidate["alias_scheme_name"]: candidate
        for candidate in all_candidates
        if candidate.get("alias_scheme_name")
    }
    selected = []
    seen = set()

    def add_selected(candidate):
        if candidate is None:
            return
        if candidate["scheme_name"] in seen:
            return
        if candidate["hard_reject"] and not args.allow_hard_reject:
            return
        seen.add(candidate["scheme_name"])
        selected.append(candidate)

    for name in [item.strip() for item in args.include_known.split(",") if item.strip()]:
        add_selected(by_name.get(name) or by_alias.get(name))
    for name in args.candidate_name:
        if name not in by_name and name not in by_alias:
            raise RuntimeError("Unknown candidate: {}".format(name))
        add_selected(by_name.get(name) or by_alias.get(name))
    for candidate in all_candidates:
        add_selected(candidate)
        if args.max_candidates > 0 and len(selected) >= args.max_candidates:
            break
    return selected, all_candidates


def static_candidate_row(candidate, args):
    threads = threads_for_candidate(args, candidate)
    row = {
        "scheme_name": candidate["scheme_name"],
        "alias_scheme_name": candidate["alias_scheme_name"],
        "candidate_id": candidate["candidate_id"],
        "status": "static",
        "stage_count": int(candidate["stage_count"]),
        "stage_devices": "/".join(scheme_devices(candidate["scheme_cfg"])),
        "stage_runtime_threads": ",".join(str(item) for item in threads),
        "selection_bucket": candidate.get("selection_bucket", ""),
        "selection_reason": candidate.get("selection_reason", ""),
        "hard_reject": bool(candidate["hard_reject"]),
        "reject_reasons": ",".join(candidate["reject_reasons"]),
        "vta_island_count": len(candidate["vta_islands"]),
        "vta_islands": json.dumps(candidate["vta_islands"], sort_keys=True),
        "vta_unit_names": " ".join(candidate["vta_unit_names"]),
        "unit_assignment": json.dumps(candidate["unit_assignment"], sort_keys=True),
        "static_score": int(candidate["static_score"]),
        "static_rank_score": float(candidate["static_rank_score"]),
        "static_score_ms": float(candidate.get("static_score_ms", 0.0)),
        "score_components_json": candidate.get("score_components_json", ""),
        "boundary_penalty_ms_est": float(candidate.get("boundary_penalty_ms_est", 0.0)),
        "imbalance_penalty_ms_est": float(candidate.get("imbalance_penalty_ms_est", 0.0)),
        "sram_risk_penalty_ms_est": float(candidate.get("sram_risk_penalty_ms_est", 0.0)),
        "dma_fragmentation_penalty_ms_est": float(
            candidate.get("dma_fragmentation_penalty_ms_est", 0.0)
        ),
        "boundary_bytes": int(candidate["boundary_bytes"]),
        "boundary_count": int(candidate["boundary_count"]),
        "cpu_total_gops_est": float(candidate["cpu_total_gops_est"]),
        "vta_stage_gops_est": float(candidate["vta_stage_gops_est"]),
        "total_gops_est": float(candidate["total_compute_ops_est"]) / 1e9,
        "static_dma_bytes_est": int(candidate["static_dma_bytes_est"]),
        "static_dma_avg_bytes_per_tile_est": float(candidate["static_dma_avg_bytes_per_tile_est"]),
        "static_dma_fragmentation_score_est": float(
            candidate["static_dma_fragmentation_score_est"]
        ),
        "static_tile_count_est": int(candidate["static_tile_count_est"]),
        "static_tile_spill_penalty": int(candidate["static_tile_spill_penalty"]),
        "static_internal_boundary_bytes": int(candidate["static_internal_boundary_bytes"]),
        "static_residual_live_bytes": int(candidate["static_residual_live_bytes"]),
        "onchip_peak_util_pct": float(candidate["onchip_peak_util_pct"]),
        "onchip_inp_util_pct": float(candidate["onchip_inp_util_pct"]),
        "onchip_wgt_util_pct": float(candidate["onchip_wgt_util_pct"]),
        "onchip_acc_util_pct": float(candidate["onchip_acc_util_pct"]),
        "onchip_out_util_pct": float(candidate["onchip_out_util_pct"]),
        "static_pipeline_cycle_ms_est": float(candidate["static_pipeline_cycle_ms_est"]),
        "static_pipeline_fps_est": float(candidate["static_pipeline_fps_est"]),
        "static_stage_imbalance_ratio_est": float(candidate["static_stage_imbalance_ratio_est"]),
        "static_bottleneck_stage": candidate["static_bottleneck_stage"],
        "static_stage_ms_json": candidate["static_stage_ms_json"],
    }
    for key, value in theory_candidate_row(candidate).items():
        if key not in row:
            row[key] = value
    for stage in candidate["stage_infos"]:
        prefix = "stage{}_".format(stage["index"])
        row[prefix + "name"] = stage["name"]
        row[prefix + "device"] = stage["device"]
        row[prefix + "gops_est"] = stage["compute_gops_est"]
        row[prefix + "static_ms_est"] = stage["static_ms_est"]
        row[prefix + "threads"] = stage["threads"]
    return row


def achieved_gops(ops, ms):
    ms = float(ms)
    if ms <= 0.0:
        return 0.0
    return (float(ops) / 1e9) / (ms / 1000.0)


def stage_summary_from_rows(candidate, pipeline_rows, args):
    manifest_stage_count = stage_count_from_rows(pipeline_rows)
    stage_count = manifest_stage_count or len(candidate["scheme_cfg"])
    stage_summary = []
    for idx in range(stage_count):
        stage_cfg = candidate["scheme_cfg"][idx] if idx < len(candidate["scheme_cfg"]) else {}
        static_info = candidate["stage_infos"][idx] if idx < len(candidate["stage_infos"]) else {}
        ms = mean_value(pipeline_rows, "stage{}_ms".format(idx), args.skip_first)
        run_ms = mean_value(pipeline_rows, "stage{}_run_ms".format(idx), args.skip_first)
        stage_summary.append(
            {
                "index": idx,
                "name": stage_cfg.get("name", "stage{}".format(idx)),
                "device": stage_cfg.get("device", ""),
                "ms": ms,
                "run_ms": run_ms,
                "gops_est": float(static_info.get("compute_gops_est", 0.0)),
                "achieved_gops": achieved_gops(
                    static_info.get("compute_ops_est", 0),
                    run_ms if run_ms > 0.0 else ms,
                ),
            }
        )
    return stage_summary


def summarize_run(
    candidate,
    output_dir,
    args,
    threads,
    poll_label,
    poll_ns,
    run_kind,
    rank_eligible=True,
    file_cache_state="",
):
    static_row = static_candidate_row(candidate, args)
    serial_rows = read_jsonl(Path(output_dir) / "stage_serial_result.jsonl")
    pipeline_rows = read_jsonl(Path(output_dir) / "native_result.jsonl")
    manifest = read_json(Path(output_dir) / "manifest.json", default={})
    pipeline_status = profile_status(output_dir, "pipeline")
    serial_status = profile_status(output_dir, "serial")
    profile_metrics = profile_bandwidth_metrics(pipeline_status)
    dma_metrics = profile_dma_fragmentation_metrics(pipeline_status)
    stages = stage_summary_from_rows(candidate, pipeline_rows, args)
    serial_pipeline_match = top1_match(serial_rows, pipeline_rows)
    rpc_gate = rpc_correctness_summary(args, serial_rows if serial_rows else pipeline_rows)
    stage_values = [item["ms"] for item in stages if item["ms"] > 0.0]
    max_stage_ms = max(stage_values) if stage_values else 0.0
    min_stage_ms = min(stage_values) if stage_values else 0.0
    bottleneck = max(stages, key=lambda item: item["ms"]) if stages else {}
    cpu_run_ms = sum(item["run_ms"] for item in stages if item["device"] == "cpu")
    vta_run_ms = sum(item["run_ms"] for item in stages if item["device"] == "vta")
    cpu_ops = sum(
        item.get("compute_ops_est", 0)
        for item in candidate["stage_infos"]
        if item["device"] == "cpu"
    )
    vta_ops = sum(
        item.get("compute_ops_est", 0)
        for item in candidate["stage_infos"]
        if item["device"] == "vta"
    )

    row = dict(static_row)
    row.update(
        {
            "status": "ok" if pipeline_rows else "missing_results",
            "run_kind": run_kind,
            "rank_eligible": bool(rank_eligible),
            "file_cache_policy": args.file_cache_policy,
            "file_cache_state": file_cache_state,
            "output_dir": str(output_dir),
            "manifest_scheme": manifest.get("scheme", ""),
            "manifest_stage_count": int(manifest.get("stage_count", 0) or 0),
            "thread_config": ",".join(str(item) for item in threads),
            "poll_label": poll_label,
            "poll_sleep_ns": int(poll_ns),
            "runs": len(pipeline_rows),
            "skip_first": int(args.skip_first),
            "top1_match": serial_pipeline_match,
            "rpc_baseline_checked": bool(args.rpc_baseline_result),
            "passes_correctness_gate": serial_pipeline_match
            and bool(rpc_gate.get("passes_rpc_baseline", False)),
            "passes_rpc_baseline": bool(rpc_gate.get("passes_rpc_baseline", False)),
            "correctness_policy": rpc_gate.get("correctness_policy", args.correctness_policy),
            "correctness_relaxed": bool(rpc_gate.get("correctness_relaxed", False)),
            "correctness_gate_reason": rpc_gate.get("correctness_gate_reason", ""),
            "baseline_top1": rpc_gate.get("baseline_top1", ""),
            "native_top1": rpc_gate.get("native_top1", ""),
            "serial_throughput_fps": throughput_from_span(serial_rows, args.skip_first),
            "pipeline_throughput_fps": throughput_from_span(pipeline_rows, args.skip_first),
            "pipeline_stage0_interval_ms": first_stage_interval_ms(
                pipeline_rows, args.skip_first
            ),
            "pipeline_total_latency_avg_ms": mean_value(
                pipeline_rows, "total_latency_ms", args.skip_first
            ),
            "pipeline_total_latency_std_ms": std_value(
                pipeline_rows, "total_latency_ms", args.skip_first
            ),
            "stage_ms_summary": json.dumps(stages, sort_keys=True),
            "stage_balance_imbalance_ratio": (
                (max_stage_ms - min_stage_ms) / max_stage_ms if max_stage_ms > 0.0 else 0.0
            ),
            "stage_balance_bottleneck": bottleneck.get("name", ""),
            "stage_balance_bottleneck_device": bottleneck.get("device", ""),
            "steady_state_stage_fps": 1000.0 / max_stage_ms if max_stage_ms > 0.0 else 0.0,
            "cpu_shared_run_ms": cpu_run_ms,
            "vta_run_ms": vta_run_ms,
            "cpu_shared_achieved_gops": achieved_gops(cpu_ops, cpu_run_ms),
            "vta_stage_achieved_gops": achieved_gops(vta_ops, vta_run_ms),
            "serial_vta_device_wait_ms": float(serial_status.get("device_run_wait_us", 0.0))
            / 1000.0,
        }
    )
    for stage in stages:
        prefix = "stage{}_".format(stage["index"])
        row[prefix + "ms"] = stage["ms"]
        row[prefix + "run_ms"] = stage["run_ms"]
        row[prefix + "achieved_gops"] = stage["achieved_gops"]
    row.update(profile_metrics)
    row.update(dma_metrics)
    return row


def scheme_json_path(root_dir, candidate):
    path = Path(root_dir) / "scheme_configs" / (safe_name(candidate["candidate_id"]) + ".json")
    payload = {
        "scheme_name": candidate["scheme_name"],
        "candidate_id": candidate["candidate_id"],
        "scheme_cfg": candidate["scheme_cfg"],
        "vta_islands": candidate["vta_islands"],
        "unit_assignment": candidate["unit_assignment"],
    }
    write_json(path, payload)
    return path


def build_deploy_command(
    args,
    candidate,
    output_dir,
    threads,
    remote_dir,
    package_only=False,
    skip_run=False,
):
    script = repo_root() / "vta" / "tutorials" / "frontend" / "deploy_classification_stage_pipeline_native.py"
    scheme_path = scheme_json_path(Path(output_dir).parents[0], candidate)
    cmd = [
        sys.executable,
        str(script),
        "--board",
        args.board,
        "--remote-dir",
        remote_dir,
        "--remote-min-free-mb",
        str(int(args.remote_min_free_mb)),
        "--scheme",
        "three_stage_a",
        "--scheme-config-json",
        str(scheme_path),
        "--candidate-id",
        candidate["candidate_id"],
        "--vta-islands-json",
        json.dumps(candidate["vta_islands"], sort_keys=True),
        "--unit-assignment-json",
        json.dumps(candidate["unit_assignment"], sort_keys=True),
        "--runs",
        str(int(args.runs)),
        "--queue-depth",
        str(int(args.queue_depth)),
        "--runtime-num-threads",
        str(int(args.runtime_num_threads)),
        "--stage-runtime-num-threads",
        ",".join(str(item) for item in threads),
        "--stage-build-cache-dir",
        str(Path(args.build_cache_dir) / "stages"),
        "--fetch-results-dir",
        str(output_dir),
        "--vta-runtime-profile-dir",
        "profile",
        "--vta-runtime-profile-events-limit",
        str(int(args.vta_runtime_profile_events_limit)),
        "--ssh-command-timeout-s",
        str(int(args.ssh_command_timeout_s)),
        "--scp-timeout-s",
        str(int(args.scp_timeout_s)),
        "--serial-timeout-s",
        str(int(args.serial_timeout_s)),
        "--pipeline-timeout-s",
        str(int(args.pipeline_timeout_s)),
        "--fetch-timeout-s",
        str(int(args.fetch_timeout_s)),
        "--correctness-policy",
        args.correctness_policy,
        "--run-serial-before-pipeline",
        "--no-ssh-control-master",
    ]
    if package_only:
        cmd.extend(
            [
                "--package-only",
                "--build-dir",
                str(Path(args.build_cache_dir) / "buildability" / candidate["candidate_id"]),
                "--keep-build-dir",
            ]
        )
    elif not args.no_cleanup_remote:
        cmd.append("--cleanup-remote-after-run")
    if skip_run:
        cmd.append("--skip-run")
    if not package_only and args.reuse_buildability_cache:
        cached_package = buildability_package_dir(args, candidate)
        if cached_package.is_dir():
            cmd.extend(["--reuse-package-dir", str(cached_package)])
    if not args.no_compare_serial_pipeline:
        cmd.append("--compare-serial-pipeline")
    for opt in args.ssh_option:
        cmd.extend(["--ssh-option", opt])
    if args.image:
        cmd.extend(["--image", args.image])
    if args.image_dir:
        cmd.extend(["--image-dir", args.image_dir])
    if args.max_images:
        cmd.extend(["--max-images", str(int(args.max_images))])
    if args.image_size:
        cmd.extend(["--image-size", str(int(args.image_size))])
    if args.rpc_baseline_result:
        cmd.extend(["--rpc-baseline-result", args.rpc_baseline_result])
    return cmd


def command_text(cmd, poll_ns):
    env_prefix = [
        "AXU5EVB_DRIVER_POST_START_SLEEP_NS={}".format(int(poll_ns)),
        "AXU5EVB_DRIVER_POLL_SLEEP_NS={}".format(int(poll_ns)),
        "TEST_DATA_ROOT_PATH=${TEST_DATA_ROOT_PATH:-/tmp/tvm_test_data}",
        "MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl}",
        "PYTHONUNBUFFERED=1",
    ]
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        + " \\\n".join(env_prefix)
        + " \\\n"
        + " \\\n".join("  " + shlex.quote(str(part)) for part in cmd)
        + "\n"
    )


def run_command(cmd, poll_ns):
    env = os.environ.copy()
    env["AXU5EVB_DRIVER_POST_START_SLEEP_NS"] = str(int(poll_ns))
    env["AXU5EVB_DRIVER_POLL_SLEEP_NS"] = str(int(poll_ns))
    env.setdefault("TEST_DATA_ROOT_PATH", "/tmp/tvm_test_data")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    env["PYTHONUNBUFFERED"] = "1"
    print("[CMD]", " ".join(shlex.quote(str(part)) for part in cmd))
    proc = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(repo_root()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
        )


def buildability_status_path(args, candidate):
    digest = hashlib.sha256(candidate["candidate_id"].encode("utf-8")).hexdigest()[:16]
    return Path(args.build_cache_dir) / "candidate_status" / (safe_name(candidate["candidate_id"]) + "_" + digest + ".json")


def buildability_package_dir(args, candidate):
    return Path(args.build_cache_dir) / "buildability" / candidate["candidate_id"] / "package"


def classify_command_failure(err, default_type):
    text = " ".join(
        [
            str(getattr(err, "output", "") or ""),
            str(getattr(err, "stderr", "") or ""),
            repr(err),
        ]
    ).lower()
    if "free space is too low" in text or "remote_space_low" in text:
        return "remote_space_low"
    if "rpc baseline" in text or "top1 comparison failed" in text:
        return "correctness_failed"
    if "profile" in text and "missing" in text:
        return "profile_missing"
    if "timeout" in text:
        return "timeout"
    if default_type in FAILURE_TYPES:
        return default_type
    return "build_failed"


def failure_row(candidate, args, failure_type, error_summary, output_dir="", run_command_path="", failed_stage=""):
    row = static_candidate_row(candidate, args)
    row.update(
        {
            "status": failure_type,
            "buildable": False,
            "failed_stage": failed_stage,
            "failure_type": failure_type,
            "error_summary": str(error_summary)[:800],
            "run_command_path": str(run_command_path),
            "output_dir": str(output_dir),
        }
    )
    return row


def read_candidate_id_file(path):
    values = set()
    path = Path(path)
    if not path.exists():
        raise RuntimeError("--exclude-candidate-id-file does not exist: {}".format(path))
    with path.open("r", encoding="utf-8") as inp:
        for line in inp:
            line = line.split("#", 1)[0].strip()
            if line:
                values.add(line)
    return values


def excluded_candidate_ids(args):
    values = set()
    for path in args.exclude_candidate_id_file:
        values.update(read_candidate_id_file(path))
    return values


def candidate_excluded(candidate, excluded):
    names = {
        candidate.get("candidate_id", ""),
        candidate.get("scheme_name", ""),
        candidate.get("alias_scheme_name", ""),
    }
    return bool(excluded.intersection(name for name in names if name))


def candidate_lookup(candidates):
    lookup = {}
    for candidate in candidates:
        for key in [
            candidate.get("candidate_id", ""),
            candidate.get("scheme_name", ""),
            candidate.get("alias_scheme_name", ""),
        ]:
            if key:
                lookup[key] = candidate
    return lookup


def effective_device_segments(candidate):
    segments = []
    for device in scheme_devices(candidate["scheme_cfg"]):
        if not segments or segments[-1] != device:
            segments.append(device)
    return segments


def effective_device_pattern(candidate):
    return "/".join(effective_device_segments(candidate))


def island_lengths(candidate):
    lengths = []
    for island in candidate.get("vta_islands", []):
        lengths.append(int(island["end_idx"]) - int(island["start_idx"]) + 1)
    return lengths


def theory_base_metrics(candidate):
    cpu_stage_ms = [
        float(stage.get("static_ms_est", 0.0))
        for stage in candidate["stage_infos"]
        if stage.get("device") == "cpu"
    ]
    vta_stages = [
        stage for stage in candidate["stage_infos"] if stage.get("device") == "vta"
    ]
    vta_dma_ms = sum(float(stage.get("static_vta_dma_ms_est", 0.0)) for stage in vta_stages)
    vta_total_occupied_ms = sum(float(stage.get("static_ms_est", 0.0)) for stage in vta_stages)
    vta_compute_submit_ms = max(0.0, vta_total_occupied_ms - vta_dma_ms)
    max_cpu_stage_ms = max(cpu_stage_ms) if cpu_stage_ms else 0.0
    tail_cpu_ms = 0.0
    if candidate["stage_infos"] and candidate["stage_infos"][-1].get("device") == "cpu":
        tail_cpu_ms = float(candidate["stage_infos"][-1].get("static_ms_est", 0.0))

    lengths = island_lengths(candidate)
    single_unit_islands = sum(1 for length in lengths if length <= 1)
    tiny_islands = sum(1 for length in lengths if length <= 2)
    two_unit_islands = sum(1 for length in lengths if length == 2)
    three_unit_islands = sum(1 for length in lengths if length == 3)
    small_island_penalty_ms = single_unit_islands * 28.0 + two_unit_islands * 18.0 + three_unit_islands * 6.0 + max(
        0, tiny_islands - single_unit_islands
    ) * 0.0

    pattern = effective_device_pattern(candidate)
    if pattern == "cpu/vta/cpu":
        effective_segment_penalty_ms = 0.0
    elif pattern == "cpu/vta/cpu/vta/cpu":
        effective_segment_penalty_ms = 8.0
    else:
        effective_segment_penalty_ms = 36.0 + max(
            0, len(effective_device_segments(candidate)) - 5
        ) * 14.0

    cpu_tail_penalty_ms = max(0.0, tail_cpu_ms - 78.0) * 0.90
    sram_spill_penalty_ms = float(candidate.get("sram_risk_penalty_ms_est", 0.0))
    if int(candidate.get("static_tile_spill_penalty", 0)) > 0:
        sram_spill_penalty_ms += 25.0
    dma_fragmentation_penalty_ms = float(
        candidate.get("dma_fragmentation_penalty_ms_est", 0.0)
    )
    boundary_ms = float(candidate.get("boundary_penalty_ms_est", 0.0))

    resource_cycle_ms = max(max_cpu_stage_ms, vta_compute_submit_ms)
    predicted_cycle_ms = (
        resource_cycle_ms
        + vta_dma_ms
        + boundary_ms
        + dma_fragmentation_penalty_ms
        + sram_spill_penalty_ms
        + cpu_tail_penalty_ms
        + small_island_penalty_ms
        + effective_segment_penalty_ms
    )
    theory_hard_reject = bool(candidate.get("hard_reject", False)) or single_unit_islands > 0
    if tail_cpu_ms > 95.0 or float(candidate.get("onchip_peak_util_pct", 0.0)) > 92.0:
        theory_hard_reject = True

    return {
        "theory_predicted_cycle_ms": float(predicted_cycle_ms),
        "theory_predicted_fps": 1000.0 / predicted_cycle_ms
        if predicted_cycle_ms > 0.0
        else 0.0,
        "theory_resource_cycle_ms": float(resource_cycle_ms),
        "theory_max_cpu_stage_ms": float(max_cpu_stage_ms),
        "theory_tail_cpu_ms": float(tail_cpu_ms),
        "theory_vta_compute_submit_ms": float(vta_compute_submit_ms),
        "theory_vta_dma_ms": float(vta_dma_ms),
        "theory_vta_total_occupied_ms": float(vta_total_occupied_ms),
        "theory_boundary_ms": float(boundary_ms),
        "theory_dma_fragmentation_penalty_ms": float(dma_fragmentation_penalty_ms),
        "theory_sram_spill_penalty_ms": float(sram_spill_penalty_ms),
        "theory_cpu_tail_penalty_ms": float(cpu_tail_penalty_ms),
        "theory_small_island_penalty_ms": float(small_island_penalty_ms),
        "theory_effective_segment_penalty_ms": float(effective_segment_penalty_ms),
        "theory_effective_device_pattern": pattern,
        "theory_vta_island_lengths": ",".join(str(item) for item in lengths),
        "theory_single_unit_vta_island_count": int(single_unit_islands),
        "theory_hard_reject": bool(theory_hard_reject),
    }


def iter_warm_summary_rows(output_root, current_search_dir):
    output_root = Path(output_root)
    current_search_dir = Path(current_search_dir).resolve()
    for path in sorted(output_root.glob("batch*/summary.json")):
        try:
            if path.parent.resolve() == current_search_dir:
                continue
        except OSError:
            pass
        payload = read_json(path, default={})
        for row in payload.get("rows", []) or []:
            yield row


def median_or_zero(values):
    values = sorted(float(item) for item in values)
    if not values:
        return 0.0
    return statistics.median(values)


def compute_theory_residuals(args, candidates, search_dir):
    lookup = candidate_lookup(candidates)
    global_residuals = []
    by_pattern = {}
    for row in iter_warm_summary_rows(args.output_root, search_dir):
        if row.get("status") != "ok":
            continue
        fps = as_float(row.get("pipeline_throughput_fps"), 0.0)
        if fps <= 0.0:
            continue
        candidate = lookup.get(row.get("candidate_id", "")) or lookup.get(
            row.get("scheme_name", "")
        )
        if not candidate:
            continue
        metrics = theory_base_metrics(candidate)
        residual = (1000.0 / fps) - float(metrics["theory_predicted_cycle_ms"])
        global_residuals.append(residual)
        by_pattern.setdefault(metrics["theory_effective_device_pattern"], []).append(residual)
    global_median = median_or_zero(global_residuals)
    return {
        "global": global_median,
        "by_pattern": {key: median_or_zero(values) for key, values in by_pattern.items()},
        "sample_count": len(global_residuals),
    }


def annotate_theory_scores(args, candidates, search_dir):
    residuals = compute_theory_residuals(args, candidates, search_dir)
    for candidate in candidates:
        metrics = theory_base_metrics(candidate)
        raw_residual = residuals["by_pattern"].get(
            metrics["theory_effective_device_pattern"], residuals["global"]
        )
        residual = max(-35.0, min(35.0, float(raw_residual) * 0.35))
        corrected_cycle_ms = max(1.0, metrics["theory_predicted_cycle_ms"] + residual)
        metrics.update(
            {
                "theory_residual_ms": float(residual),
                "theory_corrected_cycle_ms": float(corrected_cycle_ms),
                "theory_corrected_fps": 1000.0 / corrected_cycle_ms,
                "theory_residual_sample_count": int(residuals["sample_count"]),
            }
        )
        candidate.update(metrics)


def sort_candidates_theory(candidates):
    return sorted(
        candidates,
        key=lambda item: (
            1 if item.get("theory_hard_reject", False) else 0,
            1 if item.get("hard_reject", False) else 0,
            float(item.get("theory_corrected_cycle_ms", 1e9)),
            float(item.get("theory_tail_cpu_ms", 1e9)),
            float(item.get("static_dma_fragmentation_score_est", 1e9)),
            int(item.get("boundary_bytes", 0)),
            item.get("scheme_name", ""),
        ),
    )


def theory_candidate_row(candidate):
    keys = [
        "scheme_name",
        "candidate_id",
        "stage_count",
        "theory_effective_device_pattern",
        "theory_corrected_cycle_ms",
        "theory_corrected_fps",
        "theory_predicted_cycle_ms",
        "theory_predicted_fps",
        "theory_residual_ms",
        "theory_max_cpu_stage_ms",
        "theory_tail_cpu_ms",
        "theory_vta_total_occupied_ms",
        "theory_vta_compute_submit_ms",
        "theory_vta_dma_ms",
        "theory_boundary_ms",
        "theory_dma_fragmentation_penalty_ms",
        "theory_sram_spill_penalty_ms",
        "theory_cpu_tail_penalty_ms",
        "theory_small_island_penalty_ms",
        "theory_effective_segment_penalty_ms",
        "theory_vta_island_lengths",
        "theory_single_unit_vta_island_count",
        "theory_hard_reject",
        "hard_reject",
        "reject_reasons",
        "static_rank_score",
        "static_score_ms",
        "static_pipeline_cycle_ms_est",
        "static_pipeline_fps_est",
        "boundary_bytes",
        "static_dma_bytes_est",
        "static_dma_fragmentation_score_est",
        "onchip_peak_util_pct",
        "vta_unit_names",
    ]
    row = {}
    for key in keys:
        value = candidate.get(key, "")
        if isinstance(value, list):
            value = ",".join(str(item) for item in value)
        row[key] = value
    return row


def write_theory_candidate_files(search_dir, ranked_candidates, selected, candidate_prior_file=""):
    search_dir = Path(search_dir)
    write_csv(
        search_dir / "theory_ranked_candidates.csv",
        [theory_candidate_row(candidate) for candidate in ranked_candidates],
    )
    text = "".join("{}\n".format(candidate["candidate_id"]) for candidate in selected)
    (search_dir / "theory_build100_candidate_ids.txt").write_text(text, encoding="utf-8")
    if candidate_prior_file:
        prior_path = Path(candidate_prior_file)
        prior_path.parent.mkdir(parents=True, exist_ok=True)
        prior_path.write_text(text, encoding="utf-8")
    write_json(
        search_dir / "theory_build100_summary.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "candidate_count": len(selected),
            "candidate_ids": [candidate["candidate_id"] for candidate in selected],
            "top_candidates": [theory_candidate_row(candidate) for candidate in selected[:20]],
        },
    )


def read_candidate_prior(args):
    if not args.candidate_prior_file:
        return []
    path = Path(args.candidate_prior_file)
    if not path.exists():
        return []
    values = []
    seen = set()
    with path.open("r", encoding="utf-8") as inp:
        for line in inp:
            line = line.split("#", 1)[0].strip()
            if line and line not in seen:
                seen.add(line)
                values.append(line)
    return values


def select_prior_candidates(args, candidates, limit):
    prior = read_candidate_prior(args)
    if not prior:
        return []
    lookup = candidate_lookup(candidates)
    excluded = excluded_candidate_ids(args)
    selected = []
    seen = set()
    for name in prior:
        candidate = lookup.get(name)
        if not candidate:
            raise RuntimeError("--candidate-prior-file contains unknown candidate: {}".format(name))
        if candidate["scheme_name"] in seen:
            continue
        if candidate_excluded(candidate, excluded):
            continue
        if candidate["hard_reject"] and not args.allow_hard_reject:
            continue
        candidate["selection_bucket"] = "candidate_prior_file"
        candidate["selection_reason"] = "ordered by --candidate-prior-file"
        seen.add(candidate["scheme_name"])
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def write_theory_build_outputs(search_dir, args, candidates, rows):
    search_dir = Path(search_dir)
    by_id = {row.get("candidate_id"): row for row in rows}
    buildable_ids = []
    failed_ids = []
    manifest_rows = []
    for candidate in candidates:
        row = by_id.get(candidate["candidate_id"], {})
        buildable = bool(row.get("buildable", candidate.get("buildable", False)))
        if buildable:
            buildable_ids.append(candidate["candidate_id"])
        else:
            failed_ids.append(candidate["candidate_id"])
        package_dir = buildability_package_dir(args, candidate)
        manifest_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "scheme_name": candidate["scheme_name"],
                "stage_count": int(candidate["stage_count"]),
                "effective_device_pattern": candidate.get(
                    "theory_effective_device_pattern", effective_device_pattern(candidate)
                ),
                "vta_islands": candidate.get("vta_islands", []),
                "vta_island_lengths": candidate.get("theory_vta_island_lengths", ""),
                "predicted_theory_score_ms": float(
                    candidate.get("theory_corrected_cycle_ms", candidate.get("static_score_ms", 0.0))
                ),
                "predicted_cycle_ms": float(candidate.get("theory_corrected_cycle_ms", 0.0)),
                "predicted_fps": float(candidate.get("theory_corrected_fps", 0.0)),
                "predicted_raw_cycle_ms": float(candidate.get("theory_predicted_cycle_ms", 0.0)),
                "theory_residual_ms": float(candidate.get("theory_residual_ms", 0.0)),
                "max_cpu_stage_ms": float(candidate.get("theory_max_cpu_stage_ms", 0.0)),
                "tail_cpu_ms": float(candidate.get("theory_tail_cpu_ms", 0.0)),
                "vta_total_occupied_ms": float(
                    candidate.get("theory_vta_total_occupied_ms", 0.0)
                ),
                "vta_compute_submit_ms": float(
                    candidate.get("theory_vta_compute_submit_ms", 0.0)
                ),
                "vta_dma_ms": float(candidate.get("theory_vta_dma_ms", 0.0)),
                "boundary_ms": float(candidate.get("theory_boundary_ms", 0.0)),
                "boundary_bytes": int(candidate.get("boundary_bytes", 0)),
                "static_dma_bytes_est": int(candidate.get("static_dma_bytes_est", 0)),
                "dma_fragmentation_score_est": float(
                    candidate.get("static_dma_fragmentation_score_est", 0.0)
                ),
                "onchip_peak_util_pct": float(candidate.get("onchip_peak_util_pct", 0.0)),
                "buildable": buildable,
                "build_cache_hit": bool(row.get("buildability_cache_hit", False)),
                "build_failure_type": row.get("build_failure_type", ""),
                "build_error_summary": row.get("build_error_summary", ""),
                "package_path": str(package_dir) if package_dir.exists() else "",
                "status_path": str(buildability_status_path(args, candidate)),
            }
        )
    (search_dir / "theory_build100_buildable_ids.txt").write_text(
        "".join("{}\n".format(item) for item in buildable_ids), encoding="utf-8"
    )
    (search_dir / "theory_build100_failed_ids.txt").write_text(
        "".join("{}\n".format(item) for item in failed_ids), encoding="utf-8"
    )
    write_json(
        search_dir / "theory_build100_manifest.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "candidate_count": len(candidates),
            "buildable_count": len(buildable_ids),
            "failed_count": len(failed_ids),
            "build_cache_dir": str(args.build_cache_dir),
            "rows": manifest_rows,
        },
    )


def run_buildability_candidate(args, search_dir, candidate):
    status_path = buildability_status_path(args, candidate)
    if (args.skip_existing or args.reuse_buildability_cache) and status_path.exists():
        cached = read_json(status_path, default={})
        candidate.update(
            {
                "buildable": bool(cached.get("buildable", False)),
                "build_failed_stage": cached.get("build_failed_stage", ""),
                "build_failure_type": cached.get("build_failure_type", ""),
                "build_error_summary": cached.get("build_error_summary", ""),
                "buildability_cache_hit": True,
            }
        )
        print("[BUILDABILITY-CACHE] hit {}".format(candidate["scheme_name"]))
        return cached
    threads = threads_for_candidate(args, candidate)
    output_dir = Path(search_dir) / "buildability" / safe_name(candidate["candidate_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_deploy_command(
        args,
        candidate,
        output_dir,
        threads,
        args.remote_dir.rstrip("/") + "/buildability_" + safe_name(candidate["candidate_id"]),
        package_only=True,
    )
    script_path = output_dir / "build_command.sh"
    script_path.write_text(command_text(cmd, DEFAULT_POLL_VALUES[0][1]), encoding="utf-8")
    script_path.chmod(0o755)
    try:
        run_command(cmd, DEFAULT_POLL_VALUES[0][1])
        status = {
            "candidate_id": candidate["candidate_id"],
            "buildable": True,
            "build_failed_stage": "",
            "build_failure_type": "",
            "build_error_summary": "",
            "buildability_cache_hit": False,
        }
    except subprocess.CalledProcessError as err:
        failure_type = classify_command_failure(err, "build_failed")
        status = {
            "candidate_id": candidate["candidate_id"],
            "buildable": False,
            "build_failed_stage": "",
            "build_failure_type": failure_type,
            "build_error_summary": "returncode={} {}".format(
                err.returncode, (getattr(err, "stderr", "") or "")[:400]
            ),
            "buildability_cache_hit": False,
        }
    except Exception as err:  # pylint: disable=broad-except
        status = {
            "candidate_id": candidate["candidate_id"],
            "buildable": False,
            "build_failed_stage": "",
            "build_failure_type": type(err).__name__,
            "build_error_summary": repr(err)[:800],
            "buildability_cache_hit": False,
        }
    write_json(status_path, status)
    candidate.update(
        {
            "buildable": bool(status["buildable"]),
            "build_failed_stage": status["build_failed_stage"],
            "build_failure_type": status["build_failure_type"],
            "build_error_summary": status["build_error_summary"],
            "buildability_cache_hit": bool(status.get("buildability_cache_hit", False)),
        }
    )
    return status


def apply_cached_buildability(args, candidates):
    for candidate in candidates:
        status = read_json(buildability_status_path(args, candidate), default={})
        candidate["buildable"] = bool(status.get("buildable", False))
        candidate["build_failed_stage"] = status.get("build_failed_stage", "")
        candidate["build_failure_type"] = status.get("build_failure_type", "")
        candidate["build_error_summary"] = status.get("build_error_summary", "")


def select_measurement_candidates(args, candidates):
    excluded = excluded_candidate_ids(args)
    selectable = [
        item
        for item in candidates
        if not item["hard_reject"] and (args.skip_buildability or item.get("buildable", False))
        and not candidate_excluded(item, excluded)
    ]
    known_names = [item.strip() for item in args.include_known.split(",") if item.strip()]
    selected = []
    seen = set()

    def add(candidate):
        if candidate is None or candidate["scheme_name"] in seen:
            return
        seen.add(candidate["scheme_name"])
        selected.append(candidate)

    for name in known_names:
        add(next((item for item in selectable if item["scheme_name"] == name or item["alias_scheme_name"] == name), None))
    for candidate in selectable:
        add(candidate)
        if len(selected) >= int(args.measure_top_n):
            break
    return selected[: int(args.measure_top_n)]


def measured_candidate_ids_from_rows(rows):
    ids = []
    seen = set()
    ignored_status = {"buildable", "unbuildable", "dry_run"}
    for row in rows:
        if not row.get("run_kind"):
            continue
        if row.get("status") in ignored_status:
            continue
        candidate_id = row.get("candidate_id") or row.get("scheme_name")
        if candidate_id and candidate_id not in seen:
            seen.add(candidate_id)
            ids.append(candidate_id)
    return ids


def write_measured_candidate_ids(path, rows):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = measured_candidate_ids_from_rows(rows)
    path.write_text("".join("{}\n".format(item) for item in ids), encoding="utf-8")


def measured_status_counts(rows):
    ignored_status = {"buildable", "unbuildable", "dry_run"}
    success = 0
    failed = 0
    attempted = 0
    for row in rows:
        if not row.get("run_kind"):
            continue
        if row.get("status") in ignored_status:
            continue
        attempted += 1
        if row.get("status") == "ok":
            success += 1
        else:
            failed += 1
    return attempted, success, failed


def render_progress_bar(done, total, width=20):
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    filled = int(round(width * done / float(total)))
    return "{}{}".format("#" * filled, "-" * (width - filled))


def write_progress(
    search_dir,
    args,
    stats,
    rows,
    batch_total,
    current_candidate="",
    status="initializing",
):
    attempted, success, failed = measured_status_counts(rows)
    batch_done = int(stats.get("board_test_config_count", attempted))
    batch_total = int(batch_total or args.max_board_test_configs)
    total_count = int(args.progress_total_count or batch_total)
    total_done = int(args.progress_total_offset) + batch_done
    started_at = float(stats.get("progress_started_at", time.time()))
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "batch_index": int(args.progress_batch_index),
        "batch_count": int(args.progress_batch_count),
        "batch_done": batch_done,
        "batch_total": batch_total,
        "total_done": total_done,
        "total_count": total_count,
        "current_candidate": current_candidate,
        "status": status,
        "success_count": success,
        "failure_count": failed,
        "attempted_row_count": attempted,
        "elapsed_s": round(time.time() - started_at, 3),
    }
    write_json(Path(search_dir) / "progress.json", payload)
    bar = render_progress_bar(batch_done, batch_total)
    line = (
        "batch{idx:03d} [{bar}] {done}/{total} total {tdone}/{ttotal} "
        "current={current} status={status} ok={ok} failed={failed} elapsed={elapsed:.1f}s"
    ).format(
        idx=int(args.progress_batch_index),
        bar=bar,
        done=batch_done,
        total=batch_total,
        tdone=total_done,
        ttotal=total_count,
        current=current_candidate or "-",
        status=status,
        ok=success,
        failed=failed,
        elapsed=payload["elapsed_s"],
    )
    Path(search_dir, "progress.txt").write_text(line + "\n", encoding="utf-8")
    print("[PROGRESS] " + line, flush=True)


def select_static_shortlist(args, candidates, search_dir=None):
    limit = int(args.static_shortlist_n)
    if args.shortlist_policy == "theory_residual":
        if search_dir is None:
            raise RuntimeError("theory_residual shortlist requires search_dir")
        annotate_theory_scores(args, candidates, search_dir)
        excluded = excluded_candidate_ids(args)
        ranked = [
            candidate
            for candidate in sort_candidates_theory(candidates)
            if not candidate_excluded(candidate, excluded)
            and not (candidate.get("theory_hard_reject", False) and not args.allow_hard_reject)
        ]
        selected = ranked[:limit]
        for candidate in selected:
            candidate["selection_bucket"] = "theory_residual"
            candidate["selection_reason"] = "theory score plus warm100 residual correction"
        write_theory_candidate_files(search_dir, ranked, selected, args.candidate_prior_file)
        return selected

    prior_selected = select_prior_candidates(args, candidates, limit)
    if prior_selected:
        return prior_selected[:limit]

    selected = []
    seen = set()

    def add(candidate, bucket, reason):
        if candidate is None or candidate["scheme_name"] in seen:
            return
        if candidate["hard_reject"] and not args.allow_hard_reject:
            return
        candidate["selection_bucket"] = bucket
        candidate["selection_reason"] = reason
        seen.add(candidate["scheme_name"])
        selected.append(candidate)

    known_names = [item.strip() for item in args.include_known.split(",") if item.strip()]
    for name in known_names:
        add(
            next(
                (
                    item
                    for item in candidates
                    if item["scheme_name"] == name or item["alias_scheme_name"] == name
                ),
                None,
            ),
            "known_manual_baseline",
            "included baseline",
        )

    sorted_candidates = sort_candidates_static(candidates)
    top_score_quota = max(0, 160 - len(selected))
    top_score_added = 0
    for candidate in sorted_candidates:
        before = len(selected)
        add(candidate, "calibrated_score_top", "top calibrated score")
        if len(selected) != before:
            top_score_added += 1
        if top_score_added >= top_score_quota:
            break

    for island_count, quota in [(1, 24), (2, 24), (3, 24)]:
        count = 0
        for candidate in sorted_candidates:
            if len(candidate["vta_islands"]) != island_count:
                continue
            before = len(selected)
            add(
                candidate,
                "{}_island_best".format(island_count),
                "best candidate in {}-island bucket".format(island_count),
            )
            if len(selected) != before:
                count += 1
            if count >= quota:
                break

    stage_quotas = [(3, 6), (5, 5), (7, 5)]
    for stage_count, stage_quota in stage_quotas:
        stage_count_added = 0
        for candidate in sorted_candidates:
            if int(candidate["stage_count"]) != stage_count:
                continue
            before = len(selected)
            add(candidate, "stage_count_diversity", "stage_count={}".format(stage_count))
            if len(selected) != before:
                stage_count_added += 1
            if stage_count_added >= stage_quota:
                break

    low_boundary = sorted(
        [item for item in candidates if not item["hard_reject"]],
        key=lambda item: (item["boundary_bytes"], item["static_rank_score"]),
    )
    low_boundary_count = 0
    for candidate in low_boundary:
        before = len(selected)
        add(candidate, "low_boundary_bytes", "lowest boundary bytes")
        if len(selected) != before:
            low_boundary_count += 1
        if low_boundary_count >= 8:
            break

    for candidate in sorted_candidates:
        add(candidate, "calibrated_score_fill", "filled remaining shortlist slot")
        if len(selected) >= limit:
            break
    return selected[:limit]


def select_buildability_candidates(args, shortlist):
    return shortlist[: max(0, int(args.buildability_top_n))]


def candidate_run_label(candidate, threads, poll_label, run_kind):
    return "{}_threads_{}_poll_{}_{}".format(
        safe_name(candidate["candidate_id"]),
        "_".join(str(item) for item in threads),
        safe_name(poll_label),
        safe_name(run_kind),
    )


def run_or_reuse_candidate(
    args,
    search_dir,
    candidate,
    threads,
    poll_label,
    poll_ns,
    run_kind,
    rank_eligible=True,
    file_cache_state="",
):
    label = candidate_run_label(candidate, threads, poll_label, run_kind)
    output_dir = Path(search_dir) / label
    output_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = args.remote_dir.rstrip("/") + "/" + label
    cmd = build_deploy_command(args, candidate, output_dir, threads, remote_dir, package_only=False)
    run_script = output_dir / "run_command.sh"
    run_script.write_text(command_text(cmd, poll_ns), encoding="utf-8")
    run_script.chmod(0o755)

    has_results = (output_dir / "native_result.jsonl").exists() and (
        output_dir / "stage_serial_result.jsonl"
    ).exists()
    if args.dry_run:
        row = static_candidate_row(candidate, args)
        row.update(
            {
                "status": "dry_run",
                "run_kind": run_kind,
                "rank_eligible": bool(rank_eligible),
                "file_cache_policy": args.file_cache_policy,
                "file_cache_state": file_cache_state,
                "output_dir": str(output_dir),
                "thread_config": ",".join(str(item) for item in threads),
                "poll_label": poll_label,
                "poll_sleep_ns": int(poll_ns),
            }
        )
        return row

    if not (args.skip_existing and has_results):
        run_command(cmd, poll_ns)
    return summarize_run(
        candidate,
        output_dir,
        args,
        threads,
        poll_label,
        poll_ns,
        run_kind,
        rank_eligible=rank_eligible,
        file_cache_state=file_cache_state,
    )


def warm_remote_file_cache(args, search_dir, candidate, threads, poll_label, poll_ns, run_kind):
    label = candidate_run_label(
        candidate, threads, poll_label, "{}_file_cache_warmup".format(run_kind)
    )
    output_dir = Path(search_dir) / label
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / ".file_cache_warmup_done"
    if args.skip_existing and marker.exists():
        print("[FILE-CACHE] warmup cache hit:", label)
        return
    remote_dir = args.remote_dir.rstrip("/") + "/" + label
    cmd = build_deploy_command(
        args,
        candidate,
        output_dir,
        threads,
        remote_dir,
        package_only=False,
        skip_run=True,
    )
    run_script = output_dir / "run_command.sh"
    run_script.write_text(command_text(cmd, poll_ns), encoding="utf-8")
    run_script.chmod(0o755)
    print("[FILE-CACHE] prewarm candidate={} threads={} poll={}({} ns)".format(
        candidate["scheme_name"],
        ",".join(str(item) for item in threads),
        poll_label,
        poll_ns,
    ))
    run_command(cmd, poll_ns)
    marker.write_text(datetime.now().isoformat(timespec="seconds") + "\n", encoding="utf-8")


def sort_candidates_static(candidates):
    return sorted(
        candidates,
        key=lambda item: (
            1 if item["hard_reject"] else 0,
            1 if not item.get("buildable", True) else 0,
            item["static_rank_score"],
            item["boundary_bytes"],
            item["static_dma_fragmentation_score_est"],
            item["scheme_name"],
        ),
    )


def sort_rows_for_best(rows):
    def key(row):
        rank_eligible = row.get("rank_eligible", True) is not False
        ok = (
            rank_eligible
            and row.get("status") == "ok"
            and bool(row.get("passes_correctness_gate", False))
        )
        measured_fps = float(row.get("pipeline_throughput_fps", 0.0) or 0.0)
        return (
            0 if ok else 1,
            0 if rank_eligible else 1,
            -measured_fps,
            float(row.get("stage_balance_imbalance_ratio", 1.0) or 1.0),
            float(row.get("dma_fragmentation_score", row.get("static_dma_fragmentation_score_est", 0.0)) or 0.0),
            float(row.get("boundary_bytes", 0.0) or 0.0),
            -float(row.get("static_pipeline_fps_est", 0.0) or 0.0),
            float(row.get("static_rank_score", 0.0) or 0.0),
        )

    return sorted(rows, key=key)


def csv_fieldnames(rows):
    preferred = [
        "scheme_name",
        "alias_scheme_name",
        "candidate_id",
        "status",
        "run_kind",
        "rank_eligible",
        "file_cache_policy",
        "file_cache_state",
        "passes_correctness_gate",
        "top1_match",
        "rpc_baseline_checked",
        "passes_rpc_baseline",
        "correctness_policy",
        "correctness_relaxed",
        "correctness_gate_reason",
        "baseline_top1",
        "native_top1",
        "pipeline_throughput_fps",
        "steady_state_stage_fps",
        "serial_throughput_fps",
        "stage_count",
        "stage_devices",
        "thread_config",
        "selection_bucket",
        "selection_reason",
        "stage_balance_imbalance_ratio",
        "stage_balance_bottleneck",
        "stage_balance_bottleneck_device",
        "cpu_total_gops_est",
        "vta_stage_gops_est",
        "total_gops_est",
        "cpu_shared_achieved_gops",
        "vta_stage_achieved_gops",
        "ps_pl_total_bw_gbps",
        "ps_pl_total_dma_bytes",
        "vta_device_wait_ms",
        "onchip_peak_util_pct",
        "onchip_inp_util_pct",
        "onchip_wgt_util_pct",
        "onchip_acc_util_pct",
        "onchip_out_util_pct",
        "dma_avg_bytes_per_call",
        "dma_fragmentation_score",
        "static_dma_fragmentation_score_est",
        "boundary_bytes",
        "static_pipeline_cycle_ms_est",
        "static_pipeline_fps_est",
        "static_score_ms",
        "theory_effective_device_pattern",
        "theory_corrected_cycle_ms",
        "theory_corrected_fps",
        "theory_predicted_cycle_ms",
        "theory_predicted_fps",
        "theory_residual_ms",
        "theory_max_cpu_stage_ms",
        "theory_tail_cpu_ms",
        "theory_vta_total_occupied_ms",
        "theory_vta_compute_submit_ms",
        "theory_vta_dma_ms",
        "theory_boundary_ms",
        "theory_dma_fragmentation_penalty_ms",
        "theory_sram_spill_penalty_ms",
        "theory_cpu_tail_penalty_ms",
        "theory_small_island_penalty_ms",
        "theory_effective_segment_penalty_ms",
        "theory_vta_island_lengths",
        "theory_single_unit_vta_island_count",
        "theory_hard_reject",
        "score_components_json",
        "boundary_penalty_ms_est",
        "imbalance_penalty_ms_est",
        "sram_risk_penalty_ms_est",
        "dma_fragmentation_penalty_ms_est",
        "static_stage_imbalance_ratio_est",
        "static_bottleneck_stage",
        "buildable",
        "buildability_cache_hit",
        "failure_type",
        "failed_stage",
        "error_summary",
        "run_command_path",
        "build_failure_type",
        "build_error_summary",
        "vta_island_count",
        "vta_unit_names",
        "output_dir",
    ]
    keys = []
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
    for row in rows:
        for key in sorted(row):
            if key not in keys:
                keys.append(key)
    return keys


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = csv_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value):
    if value in ["", None]:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    try:
        return "{:.3f}".format(float(value))
    except (TypeError, ValueError):
        return str(value)


def render_readme(args, rows, stats):
    ranked = sort_rows_for_best(rows)
    best = ranked[0] if ranked else None
    lines = [
        "# ResNet18 N-stage Stage Split Search",
        "",
        "- Generated: {}".format(datetime.now().isoformat(timespec="seconds")),
        "- Raw island candidates: {}".format(stats.get("raw_candidate_count", "n/a")),
        "- Native candidates: {}".format(stats.get("native_candidate_count", "n/a")),
        "- Static shortlist: {}".format(stats.get("static_shortlist_count", "n/a")),
        "- Buildability tested: {}".format(stats.get("buildability_tested_count", "n/a")),
        "- Buildable candidates: {}".format(stats.get("buildable_count", "n/a")),
        "- Board test configs: {}".format(stats.get("board_test_config_count", "n/a")),
        "- File cache policy: {}".format(args.file_cache_policy),
        "- File cache warmups: {}".format(stats.get("file_cache_warmup_count", "n/a")),
        "- Max VTA islands: {}".format(args.max_vta_islands),
        "- Measure top N: {}".format(args.measure_top_n),
        "- Refine top N: {}".format(args.refine_top_n),
        "- Primary objective: measured pipeline throughput fps",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "## Best",
                "",
                "- Scheme: {}".format(best.get("scheme_name", "n/a")),
                "- Candidate: {}".format(best.get("candidate_id", "n/a")),
                "- Stage devices: {}".format(best.get("stage_devices", "n/a")),
                "- Threads: {}".format(best.get("thread_config", best.get("stage_runtime_threads", "n/a"))),
                "- Correctness gate: {}".format(best.get("passes_correctness_gate", "n/a")),
                "- Correctness policy: {} relaxed={} reason={} baseline_top1={} native_top1={}".format(
                    best.get("correctness_policy", "n/a"),
                    best.get("correctness_relaxed", "n/a"),
                    best.get("correctness_gate_reason", "n/a"),
                    best.get("baseline_top1", "n/a"),
                    best.get("native_top1", "n/a"),
                ),
                "- Pipeline throughput: {} fps".format(fmt(best.get("pipeline_throughput_fps"))),
                "- Static throughput estimate: {} fps".format(fmt(best.get("static_pipeline_fps_est"))),
                "- Bottleneck: {} ({})".format(
                    best.get("stage_balance_bottleneck", best.get("static_bottleneck_stage", "n/a")),
                    best.get("stage_balance_bottleneck_device", "n/a"),
                ),
                "- CPU/VTA GOPS est: cpu={} vta={}".format(
                    fmt(best.get("cpu_total_gops_est")),
                    fmt(best.get("vta_stage_gops_est")),
                ),
                "- CPU/VTA achieved GOPS: cpu={} vta={}".format(
                    fmt(best.get("cpu_shared_achieved_gops")),
                    fmt(best.get("vta_stage_achieved_gops")),
                ),
                "- PS-PL bandwidth: {} Gbit/s, DMA bytes={}".format(
                    fmt(best.get("ps_pl_total_bw_gbps")),
                    fmt(best.get("ps_pl_total_dma_bytes", best.get("static_dma_bytes_est"))),
                ),
                "- SRAM peak utilization: {}%".format(fmt(best.get("onchip_peak_util_pct"))),
                "- DMA fragmentation: measured={} static={}".format(
                    fmt(best.get("dma_fragmentation_score")),
                    fmt(best.get("static_dma_fragmentation_score_est")),
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Ranking",
            "",
            "| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |",
            "|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(ranked[:20], start=1):
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                idx,
                row.get("scheme_name", ""),
                row.get("status", ""),
                row.get("passes_correctness_gate", row.get("top1_match", "")),
                fmt(row.get("pipeline_throughput_fps")),
                row.get("stage_count", ""),
                row.get("thread_config", row.get("stage_runtime_threads", "")),
                row.get("stage_balance_bottleneck", row.get("static_bottleneck_stage", "")),
                fmt(row.get("ps_pl_total_bw_gbps")),
                fmt(row.get("onchip_peak_util_pct")),
                fmt(row.get("dma_fragmentation_score", row.get("static_dma_fragmentation_score_est"))),
                fmt(row.get("boundary_bytes")),
            )
        )
    lines.extend(["", "Full metrics are in `summary.csv` and `summary.json`."])
    return "\n".join(lines) + "\n"


def write_summary(search_dir, args, rows, stats):
    ranked = sort_rows_for_best(rows)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_candidate_count": int(stats.get("raw_candidate_count", 0)),
        "native_candidate_count": int(stats.get("native_candidate_count", 0)),
        "all_candidates_count": int(stats.get("native_candidate_count", 0)),
        "static_shortlist_count": int(stats.get("static_shortlist_count", 0)),
        "buildability_tested_count": int(stats.get("buildability_tested_count", 0)),
        "buildable_count": int(stats.get("buildable_count", 0)),
        "board_test_config_count": int(stats.get("board_test_config_count", 0)),
        "file_cache_policy": args.file_cache_policy,
        "file_cache_warmup_count": int(stats.get("file_cache_warmup_count", 0)),
        "excluded_candidate_count": int(stats.get("excluded_candidate_count", 0)),
        "measured_candidate_id_output": args.measured_candidate_id_output,
        "measured_candidate_count": len(measured_candidate_ids_from_rows(rows)),
        "searched_count": len(rows),
        "best": ranked[0] if ranked else None,
        "rows": ranked,
    }
    write_json(Path(search_dir) / "summary.json", payload)
    write_csv(Path(search_dir) / "summary.csv", ranked)
    write_measured_candidate_ids(args.measured_candidate_id_output, ranked)
    (Path(search_dir) / "README.md").write_text(
        render_readme(args, ranked, stats), encoding="utf-8"
    )

    if args.archive_root:
        archive_root = Path(args.archive_root)
        if not archive_root.is_absolute():
            archive_root = repo_root() / archive_root
        archive_dir = archive_root / Path(search_dir).name
        archive_dir.mkdir(parents=True, exist_ok=True)
        write_json(archive_dir / "summary.json", payload)
        write_csv(archive_dir / "summary.csv", ranked)
        (archive_dir / "README.md").write_text(
            render_readme(args, ranked, stats),
            encoding="utf-8",
        )


def validate_args(args):
    if args.runs <= 0:
        raise RuntimeError("--runs must be positive")
    if args.queue_depth <= 0:
        raise RuntimeError("--queue-depth must be positive")
    if args.skip_first < 0:
        raise RuntimeError("--skip-first must be non-negative")
    if args.max_images < 0:
        raise RuntimeError("--max-images must be non-negative")
    if args.image and args.image_dir:
        raise RuntimeError("--image and --image-dir are mutually exclusive")
    if args.max_vta_islands < 1 or args.max_vta_islands > 3:
        raise RuntimeError("--max-vta-islands must be in [1, 3]")
    if args.measure_top_n <= 0:
        raise RuntimeError("--measure-top-n must be positive")
    if args.refine_top_n < 0:
        raise RuntimeError("--refine-top-n must be non-negative")
    if args.static_shortlist_n <= 0:
        raise RuntimeError("--static-shortlist-n must be positive")
    if args.buildability_top_n < 0:
        raise RuntimeError("--buildability-top-n must be non-negative")
    if args.max_board_test_configs <= 0:
        raise RuntimeError("--max-board-test-configs must be positive")
    if args.progress_batch_index <= 0:
        raise RuntimeError("--progress-batch-index must be positive")
    if args.progress_batch_count <= 0:
        raise RuntimeError("--progress-batch-count must be positive")
    if args.progress_total_offset < 0:
        raise RuntimeError("--progress-total-offset must be non-negative")
    if args.progress_total_count < 0:
        raise RuntimeError("--progress-total-count must be non-negative")
    for name in [
        "ssh_command_timeout_s",
        "scp_timeout_s",
        "serial_timeout_s",
        "pipeline_timeout_s",
        "fetch_timeout_s",
    ]:
        if int(getattr(args, name)) <= 0:
            raise RuntimeError("--{} must be positive".format(name.replace("_", "-")))
    if args.calibration_budget <= 0:
        raise RuntimeError("--calibration-budget must be positive")
    if not args.board and not (
        args.static_only or args.dry_run or args.buildability_only or args.calibrate_cost_model
    ):
        raise RuntimeError(
            "--board is required unless --static-only, --dry-run, --buildability-only, or --calibrate-cost-model"
        )
    if (
        not (args.static_only or args.dry_run or args.buildability_only)
        and not (args.calibrate_cost_model and not args.board)
        and not args.rpc_baseline_result
        and not args.allow_missing_rpc_baseline
    ):
        raise RuntimeError(
            "--rpc-baseline-result is required for measured ranking unless "
            "--allow-missing-rpc-baseline is set"
        )


def main():
    args = parse_args()
    validate_args(args)
    poll_values = args.poll_value or DEFAULT_POLL_VALUES
    label = args.search_label or "{}_resnet18_nstage_search".format(time.strftime("%Y%m%d_%H%M%S"))
    search_dir = Path(args.output_root) / safe_name(label)
    search_dir.mkdir(parents=True, exist_ok=True)

    if args.calibrate_cost_model:
        cost_model = run_calibration(args, search_dir)
    else:
        cost_model = load_cost_model(args)
    write_json(Path(search_dir) / "cost_model.json", cost_model)
    if args.calibrate_cost_model and not args.board and not (
        args.static_only or args.dry_run or args.buildability_only
    ):
        print("[CALIBRATION] wrote", Path(search_dir) / "cost_model.json")
        return

    raw_candidate_count = len(enumerate_island_sets(args.max_vta_islands))
    _, all_candidates = load_candidates(args, cost_model)
    all_candidates = sort_candidates_static(all_candidates)
    shortlist = select_static_shortlist(args, all_candidates, search_dir)
    print("[SEARCH] raw island candidates =", raw_candidate_count)
    print("[SEARCH] native candidates     =", len(all_candidates))
    print("[SEARCH] static shortlist      =", len(shortlist))
    print("[SEARCH] output_dir =", search_dir)

    rows = []
    buildable_count = 0
    board_test_config_count = 0
    stats = {
        "raw_candidate_count": raw_candidate_count,
        "native_candidate_count": len(all_candidates),
        "static_shortlist_count": len(shortlist),
        "buildability_tested_count": 0,
        "buildable_count": 0,
        "board_test_config_count": 0,
        "file_cache_warmup_count": 0,
        "excluded_candidate_count": len(excluded_candidate_ids(args)),
        "progress_started_at": time.time(),
    }
    if args.static_only:
        rows = [static_candidate_row(candidate, args) for candidate in all_candidates]
        write_summary(search_dir, args, rows, stats)
        if rows:
            best = sort_rows_for_best(rows)[0]
            print(
                "[BEST-STATIC] scheme={} score_ms={} stages={} devices={}".format(
                    best["scheme_name"],
                    fmt(best.get("static_score_ms")),
                    best["stage_count"],
                    best["stage_devices"],
                )
            )
        return

    if args.skip_buildability or args.dry_run:
        for candidate in shortlist:
            candidate["buildable"] = not candidate["hard_reject"]
        buildable_count = sum(1 for candidate in shortlist if candidate.get("buildable", False))
        stats["buildable_count"] = buildable_count
    else:
        buildability_candidates = select_buildability_candidates(args, shortlist)
        stats["buildability_tested_count"] = len(buildability_candidates)
        for idx, candidate in enumerate(buildability_candidates, start=1):
            if candidate["hard_reject"] and not args.allow_hard_reject:
                candidate["buildable"] = False
                continue
            print(
                "[BUILDABILITY] {}/{} {}".format(
                    idx, len(buildability_candidates), candidate["scheme_name"]
                )
            )
            run_buildability_candidate(args, search_dir, candidate)
            if candidate.get("buildable", False):
                buildable_count += 1
                stats["buildable_count"] = buildable_count
            row = static_candidate_row(candidate, args)
            row.update(
                {
                    "status": "buildable" if candidate.get("buildable", False) else "unbuildable",
                    "buildable": bool(candidate.get("buildable", False)),
                    "build_failed_stage": candidate.get("build_failed_stage", ""),
                    "build_failure_type": candidate.get("build_failure_type", ""),
                    "build_error_summary": candidate.get("build_error_summary", ""),
                    "buildability_cache_hit": bool(candidate.get("buildability_cache_hit", False)),
                    "failed_stage": candidate.get("build_failed_stage", ""),
                    "failure_type": candidate.get("build_failure_type", ""),
                    "error_summary": candidate.get("build_error_summary", ""),
                }
            )
            rows.append(row)
            write_summary(search_dir, args, rows, stats)

    if args.buildability_only:
        write_theory_build_outputs(
            search_dir,
            args,
            select_buildability_candidates(args, shortlist),
            rows,
        )
        print("[SEARCH] buildable candidates =", buildable_count)
        return

    measure_candidates = select_measurement_candidates(args, shortlist)
    max_main = min(int(args.measure_top_n), int(args.max_board_test_configs))
    measure_candidates = measure_candidates[:max_main]
    print("[SEARCH] measurement candidates =", len(measure_candidates))
    write_progress(search_dir, args, stats, rows, len(measure_candidates), status="ready")
    for candidate in measure_candidates:
        if board_test_config_count >= int(args.max_board_test_configs):
            break
        base_threads = threads_for_candidate(args, candidate)
        thread_variants = [("default", base_threads)]
        for run_kind, threads in thread_variants:
            for poll_label, poll_ns in poll_values:
                run_plan = [(run_kind, True, "unmanaged")]
                if args.file_cache_policy == "prewarm":
                    run_plan = [(run_kind, True, "prewarmed")]
                elif args.file_cache_policy == "cold_then_warm":
                    run_plan = [
                        ("{}_cold_probe".format(run_kind), False, "cold_probe"),
                        (run_kind, True, "warm_verify"),
                    ]
                for actual_run_kind, rank_eligible, file_cache_state in run_plan:
                    if board_test_config_count >= int(args.max_board_test_configs):
                        break
                    if args.file_cache_policy == "prewarm" and not args.dry_run:
                        try:
                            warm_remote_file_cache(
                                args,
                                search_dir,
                                candidate,
                                threads,
                                poll_label,
                                poll_ns,
                                run_kind,
                            )
                            stats["file_cache_warmup_count"] += 1
                        except subprocess.CalledProcessError as err:
                            print(
                                "[FILE-CACHE] warmup failed, continuing to measured run: returncode={} {}".format(
                                    err.returncode, (getattr(err, "stderr", "") or "")[:300]
                                )
                            )
                        except Exception as err:  # pylint: disable=broad-except
                            print("[FILE-CACHE] warmup failed, continuing to measured run:", repr(err))
                    print(
                        "\n[SEARCH] candidate={} threads={} poll={}({} ns) run_kind={} rank_eligible={} file_cache={}".format(
                            candidate["scheme_name"],
                            ",".join(str(item) for item in threads),
                            poll_label,
                            poll_ns,
                            actual_run_kind,
                            rank_eligible,
                            file_cache_state,
                        )
                    )
                    board_test_config_count += 1
                    stats["board_test_config_count"] = board_test_config_count
                    write_progress(
                        search_dir,
                        args,
                        stats,
                        rows,
                        len(measure_candidates),
                        current_candidate=candidate["scheme_name"],
                        status="running",
                    )
                    try:
                        row = run_or_reuse_candidate(
                            args,
                            search_dir,
                            candidate,
                            threads,
                            poll_label,
                            poll_ns,
                            actual_run_kind,
                            rank_eligible=rank_eligible,
                            file_cache_state=file_cache_state,
                        )
                    except subprocess.CalledProcessError as err:
                        failure_type = classify_command_failure(err, "pipeline_run_failed")
                        failed_label = candidate_run_label(
                            candidate, threads, poll_label, actual_run_kind
                        )
                        row = failure_row(
                            candidate,
                            args,
                            failure_type,
                            "returncode={} {}".format(
                                err.returncode, (getattr(err, "stderr", "") or "")[:400]
                            ),
                            output_dir=Path(search_dir) / failed_label,
                            run_command_path=Path(search_dir) / failed_label / "run_command.sh",
                        )
                        row.update(
                            {
                                "run_kind": actual_run_kind,
                                "rank_eligible": bool(rank_eligible),
                                "file_cache_policy": args.file_cache_policy,
                                "file_cache_state": file_cache_state,
                                "thread_config": ",".join(str(item) for item in threads),
                                "poll_label": poll_label,
                                "poll_sleep_ns": int(poll_ns),
                            }
                        )
                        print("[SEARCH] failed:", row["error_summary"])
                    except Exception as err:  # pylint: disable=broad-except
                        failed_label = candidate_run_label(
                            candidate, threads, poll_label, actual_run_kind
                        )
                        row = failure_row(
                            candidate,
                            args,
                            "pipeline_run_failed",
                            repr(err),
                            output_dir=Path(search_dir) / failed_label,
                            run_command_path=Path(search_dir) / failed_label / "run_command.sh",
                        )
                        row.update(
                            {
                                "run_kind": actual_run_kind,
                                "rank_eligible": bool(rank_eligible),
                                "file_cache_policy": args.file_cache_policy,
                                "file_cache_state": file_cache_state,
                                "thread_config": ",".join(str(item) for item in threads),
                                "poll_label": poll_label,
                                "poll_sleep_ns": int(poll_ns),
                            }
                        )
                        print("[SEARCH] failed:", repr(err))
                    rows.append(row)
                    write_summary(search_dir, args, rows, stats)
                    write_progress(
                        search_dir,
                        args,
                        stats,
                        rows,
                        len(measure_candidates),
                        current_candidate=candidate["scheme_name"],
                        status=row.get("status", "recorded"),
                    )

    measured_ranked = [
        row
        for row in sort_rows_for_best(rows)
        if row.get("status") == "ok"
        and row.get("run_kind") == "default"
        and row.get("rank_eligible", True) is not False
    ]
    refine_names = [row["scheme_name"] for row in measured_ranked[: int(args.refine_top_n)]]
    refine_candidates = [candidate for candidate in measure_candidates if candidate["scheme_name"] in refine_names]
    for candidate in refine_candidates:
        if board_test_config_count >= int(args.max_board_test_configs):
            break
        threads = refine_cpu4_threads(candidate)
        if threads == threads_for_candidate(args, candidate):
            continue
        for poll_label, poll_ns in poll_values:
            if board_test_config_count >= int(args.max_board_test_configs):
                break
            run_plan = [("cpu4_refine", True, "unmanaged")]
            if args.file_cache_policy == "prewarm":
                run_plan = [("cpu4_refine", True, "prewarmed")]
            elif args.file_cache_policy == "cold_then_warm":
                run_plan = [
                    ("cpu4_refine_cold_probe", False, "cold_probe"),
                    ("cpu4_refine", True, "warm_verify"),
                ]
            for actual_run_kind, rank_eligible, file_cache_state in run_plan:
                if board_test_config_count >= int(args.max_board_test_configs):
                    break
                if args.file_cache_policy == "prewarm" and not args.dry_run:
                    try:
                        warm_remote_file_cache(
                            args,
                            search_dir,
                            candidate,
                            threads,
                            poll_label,
                            poll_ns,
                            "cpu4_refine",
                        )
                        stats["file_cache_warmup_count"] += 1
                    except subprocess.CalledProcessError as err:
                        print(
                            "[FILE-CACHE] warmup failed, continuing to measured run: returncode={} {}".format(
                                err.returncode, (getattr(err, "stderr", "") or "")[:300]
                            )
                        )
                    except Exception as err:  # pylint: disable=broad-except
                        print("[FILE-CACHE] warmup failed, continuing to measured run:", repr(err))
                print(
                    "\n[REFINE] candidate={} threads={} poll={}({} ns) run_kind={} rank_eligible={} file_cache={}".format(
                        candidate["scheme_name"],
                        ",".join(str(item) for item in threads),
                        poll_label,
                        poll_ns,
                        actual_run_kind,
                        rank_eligible,
                        file_cache_state,
                    )
                )
                board_test_config_count += 1
                stats["board_test_config_count"] = board_test_config_count
                write_progress(
                    search_dir,
                    args,
                    stats,
                    rows,
                    len(measure_candidates),
                    current_candidate=candidate["scheme_name"],
                    status="running_refine",
                )
                try:
                    row = run_or_reuse_candidate(
                        args,
                        search_dir,
                        candidate,
                        threads,
                        poll_label,
                        poll_ns,
                        actual_run_kind,
                        rank_eligible=rank_eligible,
                        file_cache_state=file_cache_state,
                    )
                except subprocess.CalledProcessError as err:
                    failure_type = classify_command_failure(err, "pipeline_run_failed")
                    failed_label = candidate_run_label(candidate, threads, poll_label, actual_run_kind)
                    row = failure_row(
                        candidate,
                        args,
                        failure_type,
                        "returncode={} {}".format(
                            err.returncode, (getattr(err, "stderr", "") or "")[:400]
                        ),
                        output_dir=Path(search_dir) / failed_label,
                        run_command_path=Path(search_dir) / failed_label / "run_command.sh",
                    )
                    row.update(
                        {
                            "run_kind": actual_run_kind,
                            "rank_eligible": bool(rank_eligible),
                            "file_cache_policy": args.file_cache_policy,
                            "file_cache_state": file_cache_state,
                            "thread_config": ",".join(str(item) for item in threads),
                            "poll_label": poll_label,
                            "poll_sleep_ns": int(poll_ns),
                        }
                    )
                    print("[REFINE] failed:", row["error_summary"])
                except Exception as err:  # pylint: disable=broad-except
                    failed_label = candidate_run_label(candidate, threads, poll_label, actual_run_kind)
                    row = failure_row(
                        candidate,
                        args,
                        "pipeline_run_failed",
                        repr(err),
                        output_dir=Path(search_dir) / failed_label,
                        run_command_path=Path(search_dir) / failed_label / "run_command.sh",
                    )
                    row.update(
                        {
                            "run_kind": actual_run_kind,
                            "rank_eligible": bool(rank_eligible),
                            "file_cache_policy": args.file_cache_policy,
                            "file_cache_state": file_cache_state,
                            "thread_config": ",".join(str(item) for item in threads),
                            "poll_label": poll_label,
                            "poll_sleep_ns": int(poll_ns),
                        }
                    )
                    print("[REFINE] failed:", repr(err))
                rows.append(row)
                write_summary(search_dir, args, rows, stats)
                write_progress(
                    search_dir,
                    args,
                    stats,
                    rows,
                    len(measure_candidates),
                    current_candidate=candidate["scheme_name"],
                    status=row.get("status", "recorded"),
                )

    write_progress(search_dir, args, stats, rows, len(measure_candidates), status="complete")
    ranked = sort_rows_for_best(rows)
    best = ranked[0] if ranked else None
    if best is not None:
        print(
            "\n[BEST] scheme={} status={} gate={} fps={} bottleneck={} ps_pl_bw_gbps={} onchip_peak_pct={} dma_frag={}".format(
                best.get("scheme_name"),
                best.get("status"),
                best.get("passes_correctness_gate"),
                fmt(best.get("pipeline_throughput_fps")),
                best.get("stage_balance_bottleneck", best.get("static_bottleneck_stage")),
                fmt(best.get("ps_pl_total_bw_gbps")),
                fmt(best.get("onchip_peak_util_pct")),
                fmt(best.get("dma_fragmentation_score", best.get("static_dma_fragmentation_score_est"))),
            )
        )
    print("[SEARCH] summary =", search_dir / "README.md")


if __name__ == "__main__":
    main()
