#!/usr/bin/env python3
"""Search ResNet18 CPU/VTA/CPU stage cuts for native pipeline throughput.

The search has two layers:
- static resource accounting for all candidate VTA windows
- optional board execution through deploy_classification_stage_pipeline_native.py

Board results are ranked primarily by measured pipeline throughput. The report
also keeps the resource terms that explain the choice: CPU/VTA GOPS, PS-PL
bandwidth, stage balance, on-chip memory utilization, and DMA fragmentation.
"""

from __future__ import absolute_import, print_function

import argparse
import csv
import json
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
    AUTO_RESOURCE_AWARE_SCHEME,
    build_resource_aware_candidates,
)


DEFAULT_KNOWN_SCHEMES = "three_stage_a,three_stage_b,three_stage_e,three_stage_f,block_stage_c"
DEFAULT_THREAD_CONFIGS = ["3,1,1"]
DEFAULT_POLL_VALUES = [("poll_1us", 1000)]


def repo_root():
    return Path(__file__).resolve().parents[3]


def parse_thread_config(text):
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("thread config must be STAGE0,STAGE1,STAGE2")
    try:
        values = tuple(int(item) for item in parts)
    except ValueError as err:
        raise argparse.ArgumentTypeError("thread config values must be integers") from err
    if min(values) <= 0:
        raise argparse.ArgumentTypeError("thread config values must be positive")
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
        description="Search ResNet18 CPU/VTA/CPU native stage-pipeline cuts."
    )
    parser.add_argument("--board", default="", help="SSH target, for example root@192.168.1.247")
    parser.add_argument("--remote-dir", default="/mnt/sd/vta_stage_pipeline_search")
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
    parser.add_argument(
        "--thread-config",
        action="append",
        type=parse_thread_config,
        default=[],
        help="Stage runtime threads as STAGE0,STAGE1,STAGE2. Repeat to sweep.",
    )
    parser.add_argument(
        "--poll-value",
        action="append",
        type=parse_poll_value,
        default=[],
        help="Driver poll/post-start sleep as LABEL:NS. Repeat to sweep.",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=12,
        help="Add this many static-ranked auto candidates after --include-known; 0 means all.",
    )
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--candidate-name", action="append", default=[])
    parser.add_argument("--include-known", default=DEFAULT_KNOWN_SCHEMES)
    parser.add_argument("--allow-hard-reject", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output-root", default="/tmp/vta_stage_split_searches")
    parser.add_argument("--search-label", default="")
    parser.add_argument(
        "--archive-root",
        default="",
        help="Optional aggregate-report archive root; relative paths are under the repo root.",
    )
    parser.add_argument("--vta-runtime-profile-events-limit", type=int, default=200)
    parser.add_argument("--rpc-baseline-result", default="")
    parser.add_argument("--no-compare-serial-pipeline", action="store_true")
    parser.add_argument(
        "--cpu-peak-gops",
        type=float,
        default=0.0,
        help="Optional CPU peak/expected GOPS for static-only time estimates.",
    )
    parser.add_argument(
        "--vta-peak-gops",
        type=float,
        default=0.0,
        help="Optional VTA peak/expected GOPS for static-only time estimates.",
    )
    parser.add_argument(
        "--ps-pl-bandwidth-gbps",
        type=float,
        default=0.0,
        help="Optional PS-PL bandwidth in Gbit/s for static-only transfer estimates.",
    )
    return parser.parse_args()


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


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


def mean_value(rows, key, skip_first=0):
    values = [float(row[key]) for row in rows[int(skip_first) :] if key in row]
    return statistics.mean(values) if values else 0.0


def std_value(rows, key, skip_first=0):
    values = [float(row[key]) for row in rows[int(skip_first) :] if key in row]
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def throughput_from_span(rows, skip_first):
    keep = rows[int(skip_first) :]
    if len(keep) < 2:
        return 0.0
    first_start = float(keep[0]["stage0_start_ms"])
    last_end = max(float(row["stage2_end_ms"]) for row in keep)
    span_ms = last_end - first_start
    return float(len(keep)) * 1000.0 / span_ms if span_ms > 0.0 else 0.0


def stage0_interval_ms(rows, skip_first):
    starts = [float(row["stage0_start_ms"]) for row in rows if "stage0_start_ms" in row]
    intervals = [starts[idx] - starts[idx - 1] for idx in range(1, len(starts))]
    if skip_first:
        intervals = intervals[max(0, int(skip_first) - 1) :]
    return statistics.mean(intervals) if intervals else 0.0


def top1_match(serial_rows, pipeline_rows):
    if not serial_rows or not pipeline_rows:
        return False
    if len(serial_rows) != len(pipeline_rows):
        return False
    for serial, pipeline in zip(serial_rows, pipeline_rows):
        for key in ["input_index", "input_file", "top1"]:
            if serial.get(key) != pipeline.get(key):
                return False
    return True


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


def candidate_devices(candidate):
    return [stage["device"] for stage in candidate["scheme_cfg"]]


def is_native_three_stage_candidate(candidate):
    return len(candidate["scheme_cfg"]) == 3 and candidate_devices(candidate) == ["cpu", "vta", "cpu"]


def load_candidates(args):
    env = vta.get_env()
    full_model = vision.get_model(args.model, pretrained=True)
    feature_blocks = list(full_model.features)
    output_block = full_model.output
    candidates = build_resource_aware_candidates(
        feature_blocks,
        output_block,
        batch=env.BATCH,
        image_size=args.image_size,
    )
    by_name = {candidate["scheme_name"]: candidate for candidate in candidates}
    selected = []
    seen = set()

    def add(candidate):
        if candidate is None:
            return
        name = candidate["scheme_name"]
        if name in seen:
            return
        if not is_native_three_stage_candidate(candidate):
            return
        if candidate["hard_reject"] and not args.allow_hard_reject:
            return
        seen.add(name)
        selected.append(candidate)

    for name in [item.strip() for item in args.include_known.split(",") if item.strip()]:
        add(by_name.get(name))
    for name in args.candidate_name:
        if name not in by_name:
            raise RuntimeError("Unknown resource-aware candidate: {}".format(name))
        add(by_name[name])
    top_limit = None if int(args.candidate_top_k) <= 0 else len(selected) + int(args.candidate_top_k)
    for candidate in candidates:
        add(candidate)
        if top_limit is not None and len(selected) >= top_limit:
            break
    if args.max_candidates > 0:
        selected = selected[: args.max_candidates]
    return selected, candidates


def estimate_static_stage_ms(candidate, args):
    cpu_peak = float(args.cpu_peak_gops)
    vta_peak = float(args.vta_peak_gops)
    bw_gbps = float(args.ps_pl_bandwidth_gbps)

    def compute_ms(ops, peak_gops):
        if peak_gops <= 0.0:
            return 0.0
        return float(ops) / (peak_gops * 1e9) * 1000.0

    def transfer_ms(byte_count):
        if bw_gbps <= 0.0:
            return 0.0
        return float(byte_count) * 8.0 / (bw_gbps * 1e9) * 1000.0

    stage0_ms = compute_ms(candidate.get("cpu_stage0_compute_ops_est", 0), cpu_peak)
    stage2_ms = compute_ms(candidate.get("cpu_stage2_compute_ops_est", 0), cpu_peak)
    vta_compute_ms = compute_ms(candidate.get("vta_compute_ops_est", 0), vta_peak)
    vta_dma_bytes = (
        candidate.get("tile_input_load_bytes_est", 0)
        + candidate.get("tile_weight_load_bytes_est", 0)
        + candidate.get("tile_output_store_bytes_est", 0)
    )
    stage1_ms = vta_compute_ms + transfer_ms(vta_dma_bytes)
    cycle_ms = max(stage0_ms, stage1_ms, stage2_ms)
    return {
        "static_stage0_ms_est": stage0_ms,
        "static_stage1_ms_est": stage1_ms,
        "static_stage2_ms_est": stage2_ms,
        "static_vta_compute_ms_est": vta_compute_ms,
        "static_vta_dma_ms_est": transfer_ms(vta_dma_bytes),
        "static_pipeline_cycle_ms_est": cycle_ms,
        "static_pipeline_fps_est": 1000.0 / cycle_ms if cycle_ms > 0.0 else 0.0,
    }


def static_candidate_row(candidate, args):
    row = {
        "scheme_name": candidate["scheme_name"],
        "alias_scheme_name": candidate["alias_scheme_name"] or "",
        "status": "static",
        "hard_reject": bool(candidate["hard_reject"]),
        "reject_reasons": ",".join(candidate["reject_reasons"]),
        "vta_unit_names": " ".join(candidate["vta_unit_names"]),
        "start_idx": int(candidate["start_idx"]),
        "end_idx": int(candidate["end_idx"]),
        "span": int(candidate["span"]),
        "static_score": int(candidate["score"]),
        "boundary_bytes": int(candidate["total_boundary_bytes"]),
        "boundary_count": int(candidate["boundary_count"]),
        "cpu_stage0_gops_est": float(candidate["cpu_stage0_compute_ops_est"]) / 1e9,
        "vta_stage_gops_est": float(candidate["vta_compute_ops_est"]) / 1e9,
        "cpu_stage2_gops_est": float(candidate["cpu_stage2_compute_ops_est"]) / 1e9,
        "cpu_total_gops_est": float(candidate["cpu_total_compute_ops_est"]) / 1e9,
        "onchip_inp_util_pct": 100.0 * float(candidate["max_inp_ratio"]),
        "onchip_wgt_util_pct": 100.0 * float(candidate["max_wgt_ratio"]),
        "onchip_acc_util_pct": 100.0 * float(candidate["max_acc_ratio"]),
        "onchip_out_util_pct": 100.0 * float(candidate["max_out_ratio"]),
        "onchip_peak_util_pct": 100.0
        * max(
            float(candidate["max_inp_ratio"]),
            float(candidate["max_wgt_ratio"]),
            float(candidate["max_acc_ratio"]),
            float(candidate["max_out_ratio"]),
        ),
        "static_dma_bytes_est": int(
            candidate["tile_input_load_bytes_est"]
            + candidate["tile_weight_load_bytes_est"]
            + candidate["tile_output_store_bytes_est"]
        ),
        "static_tile_count_est": int(candidate["tile_count_est"]),
        "static_tile_spill_penalty": int(candidate["tile_spill_penalty"]),
        "static_internal_boundary_bytes": int(candidate["total_internal_boundary_bytes"]),
        "static_residual_live_bytes": int(candidate["total_residual_live_bytes"]),
    }
    row.update(estimate_static_stage_ms(candidate, args))
    return row


def achieved_gops(ops, ms):
    ms = float(ms)
    if ms <= 0.0:
        return 0.0
    return (float(ops) / 1e9) / (ms / 1000.0)


def summarize_run(candidate, output_dir, args, thread_config, poll_label, poll_ns):
    static_row = static_candidate_row(candidate, args)
    serial_rows = read_jsonl(Path(output_dir) / "stage_serial_result.jsonl")
    pipeline_rows = read_jsonl(Path(output_dir) / "native_result.jsonl")
    manifest = read_json(Path(output_dir) / "manifest.json", default={})
    pipeline_status = profile_status(output_dir, "pipeline")
    serial_status = profile_status(output_dir, "serial")
    profile_metrics = profile_bandwidth_metrics(pipeline_status)
    dma_metrics = profile_dma_fragmentation_metrics(pipeline_status)

    stage0_ms = mean_value(pipeline_rows, "stage0_ms", args.skip_first)
    stage1_ms = mean_value(pipeline_rows, "stage1_ms", args.skip_first)
    stage2_ms = mean_value(pipeline_rows, "stage2_ms", args.skip_first)
    stage_values = [value for value in [stage0_ms, stage1_ms, stage2_ms] if value > 0.0]
    max_stage_ms = max(stage_values) if stage_values else 0.0
    min_stage_ms = min(stage_values) if stage_values else 0.0
    bottleneck = "-"
    if stage_values:
        stage_map = {"stage0_cpu": stage0_ms, "stage1_vta": stage1_ms, "stage2_cpu": stage2_ms}
        bottleneck = max(stage_map, key=stage_map.get)

    stage0_run_ms = mean_value(pipeline_rows, "stage0_run_ms", args.skip_first)
    stage1_run_ms = mean_value(pipeline_rows, "stage1_run_ms", args.skip_first)
    stage2_run_ms = mean_value(pipeline_rows, "stage2_run_ms", args.skip_first)
    cpu_shared_run_ms = stage0_run_ms + stage2_run_ms

    row = dict(static_row)
    row.update(
        {
            "status": "ok" if pipeline_rows else "missing_results",
            "output_dir": str(output_dir),
            "manifest_scheme": manifest.get("scheme", ""),
            "thread_config": "{},{},{}".format(*thread_config),
            "poll_label": poll_label,
            "poll_sleep_ns": int(poll_ns),
            "runs": len(pipeline_rows),
            "skip_first": int(args.skip_first),
            "top1_match": top1_match(serial_rows, pipeline_rows),
            "serial_throughput_fps": throughput_from_span(serial_rows, args.skip_first),
            "pipeline_throughput_fps": throughput_from_span(pipeline_rows, args.skip_first),
            "pipeline_stage0_interval_ms": stage0_interval_ms(pipeline_rows, args.skip_first),
            "pipeline_total_latency_avg_ms": mean_value(
                pipeline_rows, "total_latency_ms", args.skip_first
            ),
            "pipeline_total_latency_std_ms": std_value(
                pipeline_rows, "total_latency_ms", args.skip_first
            ),
            "stage0_ms": stage0_ms,
            "stage1_ms": stage1_ms,
            "stage2_ms": stage2_ms,
            "stage0_run_ms": stage0_run_ms,
            "stage1_run_ms": stage1_run_ms,
            "stage2_run_ms": stage2_run_ms,
            "cpu_shared_run_ms": cpu_shared_run_ms,
            "stage_balance_imbalance_ratio": (
                (max_stage_ms - min_stage_ms) / max_stage_ms if max_stage_ms > 0.0 else 0.0
            ),
            "stage_balance_bottleneck": bottleneck,
            "steady_state_stage_fps": 1000.0 / max_stage_ms if max_stage_ms > 0.0 else 0.0,
            "cpu_stage0_achieved_gops": achieved_gops(
                candidate.get("cpu_stage0_compute_ops_est", 0), stage0_run_ms
            ),
            "vta_stage_achieved_gops": achieved_gops(
                candidate.get("vta_compute_ops_est", 0), stage1_run_ms
            ),
            "cpu_stage2_achieved_gops": achieved_gops(
                candidate.get("cpu_stage2_compute_ops_est", 0), stage2_run_ms
            ),
            "cpu_shared_achieved_gops": achieved_gops(
                candidate.get("cpu_stage0_compute_ops_est", 0)
                + candidate.get("cpu_stage2_compute_ops_est", 0),
                cpu_shared_run_ms,
            ),
            "serial_vta_device_wait_ms": float(serial_status.get("device_run_wait_us", 0.0))
            / 1000.0,
        }
    )
    row.update(profile_metrics)
    row.update(dma_metrics)
    return row


def build_deploy_command(args, candidate, output_dir, thread_config, remote_dir):
    script = repo_root() / "vta" / "tutorials" / "frontend" / "deploy_classification_stage_pipeline_native.py"
    scheme_name = candidate["alias_scheme_name"] or candidate["scheme_name"]
    cmd = [
        sys.executable,
        str(script),
        "--board",
        args.board,
        "--remote-dir",
        remote_dir,
        "--scheme",
        scheme_name if candidate["alias_scheme_name"] else AUTO_RESOURCE_AWARE_SCHEME,
        "--runs",
        str(int(args.runs)),
        "--queue-depth",
        str(int(args.queue_depth)),
        "--runtime-num-threads",
        str(int(args.runtime_num_threads)),
        "--stage0-runtime-num-threads",
        str(thread_config[0]),
        "--stage1-runtime-num-threads",
        str(thread_config[1]),
        "--stage2-runtime-num-threads",
        str(thread_config[2]),
        "--fetch-results-dir",
        str(output_dir),
        "--vta-runtime-profile-dir",
        "profile",
        "--vta-runtime-profile-events-limit",
        str(int(args.vta_runtime_profile_events_limit)),
        "--run-serial-before-pipeline",
    ]
    if not args.no_compare_serial_pipeline:
        cmd.append("--compare-serial-pipeline")
    if not candidate["alias_scheme_name"]:
        cmd.extend(["--resource-aware-candidate-name", candidate["scheme_name"]])
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
    subprocess.check_call([str(part) for part in cmd], cwd=str(repo_root()), env=env)


def candidate_run_label(candidate, thread_config, poll_label):
    return "{}_threads_{}_poll_{}".format(
        safe_name(candidate["scheme_name"]),
        "{}_{}_{}".format(*thread_config),
        safe_name(poll_label),
    )


def run_or_reuse_candidate(args, search_dir, candidate, thread_config, poll_label, poll_ns):
    label = candidate_run_label(candidate, thread_config, poll_label)
    output_dir = Path(search_dir) / label
    output_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = args.remote_dir.rstrip("/") + "/" + label
    cmd = build_deploy_command(args, candidate, output_dir, thread_config, remote_dir)
    run_script = output_dir / "run_command.sh"
    run_script.write_text(command_text(cmd, poll_ns), encoding="utf-8")
    run_script.chmod(0o755)

    has_results = (output_dir / "native_result.jsonl").exists() and (
        output_dir / "stage_serial_result.jsonl"
    ).exists()
    if args.dry_run:
        print("[DRY-RUN]", run_script)
        row = static_candidate_row(candidate, args)
        row.update(
            {
                "status": "dry_run",
                "output_dir": str(output_dir),
                "thread_config": "{},{},{}".format(*thread_config),
                "poll_label": poll_label,
                "poll_sleep_ns": int(poll_ns),
            }
        )
        return row

    if not (args.skip_existing and has_results):
        run_command(cmd, poll_ns)
    return summarize_run(candidate, output_dir, args, thread_config, poll_label, poll_ns)


def sort_rows_for_best(rows):
    def key(row):
        ok = row.get("status") == "ok" and bool(row.get("top1_match", False))
        measured_fps = float(row.get("pipeline_throughput_fps", 0.0) or 0.0)
        static_fps = float(row.get("static_pipeline_fps_est", 0.0) or 0.0)
        return (
            0 if ok else 1,
            -measured_fps,
            -static_fps,
            -float(row.get("steady_state_stage_fps", 0.0)),
            float(row.get("dma_fragmentation_score", 0.0)),
            float(row.get("stage_balance_imbalance_ratio", 1.0)),
            int(row.get("static_score", 0)),
        )

    return sorted(rows, key=key)


def csv_fieldnames(rows):
    preferred = [
        "scheme_name",
        "alias_scheme_name",
        "status",
        "top1_match",
        "pipeline_throughput_fps",
        "steady_state_stage_fps",
        "serial_throughput_fps",
        "stage0_ms",
        "stage1_ms",
        "stage2_ms",
        "stage_balance_imbalance_ratio",
        "stage_balance_bottleneck",
        "cpu_stage0_gops_est",
        "vta_stage_gops_est",
        "cpu_stage2_gops_est",
        "cpu_stage0_achieved_gops",
        "vta_stage_achieved_gops",
        "cpu_stage2_achieved_gops",
        "cpu_shared_achieved_gops",
        "ps_pl_total_bw_gbps",
        "ps_pl_total_dma_bytes",
        "vta_device_wait_ms",
        "onchip_peak_util_pct",
        "onchip_inp_util_pct",
        "onchip_wgt_util_pct",
        "onchip_acc_util_pct",
        "onchip_out_util_pct",
        "dma_avg_bytes_per_call",
        "dma_small_call_ratio",
        "dma_strided_call_ratio",
        "dma_padded_call_ratio",
        "dma_thin_shape_call_ratio",
        "dma_fragmentation_score",
        "boundary_bytes",
        "thread_config",
        "poll_sleep_ns",
        "static_score",
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


def render_readme(args, rows):
    ranked = sort_rows_for_best(rows)
    best = ranked[0] if ranked else None
    lines = [
        "# ResNet18 Stage Split Search",
        "",
        "- Generated: {}".format(datetime.now().isoformat(timespec="seconds")),
        "- Runs per candidate: {}".format(args.runs),
        "- Skip first frames: {}".format(args.skip_first),
        "- Primary objective: measured pipeline throughput fps",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "## Best",
                "",
                "- Scheme: {}".format(best.get("scheme_name", "n/a")),
                "- Alias: {}".format(best.get("alias_scheme_name") or "n/a"),
                "- Pipeline throughput: {} fps".format(fmt(best.get("pipeline_throughput_fps"))),
                "- Static throughput estimate: {} fps".format(
                    fmt(best.get("static_pipeline_fps_est"))
                ),
                "- Stage ms: stage0={} stage1={} stage2={}".format(
                    fmt(best.get("stage0_ms")),
                    fmt(best.get("stage1_ms")),
                    fmt(best.get("stage2_ms")),
                ),
                "- CPU/VTA achieved GOPS: cpu_shared={} vta={}".format(
                    fmt(best.get("cpu_shared_achieved_gops")),
                    fmt(best.get("vta_stage_achieved_gops")),
                ),
                "- PS-PL bandwidth: {} Gbit/s, DMA bytes={}".format(
                    fmt(best.get("ps_pl_total_bw_gbps")),
                    fmt(best.get("ps_pl_total_dma_bytes")),
                ),
                "- On-chip peak utilization: {}%".format(fmt(best.get("onchip_peak_util_pct"))),
                "- DMA fragmentation score: {}".format(
                    fmt(best.get("dma_fragmentation_score"))
                ),
                "- VTA units: {}".format(best.get("vta_unit_names", "")),
                "",
            ]
        )
    lines.extend(
        [
            "## Ranking",
            "",
            "| rank | scheme | status | top1 | fps | static fps | stage0 | stage1 | stage2 | ps-pl Gbit/s | SRAM peak % | DMA frag |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(ranked[:20], start=1):
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                idx,
                row.get("scheme_name", ""),
                row.get("status", ""),
                row.get("top1_match", ""),
                fmt(row.get("pipeline_throughput_fps")),
                fmt(row.get("static_pipeline_fps_est")),
                fmt(row.get("stage0_ms")),
                fmt(row.get("stage1_ms")),
                fmt(row.get("stage2_ms")),
                fmt(row.get("ps_pl_total_bw_gbps")),
                fmt(row.get("onchip_peak_util_pct")),
                fmt(row.get("dma_fragmentation_score")),
            )
        )
    lines.extend(["", "Full metrics are in `summary.csv` and `summary.json`."])
    return "\n".join(lines) + "\n"


def write_summary(search_dir, args, rows, all_candidates_count):
    ranked = sort_rows_for_best(rows)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "all_candidates_count": int(all_candidates_count),
        "searched_count": len(rows),
        "best": ranked[0] if ranked else None,
        "rows": ranked,
    }
    write_json(Path(search_dir) / "summary.json", payload)
    write_csv(Path(search_dir) / "summary.csv", ranked)
    (Path(search_dir) / "README.md").write_text(render_readme(args, ranked), encoding="utf-8")

    if args.archive_root:
        archive_root = Path(args.archive_root)
        if not archive_root.is_absolute():
            archive_root = repo_root() / archive_root
        archive_dir = archive_root / Path(search_dir).name
        archive_dir.mkdir(parents=True, exist_ok=True)
        write_json(archive_dir / "summary.json", payload)
        write_csv(archive_dir / "summary.csv", ranked)
        (archive_dir / "README.md").write_text(render_readme(args, ranked), encoding="utf-8")


def main():
    args = parse_args()
    if args.runs <= 0:
        raise RuntimeError("--runs must be positive")
    if args.skip_first < 0:
        raise RuntimeError("--skip-first must be non-negative")
    if args.max_images < 0:
        raise RuntimeError("--max-images must be non-negative")
    if args.image and args.image_dir:
        raise RuntimeError("--image and --image-dir are mutually exclusive")
    if not args.board and not (args.static_only or args.dry_run):
        raise RuntimeError("--board is required unless --static-only or --dry-run is set")

    thread_configs = args.thread_config or [parse_thread_config(item) for item in DEFAULT_THREAD_CONFIGS]
    poll_values = args.poll_value or DEFAULT_POLL_VALUES
    label = args.search_label or "{}_resnet18_stage_search".format(time.strftime("%Y%m%d_%H%M%S"))
    search_dir = Path(args.output_root) / safe_name(label)
    search_dir.mkdir(parents=True, exist_ok=True)

    selected_candidates, all_candidates = load_candidates(args)
    print("[SEARCH] selected candidates =", len(selected_candidates))
    print("[SEARCH] all static candidates =", len(all_candidates))
    print("[SEARCH] output_dir =", search_dir)

    rows = []
    if args.static_only:
        rows = [static_candidate_row(candidate, args) for candidate in selected_candidates]
        write_summary(search_dir, args, rows, len(all_candidates))
        if rows:
            best = sort_rows_for_best(rows)[0]
            print(
                "[BEST-STATIC] scheme={} score={} vta_units={}".format(
                    best["scheme_name"], best["static_score"], best["vta_unit_names"]
                )
            )
        return

    for candidate in selected_candidates:
        for thread_config in thread_configs:
            for poll_label, poll_ns in poll_values:
                print(
                    "\n[SEARCH] candidate={} threads={} poll={}({} ns)".format(
                        candidate["scheme_name"],
                        "{},{},{}".format(*thread_config),
                        poll_label,
                        poll_ns,
                    )
                )
                try:
                    row = run_or_reuse_candidate(
                        args,
                        search_dir,
                        candidate,
                        thread_config,
                        poll_label,
                        poll_ns,
                    )
                except subprocess.CalledProcessError as err:
                    row = static_candidate_row(candidate, args)
                    row.update(
                        {
                            "status": "command_failed",
                            "error": "returncode={}".format(err.returncode),
                            "thread_config": "{},{},{}".format(*thread_config),
                            "poll_label": poll_label,
                            "poll_sleep_ns": int(poll_ns),
                        }
                    )
                    print("[SEARCH] failed:", row["error"])
                except Exception as err:  # pylint: disable=broad-except
                    row = static_candidate_row(candidate, args)
                    row.update(
                        {
                            "status": "error",
                            "error": repr(err),
                            "thread_config": "{},{},{}".format(*thread_config),
                            "poll_label": poll_label,
                            "poll_sleep_ns": int(poll_ns),
                        }
                    )
                    print("[SEARCH] failed:", repr(err))
                rows.append(row)
                write_summary(search_dir, args, rows, len(all_candidates))

    ranked = sort_rows_for_best(rows)
    best = ranked[0] if ranked else None
    if best is not None:
        print(
            "\n[BEST] scheme={} status={} top1_match={} fps={} stage_ms={}/{}/{} ps_pl_bw_gbps={} onchip_peak_pct={} dma_frag={}".format(
                best.get("scheme_name"),
                best.get("status"),
                best.get("top1_match"),
                fmt(best.get("pipeline_throughput_fps")),
                fmt(best.get("stage0_ms")),
                fmt(best.get("stage1_ms")),
                fmt(best.get("stage2_ms")),
                fmt(best.get("ps_pl_total_bw_gbps")),
                fmt(best.get("onchip_peak_util_pct")),
                fmt(best.get("dma_fragmentation_score")),
            )
        )
        print("[BEST] vta_units={}".format(best.get("vta_unit_names", "")))
    print("[SEARCH] summary =", search_dir / "README.md")


if __name__ == "__main__":
    main()
