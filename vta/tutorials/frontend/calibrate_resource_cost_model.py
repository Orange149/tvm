#!/usr/bin/env python3
"""Measure the complete RAMPS CPU, VTA, DMA, and bridge cost model.

The calibration protocol is deliberately strict: every bucket must have real
board measurements from every independent session.  The script never fills a
missing bucket from a default or from another bucket.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence


CPU_BUCKET_CASES = {
    "stem_large_input_conv": ["stem_conv7x7_s2"],
    "residual_3x3_conv": [
        "s1_conv3x3_64_64",
        "s2_conv3x3_128_128",
        "s4_conv3x3_512_512",
    ],
    "skip_proj_1x1_conv": [
        "s2_proj1x1_64_128_s2",
        "s3_proj1x1_128_256_s2",
        "s4_proj1x1_256_512_s2",
    ],
    "add_relu_tail_elementwise": ["residual_add_relu_256_14"],
    "head_pool_dense": ["tail_global_avg_pool", "tail_dense_512_1000"],
}

VTA_BUCKET_CASES = {
    "conv3x3_c_small": ["s1_conv3x3_64_64"],
    "conv3x3_c_large": ["s4_conv3x3_512_512"],
    "conv1x1": ["s2_proj1x1_64_128_s2"],
    "stride2_downsample": ["s2_conv3x3_64_128_s2"],
    "skip_proj": ["s3_proj1x1_128_256_s2"],
}

DMA_BUCKET_CASE = {
    "large_contiguous_load_store": "s4_conv3x3_512_512",
    "small_tensor_load_store": "s2_proj1x1_64_128_s2",
    "strided_load": "s3_proj1x1_128_256_s2",
    "padded_load": "s2_conv3x3_64_128_s2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.1.228")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument(
        "--output-dir",
        default=(
            "vta/tutorials/frontend/report_out/resource_aware_maxplus/"
            "20260713_v1/calibration"
        ),
    )
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--threads", default="1,2,3,4")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--number", type=int, default=20)
    parser.add_argument("--command-timeout-s", type=int, default=1800)
    parser.add_argument("--profile-events-limit", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Do not contact the board; aggregate existing session artifacts.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as inp:
        return list(csv.DictReader(inp))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_threads(text: str) -> List[int]:
    values = sorted({int(item.strip()) for item in str(text).split(",") if item.strip()})
    if not values or any(item < 1 or item > 4 for item in values):
        raise ValueError("--threads must be a non-empty subset of 1,2,3,4")
    return values


def median(values: Iterable[float]) -> float:
    data = [float(item) for item in values if math.isfinite(float(item))]
    if not data:
        raise RuntimeError("cannot aggregate an empty measurement set")
    return float(statistics.median(data))


def vta_sram_capacities() -> Dict[str, Any]:
    hw_root = Path(
        os.environ.get(
            "VTA_HW_PATH",
            str(Path(__file__).resolve().parents[3] / "3rdparty" / "vta-hw"),
        )
    )
    path = hw_root / "config" / "vta_config.json"
    if not path.exists():
        raise RuntimeError("missing VTA hardware configuration: {}".format(path))
    config = json.loads(path.read_text(encoding="utf-8"))
    capacities = {
        "inp_bytes": 1 << int(config["LOG_INP_BUFF_SIZE"]),
        "wgt_bytes": 1 << int(config["LOG_WGT_BUFF_SIZE"]),
        "acc_bytes": 1 << int(config["LOG_ACC_BUFF_SIZE"]),
        "out_bytes": 1 << int(config["LOG_ACC_BUFF_SIZE"]),
    }
    return {
        **capacities,
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def mad(values: Iterable[float]) -> float:
    data = [float(item) for item in values if math.isfinite(float(item))]
    center = median(data)
    return median(abs(item - center) for item in data)


def command_for_run(
    args: argparse.Namespace,
    output_dir: Path,
    device: str,
    cases: Sequence[str],
    cpu_threads: int = 0,
    profile_dir: Path | None = None,
) -> List[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "benchmark_resnet18_single_ops.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--devices",
        device,
        "--cases",
        ",".join(cases),
        "--warmup",
        str(args.warmup),
        "--number",
        str(args.number),
        "--output-dir",
        str(output_dir),
    ]
    if cpu_threads:
        command.extend(["--cpu-num-threads", str(cpu_threads)])
    if profile_dir is not None:
        command.extend(
            [
                "--vta-runtime-profile-dir",
                str(profile_dir),
                "--vta-runtime-profile-events-limit",
                str(args.profile_events_limit),
            ]
        )
    return command


def run_logged(command: Sequence[str], log_path: Path, timeout_s: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(timeout_s),
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError(
            "calibration command failed rc={} command={}\n{}".format(
                completed.returncode, " ".join(command), tail
            )
        )


def update_progress(
    output_dir: Path,
    completed: int,
    total: int,
    current: str,
    status: str,
    started: float,
) -> None:
    fraction = completed / max(1, total)
    width = 30
    filled = min(width, int(round(fraction * width)))
    line = "[{}{}] {}/{} current={} status={} elapsed={:.1f}s".format(
        "#" * filled,
        "-" * (width - filled),
        completed,
        total,
        current,
        status,
        time.monotonic() - started,
    )
    (output_dir / "progress.txt").write_text(line + "\n", encoding="utf-8")
    write_json(
        output_dir / "progress.json",
        {
            "completed": completed,
            "total": total,
            "current": current,
            "status": status,
            "elapsed_s": time.monotonic() - started,
        },
    )
    print(line, flush=True)


def collect_measurements(args: argparse.Namespace, output_dir: Path, threads: Sequence[int]) -> None:
    cpu_cases = sorted({case for cases in CPU_BUCKET_CASES.values() for case in cases})
    vta_cases = sorted({case for cases in VTA_BUCKET_CASES.values() for case in cases})
    total = int(args.sessions) * (len(threads) + 1)
    completed = 0
    started = time.monotonic()
    for session in range(1, int(args.sessions) + 1):
        session_dir = output_dir / "raw" / "session{:02d}".format(session)
        for thread_count in threads:
            run_dir = session_dir / "cpu_t{}".format(thread_count)
            csv_path = run_dir / "raw_benchmark.csv"
            current = "session{:02d}/cpu_t{}".format(session, thread_count)
            update_progress(output_dir, completed, total, current, "running", started)
            if not (args.resume and csv_path.exists()):
                command = command_for_run(
                    args, run_dir, "arm_cpu", cpu_cases, cpu_threads=thread_count
                )
                (run_dir / "command.txt").parent.mkdir(parents=True, exist_ok=True)
                (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
                run_logged(command, run_dir / "run.log", args.command_timeout_s)
            completed += 1
            update_progress(output_dir, completed, total, current, "complete", started)

        run_dir = session_dir / "vta"
        csv_path = run_dir / "raw_benchmark.csv"
        current = "session{:02d}/vta".format(session)
        update_progress(output_dir, completed, total, current, "running", started)
        if not (args.resume and csv_path.exists()):
            profile_dir = run_dir / "profile"
            command = command_for_run(
                args, run_dir, "vta", vta_cases, profile_dir=profile_dir
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
            run_logged(command, run_dir / "run.log", args.command_timeout_s)
        completed += 1
        update_progress(output_dir, completed, total, current, "complete", started)


def profile_metrics(status: Mapping[str, Any], repeats: int) -> Dict[str, float]:
    divisor = max(1, int(repeats))
    load_bytes = as_float(status.get("load_buffer_2d_bytes")) / divisor
    store_bytes = as_float(status.get("store_buffer_2d_bytes")) / divisor
    load_calls = as_float(status.get("load_buffer_2d_calls")) / divisor
    store_calls = as_float(status.get("store_buffer_2d_calls")) / divisor
    total_bytes = load_bytes + store_bytes
    total_calls = load_calls + store_calls
    wait_us = as_float(status.get("device_run_wait_us")) / divisor
    small_calls = (
        as_float(status.get("load_buffer_2d_small_calls"))
        + as_float(status.get("store_buffer_2d_small_calls"))
    ) / divisor
    strided_calls = (
        as_float(status.get("load_buffer_2d_strided_calls"))
        + as_float(status.get("store_buffer_2d_strided_calls"))
    ) / divisor
    padded_calls = as_float(status.get("load_buffer_2d_padded_calls")) / divisor
    enqueue_us = (
        as_float(status.get("load_buffer_2d_enqueue_us"))
        + as_float(status.get("store_buffer_2d_enqueue_us"))
    ) / divisor
    return {
        "load_bytes": load_bytes,
        "store_bytes": store_bytes,
        "load_calls": load_calls,
        "store_calls": store_calls,
        "avg_bytes_per_call": total_bytes / max(total_calls, 1.0),
        "small_call_ratio": small_calls / max(total_calls, 1.0),
        "strided_call_ratio": strided_calls / max(total_calls, 1.0),
        "padded_call_ratio": padded_calls / max(total_calls, 1.0),
        "ps_pl_load_bw_GBps": load_bytes / max(wait_us, 1.0e-9) / 1000.0,
        "ps_pl_store_bw_GBps": store_bytes / max(wait_us, 1.0e-9) / 1000.0,
        "fragmentation_penalty_ms_per_call": enqueue_us / max(total_calls, 1.0) / 1000.0,
        "device_wait_ms": wait_us / 1000.0,
        "driver_submit_ms": as_float(status.get("driver_submit_mmio_us")) / divisor / 1000.0,
        "driver_sync_wait_ms": as_float(status.get("driver_poll_wait_us")) / divisor / 1000.0,
    }


def load_all_rows(
    output_dir: Path,
    sessions: int,
    threads: Sequence[int],
    repeats: int,
    warmup: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    profiles: List[Dict[str, Any]] = []
    for session in range(1, int(sessions) + 1):
        session_dir = output_dir / "raw" / "session{:02d}".format(session)
        for thread_count in threads:
            path = session_dir / "cpu_t{}".format(thread_count) / "raw_benchmark.csv"
            if not path.exists():
                raise RuntimeError("missing CPU calibration artifact: {}".format(path))
            for row in read_csv(path):
                row.update({"session": session, "thread_count": thread_count})
                rows.append(row)
        path = session_dir / "vta" / "raw_benchmark.csv"
        if not path.exists():
            raise RuntimeError("missing VTA calibration artifact: {}".format(path))
        for row in read_csv(path):
            row.update({"session": session, "thread_count": 1})
            rows.append(row)
        for bucket, case_id in DMA_BUCKET_CASE.items():
            status_path = (
                session_dir
                / "vta"
                / "profile"
                / "single_op"
                / case_id
                / "benchmark_totals_status.json"
            )
            if not status_path.exists():
                raise RuntimeError("missing DMA profiler artifact: {}".format(status_path))
            status = json.loads(status_path.read_text(encoding="utf-8"))
            profiles.append(
                {
                    "session": session,
                    "bucket": bucket,
                    "case_id": case_id,
                    **profile_metrics(status, int(repeats) * 2 + int(warmup)),
                }
            )
    return rows, profiles


def successful_rows(rows: Sequence[Mapping[str, Any]], device: str, cases: Sequence[str]):
    wanted = set(cases)
    return [
        row
        for row in rows
        if row.get("device") == device
        and row.get("case_id") in wanted
        and row.get("status") == "ok"
        and str(row.get("ok", "")).lower() in {"true", "1"}
    ]


def aggregate_cost_model(
    rows: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    sessions: int,
    threads: Sequence[int],
    host: str,
    port: int,
    warmup: int,
    number: int,
) -> Dict[str, Any]:
    cpu: Dict[str, Any] = {}
    for bucket, cases in CPU_BUCKET_CASES.items():
        entry: Dict[str, Any] = {"measurement_sessions": int(sessions), "measured_from": cases}
        all_bucket_rows = successful_rows(rows, "arm_cpu", cases)
        for thread_count in threads:
            selected = [
                row
                for row in all_bucket_rows
                if int(row.get("thread_count") or 0) == int(thread_count)
                and int(row.get("cpu_num_threads_actual") or 0) == int(thread_count)
            ]
            session_ids = {int(row["session"]) for row in selected}
            if session_ids != set(range(1, int(sessions) + 1)):
                raise RuntimeError(
                    "CPU bucket {} thread {} lacks independent sessions: {}".format(
                        bucket, thread_count, sorted(session_ids)
                    )
                )
            entry[str(thread_count)] = median(as_float(row["gops"]) for row in selected)
            entry["{}_mad".format(thread_count)] = mad(
                as_float(row["gops"]) for row in selected
            )
        entry["set_input_ms"] = median(as_float(row["h2d_ms"]) for row in all_bucket_rows)
        entry["run_ms"] = median(as_float(row["kernel_ms"]) for row in all_bucket_rows)
        entry["get_output_ms"] = median(as_float(row["d2h_ms"]) for row in all_bucket_rows)
        cpu[bucket] = entry

    vta: Dict[str, Any] = {}
    for bucket, cases in VTA_BUCKET_CASES.items():
        selected = successful_rows(rows, "vta", cases)
        session_ids = {int(row["session"]) for row in selected}
        if session_ids != set(range(1, int(sessions) + 1)):
            raise RuntimeError(
                "VTA bucket {} lacks independent sessions: {}".format(bucket, sorted(session_ids))
            )
        gops_values = [as_float(row["gops"]) for row in selected]
        vta[bucket] = {
            "gops": median(gops_values),
            "gops_mad": mad(gops_values),
            "runner_submit_ms": median(as_float(row["submit_ms"]) for row in selected),
            "sync_wait_ms": median(as_float(row["sync_ms"]) for row in selected),
            "bridge_pack_ms": median(as_float(row["pack_ms"]) for row in selected),
            "set_input_ms": median(as_float(row["h2d_ms"]) for row in selected),
            "get_output_ms": median(as_float(row["d2h_ms"]) for row in selected),
            "measured_kernel_ms": median(as_float(row["kernel_ms"]) for row in selected),
            "overlap_factor": 0.0,
            "overlap_factor_semantics": (
                "conservative_roofline: measured effective GOP/s already includes DMA service"
            ),
            "measurement_sessions": int(sessions),
            "measured_from": cases,
        }

    dma: Dict[str, Any] = {}
    for bucket, case_id in DMA_BUCKET_CASE.items():
        selected = [row for row in profiles if row["bucket"] == bucket]
        session_ids = {int(row["session"]) for row in selected}
        if session_ids != set(range(1, int(sessions) + 1)):
            raise RuntimeError(
                "DMA bucket {} lacks independent sessions: {}".format(bucket, sorted(session_ids))
            )
        keys = [
            "ps_pl_load_bw_GBps",
            "ps_pl_store_bw_GBps",
            "avg_bytes_per_call",
            "small_call_ratio",
            "strided_call_ratio",
            "padded_call_ratio",
            "fragmentation_penalty_ms_per_call",
            "load_bytes",
            "store_bytes",
            "load_calls",
            "store_calls",
            "device_wait_ms",
            "driver_submit_ms",
            "driver_sync_wait_ms",
        ]
        dma[bucket] = {key: median(as_float(row[key]) for row in selected) for key in keys}
        dma[bucket].update(
            {
                "measurement_sessions": int(sessions),
                "measured_from": case_id,
                "bandwidth_semantics": "effective_bytes_over_device_service_time",
            }
        )

    cpu_rows = [row for row in rows if row.get("device") == "arm_cpu" and row.get("status") == "ok"]
    vta_rows = [row for row in rows if row.get("device") == "vta" and row.get("status") == "ok"]
    cpu_memory_rows = [
        row for row in cpu_rows if row.get("case_id") == "residual_add_relu_256_14"
    ]
    cpu_memory_bandwidth = {}
    for thread_count in threads:
        selected = [
            row
            for row in cpu_memory_rows
            if int(row.get("thread_count") or 0) == int(thread_count)
        ]
        if {int(row["session"]) for row in selected} != set(range(1, int(sessions) + 1)):
            raise RuntimeError(
                "CPU memory bandwidth thread {} lacks independent sessions".format(
                    thread_count
                )
            )
        values = [as_float(row.get("effective_memory_bandwidth_GBps")) for row in selected]
        cpu_memory_bandwidth[str(thread_count)] = median(values)
        cpu_memory_bandwidth["{}_mad".format(thread_count)] = mad(values)
    boundary_bw_samples = []
    for row in cpu_rows:
        input_bytes = (
            as_float(row.get("batch"), 1.0)
            * as_float(row.get("channels_in"))
            * as_float(row.get("height"), 1.0)
            * as_float(row.get("width"), 1.0)
            * 4.0
        )
        copy_ms = as_float(row.get("h2d_ms"))
        if input_bytes > 0.0 and copy_ms > 0.0:
            boundary_bw_samples.append(input_bytes / (copy_ms / 1000.0) / 1.0e9)

    model = {
        "version": 2,
        "model": "RAMPS hardware service calibration",
        "source": "ramps_measured_calibration",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "board": "root@{}".format(host),
        "rpc_port": int(port),
        "protocol": {
            "independent_sessions": int(sessions),
            "warmup_runs": int(warmup),
            "measured_repeats": int(number),
            "cpu_threads": list(threads),
            "aggregation": "median with MAD",
            "fallback_allowed": False,
        },
        "units": {
            "ops": "OP",
            "gops": "GOP/s",
            "time": "ms",
            "dma_bytes": "B",
            "ps_pl_bw": "decimal GB/s",
        },
        "sram_capacities": vta_sram_capacities(),
        "cpu_eff_gops": cpu,
        "cpu_memory_bandwidth_GBps": {
            **cpu_memory_bandwidth,
            "measurement_sessions": int(sessions),
            "measured_from": "residual_add_relu_256_14 compulsory tensor traffic",
        },
        "vta_eff_gops": vta,
        "dma": dma,
        "stage_overheads": {
            "cpu": {
                "set_input_ms": median(as_float(row["h2d_ms"]) for row in cpu_rows),
                "get_output_ms": median(as_float(row["d2h_ms"]) for row in cpu_rows),
                "measured_from": "all_cpu_calibration_cases",
            },
            "vta": {
                "set_input_ms": median(as_float(row["h2d_ms"]) for row in vta_rows),
                "bridge_pack_ms": median(as_float(row["pack_ms"]) for row in vta_rows),
                "runner_submit_ms": median(as_float(row["submit_ms"]) for row in vta_rows),
                "sync_wait_ms": median(as_float(row["sync_ms"]) for row in vta_rows),
                "get_output_ms": median(as_float(row["d2h_ms"]) for row in vta_rows),
                "measured_from": "all_vta_calibration_cases",
            },
        },
        "boundary_bw_GBps": median(boundary_bw_samples),
    }
    validate_complete_model(model, sessions, threads)
    return model


def validate_complete_model(
    model: Mapping[str, Any], sessions: int, threads: Sequence[int]
) -> None:
    if model.get("source") != "ramps_measured_calibration":
        raise RuntimeError("RAMPS calibration source is not measured")
    expected = {
        "cpu_eff_gops": set(CPU_BUCKET_CASES),
        "vta_eff_gops": set(VTA_BUCKET_CASES),
        "dma": set(DMA_BUCKET_CASE),
    }
    for section, bucket_names in expected.items():
        actual = set((model.get(section) or {}).keys())
        if actual != bucket_names:
            raise RuntimeError(
                "{} buckets differ: missing={} extra={}".format(
                    section, sorted(bucket_names - actual), sorted(actual - bucket_names)
                )
            )
        for bucket in bucket_names:
            entry = model[section][bucket]
            if int(entry.get("measurement_sessions", 0)) != int(sessions):
                raise RuntimeError("{}:{} lacks measured sessions".format(section, bucket))
            if "estimated_from" in entry or "fallback" in json.dumps(entry).lower():
                raise RuntimeError("{}:{} contains an estimate/fallback".format(section, bucket))
    for bucket, entry in model["cpu_eff_gops"].items():
        for thread_count in threads:
            if as_float(entry.get(str(thread_count))) <= 0.0:
                raise RuntimeError("CPU bucket {} thread {} is empty".format(bucket, thread_count))
    for thread_count in threads:
        if as_float(model.get("cpu_memory_bandwidth_GBps", {}).get(str(thread_count))) <= 0.0:
            raise RuntimeError(
                "CPU memory bandwidth thread {} is empty".format(thread_count)
            )
    for bucket, entry in model["vta_eff_gops"].items():
        if as_float(entry.get("gops")) <= 0.0:
            raise RuntimeError("VTA bucket {} is empty".format(bucket))
    for bucket, entry in model["dma"].items():
        if (
            as_float(entry.get("avg_bytes_per_call")) <= 0.0
            or as_float(entry.get("ps_pl_load_bw_GBps")) <= 0.0
            or as_float(entry.get("ps_pl_store_bw_GBps")) <= 0.0
        ):
            raise RuntimeError("DMA bucket {} is empty".format(bucket))
    if as_float(model.get("boundary_bw_GBps")) <= 0.0:
        raise RuntimeError("CPU boundary bandwidth is empty")


def main() -> None:
    args = parse_args()
    if args.sessions < 3:
        raise RuntimeError("publication calibration requires at least three independent sessions")
    if args.warmup < 5 or args.number < 20:
        raise RuntimeError("publication calibration requires warmup>=5 and number>=20")
    threads = parse_threads(args.threads)
    if threads != [1, 2, 3, 4]:
        raise RuntimeError("publication calibration requires CPU threads 1,2,3,4")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "calibration_protocol.json",
        {
            "frozen_before_measurement": True,
            "host": args.host,
            "port": args.port,
            "sessions": args.sessions,
            "threads": threads,
            "warmup": args.warmup,
            "number": args.number,
            "cpu_bucket_cases": CPU_BUCKET_CASES,
            "vta_bucket_cases": VTA_BUCKET_CASES,
            "dma_bucket_cases": DMA_BUCKET_CASE,
        },
    )
    if not args.aggregate_only:
        collect_measurements(args, output_dir, threads)
    rows, profiles = load_all_rows(
        output_dir, args.sessions, threads, args.number, args.warmup
    )
    write_csv(output_dir / "raw_measurements.csv", rows)
    write_csv(output_dir / "raw_dma_profiles.csv", profiles)
    model = aggregate_cost_model(
        rows,
        profiles,
        args.sessions,
        threads,
        args.host,
        args.port,
        args.warmup,
        args.number,
    )
    write_json(output_dir / "cost_model.json", model)
    update_progress(output_dir, 1, 1, "aggregate", "complete", time.monotonic())
    print("[RAMPS CALIBRATION] wrote", output_dir / "cost_model.json")


if __name__ == "__main__":
    main()
