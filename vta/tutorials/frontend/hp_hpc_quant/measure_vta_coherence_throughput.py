#!/usr/bin/env python3
"""Measure VTA data-path throughput and coherence overhead on board.

This script is intended for A/B comparison between:
- HP-only + non-coherent runtime
- HPC/coherent runtime

It reuses the existing single-op conv2d benchmark path, but reports a
different metric set focused on:
- DMA load/store bytes observed by the VTA runtime profiler
- device-side wait time and effective DMA throughput
- explicit cache-maintenance cost (flush/invalidate)
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

# Allow direct execution from the grouped hp_hpc_quant directory.
FRONTEND_DIR = Path(__file__).resolve().parent.parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

from benchmark_resnet18_single_ops import (  # pylint: disable=import-error
    benchmark_conv_case,
    connect_remote,
    env,
    is_vta_compatible_conv,
    selected_cases,
)
from hp_hpc_quant.vta_runtime_profile_utils import ensure_dir


DEFAULT_CASES = ",".join(
    [
        "s1_conv3x3_64_64",
        "s2_proj1x1_64_128_s2",
        "s2_conv3x3_128_128",
        "s3_proj1x1_128_256_s2",
        "s4_proj1x1_256_512_s2",
        "s4_conv3x3_512_512",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument(
        "--config-label",
        default="hp_only",
        help="Label stored in the CSV/report, e.g. hp_only or hpc_coherent",
    )
    parser.add_argument(
        "--cases",
        default=DEFAULT_CASES,
        help="Comma-separated benchmark case ids; defaults to a throughput-oriented subset",
    )
    parser.add_argument("--case-file", default="")
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument(
        "--number",
        type=int,
        default=1,
        help="Benchmark steady-state repeat count passed through to the reused conv path",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup count passed through to the reused conv path",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional CSV path; defaults to a timestamped file under --output-dir",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory to store the CSV and profiler snapshots",
    )
    parser.add_argument(
        "--profile-events-limit",
        type=int,
        default=200,
        help="Maximum profiler events to persist per snapshot",
    )
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> Dict[str, str]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.output_dir) if args.output_dir else Path(f"/tmp/vta_coherence_bw_{args.config_label}_{ts}")
    base_dir = Path(ensure_dir(str(base_dir)))
    csv_path = Path(args.output) if args.output else base_dir / "coherence_throughput.csv"
    profile_dir = base_dir / "profiles"
    ensure_dir(str(profile_dir))
    return {
        "base_dir": str(base_dir),
        "csv_path": str(csv_path),
        "profile_dir": str(profile_dir),
        "summary_path": str(base_dir / "summary.json"),
    }


def read_json(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def bytes_per_us_to_gbps(byte_count: float, duration_us: float) -> float:
    if duration_us <= 0.0:
        return 0.0
    return float(byte_count) / float(duration_us) / 1000.0


def safe_ratio(numer: float, denom: float) -> float:
    if denom == 0.0:
        return 0.0
    return float(numer) / float(denom)


def summarize_case(case_row: Dict[str, object], case_profile_dir: Path, config_label: str) -> Dict[str, object]:
    after_run = read_json(case_profile_dir / "after_run_status.json")
    after_get_output = read_json(case_profile_dir / "after_get_output_status.json")

    load_bytes = int(after_get_output.get("load_buffer_2d_bytes", 0))
    store_bytes = int(after_get_output.get("store_buffer_2d_bytes", 0))
    total_dma_bytes = load_bytes + store_bytes
    device_wait_us = float(after_get_output.get("device_run_wait_us", 0.0))
    driver_run_total_us = float(after_get_output.get("driver_run_total_us", 0.0))
    driver_submit_mmio_us = float(after_get_output.get("driver_submit_mmio_us", 0.0))

    flush_calls = int(after_get_output.get("flush_cache_calls", 0))
    flush_bytes = int(after_get_output.get("flush_cache_bytes", 0))
    flush_us = float(after_get_output.get("flush_cache_us", 0.0))
    invalidate_calls = int(after_get_output.get("invalidate_cache_calls", 0))
    invalidate_bytes = int(after_get_output.get("invalidate_cache_bytes", 0))
    invalidate_us = float(after_get_output.get("invalidate_cache_us", 0.0))
    coherence_us = flush_us + invalidate_us

    total_dma_mib = total_dma_bytes / float(1 << 20)
    load_mib = load_bytes / float(1 << 20)
    store_mib = store_bytes / float(1 << 20)

    row = {
        "config_label": config_label,
        "target": env.TARGET,
        "case_id": case_row["case_id"],
        "kernel_ms": float(case_row["kernel_ms"]),
        "total_ms": float(case_row["total_ms"]),
        "submit_ms": float(case_row["submit_ms"]),
        "sync_ms": float(case_row["sync_ms"]),
        "h2d_ms": float(case_row["h2d_ms"]),
        "d2h_ms": float(case_row["d2h_ms"]),
        "gops": float(case_row["gops"]),
        "ok": bool(case_row["ok"]),
        "load_bytes": load_bytes,
        "store_bytes": store_bytes,
        "total_dma_bytes": total_dma_bytes,
        "load_mib": load_mib,
        "store_mib": store_mib,
        "total_dma_mib": total_dma_mib,
        "device_run_wait_us": device_wait_us,
        "driver_run_total_us": driver_run_total_us,
        "driver_submit_mmio_us": driver_submit_mmio_us,
        "load_bw_gbps": bytes_per_us_to_gbps(load_bytes, device_wait_us),
        "store_bw_gbps": bytes_per_us_to_gbps(store_bytes, device_wait_us),
        "total_bw_gbps": bytes_per_us_to_gbps(total_dma_bytes, device_wait_us),
        "flush_cache_calls": flush_calls,
        "flush_cache_bytes": flush_bytes,
        "flush_cache_us": flush_us,
        "invalidate_cache_calls": invalidate_calls,
        "invalidate_cache_bytes": invalidate_bytes,
        "invalidate_cache_us": invalidate_us,
        "coherence_overhead_us": coherence_us,
        "coherence_overhead_ms": coherence_us / 1000.0,
        "coherence_us_per_dma_mib": safe_ratio(coherence_us, total_dma_mib),
        "coherence_pct_of_driver_run": 100.0 * safe_ratio(coherence_us, driver_run_total_us),
        "coherence_pct_of_device_wait": 100.0 * safe_ratio(coherence_us, device_wait_us),
        "flush_pct_of_coherence": 100.0 * safe_ratio(flush_us, coherence_us),
        "invalidate_pct_of_coherence": 100.0 * safe_ratio(invalidate_us, coherence_us),
        "profile_dir": str(case_profile_dir),
        "after_run_driver_wait_us": float(after_run.get("device_run_wait_us", 0.0)),
    }
    return row


def aggregate_summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    ok_rows = [row for row in rows if row["ok"]]
    summary = {
        "row_count": len(rows),
        "ok_count": len(ok_rows),
        "config_labels": sorted({row["config_label"] for row in rows}),
    }
    if not ok_rows:
        return summary
    numeric_keys = [
        "load_bw_gbps",
        "store_bw_gbps",
        "total_bw_gbps",
        "coherence_overhead_ms",
        "coherence_pct_of_driver_run",
        "coherence_pct_of_device_wait",
        "kernel_ms",
        "total_ms",
    ]
    for key in numeric_keys:
        values = [float(row[key]) for row in ok_rows]
        summary[key] = {
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }
    hottest = max(ok_rows, key=lambda row: float(row["total_bw_gbps"]))
    worst_coherence = max(ok_rows, key=lambda row: float(row["coherence_overhead_ms"]))
    summary["best_dma_case"] = {
        "case_id": hottest["case_id"],
        "total_bw_gbps": hottest["total_bw_gbps"],
    }
    summary["worst_coherence_case"] = {
        "case_id": worst_coherence["case_id"],
        "coherence_overhead_ms": worst_coherence["coherence_overhead_ms"],
    }
    return summary


def main() -> None:
    args = parse_args()
    paths = resolve_output_paths(args)
    cases = selected_cases(args.cases, args.case_file)
    remote = connect_remote(args.host, args.port)

    print("========== Coherence/Throughput ==========")
    print("config_label =", args.config_label)
    print("target       =", env.TARGET)
    print("host         =", args.host)
    print("port         =", args.port)
    print("cases        =", [case.case_id for case in cases])
    print("output       =", paths["csv_path"])
    print("profile_dir  =", paths["profile_dir"])
    print("==========================================")

    rows: List[Dict[str, object]] = []
    for case in cases:
        if case.op_name != "conv2d" or not is_vta_compatible_conv(case):
            print(f"[SKIP] {case.case_id} unsupported for VTA throughput experiment")
            continue
        case_profile_dir = Path(paths["profile_dir"]) / case.case_id
        print(f"\n=== {case.case_id} ({args.config_label}) ===")
        case_row = benchmark_conv_case(
            remote,
            case,
            "vta",
            number=args.number,
            warmup=args.warmup,
            check_correctness=args.check_correctness,
            runtime_profile_dir=str(case_profile_dir),
            runtime_profile_events_limit=args.profile_events_limit,
        )
        summary_row = summarize_case(case_row, case_profile_dir, args.config_label)
        rows.append(summary_row)
        print(
            "dma_mib={:.3f} total_bw={:.3f} load_bw={:.3f} store_bw={:.3f} "
            "coh={:.3f} ms ({:.2f}% driver_run) ok={}".format(
                summary_row["total_dma_mib"],
                summary_row["total_bw_gbps"],
                summary_row["load_bw_gbps"],
                summary_row["store_bw_gbps"],
                summary_row["coherence_overhead_ms"],
                summary_row["coherence_pct_of_driver_run"],
                summary_row["ok"],
            )
        )

    if not rows:
        raise RuntimeError("No valid VTA cases were run")

    fieldnames = [
        "config_label",
        "target",
        "case_id",
        "kernel_ms",
        "total_ms",
        "submit_ms",
        "sync_ms",
        "h2d_ms",
        "d2h_ms",
        "gops",
        "ok",
        "load_bytes",
        "store_bytes",
        "total_dma_bytes",
        "load_mib",
        "store_mib",
        "total_dma_mib",
        "device_run_wait_us",
        "driver_run_total_us",
        "driver_submit_mmio_us",
        "load_bw_gbps",
        "store_bw_gbps",
        "total_bw_gbps",
        "flush_cache_calls",
        "flush_cache_bytes",
        "flush_cache_us",
        "invalidate_cache_calls",
        "invalidate_cache_bytes",
        "invalidate_cache_us",
        "coherence_overhead_us",
        "coherence_overhead_ms",
        "coherence_us_per_dma_mib",
        "coherence_pct_of_driver_run",
        "coherence_pct_of_device_wait",
        "flush_pct_of_coherence",
        "invalidate_pct_of_coherence",
        "profile_dir",
        "after_run_driver_wait_us",
    ]

    with open(paths["csv_path"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = aggregate_summary(rows)
    summary["csv_path"] = paths["csv_path"]
    summary["profile_dir"] = paths["profile_dir"]
    summary["config_label"] = args.config_label
    with open(paths["summary_path"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\n[CSV] wrote {len(rows)} rows to {paths['csv_path']}")
    print(f"[JSON] wrote summary to {paths['summary_path']}")


if __name__ == "__main__":
    main()
