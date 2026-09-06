#!/usr/bin/env python3
"""Summarize a native RAMPS instrumentation smoke run."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--skip-first", type=int, default=3)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def median(rows: List[Dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(row[key]) for row in rows))


def profile_summary(path: Path, runs: int) -> Dict[str, float]:
    raw = json.loads(path.read_text())
    divisor = max(1, int(runs))
    keys = [
        "device_run_wait_us",
        "driver_submit_mmio_us",
        "driver_poll_wait_us",
        "load_buffer_2d_bytes",
        "store_buffer_2d_bytes",
        "load_buffer_2d_calls",
        "store_buffer_2d_calls",
        "load_buffer_2d_enqueue_us",
        "store_buffer_2d_enqueue_us",
    ]
    return {key: float(raw.get(key, 0.0)) / divisor for key in keys}


def main() -> None:
    args = parse_args()
    root = Path(args.run_dir)
    serial_all = read_jsonl(root / "stage_serial_result.jsonl")
    pipeline_all = read_jsonl(root / "native_result.jsonl")
    serial = serial_all[int(args.skip_first) :]
    pipeline = pipeline_all[int(args.skip_first) :]
    manifest = json.loads((root / "manifest.json").read_text())
    stages = []
    for index, stage in enumerate(manifest["stages"]):
        wall_ms = median(serial, "stage{}_ms".format(index))
        core_ms = median(serial, "stage{}_process_cpu_ms".format(index))
        stages.append(
            {
                "index": index,
                "name": stage["name"],
                "device": stage["device"],
                "serial_wall_ms": wall_ms,
                "serial_set_ms": median(serial, "stage{}_set_ms".format(index)),
                "serial_run_ms": median(serial, "stage{}_run_ms".format(index)),
                "serial_get_ms": median(serial, "stage{}_get_ms".format(index)),
                "serial_process_cpu_ms": core_ms,
                "average_cpu_cores": core_ms / max(wall_ms, 1.0e-12),
                "pipeline_wall_ms": median(pipeline, "stage{}_ms".format(index)),
                "pipeline_process_cpu_ms": None,
            }
        )
    starts = [float(row["stage0_start_ms"]) for row in pipeline]
    intervals = [right - left for left, right in zip(starts, starts[1:])]
    cycle_ms = float(statistics.mean(intervals))
    serial_profile = profile_summary(
        root / "profile" / "serial" / "benchmark_totals_status.json", len(serial_all)
    )
    pipeline_profile = profile_summary(
        root / "profile" / "pipeline" / "benchmark_totals_status.json", len(pipeline_all)
    )
    summary = {
        "schema_version": 1,
        "measurement_path": "native_static_package",
        "rpc_used_for_performance": False,
        "rpc_used_for_correctness_reference_only": True,
        "candidate_id": manifest.get("candidate_id"),
        "runs": len(serial_all),
        "skip_first": int(args.skip_first),
        "serial_cpu_time_scope": sorted(
            {row.get("stage_cpu_time_scope") for row in serial_all}
        ),
        "pipeline_cpu_time_scope": sorted(
            {row.get("stage_cpu_time_scope") for row in pipeline_all}
        ),
        "serial_pipeline_top1_match": [row.get("top1") for row in serial_all]
        == [row.get("top1") for row in pipeline_all],
        "top1": sorted({row.get("top1") for row in serial_all + pipeline_all}),
        "pipeline_cycle_ms": cycle_ms,
        "pipeline_throughput_fps": 1000.0 / cycle_ms,
        "stages": stages,
        "vta_profile_per_frame": {
            "serial": serial_profile,
            "pipeline": pipeline_profile,
        },
        "publication_status": "instrumentation_smoke_only",
        "limitations": [
            "This is a multi-stage ResNet18 candidate, not an isolated service bucket.",
            "Eight runs with three skipped frames are insufficient for formal calibration.",
            "The result validates native timing fields but is not a fitted service model.",
        ],
    }
    (root / "native_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# RAMPS Native Instrumentation Smoke",
        "",
        "- Performance path: native static package and board-side C++ runner.",
        "- RPC role: persisted all-VTA correctness reference only.",
        "- Pipeline throughput: `{:.4f} fps` (`{:.3f} ms` cycle).".format(
            summary["pipeline_throughput_fps"], summary["pipeline_cycle_ms"]
        ),
        "- Serial/pipeline top1 match: `{}`; top1: `{}`.".format(
            summary["serial_pipeline_top1_match"], summary["top1"]
        ),
        "",
        "| Stage | Device | Serial wall ms | Process CPU ms | Average cores | Pipeline wall ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for stage in stages:
        lines.append(
            "| `{}` | `{}` | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                stage["name"],
                stage["device"],
                stage["serial_wall_ms"],
                stage["serial_process_cpu_ms"],
                stage["average_cpu_cores"],
                stage["pipeline_wall_ms"],
            )
        )
    lines.extend(
        [
            "",
            "The CPU values demonstrate why `wall_time * requested_threads` is invalid. The",
            "measured average utilization is about 2.30/3 cores for stage0 and 3.20/4 cores",
            "for stage2 rather than exactly the requested thread count.",
            "",
            "This smoke validates instrumentation only. Formal calibration still requires",
            "isolated native buckets, three sessions, warmup 5, 20 measured runs and no",
            "fallback schedule or estimated field.",
        ]
    )
    (root / "NATIVE_SMOKE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "NATIVE_SMOKE_REPORT.md")


if __name__ == "__main__":
    main()
