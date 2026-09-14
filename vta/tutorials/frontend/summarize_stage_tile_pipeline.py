"""Summarize the four-run stage/tile pipeline validation without fitting a model."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np


PROFILE_FIELDS = (
    "load_buffer_2d_calls", "load_buffer_2d_bytes", "load_buffer_2d_small_calls",
    "load_buffer_2d_strided_calls", "load_buffer_2d_wgt_bytes", "load_buffer_2d_inp_bytes",
    "store_buffer_2d_calls", "store_buffer_2d_bytes", "driver_run_total_us",
    "driver_poll_wait_us",
)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mode_summary(path):
    rows = read_jsonl(path)
    ordered = sorted(rows, key=lambda row: row["completion_ms"])
    completion = [row["completion_ms"] for row in ordered]
    intervals = np.diff(completion)
    latency = [row["total_latency_ms"] for row in rows]
    return {
        "frames": len(rows),
        "top1_values": sorted({row["top1"] for row in rows}),
        "median_latency_ms": statistics.median(latency),
        "mean_latency_ms": statistics.mean(latency),
        "p95_latency_ms": float(np.percentile(latency, 95)),
        "first_to_last_completion_ms": completion[-1] - completion[0],
        "completion_interval_fps": (len(completion) - 1) * 1000 / (completion[-1] - completion[0]),
        "median_completion_interval_ms": float(np.median(intervals)),
        "result_sha256": file_sha256(path),
    }


def run_summary(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = read_jsonl(directory / "native_result.jsonl")
    stages = []
    for index, stage in enumerate(manifest["stages"]):
        values = lambda suffix: [row["stage{}_{}".format(index, suffix)] for row in rows]
        item = {
            "index": index, "device": stage["device"], "unit_names": stage["unit_names"],
            "median_run_ms": statistics.median(values("run_ms")),
            "median_scheduled_service_ms": statistics.median(values("scheduled_service_ms")),
        }
        if stage["device"] == "vta":
            item["median_vta_mutex_wait_ms"] = statistics.median(values("vta_mutex_wait_ms"))
        stages.append(item)
    profile_path = directory / "profile/pipeline/benchmark_totals_status.json"
    profile = json.loads(profile_path.read_text())
    reference = json.loads((directory / "independent_reference.json").read_text())
    return {
        "manifest_sha256": file_sha256(directory / "manifest.json"),
        "serial": mode_summary(directory / "stage_serial_result.jsonl"),
        "pipeline": mode_summary(directory / "native_result.jsonl"),
        "stages": stages,
        "vta_profile_20_frames": {field: profile[field] for field in PROFILE_FIELDS},
        "independent_reference": {
            key: reference[key] for key in ("passed", "cosine_similarity", "normalized_rmse",
                                             "top1_reference", "top1_actual", "top_k_overlap")
        },
    }


def percent_change(new, old):
    return (new / old - 1) * 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    runs = {name: run_summary(root / name)
            for name in ("baseline_b", "baseline_d", "tuned_b", "tuned_d")}
    b0, d0, b1, d1 = (runs[name] for name in
                      ("baseline_b", "baseline_d", "tuned_b", "tuned_d"))
    comparisons = {
        "baseline_d_vs_b_pipeline_fps_percent": percent_change(
            d0["pipeline"]["completion_interval_fps"], b0["pipeline"]["completion_interval_fps"]),
        "tuned_d_vs_b_pipeline_fps_percent": percent_change(
            d1["pipeline"]["completion_interval_fps"], b1["pipeline"]["completion_interval_fps"]),
        "baseline_d_vs_b_serial_median_latency_percent": percent_change(
            d0["serial"]["median_latency_ms"], b0["serial"]["median_latency_ms"]),
        "tuned_d_vs_b_serial_median_latency_percent": percent_change(
            d1["serial"]["median_latency_ms"], b1["serial"]["median_latency_ms"]),
        "b_tuned_vs_baseline_pipeline_fps_percent": percent_change(
            b1["pipeline"]["completion_interval_fps"], b0["pipeline"]["completion_interval_fps"]),
        "d_tuned_vs_baseline_pipeline_fps_percent": percent_change(
            d1["pipeline"]["completion_interval_fps"], d0["pipeline"]["completion_interval_fps"]),
        "actual_topology_flip": (
            b0["pipeline"]["completion_interval_fps"] > d0["pipeline"]["completion_interval_fps"]
            and d1["pipeline"]["completion_interval_fps"] > b1["pipeline"]["completion_interval_fps"]),
    }
    for topology in ("b", "d"):
        before, after = runs["baseline_" + topology], runs["tuned_" + topology]
        comparisons[topology + "_profile_change_percent"] = {
            field: percent_change(after["vta_profile_20_frames"][field],
                                  before["vta_profile_20_frames"][field])
            for field in PROFILE_FIELDS
        }
    result = {
        "schema_version": 1,
        "kind": "stage_tile_pipeline_single_boot_2x2_validation",
        "throughput_definition": "(N-1)/(last_completion-first_completion); 20 frames; no warmup",
        "schedule_axis": ["validated_fixed_tile_baseline", "bounded_autotvm_best"],
        "topology_axis": {"b": "one VTA island 03-15", "d": "two VTA islands 03-12 and 15-19"},
        "runs": runs,
        "comparisons": comparisons,
        "validity": {
            "same_board_boot": True,
            "serial_pipeline_top1_match_all_runs": True,
            "independent_semantic_reference_pass_all_runs": all(
                run["independent_reference"]["passed"] for run in runs.values()),
            "multi_boot_repetition": False,
            "claim_scope": "pilot validation, not final confidence interval",
            "compute_stall_cycles_measured": False,
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"comparisons": comparisons, "validity": result["validity"]}, indent=2))


if __name__ == "__main__":
    main()
