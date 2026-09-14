#!/usr/bin/env python3
"""Derive the auditable P7R454 from-scratch AutoTVM/ours Y10 comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    manifest = read(root / "artifact_hashes.json")["artifacts"]
    bad = []
    for relative, expected in manifest.items():
        path = root / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            bad.append({"path": relative, "expected": expected, "actual": actual})
    if bad:
        raise RuntimeError("artifact verification failed: " + json.dumps(bad[:3]))
    return len(manifest)


def profile_sum(rows):
    keys = (
        "load_buffer_2d_bytes", "store_buffer_2d_bytes",
        "load_buffer_2d_calls", "store_buffer_2d_calls", "driver_run_calls",
    )
    result = {key: 0 for key in keys}
    for row in rows:
        profile = row.get("runtime_profile_complete") or row.get("runtime_profile") or {}
        for key in keys:
            result[key] += int(profile.get(key, 0))
    result["dma_calls"] = result["load_buffer_2d_calls"] + result["store_buffer_2d_calls"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xgb", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    xroot, oroot, output = args.xgb.resolve(), args.ours.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    verified = {"xgb": verify(xroot), "ours": verify(oroot)}
    xgb, ours = read(xroot / "summary.json"), read(oroot / "summary.json")
    if xgb["status"] != "completed_t0_to_t1" or ours["status"] != "completed_t0_to_t1":
        raise RuntimeError("both sessions must have completed T0-to-T1")

    timeline = [json.loads(line) for line in (xroot / "timeline.jsonl").read_text().splitlines()
                if line.strip()]
    error_counts = {}
    for row in timeline:
        key = str(row["error_no"])
        error_counts[key] = error_counts.get(key, 0) + 1
    measurements = [read(path) for path in (xroot / "isolated_measurements").glob("*/measurement.json")]
    successful = [row for row in measurements if row.get("correct")]
    runtime_failed = [row for row in measurements if not row.get("correct")]
    isolated_one_profile = profile_sum(successful)
    # Each successful measurement executes 3 correctness calls, 3 timed calls,
    # and one separately profiled call. Failed device calls have no complete
    # profile and are intentionally excluded from this auditable lower bound.
    isolated_successful_calls = {key: value * 7 for key, value in isolated_one_profile.items()}
    xgraph = xgb["full_graph_result"]
    xgraph_profile = profile_sum(xgraph["correctness"] + xgraph["timings"])
    xgb_board_lower_bound = {
        key: isolated_successful_calls[key] + xgraph_profile[key]
        for key in isolated_successful_calls
    }

    xcid = xgraph["candidate_id"]
    xgb_ms = float(xgraph["median_latency_ms"][xcid])
    xgb_stock_ms = float(xgraph["median_latency_ms"]["stock_reference"])
    ours_ms = float(ours["selected_fullgraph_median_ms"])
    ours_stock_ms = float(ours["paired_stock_fullgraph_median_ms"])
    total_reduction = 1.0 - ours["t0_to_t1_seconds"] / xgb["t0_to_t1_seconds"]
    result = {
        "schema": "c3_from_scratch_autotune_ab_y10_v1",
        "status": "one_seed_matched_y10_ab_complete",
        "workload": "YOLOv3-tiny-320 conv18, CI=256, CO=128, H=W=10, K=1",
        "verified_artifact_counts": verified,
        "xgb": {
            "config_space": xgb["config_space_size"],
            "gross_proposals": xgb["gross_proposals"],
            "isolated_compile_attempts": len(timeline),
            "isolated_build_artifacts": len(measurements),
            "isolated_fpga_correct": len(successful),
            "isolated_runtime_failed": len(runtime_failed),
            "timeline_error_no_counts": error_counts,
            "fsim_invocations": 0,
            "distinct_isolated_board_dispatches": len(measurements),
            "known_correctness_invocations": len(successful) * 3,
            "known_timing_invocations": len(successful) * 3,
            "known_profile_invocations": len(successful),
            "failed_device_attempts": len(runtime_failed),
            "tuner_seconds": next(row["seconds"] for row in xgb["phases"]
                                  if row["phase"] == "empty_history_autotvm_tune"),
            "t0_to_t1_seconds": xgb["t0_to_t1_seconds"],
            "deployable_fullgraph_result_available_seconds": xgb["t0_to_t1_seconds"],
            "selected_mode": "original",
            "selected_config_index": xgb["selected_config_index"],
            "selected_fullgraph_median_ms": xgb_ms,
            "paired_stock_fullgraph_median_ms": xgb_stock_ms,
            "paired_ratio": xgraph["median_paired_latency_ratio"],
            "speedup_vs_paired_stock_percent": (1.0 - xgraph["median_paired_latency_ratio"]) * 100.0,
            "paired_wins": xgraph["candidate_paired_wins"],
            "board_logical_resource_lower_bound_excluding_failed_device_calls": xgb_board_lower_bound,
        },
        "ours": {
            "config_space_original_tiles": ours["config_space_original_tiles"],
            "generated_mode_identities": ours["generated_mode_identities"],
            "qualified_mode_identities": ours["qualified_mode_identities"],
            "static_ok": ours["static_ok"],
            "fsim_passed": ours["fsim_passed"],
            "fsim_invocations": ours["fsim_passed"] * 3,
            "fullgraph_candidate_dispatches": ours["fpga_dispatch_count"],
            "fullgraph_correctness_invocations_including_paired_stock": ours["fpga_correctness_invocations"],
            "fullgraph_timing_invocations_including_paired_stock": ours["fpga_timing_invocations"],
            "qualification_and_front_seconds": sum(
                row["seconds"] for row in ours["phase_timeline"]
                if row["phase"] != "online_fullgraph_search"
            ),
            "online_fullgraph_seconds": next(
                row["seconds"] for row in ours["phase_timeline"]
                if row["phase"] == "online_fullgraph_search"
            ),
            "t0_to_t1_seconds": ours["t0_to_t1_seconds"],
            "first_deployable_fullgraph_within_final_xgb_2_percent_seconds": ours[
                "first_final_best_plus_2_percent_t0_seconds"
            ],
            "selected_mode": ours["selected_public_mode"],
            "selected_fullgraph_median_ms": ours_ms,
            "paired_stock_fullgraph_median_ms": ours_stock_ms,
            "paired_ratio": ours["selected_over_stock_ratio"],
            "speedup_vs_paired_stock_percent": ours["selected_speedup_percent"],
            "paired_wins": ours["selected_paired_wins"],
            "board_logical_resources_including_three_paired_stock_runs": ours[
                "online_search_board_resources_including_paired_stock"
            ],
        },
        "comparison": {
            "t0_to_t1_reduction_percent": total_reduction * 100.0,
            "t0_to_t1_speedup": xgb["t0_to_t1_seconds"] / ours["t0_to_t1_seconds"],
            "ours_latency_minus_xgb_percent": (ours_ms / xgb_ms - 1.0) * 100.0,
            "paired_ratio_difference_percentage_points": (
                ours["selected_over_stock_ratio"] - xgraph["median_paired_latency_ratio"]
            ) * 100.0,
            "first_ours_xgb_plus_2_quality_time_reduction_percent": (
                1.0 - ours["first_final_best_plus_2_percent_t0_seconds"]
                / xgb["t0_to_t1_seconds"]
            ) * 100.0,
            "candidate_program_dispatch_reduction_percent": (
                1.0 - ours["fpga_dispatch_count"] / len(measurements)
            ) * 100.0,
        },
        "comparability_notes": [
            "Both sessions start after a clean bitstream reload/fresh RPC and end after three-input all-output correctness plus seven paired full-graph timing rounds.",
            "AutoTVM uses an empty tuning history and does not load TopHub during target-layer search; other graph layers use the normal deployment context in both sessions.",
            "AutoTVM searches original mode with isolated-operator FPGA timing; ours searches tile plus residence mode and tests only three promoted candidates as complete graphs. This is a system-flow comparison, not a same-space pure search-algorithm ablation.",
            "Y10 was already exposed before the cost rerun. Ours regenerated candidate identities exactly without reading target performance labels, so this is a retrospective cost reexecution, not a second prospective holdout.",
            "Logical DMA totals are not directly favorable to ours because ours validates three complete graphs and paired stocks, whereas AutoTVM searches with isolated operators. Physical AXI traffic was not measured.",
            "Only one XGB random seed and one deterministic ours clean-start session are complete; the frozen protocol requires three balanced seeds before a final statistical claim.",
        ],
        "claim_boundary": (
            "Matched one-seed Y10 case-study evidence. It supports a preliminary end-to-end "
            "time-to-quality claim, not multi-seed or cross-network generalization."
        ),
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "stock_autotvm_xgb", "ours", "note"])
        writer.writerow(["T0_to_T1_seconds", xgb["t0_to_t1_seconds"], ours["t0_to_t1_seconds"],
                         "same clean-start to deployable full-graph boundary"])
        writer.writerow(["candidate_program_dispatches", len(measurements), ours["fpga_dispatch_count"],
                         "isolated candidates for XGB; full graphs for ours"])
        writer.writerow(["FSim_invocations", 0, ours["fsim_passed"] * 3, "ours safety cost"])
        writer.writerow(["selected_fullgraph_median_ms", xgb_ms, ours_ms, "separate clean sessions"])
        writer.writerow(["paired_stock_median_ms", xgb_stock_ms, ours_stock_ms, "same-session stock"])
        writer.writerow(["selected_over_stock_ratio", xgraph["median_paired_latency_ratio"],
                         ours["selected_over_stock_ratio"], "preferred drift-resistant metric"])
        writer.writerow(["paired_wins", xgraph["candidate_paired_wins"], ours["selected_paired_wins"],
                         "out of seven"])
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n")
    artifacts = {
        path.name: sha256(path) for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_text(json.dumps({
        "artifacts": artifacts, "source_sha256": sha256(Path(__file__).resolve())
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
