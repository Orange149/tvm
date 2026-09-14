"""Compute optimistic single-operator VTA compute/DMA lower bounds.

This is an offline analytical model.  It does not contact a board and its
results are not measured latency or end-to-end network FPS.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


SCHEMA = "c3_vta_theoretical_bound_v1"


def load_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def derive_hardware(vta_config, fpga_freq_mhz, axi_data_bits):
    block_in = 1 << int(vta_config.get("LOG_BLOCK_IN", vta_config["LOG_BLOCK"]))
    block_out = 1 << int(vta_config.get("LOG_BLOCK_OUT", vta_config["LOG_BLOCK"]))
    frequency_hz = float(fpga_freq_mhz) * 1e6
    return {
        "block_in": block_in,
        "block_out": block_out,
        "fpga_freq_mhz": float(fpga_freq_mhz),
        "axi_data_bits": int(axi_data_bits),
        "peak_macs_per_second": block_in * block_out * frequency_hz,
        "optimistic_unidirectional_axi_bytes_per_second": int(axi_data_bits)
        * frequency_hz
        / 8.0,
    }


def convolution_geometry(workload):
    if workload[0] != "conv2d_packed.vta":
        raise ValueError("unsupported workload template: {}".format(workload[0]))
    input_shape = workload[1][1]
    weight_shape = workload[2][1]
    strides = workload[3]
    padding = workload[4]
    dilation = workload[5]
    n_outer, ci_outer, height, width, batch, block_in = map(int, input_shape)
    co_outer, weight_ci_outer, kernel_h, kernel_w, block_out, weight_block_in = map(
        int, weight_shape
    )
    if ci_outer != weight_ci_outer or block_in != weight_block_in:
        raise ValueError("packed input/weight channel mismatch")
    stride_h, stride_w = map(int, strides)
    pad_top, pad_left, pad_bottom, pad_right = map(int, padding)
    dilation_h, dilation_w = map(int, dilation)
    effective_kh = (kernel_h - 1) * dilation_h + 1
    effective_kw = (kernel_w - 1) * dilation_w + 1
    output_h = (height + pad_top + pad_bottom - effective_kh) // stride_h + 1
    output_w = (width + pad_left + pad_right - effective_kw) // stride_w + 1
    if output_h <= 0 or output_w <= 0:
        raise ValueError("non-positive convolution output")
    logical_batch = n_outer * batch
    input_channels = ci_outer * block_in
    output_channels = co_outer * block_out
    macs = (
        logical_batch
        * output_h
        * output_w
        * output_channels
        * input_channels
        * kernel_h
        * kernel_w
    )
    return {
        "batch": logical_batch,
        "input_channels": input_channels,
        "output_channels": output_channels,
        "input_height": height,
        "input_width": width,
        "output_height": output_h,
        "output_width": output_w,
        "kernel_height": kernel_h,
        "kernel_width": kernel_w,
        "macs": macs,
    }


def dominance(compute_ms, dma_ms, tolerance=1e-12):
    if math.isclose(compute_ms, dma_ms, rel_tol=tolerance, abs_tol=tolerance):
        return "balanced"
    return "compute" if compute_ms > dma_ms else "dma"


def analyze_workload(row, hardware, workload_id):
    geometry = convolution_geometry(row["workload"])
    unique = row["unique_tensor_bytes"]
    unique_read_bytes = int(unique["input_bytes"]) + int(unique["weight_bytes"])
    unique_write_bytes = int(unique["output_bytes"])
    unique_bytes = unique_read_bytes + unique_write_bytes
    totals = row["static_dma"]["totals"]
    static_load_bytes = int(totals["load_buffer_2d_bytes"])
    static_store_bytes = int(totals["store_buffer_2d_bytes"])
    static_total_bytes = static_load_bytes + static_store_bytes
    peak = hardware["peak_macs_per_second"]
    bandwidth = hardware["optimistic_unidirectional_axi_bytes_per_second"]
    compute_ms = geometry["macs"] / peak * 1000.0
    unique_read_ms = unique_read_bytes / bandwidth * 1000.0
    unique_write_ms = unique_write_bytes / bandwidth * 1000.0
    unique_full_duplex_dma_ms = max(unique_read_ms, unique_write_ms)
    unique_shared_serial_dma_ms = unique_bytes / bandwidth * 1000.0
    static_read_ms = static_load_bytes / bandwidth * 1000.0
    static_write_ms = static_store_bytes / bandwidth * 1000.0
    static_full_duplex_dma_ms = max(static_read_ms, static_write_ms)
    static_shared_serial_dma_ms = static_total_bytes / bandwidth * 1000.0
    return {
        "workload_id": workload_id,
        "config_index": int(row["config_index"]),
        **geometry,
        "peak_macs_per_second": peak,
        "optimistic_axi_bytes_per_second": bandwidth,
        "compute_lower_bound_ms": compute_ms,
        "unique_input_bytes": int(unique["input_bytes"]),
        "unique_weight_bytes": int(unique["weight_bytes"]),
        "unique_output_bytes": int(unique["output_bytes"]),
        "irreducible_unique_total_bytes": unique_bytes,
        "irreducible_full_duplex_read_lower_bound_ms": unique_read_ms,
        "irreducible_full_duplex_write_lower_bound_ms": unique_write_ms,
        "irreducible_full_duplex_dma_lower_bound_ms": unique_full_duplex_dma_ms,
        "irreducible_perfect_overlap_lower_bound_ms": max(
            compute_ms, unique_read_ms, unique_write_ms
        ),
        "irreducible_perfect_overlap_upper_bound_operator_rate_per_second": 1000.0
        / max(compute_ms, unique_read_ms, unique_write_ms),
        "irreducible_no_overlap_reference_ms": compute_ms + unique_read_ms + unique_write_ms,
        "irreducible_bound_dominance": dominance(compute_ms, unique_full_duplex_dma_ms),
        "irreducible_shared_serial_dma_scenario_ms": unique_shared_serial_dma_ms,
        "irreducible_shared_serial_perfect_overlap_scenario_ms": max(
            compute_ms, unique_shared_serial_dma_ms
        ),
        "irreducible_shared_serial_no_overlap_scenario_ms": (
            compute_ms + unique_shared_serial_dma_ms
        ),
        "current_static_load_bytes": static_load_bytes,
        "current_static_store_bytes": static_store_bytes,
        "current_static_total_dma_bytes": static_total_bytes,
        "current_static_traffic_amplification": static_total_bytes / unique_bytes,
        "current_static_full_duplex_read_lower_bound_ms": static_read_ms,
        "current_static_full_duplex_write_lower_bound_ms": static_write_ms,
        "current_static_full_duplex_dma_lower_bound_ms": static_full_duplex_dma_ms,
        "current_static_perfect_overlap_lower_bound_ms": max(
            compute_ms, static_read_ms, static_write_ms
        ),
        "current_static_perfect_overlap_upper_bound_operator_rate_per_second": 1000.0
        / max(compute_ms, static_read_ms, static_write_ms),
        "current_static_no_overlap_reference_ms": compute_ms + static_read_ms + static_write_ms,
        "current_static_bound_dominance": dominance(compute_ms, static_full_duplex_dma_ms),
        "compute_to_current_static_dma_time_ratio": compute_ms / static_full_duplex_dma_ms,
        "perfect_overlap_bound_gain_if_static_traffic_becomes_unique": max(
            compute_ms, static_read_ms, static_write_ms
        )
        / max(compute_ms, unique_read_ms, unique_write_ms),
        "current_static_shared_serial_dma_scenario_ms": static_shared_serial_dma_ms,
        "current_static_shared_serial_perfect_overlap_scenario_ms": max(
            compute_ms, static_shared_serial_dma_ms
        ),
        "current_static_shared_serial_no_overlap_scenario_ms": (
            compute_ms + static_shared_serial_dma_ms
        ),
    }


def load_historical_correct(selected_tasks_path, artifacts_dir, bound_by_workload):
    if not selected_tasks_path or not artifacts_dir:
        return [], {"status": "not_requested"}
    selected = load_json(selected_tasks_path)
    selected_keys = {
        (canonical_key(task["workload"]), int(candidate["config_index"]))
        for task in selected
        for candidate in task["candidates"]
    }
    workload_id_by_key = {
        canonical_key(bound["workload"]): bound["workload_id"] for bound in bound_by_workload
    }
    bound_map = {bound["workload_id"]: bound for bound in bound_by_workload}
    records = []
    duplicate_keys = set()
    seen = set()
    artifact_count = 0
    for path in sorted(Path(artifacts_dir).glob("*/measurement.json")):
        artifact_count += 1
        artifact = load_json(path)
        key = (canonical_key(artifact["workload"]), int(artifact["config"]["index"]))
        if key not in selected_keys:
            continue
        if key in seen:
            duplicate_keys.add(key)
            continue
        seen.add(key)
        if not bool(artifact.get("correct", False)):
            continue
        workload_id = workload_id_by_key.get(key[0])
        if workload_id is None:
            continue
        bound = bound_map[workload_id]
        costs_ms = [float(cost) * 1000.0 for cost in artifact.get("costs_s", [])]
        if not costs_ms:
            continue
        observed_ms = statistics.median(costs_ms)
        invariant_lower = bound["irreducible_perfect_overlap_lower_bound_ms"]
        profile = artifact.get("runtime_profile") or {}
        runtime_dma_available = all(
            name in profile for name in ("load_buffer_2d_bytes", "store_buffer_2d_bytes")
        )
        runtime_dma_bytes = None
        runtime_read_ms = None
        runtime_write_ms = None
        runtime_dma_ms = None
        runtime_lower = None
        runtime_dominance = None
        if runtime_dma_available:
            runtime_load_bytes = int(profile["load_buffer_2d_bytes"])
            runtime_store_bytes = int(profile["store_buffer_2d_bytes"])
            runtime_dma_bytes = runtime_load_bytes + runtime_store_bytes
            runtime_read_ms = runtime_load_bytes / bound["optimistic_axi_bytes_per_second"] * 1000.0
            runtime_write_ms = runtime_store_bytes / bound["optimistic_axi_bytes_per_second"] * 1000.0
            runtime_dma_ms = max(runtime_read_ms, runtime_write_ms)
            runtime_lower = max(bound["compute_lower_bound_ms"], runtime_read_ms, runtime_write_ms)
            runtime_dominance = dominance(bound["compute_lower_bound_ms"], runtime_dma_ms)
        matches_static = int(artifact["config"]["index"]) == bound["config_index"]
        records.append(
            {
                "workload_id": workload_id,
                "config_index": int(artifact["config"]["index"]),
                "matches_frozen_static_config": matches_static,
                "observed_median_latency_ms": observed_ms,
                "observed_effective_gmac_per_second": bound["macs"] / observed_ms / 1e6,
                "compute_peak_utilization_fraction": bound["compute_lower_bound_ms"] / observed_ms,
                "measurement_samples": len(costs_ms),
                "compute_lower_bound_ms": bound["compute_lower_bound_ms"],
                "irreducible_perfect_overlap_lower_bound_ms": invariant_lower,
                "observed_over_irreducible_bound_gap": observed_ms / invariant_lower,
                "irreducible_bound_efficiency_fraction": invariant_lower / observed_ms,
                "post_measurement_runtime_dma_bytes": runtime_dma_bytes,
                "post_measurement_runtime_full_duplex_read_lower_bound_ms": runtime_read_ms,
                "post_measurement_runtime_full_duplex_write_lower_bound_ms": runtime_write_ms,
                "post_measurement_runtime_full_duplex_dma_lower_bound_ms": runtime_dma_ms,
                "post_measurement_runtime_perfect_overlap_lower_bound_ms": runtime_lower,
                "post_measurement_runtime_bound_dominance": runtime_dominance,
                "post_measurement_runtime_bound_efficiency_fraction": (
                    runtime_lower / observed_ms if runtime_lower is not None else None
                ),
                "frozen_static_perfect_overlap_lower_bound_ms": (
                    bound["current_static_perfect_overlap_lower_bound_ms"] if matches_static else None
                ),
                "observed_over_frozen_static_bound_gap": (
                    observed_ms / bound["current_static_perfect_overlap_lower_bound_ms"]
                    if matches_static
                    else None
                ),
                "artifact_path": str(path),
                "evidence_scope": "historical_board_measurement_joined_offline",
            }
        )
    records.sort(key=lambda item: (item["workload_id"], item["config_index"]))
    return records, {
        "status": "completed",
        "artifacts_scanned": artifact_count,
        "selected_candidate_keys": len(selected_keys),
        "selected_artifact_keys_seen": len(seen),
        "correct_records": len(records),
        "duplicate_selected_artifact_keys": len(duplicate_keys),
    }


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0]) if records else []
    with open(path, "w", encoding="utf-8", newline="") as stream:
        if fieldnames:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def mean(values):
    return sum(values) / len(values) if values else None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-dma", required=True)
    parser.add_argument("--vta-config", required=True)
    parser.add_argument("--fpga-freq-mhz", required=True, type=float)
    parser.add_argument("--axi-data-bits", required=True, type=int)
    parser.add_argument("--selected-tasks")
    parser.add_argument("--artifacts-dir")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    static = load_json(args.static_dma)
    config = load_json(args.vta_config)
    hardware = derive_hardware(config, args.fpga_freq_mhz, args.axi_data_bits)
    bounds = []
    bound_with_workload = []
    for ordinal, row in enumerate(static["rows"]):
        analyzed = analyze_workload(row, hardware, "W{:02d}".format(ordinal))
        bounds.append(analyzed)
        bound_with_workload.append({**analyzed, "workload": row["workload"]})
    historical, join = load_historical_correct(
        args.selected_tasks, args.artifacts_dir, bound_with_workload
    )
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "theoretical_bounds.csv", bounds)
    write_json(output_dir / "theoretical_bounds.json", {"schema": SCHEMA, "rows": bounds})
    write_csv(output_dir / "historical_correct_candidates.csv", historical)
    write_json(
        output_dir / "historical_correct_candidates.json",
        {"schema": SCHEMA, "records": historical, "join_diagnostics": join},
    )
    matched = [item for item in historical if item["matches_frozen_static_config"]]
    summary = {
        "schema": SCHEMA,
        "status": "completed_offline_analytical_model",
        "scope": "idealized_single_operator_bounds_not_end_to_end_network_fps",
        "hardware": hardware,
        "assumptions": {
            "compute_peak": "BLOCK_IN * BLOCK_OUT * FPGA_FREQ; one MAC counted per lane per cycle",
            "axi_bandwidth": "AXI_DATA_BITS * FPGA_FREQ / 8; optimistic ideal unidirectional payload bandwidth",
            "optimistic_full_duplex_perfect_overlap": "max(T_compute, T_read, T_write)",
            "optimistic_full_duplex_no_overlap_reference": "T_compute + T_read + T_write",
            "shared_bandwidth_scenario": "T_dma_shared=(read_bytes+write_bytes)/B; reported separately because it assumes one shared serialized bottleneck",
            "excluded_overheads": [
                "AXI protocol/turnaround/contention",
                "DMA setup and request granularity",
                "instruction/uop fetch",
                "pipeline fill/drain and dependencies",
                "CPU/runtime/RPC/framework overhead",
                "whole-network inter-operator effects",
            ],
        },
        "workloads": len(bounds),
        "irreducible_dominance_counts": {
            name: sum(item["irreducible_bound_dominance"] == name for item in bounds)
            for name in ("compute", "dma", "balanced")
        },
        "current_static_dominance_counts": {
            name: sum(item["current_static_bound_dominance"] == name for item in bounds)
            for name in ("compute", "dma", "balanced")
        },
        "current_static_traffic_amplification": {
            "minimum": min(item["current_static_traffic_amplification"] for item in bounds),
            "median": statistics.median(
                item["current_static_traffic_amplification"] for item in bounds
            ),
            "maximum": max(item["current_static_traffic_amplification"] for item in bounds),
        },
        "closest_current_static_full_duplex_balance": min(
            (
                {
                    "workload_id": item["workload_id"],
                    "compute_to_dma_time_ratio": item[
                        "compute_to_current_static_dma_time_ratio"
                    ],
                }
                for item in bounds
            ),
            key=lambda item: abs(math.log(item["compute_to_dma_time_ratio"])),
        ),
        "workloads_with_perfect_overlap_bound_gain_from_static_to_unique": sum(
            item["perfect_overlap_bound_gain_if_static_traffic_becomes_unique"] > 1.0 + 1e-12
            for item in bounds
        ),
        "historical_correct_association": {
            **join,
            "matched_frozen_static_config_records": len(matched),
            "median_irreducible_bound_efficiency_fraction": (
                statistics.median(
                    item["irreducible_bound_efficiency_fraction"] for item in historical
                )
                if historical
                else None
            ),
            "median_observed_over_irreducible_bound_gap": (
                statistics.median(item["observed_over_irreducible_bound_gap"] for item in historical)
                if historical
                else None
            ),
            "maximum_observed_effective_gmac_per_second": (
                max(item["observed_effective_gmac_per_second"] for item in historical)
                if historical
                else None
            ),
            "maximum_compute_peak_utilization_fraction": (
                max(item["compute_peak_utilization_fraction"] for item in historical)
                if historical
                else None
            ),
            "matched_frozen_static_config_median_compute_peak_utilization_fraction": (
                statistics.median(item["compute_peak_utilization_fraction"] for item in matched)
                if matched
                else None
            ),
            "post_measurement_runtime_dominance_counts": {
                name: sum(item["post_measurement_runtime_bound_dominance"] == name for item in historical)
                for name in ("compute", "dma", "balanced")
            },
            "warning": "latencies/runtime counters are historical board observations; they are not generated or remeasured by this run",
        },
        "sources": {
            "static_dma": {"path": args.static_dma, "sha256": sha256_file(args.static_dma)},
            "vta_config": {"path": args.vta_config, "sha256": sha256_file(args.vta_config)},
            "selected_tasks": (
                {"path": args.selected_tasks, "sha256": sha256_file(args.selected_tasks)}
                if args.selected_tasks
                else None
            ),
            "artifacts_dir": args.artifacts_dir,
        },
        "board_accessed": False,
        "performance_measured_by_this_run": False,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
