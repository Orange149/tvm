#!/usr/bin/env python3
"""Compare R50A call-site holdouts to expose tile-dependent DMA fragmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_vta_resnet50_callsite_composed_graph_board import verify_build_artifacts
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


FIELDS = (
    "load_buffer_2d_bytes",
    "load_buffer_2d_calls",
    "load_buffer_2d_inp_bytes",
    "load_buffer_2d_inp_calls",
    "load_buffer_2d_wgt_bytes",
    "load_buffer_2d_wgt_calls",
    "load_buffer_2d_acc_bytes",
    "load_buffer_2d_acc_calls",
    "store_buffer_2d_bytes",
    "store_buffer_2d_calls",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def aggregate_occurrences(fused, key):
    return {
        field: sum(row[key].get(field, 0) for row in fused["graph_occurrences"])
        for field in FIELDS
    }


def percent_change(after, before):
    return (after / before - 1.0) * 100.0


def summarize(audit, fused):
    original = aggregate_occurrences(fused, "original_dma")
    residency = aggregate_occurrences(fused, "residency_dma")
    latencies = audit["group_median_latency_ms"]
    return {
        "family": audit["selected_family_id"],
        "knobs": audit["knobs"],
        "original": original,
        "residency": residency,
        "delta": audit["prospective_full_graph_dma_delta"],
        "input_reload_ratio_original_over_residency": (
            original["load_buffer_2d_inp_bytes"]
            / residency["load_buffer_2d_inp_bytes"]
        ),
        "original_average_load_request_bytes": (
            original["load_buffer_2d_bytes"] / original["load_buffer_2d_calls"]
        ),
        "residency_average_load_request_bytes": (
            residency["load_buffer_2d_bytes"] / residency["load_buffer_2d_calls"]
        ),
        "load_bytes_reduction_percent": -percent_change(
            residency["load_buffer_2d_bytes"], original["load_buffer_2d_bytes"]
        ),
        "load_calls_reduction_percent": -percent_change(
            residency["load_buffer_2d_calls"], original["load_buffer_2d_calls"]
        ),
        "latency_ms": latencies,
        "throughput_speedup_percent": audit["group_speedup_percent"],
        "latency_reduction_percent": -percent_change(
            latencies["proposal"], latencies["incumbent"]
        ),
        "paired_wins": audit["group_paired_wins"],
        "paired_rounds": audit["group_paired_rounds"],
        "singleton_selected_mask": audit["singleton_selected_mask"],
        "grouped_selected_mask": audit["grouped_selected_mask"],
        "singleton_model_execution_calls": audit["singleton_model_execution_calls"],
        "group_model_execution_calls": audit["group_model_execution_calls"],
    }


def load_verified(directory, filename):
    directory = Path(directory)
    count = len(verify_build_artifacts(directory))
    return read_json(directory / filename), count


def run(args):
    audit_a, audit_a_count = load_verified(args.audit_a, "summary.json")
    audit_b, audit_b_count = load_verified(args.audit_b, "summary.json")
    fused_a, fused_a_count = load_verified(
        args.fused_a, "fused_tir_occurrence_delta.json"
    )
    fused_b, fused_b_count = load_verified(
        args.fused_b, "fused_tir_occurrence_delta.json"
    )
    for audit in (audit_a, audit_b):
        if audit.get("status") != "callsite_and_full_graph_latency_holdout_pass":
            raise RuntimeError("input is not a completed latency holdout audit")
    for fused in (fused_a, fused_b):
        if fused.get("status") != "compiler_fused_tir_prediction_frozen_before_fpga":
            raise RuntimeError("input is not a prospective fused-TIR contract")
    rows = [summarize(audit_a, fused_a), summarize(audit_b, fused_b)]
    if len({row["family"] for row in rows}) != 2:
        raise RuntimeError("tile comparison requires two distinct families")
    if any(row["singleton_selected_mask"] != row["grouped_selected_mask"] for row in rows):
        raise RuntimeError("singleton and group policies disagree in a source holdout")

    left, right = rows
    cross_mode = {}
    for mode in ("original", "residency"):
        left_latency = left["latency_ms"][
            "incumbent" if mode == "original" else "proposal"
        ]
        right_latency = right["latency_ms"][
            "incumbent" if mode == "original" else "proposal"
        ]
        cross_mode[mode] = {
            "right_vs_left_load_bytes_percent": percent_change(
                right[mode]["load_buffer_2d_bytes"],
                left[mode]["load_buffer_2d_bytes"],
            ),
            "right_vs_left_load_calls_percent": percent_change(
                right[mode]["load_buffer_2d_calls"],
                left[mode]["load_buffer_2d_calls"],
            ),
            "right_vs_left_latency_percent": percent_change(right_latency, left_latency),
        }

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    result = {
        "schema": "c3_vta_r50a_callsite_tile_holdout_comparison_v1",
        "status": "two_callsite_latency_holdouts_expose_tile_dependent_dma_fragmentation",
        "tiles": rows,
        "cross_tile_right_vs_left": cross_mode,
        "key_observation": (
            "The right tile moves fewer LOAD bytes than the left tile in both modes but "
            "remains slower because it emits far more LOAD requests. Input residency cuts "
            "both reload bytes and request fragmentation; bytes alone cannot rank these "
            "two exact full-graph programs"
        ),
        "supported_search_feature": (
            "Use final-fused-program LOAD bytes and LOAD calls jointly in the shared-memory "
            "service proxy, while retaining real FPGA latency as the admission label"
        ),
        "source_hashes": {
            "audit_a_summary": sha256(Path(args.audit_a) / "summary.json"),
            "audit_b_summary": sha256(Path(args.audit_b) / "summary.json"),
            "fused_a": sha256(Path(args.fused_a) / "fused_tir_occurrence_delta.json"),
            "fused_b": sha256(Path(args.fused_b) / "fused_tir_occurrence_delta.json"),
        },
        "verified_artifact_counts": {
            "audit_a": audit_a_count,
            "audit_b": audit_b_count,
            "fused_a": fused_a_count,
            "fused_b": fused_b_count,
        },
        "analyzer_source_sha256": sha256(Path(__file__).resolve()),
        "claim_boundary": (
            "Two tiles of one R50A workload on one ResNet50 graph and boot. This is a "
            "controlled full-graph tile/DMA mechanism comparison, not an independent "
            "workload sample, fitted latency model, or physical AXI measurement"
        ),
    }
    write_json(output / "summary.json", result)
    (output / "command.txt").write_text(
        " ".join(__import__("sys").argv) + "\n", encoding="utf-8"
    )
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-a", required=True)
    parser.add_argument("--fused-a", required=True)
    parser.add_argument("--audit-b", required=True)
    parser.add_argument("--fused-b", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
