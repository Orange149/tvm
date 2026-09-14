#!/usr/bin/env python3
"""Assemble the prospective R50D fused-Pareto wave and full-pool oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percent_saved(part, whole):
    return (1.0 - float(part) / float(whole)) * 100.0


def resource_totals(results):
    totals = {
        "candidate_dispatches": len(results),
        "graph_api_invocations": 0,
        "fpga_kernel_invocations": 0,
        "host_call_wall_ms": 0.0,
        "logical_load_bytes": 0,
        "logical_store_bytes": 0,
        "logical_dma_calls": 0,
    }
    for result in results:
        for row in result.get("correctness", []) + result.get("timings", []):
            profile = row["runtime_profile_complete"]
            totals["graph_api_invocations"] += 1
            totals["fpga_kernel_invocations"] += int(profile["driver_run_calls"])
            totals["host_call_wall_ms"] += float(row["host_wall_ms"])
            totals["logical_load_bytes"] += int(profile["load_buffer_2d_bytes"])
            totals["logical_store_bytes"] += int(profile["store_buffer_2d_bytes"])
            totals["logical_dma_calls"] += int(
                profile["load_buffer_2d_calls"] + profile["store_buffer_2d_calls"]
            )
    totals["logical_dma_bytes"] = (
        totals["logical_load_bytes"] + totals["logical_store_bytes"]
    )
    return totals


def trials_to_target(order, results, oracle_ratio, epsilon):
    for trial, candidate_id in enumerate(order, 1):
        row = results[candidate_id]
        if row["status"] != "passed":
            continue
        if row["median_paired_latency_ratio"] <= (1.0 + epsilon) * oracle_ratio:
            return trial
    return None


def run(args):
    pool_dir = Path(args.pool_dir)
    front_dir = Path(args.front_result)
    completion_dir = Path(args.completion_result)
    local_dir = Path(args.local_qualification)
    verified = {
        "pool": len(verify_artifacts_compatible(pool_dir)),
        "front_result": len(verify_artifacts_compatible(front_dir)),
        "completion_result": len(verify_artifacts_compatible(completion_dir)),
        "local_qualification": len(verify_artifacts_compatible(local_dir)),
    }
    pool = read_json(pool_dir / "pool.json")
    front = read_json(front_dir / "summary.json")
    completion = read_json(completion_dir / "summary.json")
    if pool.get("status") != "final_fused_pareto_front_frozen_before_fpga":
        raise RuntimeError("fused pool is not pristine")
    if front.get("status") != "prospective_final_fused_pareto_front_board_complete":
        raise RuntimeError("prospective Pareto front is incomplete")
    if completion.get("status") != "dominated_final_fused_oracle_completion_board_complete":
        raise RuntimeError("dominated oracle completion is incomplete")
    pool_manifest = sha256(pool_dir / "artifact_hashes.json")
    if front["pool_artifact_manifest_sha256"] != pool_manifest:
        raise RuntimeError("front result binds a different pool")
    if completion["pool_artifact_manifest_sha256"] != pool_manifest:
        raise RuntimeError("completion result binds a different pool")
    if front["candidate_ids"] != pool["pareto_front_candidate_ids"]:
        raise RuntimeError("prospective wave differs from the frozen front")

    program_by_id = {row["candidate_id"]: row for row in pool["programs"]}
    all_results = front["candidate_results"] + completion["candidate_results"]
    result_by_id = {row["candidate_id"]: row for row in all_results}
    if len(result_by_id) != len(all_results):
        raise RuntimeError("duplicate board result identity")
    if set(result_by_id) != set(program_by_id):
        raise RuntimeError("board results do not cover the complete fused pool")
    passed = [row for row in all_results if row["status"] == "passed"]
    rejected = [row for row in all_results if row["status"] != "passed"]
    if not passed:
        raise RuntimeError("complete pool has no FPGA-correct program")

    oracle = min(
        passed,
        key=lambda row: (row["median_paired_latency_ratio"], row["candidate_id"]),
    )
    front_passed = [
        result_by_id[candidate_id] for candidate_id in pool["pareto_front_candidate_ids"]
        if result_by_id[candidate_id]["status"] == "passed"
    ]
    front_best = min(
        front_passed,
        key=lambda row: (row["median_paired_latency_ratio"], row["candidate_id"]),
    )
    front_retains_oracle = front_best["candidate_id"] == oracle["candidate_id"]

    prospective_cost = resource_totals(front["candidate_results"])
    completion_cost = resource_totals(completion["candidate_results"])
    exhaustive_cost = resource_totals(all_results)
    saved = {
        key: percent_saved(prospective_cost[key], exhaustive_cost[key])
        for key in (
            "candidate_dispatches", "graph_api_invocations", "fpga_kernel_invocations",
            "host_call_wall_ms", "logical_dma_bytes", "logical_dma_calls",
        )
    }

    bytes_order = [
        row["candidate_id"] for row in sorted(
            pool["programs"], key=lambda row: (row["dma_bytes"], row["candidate_id"])
        )
    ]
    front_order = list(pool["pareto_front_candidate_ids"])
    time_to_target = {
        name: {
            "exact_oracle": trials_to_target(order, result_by_id, oracle[
                "median_paired_latency_ratio"
            ], 0.0),
            "oracle_plus_2_percent": trials_to_target(order, result_by_id, oracle[
                "median_paired_latency_ratio"
            ], 0.02),
            "oracle_plus_5_percent": trials_to_target(order, result_by_id, oracle[
                "median_paired_latency_ratio"
            ], 0.05),
        }
        for name, order in (("pareto_frozen_order", front_order),
                            ("dma_bytes_fail_fast", bytes_order))
    }

    by_mode = {}
    for mode in ("original", "input_stationary", "weight_resident_barrier"):
        ids = [row["candidate_id"] for row in pool["programs"] if row["public_mode"] == mode]
        by_mode[mode] = {
            "fsim_eligible": len(ids),
            "fpga_correct": sum(result_by_id[cid]["status"] == "passed" for cid in ids),
            "fpga_rejected": sum(result_by_id[cid]["status"] != "passed" for cid in ids),
        }

    same_tile_pairs = []
    families = sorted({row["family_id"] for row in pool["programs"]})
    for family in families:
        programs = {
            row["public_mode"]: row for row in pool["programs"] if row["family_id"] == family
        }
        if "original" not in programs:
            continue
        original_result = result_by_id[programs["original"]["candidate_id"]]
        if original_result["status"] != "passed":
            continue
        for mode in ("input_stationary", "weight_resident_barrier"):
            if mode not in programs:
                continue
            program = programs[mode]
            result = result_by_id[program["candidate_id"]]
            pair = {
                "family_id": family,
                "mode": mode,
                "status": result["status"],
                "dma_bytes_change_percent": (
                    float(program["dma_bytes"]) / programs["original"]["dma_bytes"] - 1.0
                ) * 100.0,
                "dma_calls_change_percent": (
                    float(program["dma_calls"]) / programs["original"]["dma_calls"] - 1.0
                ) * 100.0,
            }
            if result["status"] == "passed":
                pair["paired_ratio_latency_improvement_percent"] = (
                    1.0 - result["median_paired_latency_ratio"] /
                    original_result["median_paired_latency_ratio"]
                ) * 100.0
            same_tile_pairs.append(pair)

    static_rows = read_jsonl(local_dir / "static_results.jsonl")
    fsim_rows = read_jsonl(local_dir / "fsim_results.jsonl")
    local_cost = {
        "static_diagnostic_wall_seconds": sum(
            float(row.get("diagnostic_wall_seconds", 0.0)) for row in static_rows
        ),
        "fsim_diagnostic_wall_seconds": sum(
            float(row.get("diagnostic_wall_seconds", 0.0)) for row in fsim_rows
        ),
        "final_fused_build_wall_seconds": float(pool["total_build_seconds"]),
        "interpretation": (
            "common cost paid before either Pareto or exhaustive FPGA evaluation; "
            "therefore not counted as a Pareto-only saving"
        ),
    }

    candidate_table = []
    front_ids = set(pool["pareto_front_candidate_ids"])
    for program in pool["programs"]:
        result = result_by_id[program["candidate_id"]]
        candidate_table.append({
            "candidate_id": program["candidate_id"],
            "family_id": program["family_id"],
            "public_mode": program["public_mode"],
            "pareto_front": program["candidate_id"] in front_ids,
            "dma_bytes": program["dma_bytes"],
            "dma_calls": program["dma_calls"],
            "extra_submissions": program["extra_submissions"],
            "fpga_status": result["status"],
            "median_paired_latency_ratio": result.get("median_paired_latency_ratio"),
            "candidate_median_latency_ms": result.get("median_latency_ms", {}).get(
                program["candidate_id"]
            ),
        })

    analysis = {
        "schema": "c3_vta_resnet50_final_fused_pareto_holdout_analysis_v1",
        "status": "prospective_final_fused_pareto_retains_full_pool_oracle",
        "workload_id": pool["workload_id"],
        "source_model": pool["source_model"],
        "source_layer": pool["source_layer"],
        "pool_size": len(all_results),
        "pareto_front_size": len(front["candidate_results"]),
        "fpga_correct_count": len(passed),
        "fpga_rejected_count": len(rejected),
        "oracle_candidate_id": oracle["candidate_id"],
        "oracle_family_id": oracle["family_id"],
        "oracle_public_mode": oracle["public_mode"],
        "oracle_median_paired_latency_ratio": oracle["median_paired_latency_ratio"],
        "oracle_candidate_median_latency_ms": oracle["median_latency_ms"][
            oracle["candidate_id"]
        ],
        "front_retains_full_fpga_correct_pool_oracle": front_retains_oracle,
        "front_best_regret_percent": (
            front_best["median_paired_latency_ratio"] /
            oracle["median_paired_latency_ratio"] - 1.0
        ) * 100.0,
        "all_search_candidates_slower_than_stock_reference": all(
            row["median_paired_latency_ratio"] > 1.0 for row in passed
        ),
        "prospective_front_cost": prospective_cost,
        "exhaustive_full_pool_cost": exhaustive_cost,
        "prospective_savings_percent": saved,
        "common_pre_board_cost": local_cost,
        "time_to_target_candidate_dispatches": time_to_target,
        "correctness_by_mode": by_mode,
        "same_tile_pairs": same_tile_pairs,
        "candidate_table": candidate_table,
        "verified_artifact_counts": verified,
        "bound_inputs": {
            "pool_artifact_manifest_sha256": pool_manifest,
            "front_artifact_manifest_sha256": sha256(front_dir / "artifact_hashes.json"),
            "completion_artifact_manifest_sha256": sha256(
                completion_dir / "artifact_hashes.json"
            ),
            "local_artifact_manifest_sha256": sha256(local_dir / "artifact_hashes.json"),
        },
        "analyzer_source_sha256": sha256(Path(__file__).resolve()),
        "claim_boundary": (
            "One prospectively frozen ResNet50 layer-in-full-graph pool on one board boot. "
            "The result validates high-fidelity candidate elimination, not a reduction in "
            "the common all-candidate fused-build cost or superiority to TopHub."
        ),
    }
    if not front_retains_oracle:
        analysis["status"] = "prospective_final_fused_pareto_misses_full_pool_oracle"
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    write_json(output / "analysis.json", analysis)
    (output / "command.txt").write_text(
        " ".join(__import__("sys").argv) + "\n", encoding="utf-8"
    )
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        key: analysis[key] for key in (
            "status", "pool_size", "pareto_front_size", "fpga_correct_count",
            "fpga_rejected_count", "oracle_candidate_id",
            "oracle_median_paired_latency_ratio",
            "front_retains_full_fpga_correct_pool_oracle",
            "prospective_savings_percent", "time_to_target_candidate_dispatches",
        )
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--front-result", required=True)
    parser.add_argument("--completion-result", required=True)
    parser.add_argument("--local-qualification", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
