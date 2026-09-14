#!/usr/bin/env python3
"""Bind independent frozen-control runs to a completed peeling holdout oracle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def summarize_run(name, directory, expected_ids, oracle_id, quality):
    verified = verify_artifacts_compatible(directory)
    summary = read(directory / "summary.json")
    actual_ids = summary["dispatched_candidate_ids"]
    if actual_ids != expected_ids:
        raise ValueError(f"{name} dispatch order differs from frozen prefix")
    passed = [row for row in summary["candidate_results"] if row["status"] == "passed"]
    invalid = [row for row in summary["candidate_results"] if row["status"] != "passed"]
    best = min(passed, key=lambda row: row["median_paired_latency_ratio"]) if passed else None
    elapsed_by_id = {row["candidate_id"]: row["outer_elapsed_seconds"] for row in passed}
    return {
        "strategy": name,
        "artifact_manifest_sha256": sha256(directory / "artifact_hashes.json"),
        "verified_artifact_count": len(verified),
        "boot_id": summary["boot_id"],
        "dispatch_count": summary["dispatch_count"],
        "dispatched_candidate_ids": actual_ids,
        "passed_count": len(passed),
        "invalid_count": len(invalid),
        "correctness_invocations": summary["correctness_invocations"],
        "timing_invocations": summary["timing_invocations"],
        "complete_candidate_build_action_seconds": sum(
            row["complete_candidate_build_action_seconds"] for row in summary["built_programs"]
        ),
        "outer_process_seconds_before_summary_write": summary[
            "outer_process_seconds_before_summary_write"
        ],
        "best_measured_candidate_id": best["candidate_id"] if best else None,
        "best_fresh_median_paired_latency_ratio": (
            best["median_paired_latency_ratio"] if best else None
        ),
        "exact_oracle_measured": oracle_id in summary["passed_candidate_ids"],
        "pool_oracle_plus_2pct_at_prefix": quality["oracle_plus_2pct"],
        "pool_exact_oracle_at_prefix": quality["exact_oracle"],
        "pool_oracle_plus_2pct_first_hit_candidate_id": quality["oracle_plus_2pct_id"],
        "pool_oracle_plus_2pct_first_hit_outer_elapsed_seconds": elapsed_by_id.get(
            quality["oracle_plus_2pct_id"]
        ),
        "pool_exact_oracle_first_hit_outer_elapsed_seconds": elapsed_by_id.get(oracle_id),
        "quality_source": "completed-pool candidate identity labels, not an online stopping oracle",
    }


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    front_dir = Path(args.front_contract)
    analysis_dir = Path(args.pool_analysis)
    verify_artifacts_compatible(front_dir)
    verify_artifacts_compatible(analysis_dir)
    front = read(front_dir / "front.json")
    analysis = read(analysis_dir / "analysis.json")
    if front["workload_id"] != analysis["workload_id"]:
        raise ValueError("workload mismatch")
    oracle_id = analysis["oracle"]["candidate_id"]
    budget = args.candidate_budget
    if budget <= 0:
        raise ValueError("candidate budget must be positive")

    sources = {
        "peeling": Path(args.peeling),
        "calls_lazy": Path(args.calls_lazy),
        "bytes_lazy": Path(args.bytes_lazy),
        "random_lazy": Path(args.random_lazy),
    }
    rows = []
    for name, directory in sources.items():
        if name == "peeling":
            expected = front["wave0_candidate_ids"][:budget]
            replay = analysis["frozen_order_replay"]["online_peeling"]["targets"]
            quality = {
                "oracle_plus_2pct": replay["oracle_plus_2pct"]["dispatches"] <= budget,
                "exact_oracle": replay["exact_oracle"]["dispatches"] <= budget,
                "oracle_plus_2pct_id": replay["oracle_plus_2pct"]["candidate_id"]
                if replay["oracle_plus_2pct"]["dispatches"] <= budget else None,
            }
        else:
            expected = front["control_orders"][name][:budget]
            replay = analysis["frozen_order_replay"][name]["targets"]
            quality = {
                "oracle_plus_2pct": replay["oracle_plus_2pct"]["dispatches"] <= budget,
                "exact_oracle": replay["exact_oracle"]["dispatches"] <= budget,
                "oracle_plus_2pct_id": replay["oracle_plus_2pct"]["candidate_id"]
                if replay["oracle_plus_2pct"]["dispatches"] <= budget else None,
            }
        rows.append(summarize_run(name, directory, expected, oracle_id, quality))

    boots = {row["boot_id"] for row in rows}
    if len(boots) != 1:
        raise ValueError("control runs are not from one boot")
    peeling_wall = rows[0]["outer_process_seconds_before_summary_write"]
    for row in rows:
        row["outer_wall_delta_vs_peeling_percent"] = 100.0 * (
            row["outer_process_seconds_before_summary_write"] / peeling_wall - 1.0
        )
    result = {
        "schema": "c3_invalid_dominator_online_control_comparison_v1",
        "status": "independent_frozen_control_runs_complete",
        "workload_id": analysis["workload_id"],
        "candidate_budget": budget,
        "boot_id": next(iter(boots)),
        "oracle_candidate_id": oracle_id,
        "runs": rows,
        "bound_front_manifest": sha256(front_dir / "artifact_hashes.json"),
        "bound_pool_analysis_manifest": sha256(analysis_dir / "artifact_hashes.json"),
        "analyzer_source_sha256": sha256(Path(__file__).resolve()),
        "claim_boundary": (
            "Control orders and budgets were frozen before target labels. Candidate quality is joined "
            "post hoc by immutable identity after oracle completion; "
            "no pool oracle was used for execution or stopping. Equal dispatch count does not imply equal "
            "FPGA invocations because invalid candidates fail closed before timing."
        ),
    }
    output.mkdir(parents=True)
    write_json(output / "summary.json", result)
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        path.name: sha256(path) for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-contract", required=True, type=Path)
    parser.add_argument("--pool-analysis", required=True, type=Path)
    parser.add_argument("--peeling", required=True, type=Path)
    parser.add_argument("--calls-lazy", required=True, type=Path)
    parser.add_argument("--bytes-lazy", required=True, type=Path)
    parser.add_argument("--random-lazy", required=True, type=Path)
    parser.add_argument("--candidate-budget", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
