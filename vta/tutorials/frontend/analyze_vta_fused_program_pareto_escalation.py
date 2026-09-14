#!/usr/bin/env python3
"""Post-hoc audit of Pareto-safe escalation for final fused programs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_vta_fused_program_service_proxy_norefit import (
    service_score,
    total_dma,
    verify_artifacts_compatible,
)
from prepare_vta_fused_program_reranker_holdout import aggregate_residency
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dominates(left, right):
    no_worse = left["dma_bytes"] <= right["dma_bytes"] and left["dma_calls"] <= right["dma_calls"]
    strictly_better = left["dma_bytes"] < right["dma_bytes"] or left["dma_calls"] < right["dma_calls"]
    return no_worse and strictly_better


def pareto_front(programs):
    return [
        row for row in programs
        if not any(dominates(other, row) for other in programs if other is not row)
    ]


def evaluate_group(name, programs):
    oracle = min(programs, key=lambda row: (row["latency_ms"], row["program_id"]))
    bytes_pick = min(programs, key=lambda row: (row["dma_bytes"], row["program_id"]))
    service_pick = min(
        programs,
        key=lambda row: (row["service_score_byte_equivalent"], row["program_id"]),
    )
    front = pareto_front(programs)
    front_ids = sorted(row["program_id"] for row in front)
    return {
        "group": name,
        "candidate_count": len(programs),
        "programs": programs,
        "oracle_program": oracle["program_id"],
        "bytes_only_program": bytes_pick["program_id"],
        "service_proxy_program": service_pick["program_id"],
        "pareto_front": front_ids,
        "pareto_front_size": len(front),
        "oracle_retained": oracle["program_id"] in front_ids,
        "action": "static_select" if len(front) == 1 else "escalate_all_frontier_points",
        "fpga_measurements_required_for_full_frontier_resolution": len(front),
    }


def fused_program(contract, program_id, latency, request_cost, submission_cost):
    traffic = aggregate_residency(contract)
    dma = total_dma(traffic)
    return {
        "program_id": program_id,
        "candidate_id": contract["candidate_ids"][1],
        "dma_bytes": dma["bytes"],
        "dma_calls": dma["calls"],
        "service_score_byte_equivalent": service_score(
            traffic, request_cost, 0, submission_cost
        ),
        "latency_ms": float(latency),
    }


def run(args):
    p315_dir = Path(args.p7r315)
    fused_dirs = [Path(value) for value in args.r50c_fused]
    p312_dir = Path(args.p7r312)
    p326_dir = Path(args.p7r326)
    verified = {
        "p7r315": len(verify_artifacts_compatible(p315_dir)),
        "r50c_fused": [len(verify_artifacts_compatible(path)) for path in fused_dirs],
        "p7r312": len(verify_artifacts_compatible(p312_dir)),
        "p7r326": len(verify_artifacts_compatible(p326_dir)),
    }
    p315 = read_json(p315_dir / "analysis.json")
    fused = [read_json(path / "fused_tir_occurrence_delta.json") for path in fused_dirs]
    p312 = read_json(p312_dir / "summary.json")
    p326 = read_json(p326_dir / "summary.json")
    if p315.get("status") != "frozen_request_penalty_repairs_two_byte_only_tile_inversions":
        raise RuntimeError("P7R315 analysis is incomplete")
    if p312.get("status") != "fused_tir_predicted_pair_correctness_dma_and_timing_complete":
        raise RuntimeError("P7R312 board result is incomplete")
    if p326.get("status") != "prospective_fused_service_reranker_board_evaluation_complete":
        raise RuntimeError("P7R326 board result is incomplete")
    if len(fused) != 3 or any(
        row.get("status") not in {
            "compiler_fused_tir_prediction_frozen_before_fpga",
            "compiler_fused_tir_prediction_matches_fpga_profile_exactly",
        } for row in fused
    ):
        raise RuntimeError("R50C fused-program inputs are incomplete")

    request_cost = int(p315["request_equivalent_bytes"])
    submission_cost = int(p315["extra_submission_equivalent_bytes"])
    r50a = p315["programs"]
    r50a_original = [row for row in r50a if row["mode"] == "original"]
    r50a_residency = [row for row in r50a if row["mode"] == "residency"]
    r50c = [
        fused_program(
            fused[0], "R50CF00:input_stationary",
            p312["median_latency_ms"]["input_stationary"], request_cost, submission_cost,
        ),
        fused_program(
            fused[1], "R50CF01:input_stationary",
            p326["median_latency_ms"]["tile_a:input_stationary"], request_cost, submission_cost,
        ),
        fused_program(
            fused[2], "R50CF02:input_stationary",
            p326["median_latency_ms"]["tile_b:input_stationary"], request_cost, submission_cost,
        ),
    ]
    groups = [
        evaluate_group("R50A_original_two_tile", r50a_original),
        evaluate_group("R50A_residency_two_tile", r50a_residency),
        evaluate_group("R50C_input_three_tile", r50c),
        evaluate_group("R50C_input_F01_F02_conflict", r50c[1:]),
    ]
    if not all(row["oracle_retained"] for row in groups):
        raise RuntimeError("Pareto escalation discarded an observed oracle")
    if [row["pareto_front_size"] for row in groups] != [2, 2, 1, 2]:
        raise RuntimeError("frozen Pareto audit topology changed")

    result = {
        "schema": "c3_vta_fused_program_pareto_escalation_analysis_v1",
        "status": "posthoc_pareto_front_retains_four_oracles_and_exposes_conflict_cost",
        "groups": groups,
        "aggregate": {
            "groups": len(groups),
            "oracle_retained": sum(row["oracle_retained"] for row in groups),
            "singleton_static_decisions": sum(row["pareto_front_size"] == 1 for row in groups),
            "conflict_escalations": sum(row["pareto_front_size"] > 1 for row in groups),
            "bytes_only_oracle_hits": sum(row["bytes_only_program"] == row["oracle_program"] for row in groups),
            "service_proxy_oracle_hits": sum(row["service_proxy_program"] == row["oracle_program"] for row in groups),
        },
        "rule": (
            "Eliminate a program only if another program is no worse in fused LOAD+STORE "
            "bytes and calls and strictly better in at least one; otherwise escalate every "
            "non-dominated point to the next fidelity. Submission count is an additional "
            "axis when it differs."
        ),
        "bound_inputs": {
            "p7r315_analysis_sha256": sha256(p315_dir / "analysis.json"),
            "r50c_fused_sha256": [
                sha256(path / "fused_tir_occurrence_delta.json") for path in fused_dirs
            ],
            "p7r312_summary_sha256": sha256(p312_dir / "summary.json"),
            "p7r326_summary_sha256": sha256(p326_dir / "summary.json"),
        },
        "verified_artifact_counts": verified,
        "analyzer_source_sha256": sha256(Path(__file__).resolve()),
        "board_contacted": False,
        "claim_boundary": (
            "Post-hoc design audit over exposed R50A/R50C labels. Four comparison groups "
            "are overlapping and are not independent trials; oracle containment does not "
            "prove future latency dominance or search-cost benefit."
        ),
    }
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    write_json(output / "analysis.json", result)
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
    parser.add_argument("--p7r315", required=True)
    parser.add_argument("--r50c-fused", nargs=3, required=True)
    parser.add_argument("--p7r312", required=True)
    parser.add_argument("--p7r326", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
