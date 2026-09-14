#!/usr/bin/env python3
"""Develop a cheap operator-TIR Pareto proxy from the exposed R50D pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from build_vta_resnet50_fused_program_pool import pareto_front
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def operator_proxy_program(program, static):
    descriptors = static["transfer_signature"]["descriptor_aggregates"]
    return {
        "candidate_id": program["candidate_id"],
        "family_id": program["family_id"],
        "public_mode": program["public_mode"],
        "dma_bytes": int(descriptors["expanded_bytes"]),
        "dma_calls": int(descriptors["expanded_calls"]),
        "extra_submissions": int(program["public_mode"] == "weight_resident_barrier"),
    }


def run(args):
    local_dir = Path(args.local_qualification)
    pool_dir = Path(args.fused_pool)
    board_dir = Path(args.board_analysis)
    verified = {
        "local": len(verify_artifacts_compatible(local_dir)),
        "fused_pool": len(verify_artifacts_compatible(pool_dir)),
        "board_analysis": len(verify_artifacts_compatible(board_dir)),
    }
    pool = read_json(pool_dir / "pool.json")
    board = read_json(board_dir / "analysis.json")
    if board.get("status") != "prospective_final_fused_pareto_retains_full_pool_oracle":
        raise RuntimeError("R50D board oracle analysis is incomplete")
    static_by_id = {row["candidate_id"]: row
                    for row in read_jsonl(local_dir / "static_results.jsonl")}
    proxy_programs = [
        operator_proxy_program(program, static_by_id[program["candidate_id"]])
        for program in pool["programs"]
    ]
    proxy_front = sorted(pareto_front(proxy_programs), key=lambda row: row["candidate_id"])
    proxy_ids = [row["candidate_id"] for row in proxy_front]
    exact_ids = list(pool["pareto_front_candidate_ids"])
    table = {row["candidate_id"]: row for row in board["candidate_table"]}
    oracle = board["oracle_candidate_id"]
    correct_proxy_ids = [cid for cid in proxy_ids if table[cid]["fpga_status"] == "passed"]
    proxy_build_seconds = pool["stock_reference"]["build_seconds"] + sum(
        row["build_seconds"] for row in pool["programs"] if row["candidate_id"] in proxy_ids
    )
    analysis = {
        "schema": "c3_vta_operator_proxy_pareto_development_v1",
        "status": "posthoc_operator_proxy_retains_r50d_oracle_rule_ready_for_new_holdout",
        "label_exposure": {
            "r50d_full_graph_latency_exposed": True,
            "r50d_fpga_correctness_exposed": True,
            "interpretation": "development only; not prospective evidence",
        },
        "proxy_axes": [
            "operator_lowered_tir_expanded_load_plus_store_bytes",
            "operator_lowered_tir_expanded_load_plus_store_calls",
            "weight_barrier_indicator",
        ],
        "proxy_front_candidate_ids": proxy_ids,
        "proxy_front_size": len(proxy_ids),
        "exact_fused_front_candidate_ids": exact_ids,
        "exact_fused_front_size": len(exact_ids),
        "front_overlap": sorted(set(proxy_ids) & set(exact_ids)),
        "proxy_only": sorted(set(proxy_ids) - set(exact_ids)),
        "exact_only": sorted(set(exact_ids) - set(proxy_ids)),
        "oracle_candidate_id": oracle,
        "proxy_retains_oracle": oracle in proxy_ids,
        "proxy_fpga_correct_count": len(correct_proxy_ids),
        "proxy_fpga_correct_candidate_ids": correct_proxy_ids,
        "candidate_fullgraph_builds": {
            "proxy": len(proxy_ids),
            "exhaustive": len(pool["programs"]),
            "reduction_percent": (1.0 - len(proxy_ids) / len(pool["programs"])) * 100.0,
        },
        "posthoc_actual_build_wall_seconds": {
            "stock_plus_proxy": proxy_build_seconds,
            "stock_plus_exhaustive": pool["total_build_seconds"],
            "reduction_percent": (
                1.0 - proxy_build_seconds / pool["total_build_seconds"]
            ) * 100.0,
            "interpretation": "replay using exposed per-build costs, not prospective wall time",
        },
        "next_holdout_rule": (
            "After real operator lowering and three-seed FSim, form the three-axis operator "
            "proxy front without target latency, build final full graphs only for that front, "
            "then run FPGA correctness/latency; build and measure the remainder only after "
            "the prospective wave to reveal the complete oracle."
        ),
        "verified_artifact_counts": verified,
        "bound_inputs": {
            "local_artifact_manifest_sha256": sha256(local_dir / "artifact_hashes.json"),
            "fused_pool_artifact_manifest_sha256": sha256(pool_dir / "artifact_hashes.json"),
            "board_analysis_artifact_manifest_sha256": sha256(
                board_dir / "artifact_hashes.json"
            ),
        },
        "analyzer_source_sha256": sha256(Path(__file__).resolve()),
        "claim_boundary": (
            "R50D labels are exposed; this analysis only chooses a cheaper rule for a future "
            "workload and cannot be counted as another holdout or build-cost result."
        ),
    }
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
            "status", "proxy_front_size", "exact_fused_front_size", "front_overlap",
            "proxy_only", "exact_only", "proxy_retains_oracle",
            "proxy_fpga_correct_count", "candidate_fullgraph_builds",
            "posthoc_actual_build_wall_seconds",
        )
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-qualification", required=True)
    parser.add_argument("--fused-pool", required=True)
    parser.add_argument("--board-analysis", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
