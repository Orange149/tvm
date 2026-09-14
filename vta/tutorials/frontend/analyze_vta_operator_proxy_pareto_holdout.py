#!/usr/bin/env python3
"""Audit a prospective operator-proxy front against its completed FPGA oracle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from analyze_vta_resnet50_fused_program_pareto_holdout import (
    percent_saved,
    resource_totals,
    trials_to_target,
)
from build_vta_resnet50_fused_program_pool import pareto_front
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def run(args):
    front_contract_dir = Path(args.proxy_front_contract)
    front_pool_dir = Path(args.front_pool)
    front_result_dir = Path(args.front_result)
    complete_pool_dir = Path(args.complete_pool)
    completion_result_dir = Path(args.completion_result)
    local_dir = Path(args.local_qualification)
    failed_attempt_dir = Path(args.failed_prelabel_attempt) if args.failed_prelabel_attempt else None
    verified = {
        "proxy_front_contract": len(verify_artifacts_compatible(front_contract_dir)),
        "front_pool": len(verify_artifacts_compatible(front_pool_dir)),
        "front_result": len(verify_artifacts_compatible(front_result_dir)),
        "complete_pool": len(verify_artifacts_compatible(complete_pool_dir)),
        "completion_result": len(verify_artifacts_compatible(completion_result_dir)),
        "local_qualification": len(verify_artifacts_compatible(local_dir)),
    }
    if failed_attempt_dir:
        verified["failed_prelabel_attempt"] = len(
            verify_artifacts_compatible(failed_attempt_dir)
        )
    front_contract = read_json(front_contract_dir / (
        "front.json" if (front_contract_dir / "front.json").is_file() else "contract.json"
    ))
    front_pool = read_json(front_pool_dir / "pool.json")
    front = read_json(front_result_dir / "summary.json")
    pool = read_json(complete_pool_dir / "pool.json")
    completion = read_json(completion_result_dir / "summary.json")
    if front_contract.get("status") not in {
        "operator_proxy_front_frozen_before_fullgraph_fpga_or_latency",
        "wave0_and_controls_frozen_before_fullgraph_fpga_or_latency",
    }:
        raise RuntimeError("proxy front contract mismatch")
    if front_pool.get("status") != "operator_proxy_front_fullgraphs_built_before_fpga":
        raise RuntimeError("front pool mismatch")
    if front.get("status") != "prospective_final_fused_pareto_front_board_complete":
        raise RuntimeError("prospective front result incomplete")
    if pool.get("status") != "operator_proxy_oracle_completion_pool_after_front_labels":
        raise RuntimeError("complete pool mismatch")
    if completion.get("status") != "dominated_final_fused_oracle_completion_board_complete":
        raise RuntimeError("oracle completion result incomplete")

    front_pool_manifest = sha256(front_pool_dir / "artifact_hashes.json")
    complete_pool_manifest = sha256(complete_pool_dir / "artifact_hashes.json")
    front_result_manifest = sha256(front_result_dir / "artifact_hashes.json")
    if front["pool_artifact_manifest_sha256"] != front_pool_manifest:
        raise RuntimeError("front result/pool mismatch")
    if pool["prospective_front_source_artifact_manifest_sha256"] != front_pool_manifest:
        raise RuntimeError("complete pool/front pool mismatch")
    if pool["prospective_front_board_artifact_manifest_sha256"] != front_result_manifest:
        raise RuntimeError("complete pool/front result mismatch")
    if completion["pool_artifact_manifest_sha256"] != complete_pool_manifest:
        raise RuntimeError("completion result/complete pool mismatch")
    front_ids = list(front_contract.get(
        "proxy_front_candidate_ids", front_contract.get("wave0_candidate_ids", [])
    ))
    if front_ids != front_pool["pareto_front_candidate_ids"] or front_ids != front["candidate_ids"]:
        raise RuntimeError("prospective front identity drift")
    if front_ids != pool["pareto_front_candidate_ids"]:
        raise RuntimeError("complete pool front identity drift")

    program_by_id = {row["candidate_id"]: row for row in pool["programs"]}
    all_results = front["candidate_results"] + completion["candidate_results"]
    result_by_id = {row["candidate_id"]: row for row in all_results}
    if len(result_by_id) != len(all_results) or set(result_by_id) != set(program_by_id):
        raise RuntimeError("board results do not cover the complete pool exactly once")
    passed = [row for row in all_results if row["status"] == "passed"]
    rejected = [row for row in all_results if row["status"] != "passed"]
    oracle = min(passed, key=lambda row: (
        row["median_paired_latency_ratio"], row["candidate_id"]
    ))
    front_passed = [result_by_id[cid] for cid in front_ids
                    if result_by_id[cid]["status"] == "passed"]
    if not front_passed:
        raise RuntimeError("prospective front has no FPGA-correct candidate")
    front_best = min(front_passed, key=lambda row: (
        row["median_paired_latency_ratio"], row["candidate_id"]
    ))
    retains_oracle = front_best["candidate_id"] == oracle["candidate_id"]
    front_best_regret_percent = (
        front_best["median_paired_latency_ratio"] / oracle["median_paired_latency_ratio"] - 1.0
    ) * 100.0

    prospective_cost = resource_totals(front["candidate_results"])
    exhaustive_cost = resource_totals(all_results)
    board_saved = {
        key: percent_saved(prospective_cost[key], exhaustive_cost[key])
        for key in (
            "candidate_dispatches", "graph_api_invocations", "fpga_kernel_invocations",
            "host_call_wall_ms", "logical_dma_bytes", "logical_dma_calls",
        )
    }
    build_cost = dict(pool["build_cost"])
    build_cost["model_build_wall_savings_percent"] = percent_saved(
        build_cost["prospective_stock_plus_front_seconds"],
        build_cost["counterfactual_stock_plus_all_candidate_model_build_seconds"],
    )
    build_cost["candidate_build_count_savings_percent"] = percent_saved(
        len(front_ids), len(pool["programs"])
    )
    build_cost["stock_plus_candidate_build_count_savings_percent"] = percent_saved(
        1 + len(front_ids), 1 + len(pool["programs"])
    )

    proxy_programs = front_contract["operator_proxy_programs"]
    proxy_by_id = {row["candidate_id"]: row for row in proxy_programs}
    operator_bytes_order = [row["candidate_id"] for row in sorted(
        proxy_programs, key=lambda row: (row["dma_bytes"], row["candidate_id"])
    )]
    exact_fused_front = sorted(pareto_front(pool["programs"]),
                               key=lambda row: row["candidate_id"])
    exact_fused_front_ids = [row["candidate_id"] for row in exact_fused_front]
    oracle_ratio = oracle["median_paired_latency_ratio"]
    search_orders = {
        "operator_proxy_front_frozen_order": front_ids,
        "operator_dma_bytes_fail_fast": operator_bytes_order,
        "exact_fused_dma_bytes_posthoc": [row["candidate_id"] for row in sorted(
            pool["programs"], key=lambda row: (row["dma_bytes"], row["candidate_id"])
        )],
    }
    if front_contract.get("schema") == "c3_invalid_dominator_peeling_front_v1":
        search_orders.update({
            name: list(order)
            for name, order in front_contract["control_orders"].items()
        })
        search_orders["invalid_dominator_peeling_online"] = list(front_ids)
    time_to_target = {
        name: {
            "exact_oracle": trials_to_target(order, result_by_id, oracle_ratio, 0.0),
            "oracle_plus_2_percent": trials_to_target(order, result_by_id, oracle_ratio, 0.02),
            "oracle_plus_5_percent": trials_to_target(order, result_by_id, oracle_ratio, 0.05),
        } for name, order in search_orders.items()
    }

    families = sorted({row["family_id"] for row in pool["programs"]})
    same_tile_pairs = []
    for family in families:
        programs = {row["public_mode"]: row for row in pool["programs"]
                    if row["family_id"] == family}
        if "original" not in programs:
            continue
        original = programs["original"]
        original_result = result_by_id[original["candidate_id"]]
        if original_result["status"] != "passed":
            continue
        for mode in ("input_stationary", "weight_resident_barrier"):
            if mode not in programs:
                continue
            program = programs[mode]
            result = result_by_id[program["candidate_id"]]
            row = {
                "family_id": family,
                "mode": mode,
                "status": result["status"],
                "exact_fused_dma_bytes_change_percent": (
                    float(program["dma_bytes"]) / original["dma_bytes"] - 1.0
                ) * 100.0,
                "exact_fused_dma_calls_change_percent": (
                    float(program["dma_calls"]) / original["dma_calls"] - 1.0
                ) * 100.0,
            }
            if result["status"] == "passed":
                row["paired_ratio_latency_improvement_percent"] = (
                    1.0 - result["median_paired_latency_ratio"] /
                    original_result["median_paired_latency_ratio"]
                ) * 100.0
            same_tile_pairs.append(row)

    static_rows = read_jsonl(local_dir / "static_results.jsonl")
    fsim_rows = read_jsonl(local_dir / "fsim_results.jsonl")
    common_cost = {
        "static_diagnostic_wall_seconds": sum(
            float(row.get("diagnostic_wall_seconds", 0.0)) for row in static_rows
        ),
        "fsim_diagnostic_wall_seconds": sum(
            float(row.get("diagnostic_wall_seconds", 0.0)) for row in fsim_rows
        ),
        "interpretation": "paid before the proxy front and never counted as saved",
    }

    candidate_table = []
    front_set = set(front_ids)
    exact_set = set(exact_fused_front_ids)
    for program in pool["programs"]:
        cid = program["candidate_id"]
        result = result_by_id[cid]
        proxy = proxy_by_id[cid]
        candidate_table.append({
            "candidate_id": cid,
            "family_id": program["family_id"],
            "public_mode": program["public_mode"],
            "operator_proxy_front": cid in front_set,
            "exact_fused_pareto_front_posthoc": cid in exact_set,
            "operator_dma_bytes": proxy["dma_bytes"],
            "operator_dma_calls": proxy["dma_calls"],
            "exact_fused_dma_bytes": program["dma_bytes"],
            "exact_fused_dma_calls": program["dma_calls"],
            "extra_submissions": program["extra_submissions"],
            "fpga_status": result["status"],
            "median_paired_latency_ratio": result.get("median_paired_latency_ratio"),
            "candidate_median_latency_ms": result.get("median_latency_ms", {}).get(cid),
        })

    prelabel_failure = None
    if failed_attempt_dir:
        failure = read_json(failed_attempt_dir / "failure.json")
        prelabel_failure = {
            "status": failure["status"],
            "completed_candidates": failure["completed_candidates"],
            "message": failure["message"],
            "counted_as_candidate_dispatch": False,
            "labels_collected": 0,
        }
    analysis = {
        "schema": "c3_vta_operator_proxy_pareto_holdout_analysis_v1",
        "status": (
            "prospective_operator_proxy_front_retains_full_pool_oracle_and_reduces_build_cost"
            if retains_oracle else
            "prospective_invalid_peeling_retains_oracle_2pct_but_misses_exact_oracle"
            if front_best_regret_percent <= 2.0 and
               front_contract.get("schema") == "c3_invalid_dominator_peeling_front_v1" else
            "prospective_operator_proxy_front_misses_full_pool_oracle"
        ),
        "workload_id": pool["workload_id"],
        "source_model": pool["source_model"],
        "source_layer": pool["source_layer"],
        "pool_size": len(all_results),
        "operator_proxy_front_size": len(front_ids),
        "fpga_correct_count": len(passed),
        "fpga_rejected_count": len(rejected),
        "oracle_candidate_id": oracle["candidate_id"],
        "oracle_family_id": oracle["family_id"],
        "oracle_public_mode": oracle["public_mode"],
        "oracle_median_paired_latency_ratio": oracle_ratio,
        "oracle_candidate_median_latency_ms": oracle["median_latency_ms"][
            oracle["candidate_id"]
        ],
        "front_retains_full_fpga_correct_pool_oracle": retains_oracle,
        "front_best_regret_percent": front_best_regret_percent,
        "front_best_within_oracle_2_percent": front_best_regret_percent <= 2.0,
        "invalid_dominator_peeling": ({
            "online_candidate_ids": front_ids,
            "online_dispatch_count": len(front_ids),
            "fpga_invalid_count": len([
                cid for cid in front_ids if result_by_id[cid]["status"] != "passed"
            ]),
            "stop_reason": "completed_wave_had_no_fpga_invalid_candidate",
            "exact_oracle_discovered_online": retains_oracle,
            "oracle_2_percent_discovered_online": front_best_regret_percent <= 2.0,
            "interpretation": (
                "No invalid dominator was observed, so the frozen algorithm correctly stopped. "
                "Dominated candidates were measured only after the online result became immutable."
            ),
        } if front_contract.get("schema") == "c3_invalid_dominator_peeling_front_v1" else None),
        "all_search_candidates_slower_than_stock_reference": all(
            row["median_paired_latency_ratio"] > 1.0 for row in passed
        ),
        "build_cost": build_cost,
        "prospective_front_board_cost": prospective_cost,
        "exhaustive_full_pool_board_cost": exhaustive_cost,
        "prospective_board_savings_percent": board_saved,
        "common_operator_qualification_cost": common_cost,
        "time_to_target_candidate_dispatches": time_to_target,
        "operator_proxy_front_candidate_ids": front_ids,
        "exact_fused_pareto_front_candidate_ids_posthoc": exact_fused_front_ids,
        "front_overlap": sorted(front_set & exact_set),
        "same_tile_pairs": same_tile_pairs,
        "candidate_table": candidate_table,
        "prelabel_infrastructure_failure": prelabel_failure,
        "boot_ids": {"front": front["boot_id"], "completion": completion["boot_id"]},
        "verified_artifact_counts": verified,
        "bound_inputs": {
            "proxy_front_contract_artifact_manifest_sha256": sha256(
                front_contract_dir / "artifact_hashes.json"
            ),
            "front_pool_artifact_manifest_sha256": front_pool_manifest,
            "front_result_artifact_manifest_sha256": front_result_manifest,
            "complete_pool_artifact_manifest_sha256": complete_pool_manifest,
            "completion_result_artifact_manifest_sha256": sha256(
                completion_result_dir / "artifact_hashes.json"
            ),
            "local_artifact_manifest_sha256": sha256(local_dir / "artifact_hashes.json"),
        },
        "analyzer_source_sha256": sha256(Path(__file__).resolve()),
        "claim_boundary": (
            "One source-consistent Relay-testing ResNet50 layer and one board boot. The online "
            "selection is frozen before target labels; oracle completion is post-selection only. "
            "Logical VTA DMA is not physical AXI, randomized model parameters do not establish "
            "ImageNet accuracy, and this is not TopHub superiority."
        ),
    }
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    write_json(output / "analysis.json", analysis)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({key: analysis[key] for key in (
        "status", "pool_size", "operator_proxy_front_size", "fpga_correct_count",
        "oracle_candidate_id", "oracle_median_paired_latency_ratio",
        "front_retains_full_fpga_correct_pool_oracle", "build_cost",
        "prospective_board_savings_percent", "time_to_target_candidate_dispatches",
    )}, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-front-contract", required=True)
    parser.add_argument("--front-pool", required=True)
    parser.add_argument("--front-result", required=True)
    parser.add_argument("--complete-pool", required=True)
    parser.add_argument("--completion-result", required=True)
    parser.add_argument("--local-qualification", required=True)
    parser.add_argument("--failed-prelabel-attempt")
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
