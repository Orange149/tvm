#!/usr/bin/env python3
"""Audit an immutable online peeling run against its post-selection full oracle."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pct_reduction(value, baseline):
    return 100.0 * (1.0 - value / baseline)


def invocation_cost(result):
    rows = list(result.get("correctness", [])) + list(result.get("timings", []))
    load_bytes = store_bytes = dma_calls = host_wall_ms = 0.0
    for row in rows:
        profile = row["runtime_profile_complete"]
        load_bytes += int(profile["load_buffer_2d_bytes"])
        store_bytes += int(profile["store_buffer_2d_bytes"])
        dma_calls += int(profile["load_buffer_2d_calls"])
        dma_calls += int(profile["store_buffer_2d_calls"])
        host_wall_ms += float(row["host_wall_ms"])
    return {
        "fpga_kernel_invocations": len(rows),
        "logical_load_bytes": int(load_bytes),
        "logical_store_bytes": int(store_bytes),
        "logical_dma_bytes": int(load_bytes + store_bytes),
        "logical_dma_calls": int(dma_calls),
        "invocation_host_wall_ms": host_wall_ms,
    }


def sum_cost(order, results, programs, limit=None):
    ids = order if limit is None else order[:limit]
    total = {
        "candidate_dispatches": len(ids),
        "complete_candidate_build_action_seconds": 0.0,
        "fpga_kernel_invocations": 0,
        "logical_load_bytes": 0,
        "logical_store_bytes": 0,
        "logical_dma_bytes": 0,
        "logical_dma_calls": 0,
        "invocation_host_wall_ms": 0.0,
    }
    for cid in ids:
        total["complete_candidate_build_action_seconds"] += float(
            programs[cid]["complete_candidate_build_action_seconds"]
        )
        item = invocation_cost(results[cid])
        for key, value in item.items():
            total[key] += value
    return total


def first_target(order, results, oracle_id, oracle_ratio, epsilon):
    best_id = None
    best_ratio = math.inf
    for index, cid in enumerate(order, start=1):
        row = results[cid]
        if row["status"] != "passed":
            continue
        ratio = float(row["median_paired_latency_ratio"])
        if ratio < best_ratio:
            best_id, best_ratio = cid, ratio
        if best_ratio <= (1.0 + epsilon) * oracle_ratio:
            return {
                "dispatches": index,
                "candidate_id": best_id,
                "best_ratio": best_ratio,
                "regret_percent": 100.0 * (best_ratio / oracle_ratio - 1.0),
            }
    return None


def exact_target(order, results, oracle_id):
    for index, cid in enumerate(order, start=1):
        if cid == oracle_id and results[cid]["status"] == "passed":
            return {"dispatches": index, "candidate_id": cid}
    return None


def terminal_best(order, results, oracle_ratio):
    eligible = [results[cid] for cid in order if results[cid]["status"] == "passed"]
    if not eligible:
        return None
    best = min(
        eligible,
        key=lambda row: (float(row["median_paired_latency_ratio"]), row["candidate_id"]),
    )
    ratio = float(best["median_paired_latency_ratio"])
    return {
        "candidate_id": best["candidate_id"],
        "best_ratio": ratio,
        "regret_percent": 100.0 * (ratio / oracle_ratio - 1.0),
    }


def run(args):
    dirs = {
        name: Path(getattr(args, name))
        for name in (
            "target_contract",
            "policy_contract",
            "local_qualification",
            "front_contract",
            "online_result",
            "oracle_pool",
            "oracle_completion_result",
        )
    }
    verified = {
        name: len(verify_artifacts_compatible(path)) for name, path in dirs.items()
    }
    target = read_json(dirs["target_contract"] / "contract.json")
    policy_path = dirs["policy_contract"] / "policy.json"
    front_path = dirs["front_contract"] / "front.json"
    policy = read_json(policy_path)
    front = read_json(front_path)
    online = read_json(dirs["online_result"] / "summary.json")
    pool = read_json(dirs["oracle_pool"] / "pool.json")
    completion = read_json(dirs["oracle_completion_result"] / "summary.json")
    if policy.get("status") != "frozen_before_target_lowering_fsim_fullgraph_fpga_or_latency":
        raise RuntimeError("policy was not prospectively frozen")
    if front.get("status") != "wave0_and_controls_frozen_before_fullgraph_fpga_or_latency":
        raise RuntimeError("front and controls were not prospectively frozen")
    if online.get("status") != "online_peeling_complete":
        raise RuntimeError("online run is incomplete")
    if pool.get("status") != "invalid_dominator_peeling_oracle_completion_pool_after_online_labels":
        raise RuntimeError("oracle pool status mismatch")
    if completion.get("status") != "dominated_final_fused_oracle_completion_board_complete":
        raise RuntimeError("oracle completion is incomplete")
    if pool["online_result_artifact_manifest_sha256"] != sha256(
        dirs["online_result"] / "artifact_hashes.json"
    ):
        raise RuntimeError("oracle pool does not bind online result")
    if completion["pool_artifact_manifest_sha256"] != sha256(
        dirs["oracle_pool"] / "artifact_hashes.json"
    ):
        raise RuntimeError("oracle board result does not bind oracle pool")
    workload_ids = {
        target["workload_id"], policy["workload_id"], front["workload_id"],
        online["workload_id"], pool["workload_id"],
    }
    if len(workload_ids) != 1:
        raise RuntimeError("workload identity drift")

    programs = {row["candidate_id"]: row for row in pool["programs"]}
    online_results = {row["candidate_id"]: row for row in online["candidate_results"]}
    completion_results = {
        row["candidate_id"]: row for row in completion["candidate_results"]
    }
    if set(online_results) & set(completion_results):
        raise RuntimeError("online and completion result sets overlap")
    results = {**online_results, **completion_results}
    if set(results) != set(programs):
        raise RuntimeError("complete result/program identity mismatch")
    correct = [row for row in results.values() if row["status"] == "passed"]
    invalid = [row for row in results.values() if row["status"] != "passed"]
    oracle = min(
        correct,
        key=lambda row: (float(row["median_paired_latency_ratio"]), row["candidate_id"]),
    )
    oracle_id = oracle["candidate_id"]
    oracle_ratio = float(oracle["median_paired_latency_ratio"])
    online_order = list(online["dispatched_candidate_ids"])

    orders = {"online_peeling": online_order, **front["control_orders"]}
    replay = {}
    for name, order in orders.items():
        if not set(order).issubset(results):
            raise RuntimeError("control order escaped complete pool: " + name)
        targets = {
            "oracle_plus_2pct": first_target(order, results, oracle_id, oracle_ratio, 0.02),
            "oracle_plus_0_1pct": first_target(order, results, oracle_id, oracle_ratio, 0.001),
            "exact_oracle": exact_target(order, results, oracle_id),
        }
        replay[name] = {
            "order": order,
            "terminal_best": terminal_best(order, results, oracle_ratio),
            "targets": targets,
            "terminal_cost": sum_cost(order, results, programs),
            "cost_to_target": {
                key: (sum_cost(order, results, programs, value["dispatches"])
                      if value is not None else None)
                for key, value in targets.items()
            },
        }

    exhaustive_order = [row["candidate_id"] for row in pool["programs"]]
    exhaustive_cost = sum_cost(exhaustive_order, results, programs)
    online_cost = sum_cost(online_order, results, programs)
    online_best = min(
        (results[cid] for cid in online_order if results[cid]["status"] == "passed"),
        key=lambda row: (float(row["median_paired_latency_ratio"]), row["candidate_id"]),
    )
    savings = {
        key + "_reduction_percent": pct_reduction(online_cost[key], exhaustive_cost[key])
        for key in (
            "candidate_dispatches",
            "complete_candidate_build_action_seconds",
            "fpga_kernel_invocations",
            "logical_load_bytes",
            "logical_store_bytes",
            "logical_dma_bytes",
            "logical_dma_calls",
            "invocation_host_wall_ms",
        )
    }

    families = {}
    for program in programs.values():
        families.setdefault(program["family_id"], {})[program["public_mode"]] = program
    same_tile = []
    for family_id, modes in sorted(families.items()):
        if "original" not in modes:
            continue
        base = modes["original"]
        base_result = results[base["candidate_id"]]
        for mode in ("input_stationary", "weight_resident_barrier"):
            if mode not in modes:
                continue
            program = modes[mode]
            row = results[program["candidate_id"]]
            same_tile.append({
                "family_id": family_id,
                "mode": mode,
                "candidate_id": program["candidate_id"],
                "fpga_status": row["status"],
                "operator_dma_bytes_change_percent": 100.0 * (
                    program["dma_bytes"] / base["dma_bytes"] - 1.0
                ),
                "operator_dma_calls_change_percent": 100.0 * (
                    program["dma_calls"] / base["dma_calls"] - 1.0
                ),
                "whole_model_latency_improvement_percent": (
                    100.0 * (1.0 - float(row["median_paired_latency_ratio"]) /
                             float(base_result["median_paired_latency_ratio"]))
                    if row["status"] == "passed" and base_result["status"] == "passed"
                    else None
                ),
            })

    result = {
        "schema": "c3_vta_invalid_dominator_peeling_holdout_analysis_v1",
        "status": "prospective_online_peeling_exact_oracle_complete",
        "workload_id": target["workload_id"],
        "geometry_signature": target["geometry_signature"],
        "pool": {
            "eligible": len(programs),
            "fpga_correct": len(correct),
            "fpga_invalid": len(invalid),
            "invalid_candidate_ids": [row["candidate_id"] for row in invalid],
        },
        "oracle": {
            "candidate_id": oracle_id,
            "family_id": oracle["family_id"],
            "public_mode": oracle["public_mode"],
            "median_paired_latency_ratio": oracle_ratio,
            "candidate_median_latency_ms": oracle["median_latency_ms"][oracle_id],
        },
        "online": {
            "waves": online["waves"],
            "stop_reason": online["stop_reason"],
            "dispatch_count": len(online_order),
            "invalid_count": len(online["invalid_candidate_ids"]),
            "best_candidate_id": online_best["candidate_id"],
            "best_is_exact_oracle": online_best["candidate_id"] == oracle_id,
            "best_regret_percent": 100.0 * (
                float(online_best["median_paired_latency_ratio"]) / oracle_ratio - 1.0
            ),
            "outer_process_seconds_before_summary_write": online[
                "outer_process_seconds_before_summary_write"
            ],
            "measured_component_cost": online_cost,
        },
        "exhaustive_measured_component_cost": exhaustive_cost,
        "online_vs_exhaustive_savings": savings,
        "frozen_order_replay": replay,
        "same_tile_mechanism_pairs": same_tile,
        "recorded_post_selection_audit_cost": {
            "oracle_completion_build_process_wall_seconds": pool["build_cost"][
                "oracle_completion_process_wall_seconds"
            ],
            "oracle_completion_board_outer_process_seconds": completion[
                "outer_process_seconds_before_summary_write"
            ],
        },
        "board_boot_consistent": online["boot_id"] == completion["boot_id"],
        "verified_artifact_counts": verified,
        "bound_artifact_manifests": {
            name: sha256(path / "artifact_hashes.json") for name, path in dirs.items()
        },
        "analyzer_source_sha256": sha256(Path(__file__).resolve()),
        "claim_boundary": (
            "One prospective {} {} layer. Control "
            "orders are frozen-order replays over the subsequently completed pool, not "
            "independent online runs. Runtime counters are logical VTA operations, not "
            "physical AXI bursts. Invocation host wall excludes build/reset/upload; only "
            "the online outer-process wall is a directly measured end-to-end search wall."
        ).format(target.get("source_model"), target.get("source_layer")),
    }
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    write_json(output / "analysis.json", result)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "status": result["status"],
        "pool": result["pool"],
        "oracle": result["oracle"],
        "online": result["online"],
        "online_vs_exhaustive_savings": savings,
        "target_dispatches": {
            key: {target: (value["dispatches"] if value else None)
                  for target, value in row["targets"].items()}
            for key, row in replay.items()
        },
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-contract", required=True)
    parser.add_argument("--policy-contract", required=True)
    parser.add_argument("--local-qualification", required=True)
    parser.add_argument("--front-contract", required=True)
    parser.add_argument("--online-result", required=True)
    parser.add_argument("--oracle-pool", required=True)
    parser.add_argument("--oracle-completion-result", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
