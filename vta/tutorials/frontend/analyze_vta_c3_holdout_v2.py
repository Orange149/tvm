#!/usr/bin/env python3
"""Analyze the two-boot C3 selector-v2 holdout without reusing selection timing."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from run_vta_c3_reuse_supplement import file_hash, seal, verify, write


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_BOOT1 = GROUPED / "20260924_c3_holdout_board_boot1_v2_run01"
DEFAULT_BOOT2 = GROUPED / "20260924_c3_holdout_board_boot2_v2_run03"
DEFAULT_POLICIES = GROUPED / "20260924_c3_holdout_shortlists_v2_run02"
DEFAULT_YOLO_LOCAL = GROUPED / "20260924_c3_yolo608_local_qualification_run01"
DEFAULT_R18_LOCAL = GROUPED / "20260924_c3_real_task_reference_local_run05"
DEFAULT_CROSS = GROUPED / "20260924_c3_holdout_cross_compile_v2_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_holdout_cross_boot_analysis_v2_run02"
SCHEMA = "c3_holdout_cross_boot_analysis_v2"
BUDGETS = (1, 2, 3)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def median(values):
    return float(statistics.median(values))


def paired_gate(boot_rows, minimum_improvement=2.0, minimum_wins=5):
    """Apply the preregistered gate independently to both boots."""
    per_boot = []
    for row in boot_rows:
        paired = row["paired"]
        passed = (float(paired["median_improvement_pct"]) >= minimum_improvement
                  and int(paired["wins"]) >= minimum_wins)
        per_boot.append({
            "median_improvement_pct": float(paired["median_improvement_pct"]),
            "wins": int(paired["wins"]),
            "rounds": len(paired["round_improvements_pct"]),
            "passed": passed,
        })
    return {
        "per_boot": per_boot,
        "stable_improvement": all(row["passed"] for row in per_boot),
        "threshold": {
            "minimum_median_improvement_pct_each_boot": minimum_improvement,
            "minimum_wins_each_boot": minimum_wins,
            "rounds_each_boot": 6,
        },
    }


def combined_reference(boot1_task, boot2_task):
    ids1 = set(boot1_task["fpga_correct_program_ids"])
    ids2 = set(boot2_task["fpga_correct_program_ids"])
    if ids1 != ids2:
        raise ValueError("full-pool FPGA-correct program sets differ across boots")
    stats = {}
    for program_id in sorted(ids1):
        rounds1 = boot1_task["program_statistics"][program_id]["round_medians_ms"]
        rounds2 = boot2_task["program_statistics"][program_id]["round_medians_ms"]
        if len(rounds1) != 6 or len(rounds2) != 6:
            raise ValueError("reference program does not have six rounds per boot")
        values = [float(x) for x in rounds1 + rounds2]
        stats[program_id] = {
            "boot1_round_medians_ms": [float(x) for x in rounds1],
            "boot2_round_medians_ms": [float(x) for x in rounds2],
            "twelve_round_median_ms": median(values),
        }
    oracle_id = min(stats, key=lambda item: (stats[item]["twelve_round_median_ms"], item))
    return {
        "fpga_correct_program_ids": sorted(ids1),
        "fpga_rejected_program_ids": sorted(
            set(boot1_task["fpga_rejected_program_ids"])
            | set(boot2_task["fpga_rejected_program_ids"])
        ),
        "program_statistics": stats,
        "pool_oracle_program_id": oracle_id,
        "pool_oracle_latency_ms": stats[oracle_id]["twelve_round_median_ms"],
        "oracle_definition": "minimum median over the same program's 12 round medians; tie by program_id",
    }


def replay_one(order, incumbent_id, latencies, oracle_id, budget):
    attempted = list(order[:budget])
    seen = [incumbent_id] + [item for item in attempted if item in latencies]
    best = min(seen, key=lambda item: (latencies[item], item))
    oracle_ms = latencies[oracle_id]
    incumbent_ms = latencies[incumbent_id]
    best_ms = latencies[best]
    return {
        "budget_additional_candidates": budget,
        "attempted_new_program_ids": attempted,
        "best_program_id": best,
        "best_latency_ms": best_ms,
        "simple_regret_pct": 100.0 * (best_ms / oracle_ms - 1.0),
        "improvement_vs_incumbent_pct": 100.0 * (1.0 - best_ms / incumbent_ms),
        "exact_oracle_hit": best == oracle_id,
        "oracle_plus_1pct_hit": best_ms <= oracle_ms * 1.01,
        "oracle_plus_2pct_hit": best_ms <= oracle_ms * 1.02,
        "fpga_invalid_attempts": sum(item not in latencies for item in attempted),
    }


def first_hit(order, incumbent_id, latencies, oracle_ms, factor):
    if latencies[incumbent_id] <= oracle_ms * factor:
        return 0
    best = latencies[incumbent_id]
    for rank, program_id in enumerate(order, start=1):
        if program_id in latencies:
            best = min(best, latencies[program_id])
        if best <= oracle_ms * factor:
            return rank
    return None


def deterministic_replay(order, incumbent_id, latencies, oracle_id):
    oracle_ms = latencies[oracle_id]
    return {
        "budgets": {"B" + str(b): replay_one(order, incumbent_id, latencies, oracle_id, b)
                    for b in BUDGETS},
        "first_hit_additional_candidates": {
            "exact_oracle": first_hit(order, incumbent_id, latencies, oracle_ms, 1.0),
            "oracle_plus_1pct": first_hit(order, incumbent_id, latencies, oracle_ms, 1.01),
            "oracle_plus_2pct": first_hit(order, incumbent_id, latencies, oracle_ms, 1.02),
        },
    }


def random_replay(orders, incumbent_id, latencies, oracle_id):
    if len(orders) != 1000:
        raise ValueError("fixed random baseline must contain exactly 1000 orders")
    result = {"frozen_order_count": len(orders), "budgets": {}}
    for budget in BUDGETS:
        rows = [replay_one(order, incumbent_id, latencies, oracle_id, budget) for order in orders]
        regrets = [row["simple_regret_pct"] for row in rows]
        improvements = [row["improvement_vs_incumbent_pct"] for row in rows]
        result["budgets"]["B" + str(budget)] = {
            "simple_regret_pct_mean": statistics.fmean(regrets),
            "simple_regret_pct_median": median(regrets),
            "simple_regret_pct_min": min(regrets),
            "simple_regret_pct_max": max(regrets),
            "improvement_vs_incumbent_pct_mean": statistics.fmean(improvements),
            "exact_oracle_hit_rate": statistics.fmean(row["exact_oracle_hit"] for row in rows),
            "oracle_plus_1pct_hit_rate": statistics.fmean(row["oracle_plus_1pct_hit"] for row in rows),
            "oracle_plus_2pct_hit_rate": statistics.fmean(row["oracle_plus_2pct_hit"] for row in rows),
            "fpga_invalid_attempts_mean": statistics.fmean(row["fpga_invalid_attempts"] for row in rows),
        }
    oracle_ms = latencies[oracle_id]
    for label, factor in (("exact_oracle", 1.0), ("oracle_plus_1pct", 1.01),
                          ("oracle_plus_2pct", 1.02)):
        ranks = [first_hit(order, incumbent_id, latencies, oracle_ms, factor) for order in orders]
        finite = [rank for rank in ranks if rank is not None]
        result[label + "_first_hit"] = {
            "success_rate_over_full_order": len(finite) / len(ranks),
            "median_additional_candidates_when_hit": median(finite) if finite else None,
        }
    return result


def validate_boot(directory, expected_status):
    verify(directory)
    summary = load(Path(directory) / "summary.json")
    if summary["status"] != expected_status:
        raise ValueError("unexpected boot status")
    if (not summary["all_post_qualification_outputs_correct"]
            or summary["implicit_extra_calls"] != 0 or summary["session_spliced"]):
        raise ValueError("boot validity contract failed")
    return summary


def cost_summary(args, boot1_summary, boot2_summary):
    yolo = load(Path(args.yolo_local_dir) / "summary.json")
    r18 = load(Path(args.r18_local_dir) / "summary.json")
    cross = load(Path(args.cross_compile_dir) / "summary.json")
    return {
        "local_qualification": {
            "wall_seconds": yolo["wall_seconds"] + r18["wall_seconds"],
            "proposed_identities": yolo["proposed_identities"] + r18["proposed_identities"],
            "fsim_attempted": yolo["fsim_attempted"] + r18["fsim_attempted"],
        },
        "cross_compile": {
            "wall_seconds": cross["wall_seconds"],
            "candidate_count": cross["candidate_count"],
        },
        "board": {
            "boot1_wall_seconds": boot1_summary["wall_seconds"],
            "boot2_wall_seconds": boot2_summary["wall_seconds"],
            "selection_timing_samples": (boot1_summary["selection_timing_samples"]
                                         + boot2_summary["selection_timing_samples"]),
            "confirmation_timing_samples": (boot1_summary["confirmation_timing_samples"]
                                            + boot2_summary["confirmation_timing_samples"]),
            "full_pool_reference_timing_samples": (boot1_summary["reference_timing_samples"]
                                                    + boot2_summary["reference_timing_samples"]),
            "correctness_seed_checks": (boot1_summary["correctness_seed_checks"]
                                        + boot2_summary["correctness_seed_checks"]),
            "complete_operator_calls": (boot1_summary["complete_operator_calls"]
                                        + boot2_summary["complete_operator_calls"]),
        },
        "warning": "wall time is phase accounting, not operator latency or tuning-budget equivalence",
    }


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    boot1 = Path(args.boot1_dir).resolve()
    boot2 = Path(args.boot2_dir).resolve()
    policies_dir = Path(args.policies_dir).resolve()
    try:
        s1 = validate_boot(boot1, "BOOT1_COMPLETE")
        s2 = validate_boot(boot2, "BOOT2_COMPLETE")
        if s1["boot_id"] == s2["boot_id"]:
            raise ValueError("two independent boots are required")
        verify(policies_dir)
        c1, c2 = load(boot1 / "contract.json"), load(boot2 / "contract.json")
        for field in ("pool_manifest_sha256", "shortlist_manifest_sha256", "task_manifest_sha256"):
            if c1[field] != c2[field]:
                raise ValueError(field + " differs across boots")

        conf1 = load(boot1 / "confirmation_summary.json")
        conf2 = load(boot2 / "confirmation_summary.json")
        if set(conf1) != set(conf2):
            raise ValueError("confirmation task sets differ")
        confirmations = {}
        for task_id in sorted(conf1):
            a, b = conf1[task_id], conf2[task_id]
            if (a["incumbent_program_id"], a["selected_program_id"]) != (
                    b["incumbent_program_id"], b["selected_program_id"]):
                raise ValueError("confirmation identities differ")
            gate = paired_gate([a, b])
            gate.update({
                "incumbent_program_id": a["incumbent_program_id"],
                "selected_program_id": a["selected_program_id"],
                "combined_twelve_round_improvement_pct_median": median(
                    a["paired"]["round_improvements_pct"] + b["paired"]["round_improvements_pct"]),
                "deployment_eligible": gate["stable_improvement"],
            })
            confirmations[task_id] = gate

        ref1, ref2 = load(boot1 / "reference_summary.json"), load(boot2 / "reference_summary.json")
        if set(ref1) != set(ref2):
            raise ValueError("reference task sets differ")
        policies = load(policies_dir / "task_policies.json")
        references = {}
        for task_id in sorted(ref1):
            combined = combined_reference(ref1[task_id], ref2[task_id])
            policy = policies[task_id]
            latencies = {program_id: row["twelve_round_median_ms"]
                         for program_id, row in combined["program_statistics"].items()}
            if set(policy["complete_local_program_pool"]) != set(latencies):
                raise ValueError("measured full pool differs from frozen complete local pool")
            incumbent = policy["incumbent_program_id"]
            oracle = combined["pool_oracle_program_id"]
            methods = {
                "incumbent_only": deterministic_replay([], incumbent, latencies, oracle),
                "bytes_only": deterministic_replay(policy["bytes_only_order"], incumbent, latencies, oracle),
                "calls_only": deterministic_replay(policy["calls_only_order"], incumbent, latencies, oracle),
                "selector_v2": deterministic_replay(policy["selector_v2_order"], incumbent, latencies, oracle),
                "fixed_random_1000": random_replay(
                    policy["fixed_random_orders_1000"], incumbent, latencies, oracle),
            }
            shortlist_ids = policy["shortlist"]["shortlist_candidate_ids"]
            combined.update({
                "incumbent_program_id": incumbent,
                "incumbent_latency_ms": latencies[incumbent],
                "incumbent_regret_pct": 100.0 * (latencies[incumbent] / latencies[oracle] - 1.0),
                "frozen_shortlist_program_ids": shortlist_ids,
                "frozen_shortlist_omitted_oracle": oracle not in shortlist_ids,
                "offline_budget_replay": methods,
            })
            references[task_id] = combined

        # Aggregate the deterministic simple regrets by task.  Random is kept as a distribution.
        aggregate = {}
        for method in ("incumbent_only", "bytes_only", "calls_only", "selector_v2"):
            aggregate[method] = {}
            for budget in BUDGETS:
                values = [row["offline_budget_replay"][method]["budgets"]["B" + str(budget)]
                          ["simple_regret_pct"] for row in references.values()]
                aggregate[method]["B" + str(budget)] = {
                    "mean_simple_regret_pct": statistics.fmean(values),
                    "median_simple_regret_pct": median(values),
                    "exact_oracle_tasks": sum(value == 0.0 for value in values),
                    "task_count": len(values),
                }
        selector_beats_simple = any(
            aggregate["selector_v2"]["B" + str(b)]["mean_simple_regret_pct"]
            < min(aggregate[name]["B" + str(b)]["mean_simple_regret_pct"]
                  for name in ("bytes_only", "calls_only"))
            for b in BUDGETS
        )

        result = {
            "schema": SCHEMA,
            "status": "CROSS_BOOT_ANALYSIS_COMPLETE",
            "boot_ids": [s1["boot_id"], s2["boot_id"]],
            "confirmation": confirmations,
            "stable_deployment_task_ids": [task_id for task_id, row in confirmations.items()
                                            if row["deployment_eligible"]],
            "full_pool_reference": references,
            "deterministic_method_aggregate": aggregate,
            "selector_v2_beats_at_least_one_major_simple_baseline_same_budget": selector_beats_simple,
            "cost_accounting": cost_summary(args, s1, s2),
            "claim_boundary": {
                "hardware_wait_cycles_measured": False,
                "selection_samples_used_for_confirmation": False,
                "oracle_scope": "only the two fully measured frozen legal program pools",
                "random_baseline_scope": "offline replay of 1000 orders frozen before board latency",
            },
        }
        write(output / "analysis.json", result)
        write(output / "summary.json", {
            "schema": SCHEMA,
            "status": result["status"],
            "boot_ids": result["boot_ids"],
            "full_pool_task_count": len(references),
            "stable_deployment_task_ids": result["stable_deployment_task_ids"],
            "stable_deployment_task_count": len(result["stable_deployment_task_ids"]),
            "selector_v2_beats_at_least_one_major_simple_baseline_same_budget": selector_beats_simple,
            "full_pool_oracles": {task_id: row["pool_oracle_program_id"]
                                  for task_id, row in references.items()},
        })
        write(output / "inputs.json", {
            "boot1": {"path": str(boot1), "manifest_sha256": file_hash(boot1 / "artifact_hashes.json")},
            "boot2": {"path": str(boot2), "manifest_sha256": file_hash(boot2 / "artifact_hashes.json")},
            "policies": {"path": str(policies_dir),
                         "manifest_sha256": file_hash(policies_dir / "artifact_hashes.json")},
            "source_sha256": file_hash(__file__),
        })
        print(json.dumps(load(output / "summary.json"), indent=2, sort_keys=True), flush=True)
    finally:
        seal(output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boot1-dir", default=str(DEFAULT_BOOT1))
    parser.add_argument("--boot2-dir", default=str(DEFAULT_BOOT2))
    parser.add_argument("--policies-dir", default=str(DEFAULT_POLICIES))
    parser.add_argument("--yolo-local-dir", default=str(DEFAULT_YOLO_LOCAL))
    parser.add_argument("--r18-local-dir", default=str(DEFAULT_R18_LOCAL))
    parser.add_argument("--cross-compile-dir", default=str(DEFAULT_CROSS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
