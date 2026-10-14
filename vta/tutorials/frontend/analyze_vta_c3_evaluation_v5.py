#!/usr/bin/env python3
"""Analyze the two frozen C3 v5 operator boots without reselection."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from c3_evaluation_v5 import (
    BUDGETS,
    METHODS,
    read_json,
    seal_flat,
    sha256_file,
    verify_flat,
    write_json,
)


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_FREEZE = GROUPED / "20260926_c3_yolo512_evaluation_v5_freeze_run01"
DEFAULT_OUTPUT = GROUPED / "20260926_c3_yolo512_evaluation_v5_cross_boot_analysis_run01"


def median_latency(reference, task_id, program_id):
    return float(reference[task_id][program_id]["median_of_round_medians_ms"])


def confirmation_gain(summary, task_id, incumbent, winner):
    if winner == incumbent:
        return {
            "round_gains_pct": [0.0] * 6,
            "median_gain_pct": 0.0,
            "winning_rounds": 0,
            "self_comparison": True,
        }
    left = summary[task_id][incumbent]["round_medians_ms"]
    right = summary[task_id][winner]["round_medians_ms"]
    if len(left) != 6 or len(right) != 6:
        raise ValueError("confirmation must contain six rounds")
    gains = [100.0 * (1.0 - candidate / baseline)
             for baseline, candidate in zip(left, right)]
    return {
        "round_gains_pct": gains,
        "median_gain_pct": float(statistics.median(gains)),
        "winning_rounds": sum(value > 0.0 for value in gains),
        "self_comparison": False,
    }


def replay_random(policy, selection_stats, reference, task_id, oracle_latency):
    incumbent = policy["incumbent_program_id"]
    result = {"B{}".format(budget): [] for budget in BUDGETS}
    for replay_index, order in enumerate(policy["random_replay_orders"]):
        for budget in BUDGETS:
            exposed = [incumbent] + order[:budget]
            winner = min(exposed, key=lambda program: (
                selection_stats[program]["median_of_round_medians_ms"],
                0 if program == incumbent else 1,
                program,
            ))
            latency = median_latency(reference, task_id, winner)
            result["B{}".format(budget)].append({
                "replay": replay_index,
                "winner": winner,
                "simple_regret": latency / oracle_latency - 1.0,
            })
    return {
        budget: {
            "replays": len(rows),
            "median_simple_regret": float(statistics.median(
                row["simple_regret"] for row in rows
            )),
            "minimum_simple_regret": min(row["simple_regret"] for row in rows),
            "maximum_simple_regret": max(row["simple_regret"] for row in rows),
            "oracle_hit_count": sum(row["simple_regret"] == 0.0 for row in rows),
            "rows": rows,
        }
        for budget, rows in result.items()
    }


def run(args):
    freeze_dir = Path(args.freeze_dir).resolve()
    boot_dirs = [Path(args.boot1_dir).resolve(), Path(args.boot2_dir).resolve()]
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    freeze_manifest = verify_flat(freeze_dir)
    policies = read_json(freeze_dir / "task_policies.json")
    manifests = [verify_flat(directory) for directory in boot_dirs]
    summaries = [read_json(directory / "summary.json") for directory in boot_dirs]
    for index, summary in enumerate(summaries, 1):
        if summary.get("status") != "C3_EVALUATION_V5_OPERATOR_BOOT{}_COMPLETE".format(index):
            raise ValueError("operator boot {} is incomplete".format(index))
    if summaries[0]["boot_id"] == summaries[1]["boot_id"]:
        raise ValueError("operator boots are not independent kernel boots")
    decisions_paths = [directory / "selection_decisions_frozen.json" for directory in boot_dirs]
    if sha256_file(decisions_paths[0]) != sha256_file(decisions_paths[1]):
        raise ValueError("boot 2 did not preserve the exact sealed winners")
    decision_payload = read_json(decisions_paths[0])
    decisions = decision_payload["decisions"]
    references = [read_json(directory / "production_reference_summary.json")
                  for directory in boot_dirs]
    confirmations = [read_json(directory / "confirmation_summary.json")
                     for directory in boot_dirs]

    output.mkdir(parents=True)
    per_boot = []
    random_replays = []
    for boot_index, (reference, confirmation) in enumerate(zip(references, confirmations), 1):
        tasks = {}
        for task_id, policy in sorted(policies.items()):
            production = policy["production_program_pool"]
            if set(reference.get(task_id, {})) != set(production):
                raise ValueError("boot {} lacks complete production oracle for {}".format(
                    boot_index, task_id))
            oracle = min(production, key=lambda program: (
                median_latency(reference, task_id, program), program,
            ))
            oracle_latency = median_latency(reference, task_id, oracle)
            incumbent = policy["incumbent_program_id"]
            methods = {}
            for method in METHODS:
                methods[method] = {}
                for budget in BUDGETS:
                    key = "B{}".format(budget)
                    selected = decisions[task_id]["methods"][method][key]["search_winner"]
                    latency = median_latency(reference, task_id, selected)
                    prefix = policy["orders"][method][:budget]
                    methods[method][key] = {
                        "search_winner": selected,
                        "winner_latency_ms": latency,
                        "simple_regret": latency / oracle_latency - 1.0,
                        "prefix_covers_oracle": oracle in prefix or oracle == incumbent,
                        "winner_is_oracle": selected == oracle,
                        "winner_within_oracle_1pct": latency <= 1.01 * oracle_latency,
                        "winner_within_oracle_2pct": latency <= 1.02 * oracle_latency,
                    }
            incumbent_latency = median_latency(reference, task_id, incumbent)
            winner = decisions[task_id]["methods"]["selector_v4"]["B3"]["search_winner"]
            tasks[task_id] = {
                "production_oracle_program_id": oracle,
                "production_oracle_latency_ms": oracle_latency,
                "incumbent_program_id": incumbent,
                "incumbent_latency_ms": incumbent_latency,
                "incumbent_simple_regret": incumbent_latency / oracle_latency - 1.0,
                "incumbent_within_oracle_2pct": incumbent_latency <= 1.02 * oracle_latency,
                "methods": methods,
                "selector_v4_B3_confirmation": confirmation_gain(
                    confirmation, task_id, incumbent, winner
                ),
            }
            random_replays.append({
                "boot_index": boot_index,
                "execution_task_id": task_id,
                "statistics": replay_random(
                    policy, decisions[task_id]["program_statistics"], reference,
                    task_id, oracle_latency,
                ),
            })
        per_boot.append({"boot_index": boot_index, "boot_id": summaries[boot_index - 1]["boot_id"],
                         "tasks": tasks})

    task_ids = sorted(policies)
    non_worse_each_boot = []
    strict_tasks_by_boot = []
    for boot in per_boot:
        non_worse = {
            task_id: (
                boot["tasks"][task_id]["methods"]["selector_v4"]["B3"]["simple_regret"]
                <= boot["tasks"][task_id]["methods"]["bytes_only"]["B3"]["simple_regret"]
            )
            for task_id in task_ids
        }
        strict = [task_id for task_id in task_ids if (
            boot["tasks"][task_id]["methods"]["selector_v4"]["B3"]["simple_regret"]
            < boot["tasks"][task_id]["methods"]["bytes_only"]["B3"]["simple_regret"]
        )]
        non_worse_each_boot.append(non_worse)
        strict_tasks_by_boot.append(strict)
    stable_strict_tasks = sorted(set(strict_tasks_by_boot[0]) & set(strict_tasks_by_boot[1]))
    selector_primary_passed = (
        all(all(values.values()) for values in non_worse_each_boot)
        and bool(stable_strict_tasks)
    )

    deployment = {}
    for task_id in task_ids:
        incumbent = policies[task_id]["incumbent_program_id"]
        winner = decisions[task_id]["methods"]["selector_v4"]["B3"]["search_winner"]
        mode = next(
            row["residence_mode"]
            for row in read_json(freeze_dir / "candidate_membership.json")[task_id]
            if row["program_id"] == winner
        )
        confirmation = [boot["tasks"][task_id]["selector_v4_B3_confirmation"] for boot in per_boot]
        stable = (
            winner != incumbent
            and mode == "input_weight_resident_transition_safe"
            and all(row["median_gain_pct"] >= 2.0 and row["winning_rounds"] >= 5
                    for row in confirmation)
        )
        deployment[task_id] = {
            "search_winner": winner,
            "search_winner_residence_mode": mode,
            "incumbent_program_id": incumbent,
            "confirmation_by_boot": confirmation,
            "stable_mode11": stable,
            "deploy_program": winner if stable else incumbent,
            "fallback_used": not stable,
        }

    analysis = {
        "schema": "c3_evaluation_v5_cross_boot_analysis_v1",
        "status": "C3_EVALUATION_V5_CROSS_BOOT_ANALYSIS_COMPLETE",
        "freeze_manifest_sha256": freeze_manifest,
        "boot_manifests": manifests,
        "boot_ids": [summary["boot_id"] for summary in summaries],
        "per_boot": per_boot,
        "primary_selector_comparison": {
            "method": "selector_v4", "baseline": "bytes_only", "budget": "B3",
            "non_worse_by_boot_and_task": non_worse_each_boot,
            "strictly_better_tasks_by_boot": strict_tasks_by_boot,
            "same_task_strictly_better_both_boots": stable_strict_tasks,
            "passed": selector_primary_passed,
            "interpretation_if_tiny": (
                "strict numerical differences without a practical threshold are empirical regret "
                "ordering only, not a practical-effect claim"
            ),
        },
        "deployment_decisions": deployment,
        "at_least_one_stable_mode11": any(row["stable_mode11"] for row in deployment.values()),
        "winner_reselection_observed": False,
        "selection_and_reference_data_separated": True,
        "unified_production_legacy_oracle_forbidden": True,
    }
    write_json(output / "analysis.json", analysis)
    write_json(output / "random_replay_1000.json", random_replays)
    write_json(output / "graph_deployment_decision.json", {
        "schema": "c3_evaluation_v5_graph_deployment_decision_v1",
        "status": (
            "GRAPH_BUILD_ELIGIBLE" if analysis["at_least_one_stable_mode11"]
            else "NO_CONFIRMED_MODE11_GRAPH_BUILD_FORBIDDEN"
        ),
        "task_programs": deployment,
        "decision_frozen_before_graph_build_or_graph_latency": True,
        "source_analysis_sha256": sha256_file(output / "analysis.json"),
    })
    write_json(output / "summary.json", {
        "status": analysis["status"],
        "primary_selector_comparison_passed": selector_primary_passed,
        "stable_mode11_tasks": [task for task, row in deployment.items() if row["stable_mode11"]],
        "graph_build_eligible": analysis["at_least_one_stable_mode11"],
    })
    seal_flat(output, (Path(__file__).resolve(),))
    print(json.dumps(read_json(output / "summary.json"), indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--boot1-dir", type=Path, required=True)
    parser.add_argument("--boot2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
