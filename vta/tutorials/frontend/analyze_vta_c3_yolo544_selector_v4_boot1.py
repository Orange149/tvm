#!/usr/bin/env python3
"""Analyze strict-unseen YOLO-544 selector-v4 boot-1 complete pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from c3_selector_v3_evaluation import evaluate_order, evaluate_task_pool


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_POOL = GROUPED / "20260924_c3_yolo544_selector_v4_board_pool_freeze_run01"
DEFAULT_TIMING = GROUPED / "20260924_c3_yolo544_selector_v4_timing_freeze_run01"
DEFAULT_BOOT = GROUPED / "20260924_c3_yolo544_selector_v4_timing_boot1_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo544_selector_v4_boot1_analysis_run01"
SCHEMA = "c3_yolo544_selector_v4_boot1_analysis_v1"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def verify(directory):
    directory = Path(directory).resolve()
    ledger = read(directory / "artifact_hashes.json")
    for name, expected in ledger.get("artifacts", {}).items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("artifact hash mismatch: {}".format(path))
    return sha256_file(directory / "artifact_hashes.json")


def evaluation_row(candidate, stats, incumbent_program, incumbent_latency):
    metrics = candidate["timing_freeze_metrics"]
    latency = float(stats["median_of_round_medians_ms"])
    return {
        "execution_task_id": candidate["execution_task_id"],
        "candidate_set_id": candidate["program_id"],
        "is_incumbent": candidate["program_id"] == incumbent_program,
        "fpga_valid": True,
        "candidate_latency_ms": latency,
        "incumbent_latency_ms": incumbent_latency,
        "target_log_latency_ratio": math.log(latency / incumbent_latency),
        "candidate_metrics": {name: metrics[name] for name in (
            "total_request_bytes", "total_request_calls", "submissions",
            "uop_load_bytes", "instruction_bytes",
        )},
    }


def first_oracle_budget(method):
    for budget in (1, 2, 3):
        row = method["B{}".format(budget)]
        if row["selected_candidate_set_id"] == row["oracle_candidate_set_id"]:
            return budget
    return None


def run(args):
    pool_dir = Path(args.board_pool).resolve()
    timing_dir = Path(args.timing_freeze).resolve()
    boot_dir = Path(args.boot).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    manifests = {
        "board_pool": verify(pool_dir),
        "timing_freeze": verify(timing_dir),
        "boot1": verify(boot_dir),
    }
    pool_contract = read(pool_dir / "contract.json")
    timing_contract = read(timing_dir / "contract.json")
    boot_summary = read(boot_dir / "summary.json")
    if (pool_contract.get("strict_latency_unseen") is not True
            or timing_contract.get("strict_latency_unseen") is not True):
        raise ValueError("analysis inputs are not strict-unseen freezes")
    if timing_contract.get("latency_values_collected") is not False:
        raise ValueError("timing freeze already contained latency")
    if (boot_summary.get("status") != "BOOT1_COMPLETE"
            or boot_summary.get("all_post_qualification_outputs_correct") is not True
            or boot_summary.get("session_spliced") is not False):
        raise ValueError("boot-1 input is not a complete correct session")

    candidates = read_jsonl(pool_dir / "candidates.jsonl")
    by_program = {row["program_id"]: row for row in candidates}
    policies = read(timing_dir / "task_policies.json")
    reference = read(boot_dir / "reference_summary.json")
    confirmation = read(boot_dir / "confirmation_summary.json")
    decisions = read(boot_dir / "selection_decisions_frozen.json")
    if set(reference) != set(policies) or len(reference) != 2:
        raise ValueError("reference tasks differ from the two frozen tasks")

    task_results, budget_results, same_tile_rows = {}, {}, []
    pair_gains = []
    for task_id, ref in sorted(reference.items()):
        policy = policies[task_id]
        incumbent = policy["incumbent_program_id"]
        incumbent_latency = float(
            ref["program_statistics"][incumbent]["median_of_round_medians_ms"]
        )
        task_candidates = [row for row in candidates if row["execution_task_id"] == task_id]
        eval_rows = [evaluation_row(
            row, ref["program_statistics"][row["program_id"]], incumbent, incumbent_latency
        ) for row in task_candidates]
        evaluation = evaluate_task_pool(eval_rows, budgets=(1, 2, 3), random_replicates=1000)
        v4_order = policy["selector_v4_order"]
        evaluation["deterministic_methods"]["selector_v4"] = {
            "B{}".format(budget): evaluate_order(eval_rows, v4_order, budget)
            for budget in (1, 2, 3)
        }
        evaluation["strict_latency_unseen_at_freeze"] = True
        evaluation["performance_model_claim"] = False
        budget_results[task_id] = evaluation

        families = {}
        for row in task_candidates:
            families.setdefault(row["family_id"], {})[row["residence_mode"]] = row
        for family_id, modes in sorted(families.items()):
            baseline = modes["input_stationary"]
            resident = modes["input_weight_resident_transition_safe"]
            baseline_stats = ref["program_statistics"][baseline["program_id"]]
            resident_stats = ref["program_statistics"][resident["program_id"]]
            gains = [100.0 * (1.0 - test / base) for base, test in zip(
                baseline_stats["round_medians_ms"], resident_stats["round_medians_ms"]
            )]
            row = {
                "execution_task_id": task_id,
                "workload_id": resident["workload_id"],
                "family_id": family_id,
                "tile": resident["visible_features"],
                "input_program_id": baseline["program_id"],
                "mode11_program_id": resident["program_id"],
                "input_latency_ms": baseline_stats["median_of_round_medians_ms"],
                "mode11_latency_ms": resident_stats["median_of_round_medians_ms"],
                "median_improvement_pct": float(statistics.median(gains)),
                "round_improvements_pct": gains,
                "wins": sum(value > 0.0 for value in gains),
                "input_wgt_request_bytes": baseline["dma_features"]["totals"][
                    "load_buffer_2d_wgt_bytes"
                ],
                "mode11_wgt_request_bytes": resident["dma_features"]["totals"][
                    "load_buffer_2d_wgt_bytes"
                ],
                "input_total_request_bytes": baseline["timing_freeze_metrics"][
                    "total_request_bytes"
                ],
                "mode11_total_request_bytes": resident["timing_freeze_metrics"][
                    "total_request_bytes"
                ],
                "input_submissions": baseline["timing_freeze_metrics"]["submissions"],
                "mode11_submissions": resident["timing_freeze_metrics"]["submissions"],
                "scope": "same Tile, six paired boot-1 reference rounds; strict-unseen at freeze",
            }
            same_tile_rows.append(row)
            pair_gains.append(row["median_improvement_pct"])

        oracle = ref["pool_oracle_program_id"]
        selected = decisions[task_id]["selected_program_id"]
        paired = confirmation[task_id]["paired"]
        methods = evaluation["deterministic_methods"]
        task_results[task_id] = {
            "oracle_program_id": oracle,
            "oracle_mode": by_program[oracle]["residence_mode"],
            "oracle_latency_ms": ref["pool_oracle_latency_ms"],
            "selected_program_id": selected,
            "selected_mode": by_program[selected]["residence_mode"],
            "selected_is_oracle": selected == oracle,
            "confirmation": paired,
            "boot1_stable_threshold_met": (
                paired["median_improvement_pct"] >= 2.0 and paired["wins"] >= 5
            ),
            "first_oracle_budget": {
                method: first_oracle_budget(rows)
                for method, rows in methods.items()
                if method != "incumbent_only"
            },
        }

    methods = ("bytes_only", "calls_only", "command_only", "selector_v4")
    summary = {
        "schema": SCHEMA,
        "status": "STRICT_UNSEEN_BOOT1_ANALYSIS_COMPLETE",
        "task_count": 2,
        "candidate_count": 30,
        "boot_id": boot_summary["boot_id"],
        "strict_latency_unseen_at_freeze": True,
        "cross_boot_confirmed": False,
        "same_tile_pair_count": len(same_tile_rows),
        "same_tile_mode11_six_of_six_wins": sum(row["wins"] == 6 for row in same_tile_rows),
        "same_tile_median_improvement_pct": float(statistics.median(pair_gains)),
        "same_tile_improvement_range_pct": [float(min(pair_gains)), float(max(pair_gains))],
        "oracle_mode11_tasks": sum(
            row["oracle_mode"] == "input_weight_resident_transition_safe"
            for row in task_results.values()
        ),
        "selector_v4_selected_oracle_tasks": sum(
            row["selected_is_oracle"] for row in task_results.values()
        ),
        "boot1_stable_threshold_tasks": sum(
            row["boot1_stable_threshold_met"] for row in task_results.values()
        ),
        "oracle_hits_by_B3": {
            method: sum(
                budget_results[task_id]["deterministic_methods"][method]["B3"][
                    "selected_candidate_set_id"
                ] == budget_results[task_id]["deterministic_methods"][method]["B3"][
                    "oracle_candidate_set_id"
                ]
                for task_id in sorted(task_results)
            ) for method in methods
        },
        "performance_model_claim": False,
        "safe_pruning_claim": False,
        "task_results": task_results,
        "manifests": manifests,
        "claim_boundary": (
            "valid strict-unseen boot-1 pool evidence; stable performance and deployment still "
            "require a second distinct boot and complete-graph validation"
        ),
    }
    output.mkdir(parents=True)
    write_jsonl(output / "same_tile_pairs.jsonl", same_tile_rows)
    write(output / "budget_evaluation.json", budget_results)
    write(output / "summary.json", summary)
    write(output / "artifact_hashes.json", {
        "artifacts": {path.name: sha256_file(path) for path in sorted(output.iterdir())
                      if path.is_file() and path.name != "artifact_hashes.json"},
        "source_sha256": sha256_file(Path(__file__).resolve()),
    })
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--timing-freeze", type=Path, default=DEFAULT_TIMING)
    parser.add_argument("--boot", type=Path, default=DEFAULT_BOOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

