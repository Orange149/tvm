#!/usr/bin/env python3
"""Analyze the development-only mode-11 timing pool without changing selection."""

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
DEFAULT_FREEZE = GROUPED / "20260924_c3_yolo576_transition_safe_timing_freeze_run01"
DEFAULT_BOOT = GROUPED / "20260924_c3_yolo576_transition_safe_timing_boot1_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo576_transition_safe_timing_analysis_run02"
SCHEMA = "c3_yolo576_transition_safe_timing_analysis_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def verify(directory):
    directory = Path(directory).resolve()
    manifest = directory / "artifact_hashes.json"
    ledger = read(manifest)
    for name, expected in ledger.get("artifacts", {}).items():
        if not isinstance(expected, str):
            continue
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("artifact hash mismatch: {}".format(path))
    return sha256_file(manifest)


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
        "candidate_metrics": {
            "total_request_bytes": metrics["total_request_bytes"],
            "total_request_calls": metrics["total_request_calls"],
            "submissions": metrics["submissions"],
            "uop_load_bytes": metrics["command_uop_bytes"],
            "instruction_bytes": metrics["instruction_bytes"],
        },
    }


def run(args):
    freeze_dir = Path(args.freeze).resolve()
    boot_dir = Path(args.boot).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    manifests = {"freeze": verify(freeze_dir), "boot": verify(boot_dir)}
    freeze_contract = read(freeze_dir / "contract.json")
    boot_summary = read(boot_dir / "summary.json")
    if freeze_contract.get("strict_holdout_claimed") is not False:
        raise ValueError("analysis requires the development-only freeze")
    if boot_summary.get("status") != "BOOT1_COMPLETE" or not boot_summary.get(
            "all_post_qualification_outputs_correct"):
        raise ValueError("input boot is not a complete correct development session")

    candidates = read_jsonl(freeze_dir / "candidates.jsonl")
    by_program = {row["program_id"]: row for row in candidates}
    policies = read(freeze_dir / "task_policies.json")
    reference = read(boot_dir / "reference_summary.json")
    confirmation = read(boot_dir / "confirmation_summary.json")
    decisions = read(boot_dir / "selection_decisions_frozen.json")

    task_results = {}
    same_tile_rows = []
    budget_results = {}
    all_pair_gains = []
    for task_id, ref in sorted(reference.items()):
        policy = policies[task_id]
        incumbent_program = policy["incumbent_program_id"]
        incumbent_latency = float(
            ref["program_statistics"][incumbent_program]["median_of_round_medians_ms"]
        )
        task_candidates = [row for row in candidates if row["execution_task_id"] == task_id]
        eval_rows = [evaluation_row(
            row, ref["program_statistics"][row["program_id"]],
            incumbent_program, incumbent_latency,
        ) for row in task_candidates]
        standard = evaluate_task_pool(eval_rows, budgets=(1, 2, 3), random_replicates=1000)
        static_fallback = {
            "B{}".format(budget): evaluate_order(
                eval_rows, policy["selector_v3_order"], budget
            ) for budget in (1, 2, 3)
        }
        standard["deterministic_methods"]["static_fallback"] = static_fallback
        frozen_role_order = policy["B1_B2_B3"]["B3"]
        standard["deterministic_methods"]["frozen_role_shortlist"] = {
            "B{}".format(budget): evaluate_order(
                eval_rows,
                frozen_role_order
                + [row["candidate_set_id"] for row in eval_rows
                   if not row["is_incumbent"]
                   and row["candidate_set_id"] not in frozen_role_order],
                budget,
            ) for budget in (1, 2, 3)
        }
        standard["selector_v3_model_claim"] = False
        standard["task_latency_unseen"] = False
        budget_results[task_id] = standard

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
            baseline_dma = baseline["dma_features"]["totals"]
            resident_dma = resident["dma_features"]["totals"]
            row = {
                "execution_task_id": task_id,
                "workload_id": resident["workload_id"],
                "family_id": family_id,
                "tile": resident["visible_features"],
                "input_program_id": baseline["program_id"],
                "mode11_program_id": resident["program_id"],
                "input_latency_ms": baseline_stats["median_of_round_medians_ms"],
                "mode11_latency_ms": resident_stats["median_of_round_medians_ms"],
                "round_improvements_pct": gains,
                "median_improvement_pct": statistics.median(gains),
                "wins": sum(value > 0.0 for value in gains),
                "input_wgt_request_bytes": baseline_dma["load_buffer_2d_wgt_bytes"],
                "mode11_wgt_request_bytes": resident_dma["load_buffer_2d_wgt_bytes"],
                "input_total_request_bytes": baseline["timing_freeze_metrics"]["total_request_bytes"],
                "mode11_total_request_bytes": resident["timing_freeze_metrics"]["total_request_bytes"],
                "input_submissions": baseline["timing_freeze_metrics"]["submissions"],
                "mode11_submissions": resident["timing_freeze_metrics"]["submissions"],
                "scope": "same Tile, six paired reference rounds; development task",
            }
            same_tile_rows.append(row)
            all_pair_gains.append(row["median_improvement_pct"])

        oracle = ref["pool_oracle_program_id"]
        confirmed = confirmation[task_id]["paired"]
        task_results[task_id] = {
            "oracle_program_id": oracle,
            "oracle_mode": by_program[oracle]["residence_mode"],
            "oracle_latency_ms": ref["pool_oracle_latency_ms"],
            "selected_program_id": decisions[task_id]["selected_program_id"],
            "selected_mode": by_program[decisions[task_id]["selected_program_id"]]["residence_mode"],
            "selected_is_oracle": decisions[task_id]["selected_program_id"] == oracle,
            "confirmation": confirmed,
            "stable_improvement": (
                confirmed["median_improvement_pct"] >= 2.0 and confirmed["wins"] >= 5
            ),
        }

    summary = {
        "schema": SCHEMA,
        "status": "DEVELOPMENT_MODE11_TIMING_ANALYSIS_COMPLETE",
        "task_count": len(task_results),
        "candidate_count": len(candidates),
        "same_tile_pair_count": len(same_tile_rows),
        "same_tile_mode11_wins": sum(row["wins"] == 6 for row in same_tile_rows),
        "same_tile_median_improvement_pct": statistics.median(all_pair_gains),
        "same_tile_improvement_range_pct": [min(all_pair_gains), max(all_pair_gains)],
        "oracle_mode11_tasks": sum(row["oracle_mode"] == "input_weight_resident_transition_safe"
                                    for row in task_results.values()),
        "frozen_role_shortlist_selected_oracle_tasks": sum(
            row["selected_is_oracle"] for row in task_results.values()
        ),
        "stable_confirmation_tasks": sum(row["stable_improvement"]
                                           for row in task_results.values()),
        "task_results": task_results,
        "task_latency_unseen": False,
        "strict_holdout_claimed": False,
        "selector_v3_model_claim": False,
        "boot_id": boot_summary["boot_id"],
        "manifests": manifests,
    }
    output.mkdir(parents=True)
    write_jsonl(output / "same_tile_pairs.jsonl", same_tile_rows)
    write(output / "budget_evaluation.json", budget_results)
    write(output / "summary.json", summary)
    write(output / "artifact_hashes.json", {
        "artifacts": {
            path.name: sha256_file(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "artifact_hashes.json"
        },
        "source_sha256": sha256_file(Path(__file__).resolve()),
    })
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--boot", type=Path, default=DEFAULT_BOOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
