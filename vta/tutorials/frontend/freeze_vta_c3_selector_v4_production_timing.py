#!/usr/bin/env python3
"""Freeze production-gated timing after correctness-only qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_POOL = GROUPED / "20260924_c3_yolo512_selector_v4_production_board_pool_freeze_run02"
DEFAULT_CORRECTNESS = GROUPED / "20260924_c3_yolo512_selector_v4_production_fpga_correctness_run04"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo512_selector_v4_production_timing_freeze_run01"
SCHEMA = "c3_selector_v4_production_timing_freeze_v1"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify(directory):
    directory = Path(directory).resolve()
    ledger = read(directory / "artifact_hashes.json")
    for name, expected in ledger.get("artifacts", {}).items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("artifact hash mismatch: {}".format(path))
    return sha256_file(directory / "artifact_hashes.json")


def run(args):
    pool_dir = Path(args.board_pool).resolve()
    correctness_dir = Path(args.correctness).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    manifests = {"board_pool": verify(pool_dir), "correctness": verify(correctness_dir)}
    pool_contract = read(pool_dir / "contract.json")
    if (pool_contract.get("status")
            != "FROZEN_STRICT_UNSEEN_PRODUCTION_GATED_BEFORE_BOARD_OR_LATENCY"
            or pool_contract.get("strict_latency_unseen") is not True
            or pool_contract.get("performance_labels_collected") is not False):
        raise ValueError("board pool is not a production-gated strict-unseen freeze")
    source_policies = read(pool_dir / "task_policies.json")
    expected_production_count = sum(
        len(policy["production_program_pool"]) for policy in source_policies.values()
    )
    correctness = read(correctness_dir / "summary.json")
    correctness_contract = read(correctness_dir / "contract.json")
    pool_scope = correctness_contract.get("pool_scope", "all")
    expected_correctness_count = (
        expected_production_count
        if pool_scope == "production"
        else int(pool_contract["isolated_reference_candidate_count"])
    )
    if (correctness.get("status") != "FPGA_CORRECTNESS_QUALIFICATION_COMPLETE"
            or correctness.get("candidate_count") != expected_correctness_count
            or correctness.get("latency_values_collected") is not False):
        raise ValueError("correctness-only qualification is incomplete")
    records = read_jsonl(correctness_dir / "fpga_correctness.jsonl")
    if len(records) != expected_correctness_count:
        raise ValueError("correctness record count differs from frozen pool")
    valid = {row["program_id"] for row in records if row["status"] == "passed"}
    invalid = {row["program_id"] for row in records if row["status"] != "passed"}
    if len(valid) != correctness["fpga_correct_count"] or len(invalid) != correctness[
            "fpga_rejected_count"]:
        raise ValueError("correctness summary does not match records")

    policies = {}
    for task_id, source in sorted(source_policies.items()):
        budgets = source["B1_B2_B3"]
        order = source["selector_v4_order"]
        if budgets != {"B1": order[:1], "B2": order[:2], "B3": order[:3]}:
            raise ValueError("v4 budgets are not prefixes of one frozen order")
        incumbent = source["incumbent_program_id"]
        production = source["production_program_pool"]
        reference = (
            production if pool_scope == "production"
            else source["complete_isolated_reference_program_pool"]
        )
        valid_reference = [program for program in reference if program in valid]
        invalid_reference = [program for program in reference if program in invalid]
        if incumbent not in valid_reference:
            raise ValueError("task lacks a valid incumbent or complete declared reference pool")
        if set(reference) != set(valid_reference) | set(invalid_reference):
            raise ValueError("correctness does not cover the selected reference scope")
        if pool_scope == "production" and set(reference) != set(production):
            raise ValueError("production-only qualification scope drift")
        attempted_b3 = budgets["B3"]
        valid_b3 = [program for program in attempted_b3 if program in valid]
        invalid_b3 = [program for program in attempted_b3 if program in invalid]
        shortlist = [incumbent] + valid_b3
        if len(set(shortlist)) != len(shortlist):
            raise ValueError("timing shortlist repeats a program")
        policies[task_id] = {
            "execution_task_id": task_id,
            "workload_id": source["workload_id"],
            "incumbent_program_id": incumbent,
            "complete_local_program_pool": valid_reference,
            "declared_complete_reference_pool": reference,
            "fpga_invalid_reference_programs": invalid_reference,
            "production_program_pool": production,
            "selector_v4_order": order,
            "shortlist": {"shortlist_candidate_ids": shortlist},
            "B1_B2_B3": budgets,
            "invalid_budget_attempts_through_B3": invalid_b3,
            "invalid_consumes_budget_without_replacement": True,
            "selector_kind": "production_gate_then_label_free_low_movement_tile_diversity",
            "strict_latency_unseen": True,
            "performance_model_claim": False,
        }

    tasks = [{
        "task_id": task_id,
        "role": "strict_unseen_fpga_valid_pool_reference_and_production_limited_budget",
        "latency_unseen": True,
    } for task_id in sorted(policies)]
    output.mkdir(parents=True)
    write(output / "full_pool_tasks.json", tasks)
    write(output / "limited_budget_tasks.json", tasks)
    write(output / "task_policies.json", policies)
    contract = {
        "schema": SCHEMA,
        "status": "FROZEN_AFTER_FPGA_CORRECTNESS_BEFORE_FIRST_LATENCY",
        "input_resolution": pool_contract["input_resolution"],
        "execution_task_ids": sorted(policies),
        "execution_task_count": len(policies),
        "declared_candidate_count": int(pool_contract["isolated_reference_candidate_count"]),
        "correctness_pool_scope": pool_scope,
        "correctness_candidate_count": expected_correctness_count,
        "production_candidate_count": expected_production_count,
        "fpga_correct_count": len(valid),
        "fpga_rejected_count": len(invalid),
        "correctness_boot_id": correctness["boot_id"],
        "correctness_wave_count": correctness["wave_count"],
        "budgets": {
            "B1": 1, "B2": 2, "B3": 3,
            "unit": "additional frozen attempts; invalid consumes budget without replacement",
        },
        "strict_latency_unseen": True,
        "performance_labels_collected": False,
        "board_contacted_for_correctness_only": True,
        "latency_values_collected": False,
        "performance_model_claim": False,
        "safe_pruning_claim": False,
        "manifests": manifests,
        "claim_boundary": (
            "timing identities and orders frozen after correctness only; no latency, oracle, "
            "selector superiority, stable improvement, cross-boot, graph or deployment result"
        ),
    }
    write(output / "contract.json", contract)
    write(output / "summary.json", contract)
    write(output / "artifact_hashes.json", {
        "artifacts": {
            path.name: sha256_file(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "artifact_hashes.json"
        },
        "source_sha256": sha256_file(Path(__file__).resolve()),
    })
    print(json.dumps(contract, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--correctness", type=Path, default=DEFAULT_CORRECTNESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
