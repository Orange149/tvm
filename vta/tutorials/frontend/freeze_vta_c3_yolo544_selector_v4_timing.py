#!/usr/bin/env python3
"""Freeze YOLO-544 selector-v4 timing tasks after complete FPGA correctness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_POOL = GROUPED / "20260924_c3_yolo544_selector_v4_board_pool_freeze_run01"
DEFAULT_CORRECTNESS = GROUPED / "20260924_c3_yolo544_selector_v4_fpga_correctness_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo544_selector_v4_timing_freeze_run01"
SCHEMA = "c3_yolo544_selector_v4_timing_freeze_v1"


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
    if (pool_contract.get("status") != "FROZEN_STRICT_UNSEEN_BEFORE_BOARD_CONTACT_OR_LATENCY"
            or pool_contract.get("strict_latency_unseen") is not True
            or pool_contract.get("performance_labels_collected") is not False):
        raise ValueError("board pool is not a strict-unseen label-free freeze")
    correctness = read(correctness_dir / "summary.json")
    if (correctness.get("status") != "FPGA_CORRECTNESS_QUALIFICATION_COMPLETE"
            or correctness.get("candidate_count") != 30
            or correctness.get("fpga_correct_count") != 30
            or correctness.get("fpga_rejected_count") != 0
            or correctness.get("latency_values_collected") is not False
            or correctness.get("wave_count") != 1):
        raise ValueError("timing requires complete one-wave 30/30 FPGA correctness")
    records = read_jsonl(correctness_dir / "fpga_correctness.jsonl")
    if len(records) != 30 or any(row.get("status") != "passed" for row in records):
        raise ValueError("FPGA correctness record set is incomplete")

    source_policies = read(pool_dir / "task_policies.json")
    task_ids = sorted(source_policies)
    if task_ids != sorted(pool_contract["execution_task_ids"]) or len(task_ids) != 2:
        raise ValueError("expected the two frozen strict-unseen tasks")
    policies = {}
    for task_id in task_ids:
        source = source_policies[task_id]
        budgets = source["B1_B2_B3"]
        order = source["selector_v4"]["ordered_candidate_ids"]
        if budgets != {"B1": order[:1], "B2": order[:2], "B3": order[:3]}:
            raise ValueError("v4 budgets are not prefixes of one frozen order")
        incumbent = source["incumbent_program_id"]
        complete = source["complete_program_pool"]
        if incumbent not in complete or len(complete) != 15:
            raise ValueError("task lacks complete pool or incumbent")
        shortlist = [incumbent] + budgets["B3"]
        if len(shortlist) != 4 or len(set(shortlist)) != 4:
            raise ValueError("selector-v4 shortlist must contain incumbent plus three new programs")
        policies[task_id] = {
            "execution_task_id": task_id,
            "workload_id": source["workload_id"],
            "incumbent_program_id": incumbent,
            "complete_local_program_pool": complete,
            "selector_v4_order": order,
            "shortlist": {"shortlist_candidate_ids": shortlist},
            "B1_B2_B3": budgets,
            "selector_kind": "label_free_low_movement_tile_diversity",
            "strict_latency_unseen": True,
            "performance_model_claim": False,
        }

    tasks = [{
        "task_id": task_id,
        "role": "strict_unseen_complete_pool_reference_and_limited_budget",
        "latency_unseen": True,
    } for task_id in task_ids]
    output.mkdir(parents=True)
    write(output / "full_pool_tasks.json", tasks)
    write(output / "limited_budget_tasks.json", tasks)
    write(output / "task_policies.json", policies)
    contract = {
        "schema": SCHEMA,
        "status": "FROZEN_AFTER_FPGA_CORRECTNESS_BEFORE_FIRST_LATENCY",
        "input_resolution": 544,
        "execution_task_ids": task_ids,
        "execution_task_count": 2,
        "candidate_count": 30,
        "fpga_correct_count": 30,
        "correctness_boot_id": correctness["boot_id"],
        "budgets": pool_contract["budgets"],
        "selector_contract_schema": pool_contract["selector_contract_schema"],
        "strict_latency_unseen": True,
        "performance_labels_collected": False,
        "board_contacted_for_correctness_only": True,
        "latency_values_collected": False,
        "performance_model_claim": False,
        "safe_pruning_claim": False,
        "manifests": manifests,
        "claim_boundary": (
            "timing identities and orders frozen; no latency, oracle, selector superiority, "
            "stable-improvement, cross-boot, or complete-graph result yet"
        ),
    }
    write(output / "contract.json", contract)
    write(output / "summary.json", contract)
    write(output / "artifact_hashes.json", {
        "artifacts": {path.name: sha256_file(path) for path in sorted(output.iterdir())
                      if path.is_file() and path.name != "artifact_hashes.json"},
        "source_sha256": sha256_file(Path(__file__).resolve()),
    })
    print(json.dumps(contract, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--correctness", type=Path, default=DEFAULT_CORRECTNESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

