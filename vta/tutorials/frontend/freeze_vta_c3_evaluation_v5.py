#!/usr/bin/env python3
"""Create the superseding YOLO-512 C3 v5 evaluation freeze.

The earlier timing freeze remains immutable evidence.  This freeze consumes
the same latency-unseen candidates and completed FPGA correctness gate, but
separates production and legacy pools and removes known FPGA-invalid programs
from post-qualification B1/B2/B3 budgets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from c3_evaluation_v5 import (
    SCHEMA,
    confirmation_schedule,
    freeze_task_policies,
    qualification_schedule,
    read_json,
    read_jsonl,
    reference_schedule,
    seal_flat,
    selection_schedule,
    sha256_file,
    validate_call_counts,
    verify_flat,
    write_json,
)


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_POOL = GROUPED / "20260924_c3_yolo512_selector_v4_production_board_pool_freeze_run02"
DEFAULT_CORRECTNESS = GROUPED / "20260924_c3_yolo512_selector_v4_production_fpga_correctness_run04"
DEFAULT_PRIOR_FREEZE = GROUPED / "20260924_c3_yolo512_selector_v4_production_timing_freeze_run01"
DEFAULT_CROSS = GROUPED / "20260924_c3_yolo512_selector_v4_production_cross_compile_run01"
DEFAULT_HELPER = GROUPED / "20260926_c3_evaluation_v5_helper_build_run02"
DEFAULT_RUNTIME = GROUPED / "20260922_c3_thread_sync_freeze_run04/contract.json"
DEFAULT_OUTPUT = GROUPED / "20260926_c3_yolo512_evaluation_v5_freeze_run01"


def run(args):
    pool = Path(args.board_pool).resolve()
    correctness = Path(args.correctness).resolve()
    prior = Path(args.prior_timing_freeze).resolve()
    cross = Path(args.cross_compile).resolve()
    helper = Path(args.helper_build).resolve()
    runtime_path = Path(args.runtime_contract).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))

    manifests = {
        "board_pool": verify_flat(pool),
        "correctness": verify_flat(correctness),
        "prior_timing_freeze": verify_flat(prior),
        "cross_compile": verify_flat(cross),
        "helper_build": verify_flat(helper),
    }
    prior_contract = read_json(prior / "contract.json")
    if (prior_contract.get("status") != "FROZEN_AFTER_FPGA_CORRECTNESS_BEFORE_FIRST_LATENCY"
            or prior_contract.get("latency_values_collected") is not False):
        raise ValueError("prior freeze is not the immutable pre-latency source")
    correctness_summary = read_json(correctness / "summary.json")
    if (correctness_summary.get("status") != "FPGA_CORRECTNESS_QUALIFICATION_COMPLETE"
            or correctness_summary.get("latency_values_collected") is not False):
        raise ValueError("correctness source is incomplete or contains latency")
    helper_contract = read_json(helper / "contract.json")
    helper_binary = Path(helper_contract["binary"])
    if sha256_file(helper_binary) != helper_contract["binary_sha256"]:
        raise ValueError("v5 helper binary changed")
    runtime = read_json(runtime_path)
    for name, path in runtime["runtime_paths"].items():
        if sha256_file(path) != runtime["runtime_sha256"][name]:
            raise ValueError("frozen runtime changed: {}".format(name))

    candidates = read_jsonl(pool / "candidates.jsonl")
    source_policies = read_json(pool / "task_policies.json")
    correctness_rows = read_jsonl(correctness / "fpga_correctness.jsonl")
    policies = freeze_task_policies(candidates, source_policies, correctness_rows)
    counts = validate_call_counts(policies)
    by_program = {row["program_id"]: row for row in candidates}

    output.mkdir(parents=True)
    write_json(output / "task_policies.json", policies)
    write_json(output / "candidate_membership.json", {
        task_id: [{
            "program_id": program,
            "candidate_id": by_program[program]["candidate_id"],
            "residence_mode": by_program[program]["residence_mode"],
            "implementation_mode": by_program[program]["implementation_mode"],
            "pool": (
                "production" if program in policy["production_program_pool"]
                else "legacy_controlled" if program in policy["legacy_controlled_program_pool"]
                else "fpga_invalid"
            ),
        } for program in policy["declared_program_pool"]]
        for task_id, policy in policies.items()
    })
    write_json(output / "qualification_schedules.json", {
        "production": qualification_schedule(policies, "production"),
        "legacy": qualification_schedule(policies, "legacy"),
    })
    write_json(output / "selection_schedule.json", selection_schedule(policies))
    write_json(output / "reference_schedules.json", {
        "production": reference_schedule(policies, "production"),
        "legacy": reference_schedule(policies, "legacy"),
    })
    # This is a schema example only.  Actual winners are sealed after the valid
    # Boot-1 selection panel and generate the real confirmation schedule.
    write_json(output / "confirmation_schedule_schema.json", {
        "rounds": 6,
        "seeds": [0, 20250901, 20260910],
        "generator": "c3_evaluation_v5.confirmation_schedule",
        "example_incumbent_only": confirmation_schedule(
            policies, {task: row["incumbent_program_id"] for task, row in policies.items()}
        ),
    })
    contract = {
        "schema": SCHEMA,
        "status": "PRE_LATENCY_FROZEN",
        "supersedes_budget_semantics_of": str(prior),
        "does_not_modify_prior_evidence": True,
        "input_resolution": 512,
        "execution_task_ids": sorted(policies),
        "task_count": len(policies),
        "declared_candidate_count": 30,
        "fpga_valid_count": 28,
        "production_candidate_count": 16,
        "legacy_controlled_candidate_count": 12,
        "fpga_invalid_count": 2,
        "selection_panel": {
            "actual_acquisition_calls": counts["selection"],
            "per_method_B3_exposed_calls_two_tasks": 24,
            "rounds": 3,
            "one_seed_per_round": True,
        },
        "budgets": {
            "B1": 1, "B2": 2, "B3": 3,
            "unit": "additional FPGA-valid production candidate after qualification",
            "known_invalid_policy": "qualification cost only; absent from measurement budgets",
        },
        "primary_comparison": {"method": "selector_v4", "baseline": "bytes_only", "budget": "B3"},
        "oracle_scopes": {
            "production": "eight FPGA-valid production programs per task",
            "legacy": "six legacy programs per task under the common controlled-isolation wrapper",
            "unified_fourteen_program_oracle_forbidden": True,
        },
        "call_counts_per_boot": counts,
        "winner_state_machine": [
            "PRE_LATENCY_FROZEN", "SELECTION_RUNNING", "WINNERS_SEALED",
            "CONFIRMATION", "REFERENCE", "COMPLETE",
        ],
        "winner_retry_policy": (
            "after WINNERS_SEALED no later failure may trigger reselection; a new attempt keeps "
            "the sealed winners and never splices failed timing rows"
        ),
        "operator_success_gate": "both boots >=2% median gain and >=5/6 winning rounds",
        "graph_success_gate": "both boots >=1% median gain and all four five-call blocks positive",
        "strict_latency_unseen_before_this_freeze": True,
        "performance_labels_collected": False,
        "correctness_boot_id": correctness_summary["boot_id"],
        "requires_new_timing_boot_id": True,
        "helper_binary": str(helper_binary),
        "helper_sha256": helper_contract["binary_sha256"],
        "runtime_contract": str(runtime_path),
        "runtime_contract_sha256": sha256_file(runtime_path),
        "runtime_sha256": runtime["runtime_sha256"],
        "manifests": manifests,
        "claim_boundary": (
            "protocol, membership, orders, budgets and call schedules only; no latency, winner, "
            "oracle, stable improvement or graph result"
        ),
    }
    write_json(output / "contract.json", contract)
    write_json(output / "summary.json", contract)
    seal_flat(output, (Path(__file__).resolve(), Path(__import__("c3_evaluation_v5").__file__)))
    print(json.dumps(contract, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--correctness", type=Path, default=DEFAULT_CORRECTNESS)
    parser.add_argument("--prior-timing-freeze", type=Path, default=DEFAULT_PRIOR_FREEZE)
    parser.add_argument("--cross-compile", type=Path, default=DEFAULT_CROSS)
    parser.add_argument("--helper-build", type=Path, default=DEFAULT_HELPER)
    parser.add_argument("--runtime-contract", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
