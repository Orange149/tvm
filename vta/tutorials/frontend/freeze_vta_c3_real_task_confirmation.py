#!/usr/bin/env python3
"""Freeze boot-2 identities after boot-1 selection, without board contact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_vta_c3_real_task_board_boot import (
    DEFAULT_POOL,
    DEFAULT_SELECTOR,
    GROUPED,
    sha256_file,
    verify_flat,
    write_json,
)


DEFAULT_BOOT1 = GROUPED / "20260924_c3_real_task_board_boot1_run02"
SCHEMA = "c3_real_task_confirmation_freeze_v1"


def confirmation_programs(selector, decisions):
    result = {}
    for workload_id, frozen in selector["workloads"].items():
        identities = {frozen["incumbent_program_id"]}
        decision = decisions[workload_id]
        for budgets in decision.get("budget_decisions", {}).values():
            for row in budgets.values():
                identities.add(row["selected_program_id"])
        result[workload_id] = sorted(identities)
    return result


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    boot1 = Path(args.boot1_dir).resolve()
    selector_dir = Path(args.selector_dir).resolve()
    pool_dir = Path(args.board_pool_dir).resolve()
    summary = json.loads((boot1 / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "BOOT1_COMPLETE" or summary.get("session_spliced"):
        raise ValueError("boot1 is not a complete non-spliced session")
    decisions = json.loads(
        (boot1 / "selection_decisions_frozen.json").read_text(encoding="utf-8")
    )
    correctness = json.loads(
        (boot1 / "fpga_correctness_summary.json").read_text(encoding="utf-8")
    )
    selector = json.loads((selector_dir / "contract.json").read_text(encoding="utf-8"))
    programs = confirmation_programs(selector, decisions)
    correct_programs = {
        program["program_id"]
        for workload in selector["workloads"].values()
        for program in workload["programs"]
        if program["candidate_id"] in set(correctness["fpga_correct_candidate_ids"])
    }
    if any(program not in correct_programs for values in programs.values() for program in values):
        raise ValueError("boot1 selected a program that did not pass FPGA correctness")
    output.mkdir(parents=True)
    contract = {
        "schema": SCHEMA,
        "status": "frozen_after_boot1_before_boot2",
        "board_contacted": False,
        "performance_labels_collected_by_this_step": False,
        "boot1_dir": str(boot1),
        "boot1_boot_id": summary["boot_id"],
        "boot1_summary_sha256": sha256_file(boot1 / "summary.json"),
        "boot1_decisions_sha256": sha256_file(
            boot1 / "selection_decisions_frozen.json"
        ),
        "boot1_reference_sha256": sha256_file(boot1 / "reference_summary.json"),
        "boot1_correctness_sha256": sha256_file(
            boot1 / "fpga_correctness_summary.json"
        ),
        "selector_manifest_sha256": verify_flat(selector_dir),
        "pool_manifest_sha256": verify_flat(pool_dir),
        "requires_changed_boot_id": True,
        "selected_identities_may_change": False,
        "confirmation_program_ids": programs,
        "confirmation_rounds": 8,
        "confirmation_seeds": [0, 20250901, 20260910],
        "full_pool_reference_rounds": 6,
        "full_pool_policy": (
            "re-run FPGA correctness for every frozen local candidate; time the boot-2 "
            "FPGA-correct subset; cross-boot claims use the intersection"
        ),
        "oracle_scope": "per-boot locally qualified and FPGA-correct frozen pool",
        "latency_exposure_status": "nonadaptive_confirmation_after_disclosed_invalid_attempt",
    }
    write_json(output / "contract.json", contract)
    write_json(
        output / "summary.json",
        {
            "schema": SCHEMA,
            "status": contract["status"],
            "workload_count": len(programs),
            "unique_confirmation_programs": sum(len(values) for values in programs.values()),
            "confirmation_program_ids": programs,
        },
    )
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                "contract.json": sha256_file(output / "contract.json"),
                "summary.json": sha256_file(output / "summary.json"),
            },
            "source_sha256": sha256_file(Path(__file__).resolve()),
        },
    )
    print(json.dumps(json.loads((output / "summary.json").read_text()), indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boot1-dir", type=Path, default=DEFAULT_BOOT1)
    parser.add_argument("--selector-dir", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--board-pool-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
