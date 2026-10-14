#!/usr/bin/env python3
"""Freeze the FPGA-correct YOLO-576 timing pool without replacement.

The prospective 34-program pool and its shortlist order remain immutable.
Programs rejected by isolated FPGA correctness consume their frozen positions,
receive no timing, and are not replaced.  The emitted board pool contains only
FPGA-correct programs so an invalid program cannot poison a formal timing
session.  No latency is read or created here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_FREEZE = GROUPED / "20260924_c3_yolo576_evaluation_freeze_run01"
DEFAULT_CORRECTNESS = GROUPED / "20260924_c3_yolo576_fpga_correctness_waves_run02"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo576_fpga_admitted_timing_run01"
SCHEMA = "c3_yolo576_fpga_admitted_timing_v1"


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
    Path(path).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                          encoding="utf-8")


def verify(directory):
    directory = Path(directory).resolve()
    ledger_path = directory / "artifact_hashes.json"
    ledger = read(ledger_path)
    artifacts = ledger.get("artifacts", ledger)
    for name, expected in artifacts.items():
        if not isinstance(expected, str) or name in {"source", "source_sha256"}:
            continue
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("artifact hash mismatch: {}".format(path))
    return sha256_file(ledger_path)


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    manifests = {
        "prospective_freeze": verify(args.freeze),
        "fpga_correctness": verify(args.correctness),
    }
    source_contract = read(Path(args.freeze) / "contract.json")
    correctness_summary = read(Path(args.correctness) / "summary.json")
    if source_contract.get("latency_values_read") is not False:
        raise ValueError("prospective freeze is not latency blind")
    if correctness_summary.get("status") != "FPGA_CORRECTNESS_QUALIFICATION_COMPLETE":
        raise ValueError("FPGA correctness qualification is incomplete")
    if correctness_summary.get("latency_values_collected") is not False:
        raise ValueError("correctness qualification exposed latency")

    source_rows = read_jsonl(Path(args.freeze) / "candidates.jsonl")
    qualification = read_jsonl(Path(args.correctness) / "fpga_correctness.jsonl")
    if [row["candidate_id"] for row in qualification] != [row["candidate_id"] for row in source_rows]:
        raise ValueError("FPGA qualification order differs from prospective board pool")
    if len({row["candidate_id"] for row in qualification}) != len(source_rows):
        raise ValueError("FPGA qualification is not one-to-one")
    status_by_program = {row["program_id"]: row for row in qualification}
    passed = {row["program_id"] for row in qualification if row["status"] == "passed"}
    rejected = {row["program_id"] for row in qualification if row["status"] != "passed"}
    if len(passed) != correctness_summary["fpga_correct_count"] or not rejected:
        raise ValueError("FPGA correctness counts do not match summary")

    full_tasks = read(Path(args.freeze) / "full_pool_tasks.json")
    limited_tasks = read(Path(args.freeze) / "limited_budget_tasks.json")
    policies = read(Path(args.freeze) / "task_policies.json")
    revised_policies = {}
    for task_id, policy in policies.items():
        row = json.loads(json.dumps(policy))
        declared_pool = list(row["complete_local_program_pool"])
        frozen_shortlist = list(row["shortlist"]["shortlist_candidate_ids"])
        row["declared_complete_local_program_pool"] = declared_pool
        row["complete_local_program_pool"] = [value for value in declared_pool if value in passed]
        row["fpga_rejected_program_ids"] = [value for value in declared_pool if value in rejected]
        row["shortlist"]["frozen_attempt_candidate_ids"] = frozen_shortlist
        row["shortlist"]["shortlist_candidate_ids"] = [
            value for value in frozen_shortlist if value in passed
        ]
        row["shortlist"]["fpga_rejected_attempt_candidate_ids"] = [
            value for value in frozen_shortlist if value in rejected
        ]
        row["shortlist"]["post_correctness_policy"] = (
            "remove FPGA-invalid identities from timing only; preserve original B1/B2/B3 attempts; "
            "do not fill vacated positions"
        )
        if row["incumbent_program_id"] not in passed:
            raise ValueError("incumbent is FPGA-invalid for task {}".format(task_id))
        revised_policies[task_id] = row

    admitted_rows = [row for row in source_rows if row["program_id"] in passed]
    feature_rows = read_jsonl(Path(args.freeze) / "program_features.jsonl")
    for row in feature_rows:
        qrow = status_by_program.get(row["program_id"])
        if qrow is not None:
            row["fpga_correct"] = qrow["status"] == "passed"
            row["fpga_correctness_wave"] = qrow["wave"]
            row["fpga_correctness_manifest_sha256"] = manifests["fpga_correctness"]

    full_ids = {row["task_id"] for row in full_tasks}
    limited_ids = {row["task_id"] for row in limited_tasks}
    available_by_task = {}
    for row in admitted_rows:
        available_by_task.setdefault(row["execution_task_id"], set()).add(row["program_id"])
    for task_id, policy in revised_policies.items():
        expected = set()
        if task_id in full_ids:
            expected.update(policy["complete_local_program_pool"])
        if task_id in limited_ids:
            expected.update(policy["shortlist"]["shortlist_candidate_ids"])
        if available_by_task.get(task_id, set()) != expected:
            raise ValueError("admitted timing pool/task policy mismatch for {}".format(task_id))

    output.mkdir(parents=True)
    write(output / "full_pool_tasks.json", full_tasks)
    write(output / "limited_budget_tasks.json", limited_tasks)
    write(output / "task_policies.json", revised_policies)
    write_jsonl(output / "program_features.jsonl", feature_rows)
    write_jsonl(output / "candidates.jsonl", admitted_rows)
    write_jsonl(output / "fpga_rejected_programs.jsonl", [
        {
            "program_id": row["program_id"],
            "candidate_id": row["candidate_id"],
            "execution_task_id": row["execution_task_id"],
            "workload_id": row["workload_id"],
            "residence_mode": row["residence_mode"],
            "wave": row["wave"],
            "seed_checks": row["seed_checks"],
            "mismatch_count": row["first_failure"]["mismatch_count"],
            "timing_eligible": False,
            "replacement_allowed": False,
        }
        for row in qualification if row["status"] != "passed"
    ])
    contract = {
        "schema": SCHEMA,
        "status": "FPGA_CORRECT_TIMING_POOL_FROZEN_BEFORE_LATENCY",
        "boot_id_used_for_correctness_only": correctness_summary["boot_id"],
        "prospective_candidate_count": len(source_rows),
        "timing_candidate_count": len(admitted_rows),
        "fpga_rejected_count": len(rejected),
        "candidate_order": [row["candidate_id"] for row in admitted_rows],
        "program_order": [row["program_id"] for row in admitted_rows],
        "full_pool_task_ids": sorted(full_ids),
        "limited_budget_task_ids": sorted(limited_ids),
        "workload_counts": {
            workload_id: sum(row["workload_id"] == workload_id for row in admitted_rows)
            for workload_id in sorted({row["workload_id"] for row in admitted_rows})
        },
        "invalid_candidate_policy": "frozen attempt consumes budget; no replacement; no timing",
        "oracle_scope": "all FPGA-correct programs in each originally declared complete pool",
        "selection_confirmation_separation": True,
        "manifests": manifests,
        "board_contacted": True,
        "board_contact_scope": "correctness only",
        "latency_values_read": False,
        "performance_labels_collected": False,
    }
    write(output / "contract.json", contract)
    write(output / "summary.json", contract)
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "results.jsonl").write_text("", encoding="utf-8")
    write(output / "artifact_hashes.json", {
        "artifacts": {
            path.name: sha256_file(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "artifact_hashes.json"
        },
        "source": sha256_file(Path(__file__).resolve()),
    })
    print(json.dumps(contract, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--correctness", type=Path, default=DEFAULT_CORRECTNESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
