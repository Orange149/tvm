#!/usr/bin/env python3
"""Post-hoc audit of the missing selector-v4 production eligibility gate.

This audit deliberately reads boot-1 latency and therefore cannot create a
new latency-unseen claim.  It preserves the original isolated-operator pool
oracle, applies the already existing completion-policy gate, and reports the
counterfactual production-eligible subpool separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from c3_selector_v4_deployment import production_static_order


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_POOL = GROUPED / "20260924_c3_yolo544_selector_v4_board_pool_freeze_run01"
DEFAULT_CORRECTNESS = GROUPED / "20260924_c3_yolo544_selector_v4_fpga_correctness_run01"
DEFAULT_BOOT = GROUPED / "20260924_c3_yolo544_selector_v4_timing_boot1_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo544_selector_v4_deployment_gate_posthoc_audit_run01"
SCHEMA = "c3_yolo544_selector_v4_deployment_gate_posthoc_audit_v1"


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


def best(program_ids, statistics):
    return min(program_ids, key=lambda value: (
        float(statistics[value]["median_of_round_medians_ms"]), value
    ))


def run(args):
    pool_dir = Path(args.board_pool).resolve()
    correctness_dir = Path(args.correctness).resolve()
    boot_dir = Path(args.boot).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    manifests = {
        "board_pool": verify(pool_dir),
        "correctness": verify(correctness_dir),
        "boot1": verify(boot_dir),
    }
    correctness_summary = read(correctness_dir / "summary.json")
    boot_summary = read(boot_dir / "summary.json")
    if (correctness_summary.get("fpga_correct_count") != 30
            or correctness_summary.get("latency_values_collected") is not False):
        raise ValueError("correctness-only input is incomplete or contains latency")
    if boot_summary.get("status") != "BOOT1_COMPLETE":
        raise ValueError("boot-1 timing input is incomplete")

    candidates = read_jsonl(pool_dir / "candidates.jsonl")
    certificates = {
        row["identity"]["candidate_id"]: row
        for row in read_jsonl(pool_dir / "mode11_composability_certificates.jsonl")
    }
    correct = {
        row["candidate_id"] for row in read_jsonl(correctness_dir / "fpga_correctness.jsonl")
        if row.get("status") == "passed"
    }
    policies = read(pool_dir / "task_policies.json")
    reference = read(boot_dir / "reference_summary.json")

    task_results = {}
    rejected_modes = {}
    for task_id, policy in sorted(policies.items()):
        task_rows = []
        for source in candidates:
            if source["execution_task_id"] != task_id:
                continue
            row = json.loads(json.dumps(source))
            row["is_incumbent"] = row["program_id"] == policy["incumbent_program_id"]
            row["local_qualification_passed"] = (
                row.get("local_qualification", {}).get("status") == "three_seed_fsim_passed"
            )
            row["fpga_correct"] = row["candidate_id"] in correct
            task_rows.append(row)
        gate = production_static_order(task_rows, certificates)
        stats = reference[task_id]["program_statistics"]
        eligible_new = gate["ordered_program_ids"]
        eligible_pool = [gate["incumbent_program_id"]] + eligible_new
        deployable_oracle = best(eligible_pool, stats)
        isolated_oracle = reference[task_id]["pool_oracle_program_id"]
        by_program = {row["program_id"]: row for row in task_rows}
        budgets = {}
        for budget in (1, 2, 3):
            attempted = gate["budget_prefixes"]["B{}".format(budget)]
            selected = best([gate["incumbent_program_id"]] + attempted, stats)
            budgets["B{}".format(budget)] = {
                "attempted_program_ids": attempted,
                "selected_program_id": selected,
                "selected_mode": by_program[selected]["residence_mode"],
                "hits_deployable_subpool_oracle": selected == deployable_oracle,
            }
        for candidate_id in gate["production_rejected_candidate_ids"]:
            mode = gate["candidate_dispositions"][candidate_id]["residence_mode"]
            rejected_modes[mode] = rejected_modes.get(mode, 0) + 1
        task_results[task_id] = {
            "isolated_operator_pool_oracle_program_id": isolated_oracle,
            "isolated_operator_pool_oracle_mode": by_program[isolated_oracle]["residence_mode"],
            "deployable_subpool_oracle_program_id": deployable_oracle,
            "deployable_subpool_oracle_mode": by_program[deployable_oracle]["residence_mode"],
            "deployable_subpool_program_count": len(eligible_pool),
            "gate": gate,
            "budget_replay": budgets,
        }

    summary = {
        "schema": SCHEMA,
        "status": "POSTHOC_PROTOCOL_DEFECT_AUDITED",
        "task_count": len(task_results),
        "source_candidate_count": len(candidates),
        "production_rejected_mode_counts": rejected_modes,
        "tasks_whose_isolated_oracle_is_not_deployable": sum(
            row["isolated_operator_pool_oracle_program_id"]
            != row["deployable_subpool_oracle_program_id"]
            for row in task_results.values()
        ),
        "v4_B1_deployable_oracle_hits": sum(
            row["budget_replay"]["B1"]["hits_deployable_subpool_oracle"]
            for row in task_results.values()
        ),
        "performance_labels_read": True,
        "strict_unseen_claim": False,
        "deployment_authorized": False,
        "protocol_defect": (
            "the original strict-unseen freeze ordered all locally correct isolated-operator "
            "programs before applying the pre-existing production completion-policy gate"
        ),
        "repair": (
            "future holdouts must call production_static_order before freezing B1/B2/B3; legacy "
            "mode4/mode5 programs may remain only in the separately named reference pool"
        ),
        "claim_boundary": (
            "post-hoc diagnostic after YOLO-544 latency exposure; no new selector validation, "
            "cross-boot acceptance, graph correctness, or deployment claim"
        ),
        "task_results": task_results,
        "manifests": manifests,
    }
    output.mkdir(parents=True)
    write(output / "summary.json", summary)
    write(output / "artifact_hashes.json", {
        "artifacts": {"summary.json": sha256_file(output / "summary.json")},
        "source_sha256": sha256_file(Path(__file__).resolve()),
    })
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--correctness", type=Path, default=DEFAULT_CORRECTNESS)
    parser.add_argument("--boot", type=Path, default=DEFAULT_BOOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
