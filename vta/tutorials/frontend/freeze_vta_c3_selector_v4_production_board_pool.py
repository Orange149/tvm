#!/usr/bin/env python3
"""Freeze strict-unseen complete reference and production selector pools.

The isolated-operator reference pool retains every locally correct program so
its oracle is well defined.  Deployment budgets are frozen separately after
the production completion-policy/composability gate.  No board contact or
latency label is permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from c3_composability import candidate_certificate
from c3_selector_v4_deployment import production_static_order


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_POOLS = GROUPED / "20260924_c3_yolo512_transition_safe_route_pools_run01"
DEFAULT_LOCAL = GROUPED / "20260924_c3_yolo512_transition_safe_local_qualification_run01"
DEFAULT_GATE = GROUPED / "20260924_c3_selector_v4_production_gate_freeze_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo512_selector_v4_production_board_pool_freeze_run02"
SCHEMA = "c3_selector_v4_production_board_pool_v1"
EXPECTED_MODES = (
    "original",
    "input_stationary",
    "weight_resident_barrier",
    "input_weight_resident_barrier",
    "input_weight_resident_transition_safe",
)


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


def timing_metrics(row):
    dma = row["dma_features"]["totals"]
    command = row["command_features"]["values"]
    totals = command.get("totals") or {}
    return {
        "submissions": int(command["submissions"]),
        "uop_load_bytes": int(totals.get("uop_bytes", 0)),
        "instruction_bytes": int(totals.get("insn_bytes", 0)),
        "total_request_bytes": int(dma.get("load_buffer_2d_bytes", 0))
        + int(dma.get("store_buffer_2d_bytes", 0)),
        "total_request_calls": int(dma.get("load_buffer_2d_calls", 0))
        + int(dma.get("store_buffer_2d_calls", 0)),
    }


def is_complete(value):
    return (
        value.get("proposed_identities") == 15
        and value.get("static_ok") == 15
        and value.get("static_failed") == 0
        and value.get("fsim_passed_identities") == 15
        and value.get("fsim_failed_identities") == 0
        and value.get("distinct_fsim_correct_programs") == 15
        and value.get("incumbent_fsim_passed") is True
    )


def run(args):
    pools_dir = Path(args.pools).resolve()
    local_dir = Path(args.local).resolve()
    gate_dir = Path(args.production_gate).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    manifests = {
        "pools": verify(pools_dir),
        "local": verify(local_dir),
        "production_gate": verify(gate_dir),
    }
    pool_contract = read(pools_dir / "contract.json")
    local_summary = read(local_dir / "summary.json")
    gate_contract = read(gate_dir / "contract.json")
    resolution = int(pool_contract["input_resolution"])
    if tuple(pool_contract.get("candidate_modes") or ()) != EXPECTED_MODES:
        raise ValueError("candidate modes differ from the frozen five-mode contract")
    if pool_contract.get("performance_labels_read") is not False:
        raise ValueError("candidate pool is not latency blind")
    if (local_summary.get("latency_values_read") is not False
            or local_summary.get("board_contacted") is not False):
        raise ValueError("local qualification is not board/latency blind")
    if gate_contract.get("status") != "FROZEN_FOR_FUTURE_LATENCY_UNSEEN_TASKS":
        raise ValueError("production gate is not frozen for future tasks")
    if gate_contract.get("performance_labels_read_by_freeze") is not False:
        raise ValueError("production gate contract is not label isolated")

    complete_workloads = sorted(
        workload_id for workload_id, value in local_summary["workloads"].items()
        if is_complete(value)
    )
    if len(complete_workloads) < int(args.reference_tasks):
        raise ValueError("not enough complete reference tasks")
    selected_workloads = complete_workloads[:int(args.reference_tasks)]
    pools = {
        read(pools_dir / name)["workload_id"]: read(pools_dir / name)
        for name in pool_contract["pool_files"]
    }
    static = {row["candidate_id"]: row for row in read_jsonl(local_dir / "static_results.jsonl")}
    fsim = {row["candidate_id"]: row for row in read_jsonl(local_dir / "fsim_results.jsonl")}
    programs = read_jsonl(local_dir / "qualified_programs.jsonl")
    program_by_candidate = {}
    for program in programs:
        representative = program["representative_candidate_id"]
        if program["equivalent_candidate_ids"] != [representative]:
            raise ValueError("unexpected equivalent-program collapse")
        program_by_candidate[representative] = program

    rows, certificates, policies = [], [], {}
    for workload_id in selected_workloads:
        pool = pools[workload_id]
        task_rows, task_certificates = [], {}
        incumbent_candidate = pool["incumbent"]["candidate_id"]
        for candidate in pool["candidates"]:
            candidate_id = candidate["candidate_id"]
            srow = static.get(candidate_id)
            frow = fsim.get(candidate_id)
            program = program_by_candidate.get(candidate_id)
            if srow is None or frow is None or program is None:
                raise ValueError("complete task candidate lacks local evidence")
            if srow["status"] != "ok" or frow["status"] != "passed":
                raise ValueError("complete task contains a rejected candidate")
            if frow["relowered_tir_sha256"] != srow["lowered_tir_sha256"]:
                raise ValueError("FSim re-lowering changed the program")
            raw = json.loads(json.dumps(candidate))
            if raw["residence_mode"] == "input_weight_resident_transition_safe":
                certificate = candidate_certificate(raw, srow, frow)
                certificates.append(certificate)
                task_certificates[candidate_id] = certificate
            raw.update({
                "identity": {
                    "workload": pool["workload"],
                    "complete_config_entity": raw["complete_config_entity"],
                },
                "execution_task_id": pool["execution_task_id"],
                "program_id": program["program_id"],
                "tir_sha256": srow["lowered_tir_sha256"],
                "implementation_candidate_id": srow["implementation_candidate_id"],
                "dma_features": srow["dma_features"],
                "dependency_audit": srow["dependency_audit"],
                "sync": srow["sync"],
                "command_features": frow["command_features"],
                "local_qualification_passed": True,
                "is_incumbent": candidate_id == incumbent_candidate,
                "performance_label": None,
                "board_contacted": False,
            })
            raw["timing_freeze_metrics"] = timing_metrics(raw)
            task_rows.append(raw)
            rows.append(raw)

        incumbent_rows = [row for row in task_rows if row["is_incumbent"]]
        if len(task_rows) != 15 or len(incumbent_rows) != 1:
            raise ValueError("complete reference task requires 15 programs and one incumbent")
        if len({row["program_id"] for row in task_rows}) != 15:
            raise ValueError("complete reference task contains duplicate programs")
        gate = production_static_order(task_rows, task_certificates)
        production_pool = [gate["incumbent_program_id"]] + gate["ordered_program_ids"]
        if len(production_pool) != 9:
            raise ValueError("expected incumbent plus eight production-eligible new programs")
        policies[pool["execution_task_id"]] = {
            "workload_id": workload_id,
            "execution_task_id": pool["execution_task_id"],
            "incumbent_candidate_id": incumbent_candidate,
            "incumbent_program_id": gate["incumbent_program_id"],
            "complete_isolated_reference_program_pool": sorted(
                row["program_id"] for row in task_rows
            ),
            "production_program_pool": production_pool,
            "production_gate": gate,
            "selector_v4_order": gate["ordered_program_ids"],
            "B1_B2_B3": gate["budget_prefixes"],
            "latency_exposure": "strict_unseen_at_freeze",
        }

    rows.sort(key=lambda row: (
        row["execution_task_id"], row["family_id"], row["residence_mode"]
    ))
    certificates.sort(key=lambda row: row["identity"]["candidate_id"])
    output.mkdir(parents=True)
    write_jsonl(output / "candidates.jsonl", rows)
    write_jsonl(output / "mode11_composability_certificates.jsonl", certificates)
    write(output / "task_policies.json", policies)
    contract = {
        "schema": SCHEMA,
        "status": "FROZEN_STRICT_UNSEEN_PRODUCTION_GATED_BEFORE_BOARD_OR_LATENCY",
        "input_resolution": resolution,
        "selected_workload_ids": selected_workloads,
        "excluded_incomplete_workload_ids": sorted(
            set(local_summary["workloads"]) - set(selected_workloads)
        ),
        "reference_task_selection_rule": (
            "stable workload-id order among tasks with exactly 15/15 static, FSim, "
            "distinct-program and incumbent qualification"
        ),
        "execution_task_count": len(policies),
        "isolated_reference_candidate_count": len(rows),
        "candidate_order": [row["candidate_id"] for row in rows],
        "program_order": [row["program_id"] for row in rows],
        "production_candidate_count": sum(
            len(value["production_program_pool"]) for value in policies.values()
        ),
        "mode11_certificate_count": len(certificates),
        "strict_latency_unseen": True,
        "board_contacted": False,
        "performance_labels_collected": False,
        "performance_model_claim": False,
        "safe_pruning_claim": False,
        "reference_pool_oracle_scope": "complete isolated-operator pool only",
        "deployment_budget_scope": "production-policy-eligible subpool only",
        "manifests": manifests,
        "claim_boundary": (
            "frozen identities, semantic eligibility and label-free budget orders only; "
            "no FPGA correctness, latency, oracle, cross-boot, graph or deployment result"
        ),
    }
    write(output / "contract.json", contract)
    write(output / "summary.json", contract)
    (output / "results.jsonl").write_text("", encoding="utf-8")
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
    parser.add_argument("--pools", type=Path, default=DEFAULT_POOLS)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--production-gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-tasks", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
