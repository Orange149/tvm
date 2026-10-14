#!/usr/bin/env python3
"""Freeze strict-unseen YOLO complete pools for selector-v4 evaluation.

Only locally complete tasks are admitted: every declared candidate must have
passed final-address qualification and three-seed FSim, and the originally
frozen incumbent must remain present.  Selector-v4 orders are computed before
any board contact or latency label exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import c3_selector_v4_diverse
from c3_composability import candidate_certificate


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_POOLS = GROUPED / "20260924_c3_yolo544_transition_safe_route_pools_run01"
DEFAULT_LOCAL = GROUPED / "20260924_c3_yolo544_transition_safe_local_qualification_run01"
DEFAULT_SELECTOR = GROUPED / "20260924_c3_selector_v4_diverse_development_freeze_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo544_selector_v4_board_pool_freeze_run01"
SCHEMA = "c3_yolo544_selector_v4_board_pool_v1"
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
    artifacts = ledger.get("artifacts")
    if artifacts is None:
        raise ValueError("artifact manifest lacks artifacts mapping: {}".format(directory))
    for name, expected in artifacts.items():
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


def run(args):
    pools_dir = Path(args.pools).resolve()
    local_dir = Path(args.local).resolve()
    selector_dir = Path(args.selector_contract).resolve()
    output = Path(args.output).resolve()
    expected_resolution = int(getattr(args, "expected_resolution", 544))
    expected_task_count = int(getattr(args, "expected_task_count", 2))
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))

    manifests = {
        "pools": verify(pools_dir),
        "local": verify(local_dir),
        "selector_v4": verify(selector_dir),
    }
    selector_contract = read(selector_dir / "selector_v4_contract.json")
    selector_manifest = read(selector_dir / "artifact_hashes.json")
    if selector_contract.get("status") != "FROZEN_FOR_FUTURE_LATENCY_UNSEEN_TASKS":
        raise ValueError("selector-v4 contract is not frozen for future tasks")
    if selector_contract.get("performance_labels_read_by_freeze") is not False:
        raise ValueError("selector-v4 contract is not label isolated")
    if selector_manifest.get("selector_source_sha256") != sha256_file(
            Path(c3_selector_v4_diverse.__file__).resolve()):
        raise ValueError("selector-v4 source changed after contract freeze")

    pool_contract = read(pools_dir / "contract.json")
    local_summary = read(local_dir / "summary.json")
    resolution = int(pool_contract.get("input_resolution"))
    if resolution != expected_resolution:
        raise ValueError(
            "expected frozen YOLO-{} resolution, got {}".format(
                expected_resolution, resolution
            )
        )
    output_schema = SCHEMA if resolution == 544 else "c3_yolo_selector_v4_board_pool_v2"
    if tuple(pool_contract.get("candidate_modes") or ()) != EXPECTED_MODES:
        raise ValueError("candidate modes differ from the frozen five-mode contract")
    if pool_contract.get("performance_labels_read") is not False:
        raise ValueError("candidate pools are not latency blind")
    if local_summary.get("latency_values_read") is not False:
        raise ValueError("local qualification is not latency blind")
    eligible_workloads = sorted(
        workload_id for workload_id, value in local_summary["workloads"].items()
        if value.get("reference_eligible") is True
    )
    if len(eligible_workloads) != expected_task_count:
        raise ValueError(
            "expected exactly {} complete strict-unseen reference tasks, got {}".format(
                expected_task_count, len(eligible_workloads)
            )
        )

    pools = {read(pools_dir / name)["workload_id"]: read(pools_dir / name)
             for name in pool_contract["pool_files"]}
    static = {row["candidate_id"]: row for row in read_jsonl(local_dir / "static_results.jsonl")}
    fsim = {row["candidate_id"]: row for row in read_jsonl(local_dir / "fsim_results.jsonl")}
    programs = read_jsonl(local_dir / "qualified_programs.jsonl")
    program_by_candidate = {}
    for program in programs:
        representative = program["representative_candidate_id"]
        if program["equivalent_candidate_ids"] != [representative]:
            raise ValueError("unexpected equivalent-program collapse")
        program_by_candidate[representative] = program

    rows, certificates, task_policies = [], [], {}
    for workload_id in eligible_workloads:
        pool = pools[workload_id]
        task_rows = []
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
                address = srow["final_address_audit"]
                if address.get("debug_mode_transitions") != [32, 0]:
                    raise ValueError("mode11 debug scope is not closed")
                if (address.get("sync_event_positions") or [None])[0] != 0:
                    raise ValueError("mode11 lacks entry drain")
                certificates.append(candidate_certificate(raw, srow, frow))
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
                "timing_freeze_metrics": None,
                "local_qualification": {
                    "status": "three_seed_fsim_passed",
                    "seeds": frow["seeds"],
                    "source_tir_hash_matches": True,
                    "buffer_protocol": frow["buffer_protocol"],
                    "strict_final_address_audit": srow["final_address_audit"],
                    "arena_capacity_certificate": srow["arena_capacity_certificate"],
                },
                "board_role": "strict_unseen_complete_pool_reference_and_limited_budget",
                "performance_label": None,
                "board_contacted": False,
            })
            raw["timing_freeze_metrics"] = timing_metrics(raw)
            task_rows.append(raw)
            rows.append(raw)

        incumbent_candidate = pool["incumbent"]["candidate_id"]
        incumbent_rows = [row for row in task_rows if row["candidate_id"] == incumbent_candidate]
        if len(task_rows) != 15 or len(incumbent_rows) != 1:
            raise ValueError("complete task requires 15 programs and one original incumbent")
        incumbent_program = incumbent_rows[0]["program_id"]
        new_rows = [row for row in task_rows if row["program_id"] != incumbent_program]
        selector_output = c3_selector_v4_diverse.diverse_static_order(new_rows)
        task_policies[pool["execution_task_id"]] = {
            "workload_id": workload_id,
            "execution_task_id": pool["execution_task_id"],
            "incumbent_candidate_id": incumbent_candidate,
            "incumbent_program_id": incumbent_program,
            "complete_program_pool": sorted(row["program_id"] for row in task_rows),
            "selector_v4": selector_output,
            "B1_B2_B3": selector_output["budget_prefixes"],
            "latency_exposure": "strict_unseen_at_freeze",
        }

    rows.sort(key=lambda row: (row["execution_task_id"], row["family_id"], row["residence_mode"]))
    certificates.sort(key=lambda row: row["identity"]["candidate_id"])
    mode_counts = {mode: sum(row["residence_mode"] == mode for row in rows)
                   for mode in EXPECTED_MODES}
    expected_rows = expected_task_count * 15
    expected_per_mode = expected_task_count * 3
    if (len(rows) != expected_rows or set(mode_counts.values()) != {expected_per_mode}
            or len(certificates) != expected_per_mode):
        raise ValueError("unexpected complete-pool distribution: {}".format(mode_counts))

    output.mkdir(parents=True)
    write_jsonl(output / "candidates.jsonl", rows)
    write_jsonl(output / "mode11_composability_certificates.jsonl", certificates)
    write(output / "task_policies.json", task_policies)
    contract = {
        "schema": output_schema,
        "status": "FROZEN_STRICT_UNSEEN_BEFORE_BOARD_CONTACT_OR_LATENCY",
        "input_resolution": resolution,
        "candidate_count": len(rows),
        "candidate_order": [row["candidate_id"] for row in rows],
        "program_order": [row["program_id"] for row in rows],
        "execution_task_ids": sorted(task_policies),
        "execution_task_count": len(task_policies),
        "mode_counts": mode_counts,
        "mode11_certificate_count": len(certificates),
        "selector_contract_schema": selector_contract["selector_schema"],
        "budgets": {"B1": 1, "B2": 2, "B3": 3,
                    "unit": "additional new programs; incumbent measured separately"},
        "task_selection_rule": (
            "all locally reference-eligible YOLO-{} K1/CO256 tasks; exactly {} qualified".format(
                resolution, expected_task_count
            )
        ),
        "strict_latency_unseen": True,
        "board_contacted": False,
        "performance_labels_collected": False,
        "performance_model_claim": False,
        "safe_pruning_claim": False,
        "manifests": manifests,
        "claim_boundary": (
            "frozen candidate identities and label-free orders only; no FPGA correctness, "
            "latency, oracle, selector superiority, cross-boot, or graph benefit yet"
        ),
    }
    write(output / "contract.json", contract)
    write(output / "summary.json", contract)
    (output / "results.jsonl").write_text("", encoding="utf-8")
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    write(output / "artifact_hashes.json", {
        "artifacts": {path.name: sha256_file(path) for path in sorted(output.iterdir())
                      if path.is_file() and path.name != "artifact_hashes.json"},
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "selector_source_sha256": sha256_file(Path(c3_selector_v4_diverse.__file__).resolve()),
    })
    print(json.dumps(contract, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", type=Path, default=DEFAULT_POOLS)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--selector-contract", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-resolution", type=int, default=544)
    parser.add_argument("--expected-task-count", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
