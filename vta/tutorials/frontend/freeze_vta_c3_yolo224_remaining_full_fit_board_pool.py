#!/usr/bin/env python3
"""Freeze the FPGA board pool for remaining full-weight-fit YOLO-224 tasks.

Only locally qualified identities are admitted.  Failed identities remain in
the input qualification ledger but are not silently replaced.  Same-family
Input/mode11 pairs are recorded explicitly because they are the cleanest
attribution of the incremental weight-residency mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from c3_composability import candidate_certificate


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_POOLS = GROUPED / "20260927_c3_yolo224_remaining_full_fit_pools_run01"
DEFAULT_LOCAL = GROUPED / "20260927_c3_yolo224_remaining_full_fit_local_qualification_run01"
DEFAULT_OUTPUT = GROUPED / "20260927_c3_yolo224_remaining_full_fit_board_pool_freeze_run01"
SCHEMA = "c3_yolo224_remaining_full_fit_board_pool_v1"
EXPECTED_MODES = (
    "original",
    "input_stationary",
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
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def verify(directory):
    directory = Path(directory).resolve()
    ledger = read(directory / "artifact_hashes.json")
    artifacts = ledger.get("artifacts", {})
    for name, expected in artifacts.items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("artifact hash mismatch: {}".format(path))
    return sha256_file(directory / "artifact_hashes.json")


def timing_metrics(static, fsim):
    dma = static["dma_features"]["totals"]
    command = fsim["command_features"]["values"]
    totals = command.get("totals") or {}
    return {
        "submissions": int(command["submissions"]),
        "uop_load_bytes": int(totals.get("uop_bytes", 0)),
        "instruction_bytes": int(totals.get("insn_bytes", 0)),
        "total_request_bytes": (
            int(dma.get("load_buffer_2d_bytes", 0))
            + int(dma.get("store_buffer_2d_bytes", 0))
        ),
        "total_request_calls": (
            int(dma.get("load_buffer_2d_calls", 0))
            + int(dma.get("store_buffer_2d_calls", 0))
        ),
    }


def run(args):
    pools_dir = Path(args.pools).resolve()
    local_dir = Path(args.local).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))

    manifests = {"pools": verify(pools_dir), "local": verify(local_dir)}
    pool_contract = read(pools_dir / "contract.json")
    local_summary = read(local_dir / "summary.json")
    if tuple(pool_contract.get("candidate_modes") or ()) != EXPECTED_MODES:
        raise ValueError("candidate modes differ from the frozen three-mode contract")
    if (pool_contract.get("board_contacted") is not False
            or pool_contract.get("performance_labels_read") is not False):
        raise ValueError("candidate pool is not board/latency blind")
    if (local_summary.get("board_contacted") is not False
            or local_summary.get("latency_values_read") is not False):
        raise ValueError("local qualification is not board/latency blind")
    if local_summary.get("reference_eligible_workload_count") != 4:
        raise ValueError("all four tasks must remain reference eligible")

    pools = {
        read(pools_dir / name)["workload_id"]: read(pools_dir / name)
        for name in pool_contract["pool_files"]
    }
    static = {row["candidate_id"]: row for row in read_jsonl(local_dir / "static_results.jsonl")}
    fsim = {row["candidate_id"]: row for row in read_jsonl(local_dir / "fsim_results.jsonl")}
    program_by_candidate = {}
    for program in read_jsonl(local_dir / "qualified_programs.jsonl"):
        representative = program["representative_candidate_id"]
        for candidate_id in program["equivalent_candidate_ids"]:
            if candidate_id in program_by_candidate:
                raise ValueError("candidate appears in multiple qualified programs")
            program_by_candidate[candidate_id] = program

    candidates = []
    certificates = []
    rejected = []
    task_records = {}
    for workload_id, pool in sorted(pools.items()):
        admitted = []
        by_family = defaultdict(dict)
        for candidate in pool["candidates"]:
            candidate_id = candidate["candidate_id"]
            srow = static.get(candidate_id)
            frow = fsim.get(candidate_id)
            program = program_by_candidate.get(candidate_id)
            if (srow is None or srow.get("status") != "ok" or frow is None
                    or frow.get("status") != "passed" or program is None):
                rejected.append({
                    "candidate_id": candidate_id,
                    "workload_id": workload_id,
                    "family_id": candidate["family_id"],
                    "residence_mode": candidate["residence_mode"],
                    "static_status": None if srow is None else srow.get("status"),
                    "fsim_status": None if frow is None else frow.get("status"),
                    "failure": None if frow is None else frow.get("failure"),
                    "replacement_generated": False,
                })
                continue
            if frow["relowered_tir_sha256"] != srow["lowered_tir_sha256"]:
                raise ValueError("FSim re-lowering changed {}".format(candidate_id))
            row = json.loads(json.dumps(candidate))
            row.update({
                "identity": {
                    "workload": pool["workload"],
                    "complete_config_entity": row["complete_config_entity"],
                },
                "execution_task_id": pool["execution_task_id"],
                "graph_call_count": pool["call_count"],
                "graph_call_sites": pool["graph_call_sites"],
                "program_id": program["program_id"],
                "tir_sha256": srow["lowered_tir_sha256"],
                "implementation_candidate_id": srow["implementation_candidate_id"],
                "dma_features": srow["dma_features"],
                "dependency_audit": srow["dependency_audit"],
                "sync": srow["sync"],
                "command_features": frow["command_features"],
                "local_qualification_passed": True,
                "is_incumbent": candidate_id == pool["incumbent"]["candidate_id"],
                "performance_label": None,
                "board_contacted": False,
                "timing_freeze_metrics": timing_metrics(srow, frow),
            })
            if row["residence_mode"] == "input_weight_resident_transition_safe":
                certificates.append(candidate_certificate(row, srow, frow))
            admitted.append(row)
            by_family[row["family_id"]][row["residence_mode"]] = row
            candidates.append(row)

        pairs = []
        for family_id, modes in sorted(by_family.items()):
            if {"input_stationary", "input_weight_resident_transition_safe"}.issubset(modes):
                pairs.append({
                    "family_id": family_id,
                    "input_program_id": modes["input_stationary"]["program_id"],
                    "mode11_program_id": modes["input_weight_resident_transition_safe"]["program_id"],
                    "tile": modes["input_stationary"]["visible_features"],
                    "performance_label": None,
                })
        if len(pairs) != 3:
            raise ValueError("{} lacks three complete same-Tile Input/mode11 pairs".format(workload_id))
        if not any(row["is_incumbent"] for row in admitted):
            raise ValueError("{} incumbent failed local qualification".format(workload_id))
        task_records[pool["execution_task_id"]] = {
            "workload_id": workload_id,
            "execution_task_id": pool["execution_task_id"],
            "geometry": pool["geometry"],
            "graph_call_count": pool["call_count"],
            "graph_call_sites": pool["graph_call_sites"],
            "incumbent_candidate_id": pool["incumbent"]["candidate_id"],
            "candidate_ids": sorted(row["candidate_id"] for row in admitted),
            "program_ids": sorted(row["program_id"] for row in admitted),
            "same_tile_input_mode11_pairs": pairs,
        }

    candidates.sort(key=lambda row: (
        row["execution_task_id"], row["family_id"], EXPECTED_MODES.index(row["residence_mode"])
    ))
    certificates.sort(key=lambda row: row["identity"]["candidate_id"])
    rejected.sort(key=lambda row: row["candidate_id"])
    if len(candidates) != 34 or len(rejected) != 2 or len(certificates) != 12:
        raise ValueError("unexpected board-pool cardinality")
    if len({row["program_id"] for row in candidates}) != len(candidates):
        raise ValueError("qualified board pool contains equivalent programs")

    output.mkdir(parents=True)
    write_jsonl(output / "candidates.jsonl", candidates)
    write_jsonl(output / "mode11_composability_certificates.jsonl", certificates)
    write_jsonl(output / "rejected_candidates.jsonl", rejected)
    write(output / "task_records.json", task_records)
    contract = {
        "schema": SCHEMA,
        "status": "FROZEN_AFTER_LOCAL_QUALIFICATION_BEFORE_FPGA_OR_LATENCY",
        "input_resolution": 224,
        "execution_task_count": len(task_records),
        "candidate_count": len(candidates),
        "rejected_candidate_count": len(rejected),
        "candidate_modes": list(EXPECTED_MODES),
        "candidate_order": [row["candidate_id"] for row in candidates],
        "program_order": [row["program_id"] for row in candidates],
        "same_tile_input_mode11_pair_count": sum(
            len(row["same_tile_input_mode11_pairs"]) for row in task_records.values()
        ),
        "mode11_certificate_count": len(certificates),
        "task_records_file": "task_records.json",
        "manifests": manifests,
        "board_contacted": False,
        "performance_labels_collected": False,
        "failed_identities_replaced": False,
        "claim_boundary": (
            "frozen locally correct identities and same-Tile comparison pairs only; FPGA "
            "correctness, latency, cross-boot stability and graph benefit remain pending"
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
