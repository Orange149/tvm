#!/usr/bin/env python3
"""Freeze the mode-11 YOLO-576 pool for FPGA correctness only.

This stage consumes latency-blind lowering and FSim artifacts.  It deliberately
does not assign selector, timing, or holdout roles: the next board experiment
tests whether the entry drain makes force-serial residency safe after arbitrary
predecessor programs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from c3_composability import candidate_certificate


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_POOLS = GROUPED / "20260924_c3_yolo576_transition_safe_route_pools_run01"
DEFAULT_LOCAL = GROUPED / "20260924_c3_yolo576_transition_safe_local_qualification_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo576_transition_safe_board_pool_freeze_run01"
SCHEMA = "c3_yolo576_transition_safe_board_pool_v1"
EXPECTED_MODES = (
    "original",
    "input_stationary",
    "weight_resident_barrier",
    "input_weight_resident_barrier",
    "input_weight_resident_transition_safe",
)


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
    ledger_path = directory / "artifact_hashes.json"
    ledger = read(ledger_path)
    for name, expected in ledger.get("artifacts", {}).items():
        if not isinstance(expected, str):
            continue
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("artifact hash mismatch: {}".format(path))
    return sha256_file(ledger_path)


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
        raise ValueError("candidate modes do not match the transition-safe five-mode contract")
    if pool_contract.get("performance_labels_read") is not False:
        raise ValueError("input pools are not latency blind")
    if local_summary.get("latency_values_read") is not False:
        raise ValueError("local qualification is not latency blind")
    if local_summary.get("static_ok") != local_summary.get("proposed_identities"):
        raise ValueError("not every proposed identity passed static qualification")
    if local_summary.get("fsim_passed") != local_summary.get("proposed_identities"):
        raise ValueError("not every proposed identity passed FSim")

    candidates = {}
    pool_by_workload = {}
    for name in pool_contract["pool_files"]:
        pool = read(pools_dir / name)
        pool_by_workload[pool["workload_id"]] = pool
        for candidate in pool["candidates"]:
            if candidate["candidate_id"] in candidates:
                raise ValueError("duplicate candidate identity")
            candidates[candidate["candidate_id"]] = candidate
    static = {row["candidate_id"]: row for row in read_jsonl(local_dir / "static_results.jsonl")}
    fsim = {row["candidate_id"]: row for row in read_jsonl(local_dir / "fsim_results.jsonl")}
    programs = read_jsonl(local_dir / "qualified_programs.jsonl")
    if len(programs) != len(candidates):
        raise ValueError("this correctness pool requires one distinct program per candidate")

    rows, certificates = [], []
    for program in programs:
        candidate_id = program["representative_candidate_id"]
        if program["equivalent_candidate_ids"] != [candidate_id]:
            raise ValueError("unexpected equivalent-program collapse")
        raw = json.loads(json.dumps(candidates[candidate_id]))
        srow, frow = static[candidate_id], fsim[candidate_id]
        pool = pool_by_workload[raw["workload_id"]]
        if srow["status"] != "ok" or frow["status"] != "passed":
            raise ValueError("candidate lacks complete local qualification")
        if frow["relowered_tir_sha256"] != srow["lowered_tir_sha256"]:
            raise ValueError("FSim re-lowering changed the program")
        if raw["residence_mode"] == "input_weight_resident_composable":
            raise ValueError("predecessor-sensitive mode 10 is forbidden in this pool")
        if raw["residence_mode"] == "input_weight_resident_transition_safe":
            address = srow["final_address_audit"]
            if address.get("debug_mode_transitions") != [32, 0]:
                raise ValueError("mode 11 debug scope is not closed")
            if (address.get("sync_event_positions") or [None])[0] != 0:
                raise ValueError("mode 11 lacks an entry drain before debug mode")
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
            "local_qualification": {
                "status": "three_seed_fsim_passed",
                "seeds": frow["seeds"],
                "source_tir_hash_matches": True,
                "buffer_protocol": frow["buffer_protocol"],
                "strict_final_address_audit": srow["final_address_audit"],
                "arena_capacity_certificate": srow["arena_capacity_certificate"],
            },
            "board_role": "cross_program_transition_correctness_only",
            "performance_label": None,
            "board_contacted": False,
        })
        rows.append(raw)

    rows.sort(key=lambda row: (row["execution_task_id"], row["family_id"], row["residence_mode"]))
    certificates.sort(key=lambda row: row["identity"]["candidate_id"])
    mode_counts = {
        mode: sum(row["residence_mode"] == mode for row in rows) for mode in EXPECTED_MODES
    }
    if set(mode_counts.values()) != {9} or len(certificates) != 9:
        raise ValueError("unexpected mode distribution: {}".format(mode_counts))

    output.mkdir(parents=True)
    write_jsonl(output / "candidates.jsonl", rows)
    write_jsonl(output / "mode11_composability_certificates.jsonl", certificates)
    contract = {
        "schema": SCHEMA,
        "status": "FROZEN_FOR_FPGA_CROSS_PROGRAM_CORRECTNESS_BEFORE_BOARD_CONTACT",
        "candidate_count": len(rows),
        "candidate_order": [row["candidate_id"] for row in rows],
        "workload_count": len(pool_by_workload),
        "mode_counts": mode_counts,
        "mode11_certificate_count": len(certificates),
        "test_scope": "correctness and predecessor-transition safety only",
        "timing_authorized": False,
        "oracle_authorized": False,
        "latency_unseen_claimed": False,
        "board_contacted": False,
        "performance_labels_collected": False,
        "manifests": manifests,
    }
    write(output / "contract.json", contract)
    write(output / "summary.json", contract)
    (output / "results.jsonl").write_text("", encoding="utf-8")
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    write(output / "artifact_hashes.json", {
        "artifacts": {
            path.name: sha256_file(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "artifact_hashes.json"
        },
        "source_sha256": sha256_file(Path(__file__).resolve()),
    })
    print(json.dumps(contract, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", type=Path, default=DEFAULT_POOLS)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
