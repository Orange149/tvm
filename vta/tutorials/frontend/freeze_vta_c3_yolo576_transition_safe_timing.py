#!/usr/bin/env python3
"""Freeze a development-only timing pool for transition-safe residency.

The execution tasks have prior latency exposure, while the mode-11 programs
have not been timed.  This artifact is therefore for mechanism development,
not a strict-unseen selector evaluation.  Only production-eligible Original,
Input, and mode-11 programs are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_POOLS = GROUPED / "20260924_c3_yolo576_transition_safe_route_pools_run01"
DEFAULT_BOARD_POOL = GROUPED / "20260924_c3_yolo576_transition_safe_board_pool_freeze_run01"
DEFAULT_CORRECTNESS = GROUPED / "20260924_c3_yolo576_transition_safe_full_pool_correctness_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo576_transition_safe_timing_freeze_run01"
SCHEMA = "c3_yolo576_transition_safe_timing_freeze_v1"
ALLOWED_MODES = frozenset((
    "original", "input_stationary", "input_weight_resident_transition_safe",
))


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
    manifest = directory / "artifact_hashes.json"
    ledger = read(manifest)
    for name, expected in ledger.get("artifacts", {}).items():
        if not isinstance(expected, str):
            continue
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("artifact hash mismatch: {}".format(path))
    return sha256_file(manifest)


def metric(candidate):
    dma = candidate["dma_features"]["totals"]
    command = candidate["command_features"]["values"]
    totals = command.get("totals") or {}
    return {
        "submissions": int(command["submissions"]),
        "command_uop_bytes": int(totals.get("uop_bytes", 0)),
        "instruction_bytes": int(totals.get("insn_bytes", 0)),
        "total_request_bytes": int(dma.get("load_buffer_2d_bytes", 0))
        + int(dma.get("store_buffer_2d_bytes", 0)),
        "total_request_calls": int(dma.get("load_buffer_2d_calls", 0))
        + int(dma.get("store_buffer_2d_calls", 0)),
    }


def ordered(programs, fields):
    return [row["program_id"] for row in sorted(
        programs,
        key=lambda row: tuple(row["timing_freeze_metrics"][field] for field in fields)
        + (row["program_id"],),
    )]


def run(args):
    pools_dir = Path(args.pools).resolve()
    board_pool_dir = Path(args.board_pool).resolve()
    correctness_dir = Path(args.correctness).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    manifests = {
        "pools": verify(pools_dir),
        "board_pool": verify(board_pool_dir),
        "correctness": verify(correctness_dir),
    }
    correctness_summary = read(correctness_dir / "summary.json")
    if (correctness_summary.get("candidate_count") != 45
            or correctness_summary.get("fpga_correct_count") != 45
            or correctness_summary.get("wave_count") != 1
            or correctness_summary.get("latency_values_collected") is not False):
        raise ValueError("input is not the complete one-wave correctness qualification")
    correctness = {
        row["candidate_id"]: row for row in read_jsonl(correctness_dir / "fpga_correctness.jsonl")
    }
    source_rows = read_jsonl(board_pool_dir / "candidates.jsonl")
    rows = []
    for source in source_rows:
        if source["residence_mode"] not in ALLOWED_MODES:
            continue
        if correctness.get(source["candidate_id"], {}).get("status") != "passed":
            raise ValueError("timing candidate lacks complete FPGA correctness")
        row = json.loads(json.dumps(source))
        row["timing_freeze_metrics"] = metric(row)
        row["board_role"] = "development_full_pool_and_limited_budget"
        rows.append(row)
    rows.sort(key=lambda row: (row["execution_task_id"], row["program_id"]))
    if len(rows) != 27:
        raise ValueError("expected three tasks x three Tiles x three production modes")

    pool_contract = read(pools_dir / "contract.json")
    incumbent_candidate_by_task = {}
    for filename in pool_contract["pool_files"]:
        pool = read(pools_dir / filename)
        incumbent_candidate_by_task[pool["execution_task_id"]] = pool["incumbent"]["candidate_id"]
    by_candidate = {row["candidate_id"]: row for row in rows}
    task_ids = sorted({row["execution_task_id"] for row in rows})
    policies = {}
    for task_id in task_ids:
        task_rows = [row for row in rows if row["execution_task_id"] == task_id]
        if len(task_rows) != 9:
            raise ValueError("task lacks nine production timing programs")
        incumbent_candidate_id = incumbent_candidate_by_task[task_id]
        if incumbent_candidate_id not in by_candidate:
            raise ValueError("frozen incumbent is absent from production timing pool")
        incumbent = by_candidate[incumbent_candidate_id]["program_id"]
        new_rows = [row for row in task_rows if row["program_id"] != incumbent]
        fallback = ordered(new_rows, (
            "submissions", "command_uop_bytes", "total_request_bytes",
            "total_request_calls", "instruction_bytes",
        ))
        bytes_order = ordered(new_rows, ("total_request_bytes", "total_request_calls"))
        calls_order = ordered(new_rows, ("total_request_calls", "total_request_bytes"))
        command_order = ordered(new_rows, (
            "submissions", "command_uop_bytes", "instruction_bytes",
        ))
        shortlist = [incumbent]
        for program_id in (fallback[0], bytes_order[0], command_order[0], *fallback):
            if program_id not in shortlist and len(shortlist) < 4:
                shortlist.append(program_id)
        budget_order = [program_id for program_id in shortlist if program_id != incumbent]
        policies[task_id] = {
            "execution_task_id": task_id,
            "incumbent_program_id": incumbent,
            "complete_local_program_pool": sorted(row["program_id"] for row in task_rows),
            "selector_v3_status": "STATIC_FALLBACK_DEVELOPMENT_ONLY",
            "performance_model_claim": False,
            "task_latency_exposure": "task_exposed_mode11_programs_unmeasured",
            "selector_v3_order": fallback,
            "bytes_only_order": bytes_order,
            "calls_only_order": calls_order,
            "command_only_order": command_order,
            "shortlist": {"shortlist_candidate_ids": shortlist},
            "B1_B2_B3": {
                name: budget_order[:budget]
                for name, budget in (("B1", 1), ("B2", 2), ("B3", 3))
            },
        }

    tasks = [{
        "task_id": task_id,
        "role": "development_full_pool_and_limited_budget",
        "latency_unseen": False,
        "latency_exposure_status": "task_exposed_mode11_programs_unmeasured",
    } for task_id in task_ids]
    output.mkdir(parents=True)
    write_jsonl(output / "candidates.jsonl", rows)
    write(output / "full_pool_tasks.json", tasks)
    write(output / "limited_budget_tasks.json", tasks)
    write(output / "task_policies.json", policies)
    contract = {
        "schema": SCHEMA,
        "status": "DEVELOPMENT_TIMING_FROZEN_BEFORE_MODE11_LATENCY",
        "candidate_count": len(rows),
        "candidate_order": [row["candidate_id"] for row in rows],
        "program_order": [row["program_id"] for row in rows],
        "execution_task_count": len(task_ids),
        "full_pool_task_ids": task_ids,
        "limited_budget_task_ids": task_ids,
        "allowed_modes": sorted(ALLOWED_MODES),
        "task_latency_unseen": False,
        "mode11_program_latency_unseen_before_freeze": True,
        "strict_holdout_claimed": False,
        "performance_model_claim": False,
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
    parser.add_argument("--board-pool", type=Path, default=DEFAULT_BOARD_POOL)
    parser.add_argument("--correctness", type=Path, default=DEFAULT_CORRECTNESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
