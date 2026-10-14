#!/usr/bin/env python3
"""Freeze YOLO-576 task roles, static-fallback shortlists, and board pool.

This stage consumes only immutable lowering/FSim evidence.  It does not contact
the board or read latency.  Two tasks are assigned complete-pool roles by a
stable task-ID hash; all locally eligible tasks receive a limited-budget role.
Because selector-v3's completion-aware ridge readiness gate is not yet met,
the frozen shortlist uses the declared static fallback instead of presenting a
learned-model claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_POOLS = GROUPED / "20260924_c3_yolo576_lowering_aware_route_pools_run01"
DEFAULT_LOCAL = GROUPED / "20260924_c3_yolo576_lowering_aware_local_qualification_run01"
DEFAULT_READINESS = GROUPED / "20260924_c3_selector_v3_model_readiness_run02"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo576_evaluation_freeze_run01"
SCHEMA = "c3_yolo576_evaluation_freeze_v1"


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


def metric_value(value):
    return (value is None, 0.0 if value is None else float(value))


def metrics(static, fsim):
    totals = static["dma_features"]["totals"]
    command = fsim["command_features"]["values"]
    command_totals = command.get("totals", {})
    arenas = static.get("arena_capacity_certificate", {}).get("domains", {})

    def fraction(scope):
        row = arenas.get(scope) or {}
        capacity = row.get("capacity_bytes")
        required = row.get("required_upper_bound_bytes")
        return None if not capacity or required is None else float(required) / float(capacity)

    return {
        "inp_request_bytes": totals.get("load_buffer_2d_inp_bytes"),
        "wgt_request_bytes": totals.get("load_buffer_2d_wgt_bytes"),
        "acc_request_bytes": totals.get("load_buffer_2d_acc_bytes"),
        "out_request_bytes": totals.get("store_buffer_2d_out_bytes"),
        "total_request_bytes": (
            int(totals.get("load_buffer_2d_bytes", 0))
            + int(totals.get("store_buffer_2d_bytes", 0))
        ),
        "inp_request_calls": totals.get("load_buffer_2d_inp_calls"),
        "wgt_request_calls": totals.get("load_buffer_2d_wgt_calls"),
        "acc_request_calls": totals.get("load_buffer_2d_acc_calls"),
        "out_request_calls": totals.get("store_buffer_2d_out_calls"),
        "total_request_calls": (
            int(totals.get("load_buffer_2d_calls", 0))
            + int(totals.get("store_buffer_2d_calls", 0))
        ),
        "uop_load_bytes": None,
        "uop_load_calls": None,
        "command_uop_bytes": command_totals.get("uop_bytes"),
        "instruction_bytes": command_totals.get("insn_bytes"),
        "submissions": command.get("submissions"),
        "sync_count": static.get("sync", {}).get("explicit_residency_drains"),
        "inp_arena_fraction": fraction("local.inp_buffer"),
        "wgt_arena_fraction": fraction("local.wgt_buffer"),
        "acc_arena_fraction": fraction("local.acc_buffer"),
    }


def static_fallback_key(row):
    values = row["candidate_metrics"]
    return (
        metric_value(values.get("submissions")),
        metric_value(values.get("uop_load_bytes")),
        metric_value(values.get("total_request_bytes")),
        metric_value(values.get("total_request_calls")),
        metric_value(values.get("instruction_bytes")),
        row["program_id"],
    )


def metric_order(rows, fields):
    return [row["program_id"] for row in sorted(
        (row for row in rows if not row["is_incumbent"]),
        key=lambda row: tuple(metric_value(row["candidate_metrics"].get(name)) for name in fields)
        + (row["program_id"],),
    )]


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    manifests = {
        "pools": verify(args.pools),
        "local": verify(args.local),
        "model_readiness": verify(args.model_readiness),
    }
    pool_contract = read(Path(args.pools) / "contract.json")
    local_summary = read(Path(args.local) / "summary.json")
    readiness = read(Path(args.model_readiness) / "summary.json")
    if pool_contract.get("performance_labels_read") is not False:
        raise ValueError("candidate pool is not latency blind")
    if local_summary.get("latency_values_read") is not False:
        raise ValueError("local qualification is not latency blind")
    if readiness.get("status") != "STATIC_FALLBACK_REQUIRED":
        raise ValueError("this freeze expects selector-v3 static fallback readiness")

    pools = {}
    candidates = {}
    for name in pool_contract["pool_files"]:
        pool = read(Path(args.pools) / name)
        pools[pool["workload_id"]] = pool
        for candidate in pool["candidates"]:
            candidates[candidate["candidate_id"]] = candidate
    static = {row["candidate_id"]: row for row in read_jsonl(Path(args.local) / "static_results.jsonl")}
    fsim = {row["candidate_id"]: row for row in read_jsonl(Path(args.local) / "fsim_results.jsonl")}
    programs = read_jsonl(Path(args.local) / "qualified_programs.jsonl")
    candidate_to_program = {
        candidate_id: program for program in programs for candidate_id in program["equivalent_candidate_ids"]
    }

    tasks, feature_rows = [], []
    for workload_id, pool in sorted(pools.items()):
        local = local_summary["workloads"][workload_id]
        if not local.get("reference_eligible"):
            continue
        task_programs = sorted(
            (program for program in programs if program["workload_id"] == workload_id),
            key=lambda row: row["program_id"],
        )
        incumbent_candidate = pool["incumbent"]["candidate_id"]
        incumbent_program = candidate_to_program[incumbent_candidate]["program_id"]
        incumbent_rep = candidate_to_program[incumbent_candidate]["representative_candidate_id"]
        incumbent_metrics = metrics(static[incumbent_rep], fsim[incumbent_rep])
        incumbent_tile = candidates[incumbent_candidate]["visible_features"]
        for program in task_programs:
            representative = program["representative_candidate_id"]
            candidate = candidates[representative]
            feature_rows.append({
                "schema": SCHEMA,
                "execution_task_id": pool["execution_task_id"],
                "workload_id": workload_id,
                "network": "yolov3_tiny_576",
                "candidate_id": program["program_id"],
                "program_id": program["program_id"],
                "representative_candidate_id": representative,
                "equivalent_candidate_ids": program["equivalent_candidate_ids"],
                "incumbent_id": incumbent_program,
                "is_incumbent": program["program_id"] == incumbent_program,
                "candidate_mode": candidate["residence_mode"],
                "incumbent_mode": candidates[incumbent_candidate]["residence_mode"],
                "candidate_tile": candidate["visible_features"],
                "incumbent_tile": incumbent_tile,
                "candidate_metrics": metrics(static[representative], fsim[representative]),
                "incumbent_metrics": incumbent_metrics,
                "local_fsim_correct": True,
                "fpga_correct": None,
                "target_log_latency_ratio": None,
                "board_latency_ms": None,
                "prediction_source": "static_fallback_no_performance_model_claim",
            })
        tasks.append({
            "network": "yolov3_tiny_576",
            "task_id": pool["execution_task_id"],
            "workload_id": workload_id,
            "geometry": pool["geometry"],
            "stable_rank": hashlib.sha256(pool["execution_task_id"].encode()).hexdigest(),
            "program_ids": [row["program_id"] for row in task_programs],
            "distinct_program_count": len(task_programs),
            "incumbent_candidate_id": incumbent_candidate,
            "incumbent_program_id": incumbent_program,
            "local_qualified": True,
            "latency_unseen": True,
        })
    tasks.sort(key=lambda row: row["stable_rank"])
    if len(tasks) < 2:
        raise ValueError("fewer than two complete locally qualified tasks")
    full_tasks = [dict(row, role="full_pool_reference", role_ordinal=index + 1)
                  for index, row in enumerate(tasks[:2])]
    limited_tasks = [dict(row, role="limited_budget_confirmation", role_ordinal=index + 1)
                     for index, row in enumerate(tasks)]

    rows_by_task = {}
    for row in feature_rows:
        rows_by_task.setdefault(row["execution_task_id"], []).append(row)
    policies = {}
    for task in tasks:
        rows = rows_by_task[task["task_id"]]
        incumbent = task["incumbent_program_id"]
        new_rows = [row for row in rows if not row["is_incumbent"]]
        fallback_order = [row["program_id"] for row in sorted(new_rows, key=static_fallback_key)]
        bytes_order = metric_order(rows, ("total_request_bytes", "total_request_calls"))
        calls_order = metric_order(rows, ("total_request_calls", "total_request_bytes"))
        command_order = metric_order(
            rows, ("submissions", "command_uop_bytes", "instruction_bytes")
        )
        shortlist = [incumbent]
        for candidate_id in (
            fallback_order[0], bytes_order[0], command_order[0], *fallback_order
        ):
            if candidate_id not in shortlist and len(shortlist) < 4:
                shortlist.append(candidate_id)
        budget_order = [value for value in shortlist if value != incumbent]
        policies[task["task_id"]] = {
            "execution_task_id": task["task_id"],
            "workload_id": task["workload_id"],
            "network": task["network"],
            "incumbent_program_id": incumbent,
            "complete_local_program_pool": sorted(row["program_id"] for row in rows),
            "selector_v3_status": "STATIC_FALLBACK",
            "performance_model_claim": False,
            "model_fallback_reason": "completion-aware ridge readiness thresholds not met",
            "selector_v3_order": fallback_order,
            "bytes_only_order": bytes_order,
            "calls_only_order": calls_order,
            "command_only_order": command_order,
            "shortlist": {
                "shortlist_candidate_ids": shortlist,
                "roles": ["incumbent", "static_fallback_best", "bytes_best", "command_best"],
                "fill_policy": "deduplicate roles, then fill by frozen static-fallback order",
            },
            "B1_B2_B3": {
                name: budget_order[:budget] for name, budget in (("B1", 1), ("B2", 2), ("B3", 3))
            },
            "fpga_invalid_policy": "attempt consumes budget; no replacement",
        }

    full_ids = {row["task_id"] for row in full_tasks}
    limited_ids = {row["task_id"] for row in limited_tasks}
    selected_programs = set()
    roles = {}
    for task in tasks:
        task_id = task["task_id"]
        chosen = set()
        task_roles = []
        if task_id in full_ids:
            chosen.update(policies[task_id]["complete_local_program_pool"])
            task_roles.append("full_pool_reference")
        if task_id in limited_ids:
            chosen.update(policies[task_id]["shortlist"]["shortlist_candidate_ids"])
            task_roles.append("limited_budget_confirmation")
        for program_id in chosen:
            roles[program_id] = list(task_roles)
        selected_programs.update(chosen)

    feature_by_program = {row["program_id"]: row for row in feature_rows}
    board_rows = []
    for program_id in sorted(selected_programs):
        feature = feature_by_program[program_id]
        workload_id = feature["workload_id"]
        pool = pools[workload_id]
        candidate_id = feature["representative_candidate_id"]
        raw = json.loads(json.dumps(candidates[candidate_id]))
        srow, frow = static[candidate_id], fsim[candidate_id]
        if srow["status"] != "ok" or frow["status"] != "passed":
            raise ValueError("board representative lacks local qualification")
        if frow["relowered_tir_sha256"] != srow["lowered_tir_sha256"]:
            raise ValueError("representative TIR changed")
        raw.update({
            "identity": {
                "workload": pool["workload"],
                "complete_config_entity": raw["complete_config_entity"],
            },
            "execution_task_id": feature["execution_task_id"],
            "program_id": program_id,
            "board_roles": roles[program_id],
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
            "performance_label": None,
            "board_contacted": False,
        })
        board_rows.append(raw)
    board_rows.sort(key=lambda row: (row["execution_task_id"], row["program_id"]))

    output.mkdir(parents=True)
    write(output / "full_pool_tasks.json", full_tasks)
    write(output / "limited_budget_tasks.json", limited_tasks)
    write_jsonl(output / "program_features.jsonl", feature_rows)
    write(output / "task_policies.json", policies)
    write_jsonl(output / "candidates.jsonl", board_rows)
    contract = {
        "schema": SCHEMA,
        "status": "EVALUATION_AND_BOARD_POOL_FROZEN_BEFORE_BOARD_LATENCY",
        "input_resolution": 576,
        "eligible_task_count": len(tasks),
        "planned_limited_task_count": 4,
        "actual_limited_task_count": len(limited_tasks),
        "task_shortfall_policy": "retain actual eligible count; do not substitute latency-exposed tasks",
        "full_pool_task_ids": [row["task_id"] for row in full_tasks],
        "limited_budget_task_ids": [row["task_id"] for row in limited_tasks],
        "role_overlap_task_ids": sorted(full_ids & limited_ids),
        "role_selection": "ascending SHA256(task_id); first two full-pool, all eligible limited-budget",
        "selector_kind": "static_fallback",
        "performance_model_claim": False,
        "shortlist_maximum_including_incumbent": 4,
        "budgets": {"B1": 1, "B2": 2, "B3": 3, "unit": "additional new candidates"},
        "candidate_count": len(board_rows),
        "program_count": len(board_rows),
        "candidate_order": [row["candidate_id"] for row in board_rows],
        "program_order": [row["program_id"] for row in board_rows],
        "workload_counts": {
            workload_id: sum(row["workload_id"] == workload_id for row in board_rows)
            for workload_id in sorted(pools)
        },
        "correctness_policy": "three FPGA seeds before timing; invalid consumes budget and is not replaced",
        "reference_policy": "oracle only after all FPGA-correct programs in each declared full pool are timed",
        "selection_confirmation_separation": True,
        "manifests": manifests,
        "board_contacted": False,
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
    parser.add_argument("--pools", type=Path, default=DEFAULT_POOLS)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--model-readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
