#!/usr/bin/env python3
"""Freeze the exact C3 v2 board program union before board contact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_TASKS = GROUPED / "20260924_c3_holdout_task_freeze_v2_run01"
DEFAULT_SHORTLISTS = GROUPED / "20260924_c3_holdout_shortlists_v2_run02"
DEFAULT_R18_PREPARE = GROUPED / "20260924_c3_real_task_reference_prepare_run03"
DEFAULT_R18_LOCAL = GROUPED / "20260924_c3_real_task_reference_local_run05"
DEFAULT_YOLO_POOLS = GROUPED / "20260924_c3_yolo608_candidate_pools_run01"
DEFAULT_YOLO_LOCAL = GROUPED / "20260924_c3_yolo608_local_qualification_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_holdout_board_pool_freeze_v2_run01"
SCHEMA = "c3_holdout_board_pool_v2"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pool_index(directory):
    result = {}
    for path in Path(directory).glob("*.json"):
        payload = load(path)
        if "workload_id" in payload and "candidates" in payload:
            result[payload["workload_id"]] = payload
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--shortlists", type=Path, default=DEFAULT_SHORTLISTS)
    parser.add_argument("--r18-prepare", type=Path, default=DEFAULT_R18_PREPARE)
    parser.add_argument("--r18-local", type=Path, default=DEFAULT_R18_LOCAL)
    parser.add_argument("--yolo-pools", type=Path, default=DEFAULT_YOLO_POOLS)
    parser.add_argument("--yolo-local", type=Path, default=DEFAULT_YOLO_LOCAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite immutable output: {}".format(args.output))
    task_contract = load(args.tasks / "contract.json")
    shortlist_contract = load(args.shortlists / "contract.json")
    if task_contract.get("latency_values_read") is not False or shortlist_contract.get("latency_values_read") is not False:
        raise ValueError("upstream freeze is not latency blind")
    full_tasks = {row["task_id"] for row in load(args.tasks / "full_pool_tasks.json")}
    limited_tasks = {row["task_id"] for row in load(args.tasks / "limited_budget_tasks.json")}
    policies = load(args.shortlists / "task_policies.json")
    features = load_jsonl(args.shortlists / "program_features.jsonl")
    feature_by_program = {row["program_id"]: row for row in features}
    if len(feature_by_program) != len(features):
        raise ValueError("program IDs are not globally unique")
    selected_programs = set()
    roles = {}
    for task_id, policy in policies.items():
        task_programs = set()
        task_roles = []
        if task_id in full_tasks:
            task_programs.update(policy["complete_local_program_pool"])
            task_roles.append("full_pool_reference")
        if task_id in limited_tasks:
            task_programs.update(policy["shortlist"]["shortlist_candidate_ids"])
            task_roles.append("limited_budget_confirmation")
        if not task_roles:
            continue
        for program_id in task_programs:
            if feature_by_program[program_id]["execution_task_id"] != task_id:
                raise ValueError("program/task identity mismatch")
            roles[program_id] = list(task_roles)
        selected_programs.update(task_programs)

    r18_pools, yolo_pools = pool_index(args.r18_prepare), pool_index(args.yolo_pools)
    local_sources = {
        "resnet18": args.r18_local,
        "yolov3_tiny_608": args.yolo_local,
    }
    pool_sources = {
        "resnet18": r18_pools,
        "yolov3_tiny_608": yolo_pools,
    }
    local_cache = {}
    rows = []
    for program_id in sorted(selected_programs):
        feature = feature_by_program[program_id]
        network = feature["network"]
        workload_id = feature["workload_id"]
        pool = pool_sources[network][workload_id]
        candidate_id = feature["representative_candidate_id"]
        raw = next(row for row in pool["candidates"] if row["candidate_id"] == candidate_id)
        local_dir = local_sources[network]
        cache_key = str(local_dir)
        if cache_key not in local_cache:
            local_cache[cache_key] = (
                {row["candidate_id"]: row for row in load_jsonl(local_dir / "static_results.jsonl")},
                {row["candidate_id"]: row for row in load_jsonl(local_dir / "fsim_results.jsonl")},
            )
        static, fsim = local_cache[cache_key]
        srow, frow = static[candidate_id], fsim[candidate_id]
        if srow["status"] != "ok" or frow["status"] != "passed":
            raise ValueError("selected representative lacks local qualification")
        if frow["relowered_tir_sha256"] != srow["lowered_tir_sha256"]:
            raise ValueError("selected representative TIR changed")
        row = json.loads(json.dumps(raw))
        row.update({
            "identity": {"workload": pool["workload"], "complete_config_entity": row["complete_config_entity"]},
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
                "strict_final_address_audit": srow.get("final_address_audit"),
                "arena_capacity_certificate": srow.get("arena_capacity_certificate"),
            },
            "performance_label": None,
            "board_contacted": False,
        })
        rows.append(row)
    rows.sort(key=lambda row: (row["execution_task_id"], row["program_id"]))
    if len(rows) != 24:
        raise ValueError("expected 24 unique board representatives, got {}".format(len(rows)))
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("representative candidate identity reused across programs")

    args.output.mkdir(parents=True)
    (args.output / "candidates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_cross_compile_or_board_contact",
        "candidate_count": len(rows),
        "program_count": len(rows),
        "candidate_order": [row["candidate_id"] for row in rows],
        "program_order": [row["program_id"] for row in rows],
        "full_pool_task_ids": sorted(full_tasks),
        "limited_budget_task_ids": sorted(limited_tasks),
        "workload_counts": {
            workload_id: sum(row["workload_id"] == workload_id for row in rows)
            for workload_id in sorted({row["workload_id"] for row in rows})
        },
        "correctness_policy": "three FPGA seeds before timing; invalid consumes budget and is not replaced",
        "selection_policy": "limited tasks may time only the frozen four-program shortlist",
        "reference_policy": "only full-pool-role tasks expose every locally qualified program",
        "task_contract_sha256": sha256_file(args.tasks / "contract.json"),
        "shortlist_contract_sha256": sha256_file(args.shortlists / "contract.json"),
        "board_contacted": False,
        "performance_labels_collected": False,
    }
    write(args.output / "contract.json", contract)
    (args.output / "timeline.jsonl").write_text("", encoding="utf-8")
    (args.output / "results.jsonl").write_text("", encoding="utf-8")
    write(args.output / "summary.json", contract)
    write(args.output / "artifact_hashes.json", {
        "artifacts": {
            path.name: sha256_file(path) for path in sorted(args.output.iterdir())
            if path.is_file() and path.name != "artifact_hashes.json"
        },
        "source": sha256_file(Path(__file__).resolve()),
    })
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
