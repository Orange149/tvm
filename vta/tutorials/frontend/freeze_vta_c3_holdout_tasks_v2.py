#!/usr/bin/env python3
"""Freeze C3 v2 full-pool and limited-budget holdout task roles.

Selection uses only audited exposure status, local legality/correctness, and a
stable hash of the execution-task identity.  It does not read board latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_EXPOSURE = GROUPED / "20260924_c3_real_task_exposure_audit_v2_run01"
DEFAULT_R18_PREPARE = GROUPED / "20260924_c3_real_task_reference_prepare_run03"
DEFAULT_R18_LOCAL = GROUPED / "20260924_c3_real_task_reference_local_run05"
DEFAULT_YOLO_PREPARE = GROUPED / "20260924_c3_yolo_unseen_resolution_prepare_run02"
DEFAULT_YOLO_POOLS = GROUPED / "20260924_c3_yolo608_candidate_pools_run01"
DEFAULT_YOLO_LOCAL = GROUPED / "20260924_c3_yolo608_local_qualification_run01"
DEFAULT_SELECTOR = GROUPED / "20260924_c3_selector_v2_freeze_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_holdout_task_freeze_v2_run01"
SCHEMA = "c3_holdout_task_freeze_v2"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_ledger(directory):
    directory = Path(directory).resolve()
    ledger_path = directory / "artifact_hashes.json"
    ledger = load(ledger_path)
    artifacts = ledger.get("artifacts", ledger)
    ignored = {"source", "source_sha256", "source_hashes"}
    checked = {}
    for name, expected in artifacts.items():
        if name in ignored or name.startswith("source.") or not isinstance(expected, str):
            continue
        path = directory / name
        if not path.is_file():
            raise ValueError("missing frozen artifact {} in {}".format(name, directory))
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError("artifact hash mismatch: {}".format(path))
        checked[name] = observed
    return {"directory": str(directory), "ledger_sha256": sha256_file(ledger_path), "checked": checked}


def candidate_program_map(local_dir):
    programs = load_jsonl(Path(local_dir) / "qualified_programs.jsonl")
    result = {}
    for program in programs:
        for candidate_id in program["equivalent_candidate_ids"]:
            result[candidate_id] = program["program_id"]
    return result, programs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposure", type=Path, default=DEFAULT_EXPOSURE)
    parser.add_argument("--r18-prepare", type=Path, default=DEFAULT_R18_PREPARE)
    parser.add_argument("--r18-local", type=Path, default=DEFAULT_R18_LOCAL)
    parser.add_argument("--yolo-prepare", type=Path, default=DEFAULT_YOLO_PREPARE)
    parser.add_argument("--yolo-pools", type=Path, default=DEFAULT_YOLO_POOLS)
    parser.add_argument("--yolo-local", type=Path, default=DEFAULT_YOLO_LOCAL)
    parser.add_argument("--selector", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite immutable output: {}".format(args.output))

    inputs = {
        name: verify_ledger(path)
        for name, path in (
            ("exposure", args.exposure), ("r18_prepare", args.r18_prepare),
            ("r18_local", args.r18_local), ("yolo_pools", args.yolo_pools),
            ("yolo_local", args.yolo_local), ("selector", args.selector),
        )
    }
    exposure = load_jsonl(args.exposure / "exposure_bindings.jsonl")
    strict_dispatch = {
        row["dispatch_scope_id"]: row for row in exposure
        if row["strict_latency_unseen_eligible"]
    }
    if len(strict_dispatch) != 2:
        raise ValueError("expected exactly two audited strict-unseen ResNet18 dispatch tasks")

    r18_contract = load(args.r18_prepare / "contract.json")
    r18_summary = load(args.r18_local / "summary.json")
    r18_candidate_to_program, r18_programs = candidate_program_map(args.r18_local)
    r18_tasks = []
    for name in r18_contract["pool_files"]:
        pool = load(args.r18_prepare / name)
        dispatch_id = pool["dispatch_scope_id"]
        if dispatch_id not in strict_dispatch:
            raise ValueError("R18 pool is not bound to an audited strict-unseen dispatch")
        local = r18_summary["workloads"][pool["workload_id"]]
        program_ids = sorted({
            r18_candidate_to_program[row["candidate_id"]]
            for row in pool["candidates"] if row["candidate_id"] in r18_candidate_to_program
        })
        incumbent_program = r18_candidate_to_program.get(pool["incumbent"]["candidate_id"])
        eligible = (
            local["incumbent_fsim_passed"]
            and local["source_adapter_equivalence"] == "equivalent"
            and incumbent_program is not None
            and len(program_ids) >= 3
        )
        r18_tasks.append({
            "network": "resnet18", "task_id": dispatch_id,
            "workload_id": pool["workload_id"], "pool_file": name,
            "stable_rank": stable_rank(dispatch_id),
            "program_ids": program_ids, "distinct_program_count": len(program_ids),
            "incumbent_candidate_id": pool["incumbent"]["candidate_id"],
            "incumbent_program_id": incumbent_program,
            "local_qualified": eligible,
            "latency_unseen": True,
            "qualification_rule": "incumbent correct and source-equivalent plus at least two legal new programs",
        })
    r18_eligible = sorted((row for row in r18_tasks if row["local_qualified"]), key=lambda row: row["stable_rank"])

    resolution = load(args.yolo_prepare / "contract.json")
    yolo_local = load(args.yolo_local / "summary.json")
    if resolution.get("selected_resolution") != 608:
        raise ValueError("frozen resolution order did not select 608")
    if yolo_local.get("status") != "yolo_resolution_eligible":
        raise ValueError("YOLO-608 did not pass local resolution eligibility")
    yolo_contract = load(args.yolo_pools / "contract.json")
    yolo_candidate_to_program, yolo_programs = candidate_program_map(args.yolo_local)
    yolo_tasks = []
    for name in yolo_contract["pool_files"]:
        pool = load(args.yolo_pools / name)
        local = yolo_local["workloads"][pool["workload_id"]]
        if not local["reference_eligible"]:
            continue
        program_ids = sorted({
            yolo_candidate_to_program[row["candidate_id"]]
            for row in pool["candidates"] if row["candidate_id"] in yolo_candidate_to_program
        })
        incumbent_program = yolo_candidate_to_program.get(pool["incumbent"]["candidate_id"])
        task_id = pool["execution_task_id"]
        yolo_tasks.append({
            "network": "yolov3_tiny_608", "task_id": task_id,
            "workload_id": pool["workload_id"], "pool_file": name,
            "stable_rank": stable_rank(task_id),
            "program_ids": program_ids, "distinct_program_count": len(program_ids),
            "incumbent_candidate_id": pool["incumbent"]["candidate_id"],
            "incumbent_program_id": incumbent_program,
            "local_qualified": len(program_ids) >= 8 and incumbent_program is not None,
            "latency_unseen": True,
            "qualification_rule": "at least eight address-certified, three-seed-correct, non-equivalent programs",
        })
    yolo_eligible = sorted((row for row in yolo_tasks if row["local_qualified"]), key=lambda row: row["stable_rank"])
    if not r18_eligible or len(yolo_eligible) < 3:
        raise ValueError("insufficient locally qualified tasks for the frozen role allocation")

    full_pool = [r18_eligible[0], yolo_eligible[0]]
    limited = r18_eligible[:2] + yolo_eligible[1:3]
    if len(limited) < 4:
        limited += [row for row in yolo_eligible if row not in limited][:4 - len(limited)]
    if len(limited) != 4:
        raise ValueError("unable to freeze four limited-budget tasks")
    for role, rows in (("full_pool_reference", full_pool), ("limited_budget_confirmation", limited)):
        for ordinal, row in enumerate(rows, 1):
            row["role"] = role
            row["role_ordinal"] = ordinal

    args.output.mkdir(parents=True)
    write(args.output / "full_pool_tasks.json", full_pool)
    write(args.output / "limited_budget_tasks.json", limited)
    decision = {
        "schema": SCHEMA,
        "status": "holdout_task_roles_frozen_before_board_latency",
        "selector_freeze": inputs["selector"],
        "resolution_decision": {
            "frozen_order": [608, 576, 544, 512, 480, 448, 384, 352],
            "selected": 608,
            "reason": "first static-feasible resolution and three tasks have at least eight distinct locally correct programs",
            "eligible_yolo_task_count": len(yolo_eligible),
        },
        "full_pool_task_ids": [row["task_id"] for row in full_pool],
        "limited_budget_task_ids": [row["task_id"] for row in limited],
        "role_overlap_task_ids": sorted(set(row["task_id"] for row in full_pool) & set(row["task_id"] for row in limited)),
        "role_overlap_policy": "allowed by protocol; selection observations and independent confirmation observations remain disjoint",
        "selection_method": "ascending sha256(task_id), independently within network and role priority",
        "inputs": inputs,
        "board_contacted": False,
        "latency_values_read": False,
        "claim_boundary": "task identity and local qualification only; no board winner or oracle is known",
    }
    write(args.output / "contract.json", decision)
    write(args.output / "summary.json", decision)
    write(args.output / "artifact_hashes.json", {
        "artifacts": {
            path.name: sha256_file(path) for path in sorted(args.output.iterdir())
            if path.is_file() and path.name != "artifact_hashes.json"
        },
        "source": sha256_file(Path(__file__).resolve()),
    })
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
