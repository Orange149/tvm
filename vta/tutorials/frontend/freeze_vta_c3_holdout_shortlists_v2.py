#!/usr/bin/env python3
"""Freeze selector-v2 feature rows, rankings, and holdout shortlists."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from c3_selector_v2 import fixed_random_orders, rank_candidates, shortlist
from build_vta_c3_selector_v2_dataset import blank_metrics, normalize_mode


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_TASKS = GROUPED / "20260924_c3_holdout_task_freeze_v2_run01"
DEFAULT_SELECTOR = GROUPED / "20260924_c3_selector_v2_freeze_run01"
DEFAULT_R18_PREPARE = GROUPED / "20260924_c3_real_task_reference_prepare_run03"
DEFAULT_R18_LOCAL = GROUPED / "20260924_c3_real_task_reference_local_run05"
DEFAULT_YOLO_POOLS = GROUPED / "20260924_c3_yolo608_candidate_pools_run01"
DEFAULT_YOLO_LOCAL = GROUPED / "20260924_c3_yolo608_local_qualification_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_holdout_shortlists_v2_run02"
SCHEMA = "c3_holdout_shortlists_v2"


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


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def pool_index(directory):
    result = {}
    for path in Path(directory).glob("*.json"):
        payload = load(path)
        if "workload_id" in payload and "candidates" in payload:
            result[payload["workload_id"]] = (path.name, payload)
    return result


def metrics(static, fsim):
    result = blank_metrics()
    dma = static.get("dma_features", {})
    totals = dma.get("totals", {})
    command = fsim.get("command_features", {}).get("values") or {}
    command_totals = command.get("totals", {})
    result.update({
        "inp_request_bytes": totals.get("load_buffer_2d_inp_bytes"),
        "wgt_request_bytes": totals.get("load_buffer_2d_wgt_bytes"),
        "acc_request_bytes": totals.get("load_buffer_2d_acc_bytes"),
        "out_request_bytes": totals.get("store_buffer_2d_out_bytes"),
        "total_request_bytes": (
            totals.get("load_buffer_2d_bytes", 0) + totals.get("store_buffer_2d_bytes", 0)
            if "load_buffer_2d_bytes" in totals or "store_buffer_2d_bytes" in totals else None
        ),
        "inp_request_calls": totals.get("load_buffer_2d_inp_calls"),
        "wgt_request_calls": totals.get("load_buffer_2d_wgt_calls"),
        "acc_request_calls": totals.get("load_buffer_2d_acc_calls"),
        "out_request_calls": totals.get("store_buffer_2d_out_calls"),
        "total_request_calls": (
            totals.get("load_buffer_2d_calls", 0) + totals.get("store_buffer_2d_calls", 0)
            if "load_buffer_2d_calls" in totals or "store_buffer_2d_calls" in totals else None
        ),
        # The command signature records packed command UOP bytes, not LOAD-UOP
        # transfer bytes.  Keep the latter unknown rather than relabel it.
        "uop_load_bytes": None,
        "uop_load_calls": None,
        "command_uop_bytes": command_totals.get("uop_bytes"),
        "instruction_bytes": command_totals.get("insn_bytes"),
        "submissions": command.get("submissions"),
        "sync_count": static.get("sync", {}).get("explicit_residency_drains"),
    })
    certificate = static.get("arena_capacity_certificate", {}).get("domains", {})
    scopes = {"inp": "local.inp_buffer", "wgt": "local.wgt_buffer", "acc": "local.acc_buffer"}
    for name, scope in scopes.items():
        domain = certificate.get(scope)
        if domain and domain.get("capacity_bytes"):
            result[name + "_arena_fraction"] = (
                float(domain["required_upper_bound_bytes"]) / float(domain["capacity_bytes"])
            )
    return result


def load_task_rows(task, pool, local_dir):
    static = {row["candidate_id"]: row for row in load_jsonl(Path(local_dir) / "static_results.jsonl")}
    fsim = {row["candidate_id"]: row for row in load_jsonl(Path(local_dir) / "fsim_results.jsonl")}
    programs = [row for row in load_jsonl(Path(local_dir) / "qualified_programs.jsonl")
                if row["workload_id"] == task["workload_id"]]
    candidates = {row["candidate_id"]: row for row in pool["candidates"]}
    incumbent_candidate = task["incumbent_candidate_id"]
    incumbent_program = task["incumbent_program_id"]
    representative_by_program = {row["program_id"]: row["representative_candidate_id"] for row in programs}
    if incumbent_program not in representative_by_program:
        raise ValueError("incumbent program missing from locally qualified program set")
    incumbent_rep = representative_by_program[incumbent_program]
    incumbent_metrics = metrics(static[incumbent_rep], fsim[incumbent_rep])
    incumbent_tile = candidates[incumbent_candidate]["visible_features"]
    rows = []
    for program in sorted(programs, key=lambda row: row["program_id"]):
        representative = program["representative_candidate_id"]
        candidate = candidates[representative]
        rows.append({
            "schema": SCHEMA,
            "execution_task_id": task["task_id"],
            "workload_id": task["workload_id"],
            "network": task["network"],
            "candidate_id": program["program_id"],
            "program_id": program["program_id"],
            "representative_candidate_id": representative,
            "equivalent_candidate_ids": program["equivalent_candidate_ids"],
            "incumbent_id": incumbent_program,
            "is_incumbent": program["program_id"] == incumbent_program,
            "candidate_mode": normalize_mode(candidate["residence_mode"]),
            "incumbent_mode": normalize_mode(candidates[incumbent_candidate]["residence_mode"]),
            "candidate_tile": candidate["visible_features"],
            "incumbent_tile": incumbent_tile,
            "candidate_metrics": metrics(static[representative], fsim[representative]),
            "incumbent_metrics": incumbent_metrics,
            "local_fsim_correct": True,
            "fpga_correct": None,
            "target_log_latency_ratio": None,
            "board_latency_ms": None,
            "feature_provenance": {
                "requests_calls": "real lowered logical DMA descriptors",
                "command_uop_bytes": "three-seed-consistent command totals.uop_bytes; not LOAD-UOP bytes",
                "instruction_bytes_submissions": "three-seed-consistent command signature",
                "arena": "StorageRewrite span/capacity where present; otherwise unknown",
            },
        })
    if sum(row["is_incumbent"] for row in rows) != 1:
        raise ValueError("task does not contain exactly one incumbent program")
    return rows


def metric_order(rows, field):
    def key(row):
        value = row["candidate_metrics"].get(field)
        return (value is None, 0 if value is None else float(value), row["candidate_id"])
    return [row["candidate_id"] for row in sorted((row for row in rows if not row["is_incumbent"]), key=key)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--selector", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--r18-prepare", type=Path, default=DEFAULT_R18_PREPARE)
    parser.add_argument("--r18-local", type=Path, default=DEFAULT_R18_LOCAL)
    parser.add_argument("--yolo-pools", type=Path, default=DEFAULT_YOLO_POOLS)
    parser.add_argument("--yolo-local", type=Path, default=DEFAULT_YOLO_LOCAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite immutable output: {}".format(args.output))
    selector_contract = load(args.selector / "selector_v2_contract.json")
    if selector_contract.get("status") != "frozen":
        raise ValueError("selector v2 is not frozen")
    full = load(args.tasks / "full_pool_tasks.json")
    limited = load(args.tasks / "limited_budget_tasks.json")
    union = {row["task_id"]: row for row in full + limited}
    r18_pools, yolo_pools = pool_index(args.r18_prepare), pool_index(args.yolo_pools)
    all_rows, policies = [], {}
    for task_id, task in sorted(union.items()):
        if task["network"] == "resnet18":
            _, pool = r18_pools[task["workload_id"]]
            local_dir = args.r18_local
        else:
            _, pool = yolo_pools[task["workload_id"]]
            local_dir = args.yolo_local
        rows = load_task_rows(task, pool, local_dir)
        ranked = rank_candidates(rows, selector_contract)
        short = shortlist(rows, selector_contract)
        incumbent_id = next(row["candidate_id"] for row in rows if row["is_incumbent"])
        selector_order = [row["candidate_id"] for row in ranked if not row["is_incumbent"]]
        frozen_shortlist = list(short["shortlist_candidate_ids"])
        for candidate_id in selector_order:
            if candidate_id not in frozen_shortlist and len(frozen_shortlist) < min(4, len(rows)):
                frozen_shortlist.append(candidate_id)
        short["shortlist_candidate_ids"] = frozen_shortlist
        short["fill_policy"] = (
            "deduplicate incumbent/selector/bytes/command roles, then fill to incumbent plus "
            "at most three new programs by frozen selector-v2 rank"
        )
        budget_order = [candidate_id for candidate_id in frozen_shortlist if candidate_id != incumbent_id]
        bytes_order = metric_order(rows, "total_request_bytes")
        calls_order = metric_order(rows, "total_request_calls")
        command_order = sorted(
            (row["candidate_id"] for row in rows if not row["is_incumbent"]),
            key=lambda cid: next(
                (
                    row["candidate_metrics"].get("instruction_bytes") is None,
                    row["candidate_metrics"].get("instruction_bytes") or 0,
                    row["candidate_metrics"].get("command_uop_bytes") is None,
                    row["candidate_metrics"].get("command_uop_bytes") or 0,
                    cid,
                ) for row in rows if row["candidate_id"] == cid
            ),
        )
        policies[task_id] = {
            "execution_task_id": task_id,
            "workload_id": task["workload_id"],
            "network": task["network"],
            "incumbent_program_id": incumbent_id,
            "complete_local_program_pool": [row["candidate_id"] for row in sorted(rows, key=lambda row: row["candidate_id"])],
            "selector_v2_order": selector_order,
            "bytes_only_order": bytes_order,
            "calls_only_order": calls_order,
            "command_only_order": command_order,
            "shortlist": short,
            "B1_B2_B3": {
                name: budget_order[:budget] for name, budget in (("B1", 1), ("B2", 2), ("B3", 3))
            },
            "fpga_invalid_policy": "attempt consumes budget; no replacement",
            "random_orders_available_for_offline_full_pool_only": task_id in {row["task_id"] for row in full},
        }
        if task_id in {row["task_id"] for row in full}:
            policies[task_id]["fixed_random_orders_1000"] = fixed_random_orders(
                [row["candidate_id"] for row in rows if not row["is_incumbent"]]
            )
        all_rows.extend(ranked)

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "program_features.jsonl", all_rows)
    write(args.output / "task_policies.json", policies)
    contract = {
        "schema": SCHEMA,
        "status": "features_rankings_and_shortlists_frozen_before_board_latency",
        "selector_contract_sha256": sha256_file(args.selector / "selector_v2_contract.json"),
        "task_contract_sha256": sha256_file(args.tasks / "contract.json"),
        "execution_task_count": len(policies),
        "program_count": len(all_rows),
        "shortlist_maximum_including_incumbent": 4,
        "budgets": {"B1": 1, "B2": 2, "B3": 3, "unit": "additional new candidates"},
        "missing_data_policy": "unknown remains null; frozen selector preprocessor supplies training medians plus missing indicators",
        "uop_semantics": "command_uop_bytes is not cumulative LOAD-UOP bytes; LOAD-UOP remains unknown",
        "board_contacted": False,
        "latency_values_read": False,
    }
    write(args.output / "contract.json", contract)
    write(args.output / "summary.json", contract)
    write(args.output / "artifact_hashes.json", {
        "artifacts": {path.name: sha256_file(path) for path in sorted(args.output.iterdir())
                      if path.is_file() and path.name != "artifact_hashes.json"},
        "source": sha256_file(Path(__file__).resolve()),
    })
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
