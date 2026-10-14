#!/usr/bin/env python3
"""Freeze label-blind selector orders for the two real-task reference pools."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
C3 = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
GROUPED = C3 / "07_grouped_holdout"
DEFAULT_PREPARE = GROUPED / "20260924_c3_real_task_reference_prepare_run02"
DEFAULT_LOCAL = GROUPED / "20260924_c3_real_task_reference_local_run04"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_real_task_selector_freeze_run01"
SCHEMA = "c3_real_task_selector_freeze_v1"
FEATURES = (
    "data_request_bytes",
    "command_insn_bytes",
    "command_uop_bytes",
    "submissions",
    "explicit_residency_drains",
)
RANDOM_SEED = "c3-real-task-fixed-random-20260924"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def verify_directory(directory):
    directory = Path(directory).resolve()
    ledger = json.loads((directory / "artifact_hashes.json").read_text(encoding="utf-8"))
    for name, expected in ledger["artifacts"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: {}".format(name))
    return sha256_file(directory / "artifact_hashes.json")


def dominates(left, right):
    return all(left[key] <= right[key] for key in FEATURES) and any(
        left[key] < right[key] for key in FEATURES
    )


def pareto_fronts(rows):
    remaining = list(rows)
    fronts = []
    while remaining:
        front = [
            row
            for row in remaining
            if not any(dominates(other["metrics"], row["metrics"]) for other in remaining if other is not row)
        ]
        fronts.append(sorted(front, key=lambda row: row["program_id"]))
        selected = {row["program_id"] for row in front}
        remaining = [row for row in remaining if row["program_id"] not in selected]
    return fronts


def percentile_ranks(rows):
    result = {row["program_id"]: {} for row in rows}
    denominator = max(1, len(rows) - 1)
    for feature in FEATURES:
        ordered = sorted(rows, key=lambda row: (row["metrics"][feature], row["program_id"]))
        for position, row in enumerate(ordered):
            result[row["program_id"]][feature] = position / denominator
    return result


def cost_aware_order(rows):
    front_by_id = {}
    for front_index, front in enumerate(pareto_fronts(rows)):
        for row in front:
            front_by_id[row["program_id"]] = front_index
    ranks = percentile_ranks(rows)
    return sorted(
        rows,
        key=lambda row: (
            front_by_id[row["program_id"]],
            max(ranks[row["program_id"]].values()),
            sum(ranks[row["program_id"]].values()) / len(FEATURES),
            row["program_id"],
        ),
    ), front_by_id, ranks


def program_rows(local_dir, prepare_dir):
    programs = read_jsonl(Path(local_dir) / "qualified_programs.jsonl")
    static = {row["candidate_id"]: row for row in read_jsonl(Path(local_dir) / "static_results.jsonl")}
    fsim = {row["candidate_id"]: row for row in read_jsonl(Path(local_dir) / "fsim_results.jsonl")}
    incumbents = {}
    for path in Path(prepare_dir).glob("r18-h*_four_mode_96.json"):
        pool = json.loads(path.read_text(encoding="utf-8"))
        incumbents[pool["workload_id"]] = pool["incumbent"]["candidate_id"]
    rows = []
    for program in programs:
        candidate_id = program["representative_candidate_id"]
        srow, frow = static[candidate_id], fsim[candidate_id]
        dma = srow["dma_features"]["logical_bytes"]
        commands = frow["command_features"]["values"]
        rows.append(
            {
                "schema": SCHEMA,
                "program_id": program["program_id"],
                "workload_id": program["workload_id"],
                "candidate_id": candidate_id,
                "equivalent_candidate_ids": program["equivalent_candidate_ids"],
                "is_incumbent": incumbents[program["workload_id"]] in program["equivalent_candidate_ids"],
                "identity": program["representative_identity"],
                "metrics": {
                    "data_request_bytes": int(sum(dma.values())),
                    "command_insn_bytes": int(commands["totals"]["insn_bytes"]),
                    "command_uop_bytes": int(commands["totals"]["uop_bytes"]),
                    "submissions": int(commands["submissions"]),
                    "explicit_residency_drains": int(srow["sync"]["explicit_residency_drains"]),
                },
            }
        )
    return rows


def freeze_workload(rows):
    incumbents = [row for row in rows if row["is_incumbent"]]
    if len(incumbents) != 1:
        raise ValueError("expected one incumbent program")
    incumbent = incumbents[0]
    new_rows = [row for row in rows if not row["is_incumbent"]]
    cost_rows, fronts, ranks = cost_aware_order(new_rows)
    bytes_rows = sorted(
        new_rows,
        key=lambda row: (
            row["metrics"]["data_request_bytes"],
            row["metrics"]["command_insn_bytes"],
            row["program_id"],
        ),
    )
    random_rows = sorted(
        new_rows,
        key=lambda row: hashlib.sha256(
            (RANDOM_SEED + row["program_id"]).encode("utf-8")
        ).hexdigest(),
    )
    enriched = []
    for row in rows:
        item = json.loads(json.dumps(row))
        if not row["is_incumbent"]:
            item["cost_aware_pareto_front"] = fronts[row["program_id"]]
            item["cost_aware_percentile_ranks"] = ranks[row["program_id"]]
        enriched.append(item)
    return {
        "incumbent_program_id": incumbent["program_id"],
        "full_pool_program_ids": [row["program_id"] for row in sorted(rows, key=lambda row: row["program_id"])],
        "orders": {
            "compiler_cost_aware": [row["program_id"] for row in cost_rows],
            "bytes_only": [row["program_id"] for row in bytes_rows],
            "fixed_random": [row["program_id"] for row in random_rows],
        },
        "extra_candidate_budgets": {
            method: {str(budget): order[:budget] for budget in (1, 2, 3)}
            for method, order in {
                "compiler_cost_aware": [row["program_id"] for row in cost_rows],
                "bytes_only": [row["program_id"] for row in bytes_rows],
                "fixed_random": [row["program_id"] for row in random_rows],
            }.items()
        },
        "programs": enriched,
    }


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    prepare_manifest = verify_directory(args.prepare_dir)
    local_manifest = verify_directory(args.local_dir)
    local_summary = json.loads((Path(args.local_dir) / "summary.json").read_text(encoding="utf-8"))
    if local_summary.get("status") != "ready_for_rule_freeze":
        raise ValueError("local pools are not reference eligible")
    if local_summary.get("board_contacted") is not False:
        raise ValueError("local qualification contacted board")
    rows = program_rows(args.local_dir, args.prepare_dir)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["workload_id"]].append(row)
    frozen = {workload_id: freeze_workload(items) for workload_id, items in sorted(grouped.items())}
    output.mkdir(parents=True)
    all_rows = [row for value in frozen.values() for row in value["programs"]]
    write_jsonl(output / "program_features.jsonl", all_rows)
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_current_board_latency",
        "prepare_manifest_sha256": prepare_manifest,
        "local_manifest_sha256": local_manifest,
        "features": list(FEATURES),
        "compiler_cost_aware_rule": (
            "non-dominated front, then minimum worst within-pool percentile, then minimum mean "
            "percentile, then program_id; all five features minimized"
        ),
        "bytes_only_rule": "data_request_bytes, then command_insn_bytes, then program_id",
        "fixed_random_rule": "sha256(fixed_seed || program_id)",
        "fixed_random_seed": RANDOM_SEED,
        "budget_definition": "1/2/3 extra new programs; incumbent is always measured and reported separately",
        "selection_measurement": {
            "purpose": "choose one program after the static order is frozen",
            "rounds": 2,
            "seeds_per_round": 3,
            "winner_statistic": "median of the six single-call observations",
        },
        "independent_confirmation": {
            "required": True,
            "may_not_change_selected_identity": True,
            "requires_later_boot": True,
            "rounds": 8,
            "seeds_per_round": 3,
        },
        "full_pool_reference": {
            "purpose": "reference oracle and missed-candidate audit only",
            "may_not_modify_frozen_orders": True,
            "rounds_per_boot": 6,
            "seeds_per_round": 3,
            "two_boots_required_for_stable_performance_claim": True,
        },
        "workloads": frozen,
        "board_contacted": False,
        "current_board_performance_labels_read": False,
        "claim_boundary": "static priority heuristic, not a speed guarantee",
    }
    write_json(output / "contract.json", contract)
    write_json(
        output / "summary.json",
        {
            "status": contract["status"],
            "workloads": {
                key: {
                    "pool_programs": len(value["full_pool_program_ids"]),
                    "incumbent_program_id": value["incumbent_program_id"],
                    "cost_aware_top3": value["orders"]["compiler_cost_aware"][:3],
                    "bytes_only_top3": value["orders"]["bytes_only"][:3],
                    "fixed_random_top3": value["orders"]["fixed_random"][:3],
                }
                for key, value in frozen.items()
            },
            "board_contacted": False,
        },
    )
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_sha256": sha256_file(Path(__file__).resolve()),
        },
    )
    print(json.dumps(json.loads((output / "summary.json").read_text()), indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-dir", type=Path, default=DEFAULT_PREPARE)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
