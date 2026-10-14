#!/usr/bin/env python3
"""Freeze selector-v4's label-free low-movement/Tile-diversity contract.

The source YOLO-576 tasks are development data whose latency is already
exposed.  This script intentionally reads only their pre-latency static freeze
artifact and never reads board timing output.  The resulting rule may be used
on future tasks, but the source tasks cannot become holdout evidence again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import c3_selector_v4_diverse


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_INPUT = GROUPED / "20260924_c3_yolo576_transition_safe_timing_freeze_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_selector_v4_diverse_development_freeze_run01"
SCHEMA = "c3_selector_v4_diverse_contract_freeze_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify(directory):
    directory = Path(directory).resolve()
    manifest = read(directory / "artifact_hashes.json")
    for name, expected in manifest.get("artifacts", {}).items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("artifact hash mismatch: {}".format(path))
    return sha256(directory / "artifact_hashes.json")


def build_contract(rows, policies, input_manifest_sha256):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("execution_task_id", ""))].append(row)
    if not grouped or "" in grouped:
        raise ValueError("all rows require execution_task_id")
    if set(grouped) != set(policies):
        raise ValueError("candidate tasks and policy tasks differ")

    task_orders = {}
    for task_id in sorted(grouped):
        incumbent = str(policies[task_id].get("incumbent_program_id", ""))
        task_rows = grouped[task_id]
        if not incumbent or sum(row.get("program_id") == incumbent for row in task_rows) != 1:
            raise ValueError("task requires exactly one frozen incumbent program")
        new_rows = [row for row in task_rows if row.get("program_id") != incumbent]
        result = c3_selector_v4_diverse.diverse_static_order(new_rows)
        task_orders[task_id] = {
            "incumbent_program_id": incumbent,
            "new_candidate_count": len(new_rows),
            "selector_output": result,
        }
    return {
        "schema": SCHEMA,
        "status": "FROZEN_FOR_FUTURE_LATENCY_UNSEEN_TASKS",
        "selector_schema": c3_selector_v4_diverse.SCHEMA,
        "selector_kind": "label_free_low_movement_tile_diversity",
        "max_budget": c3_selector_v4_diverse.DEFAULT_MAX_BUDGET,
        "window_multiplier": c3_selector_v4_diverse.DEFAULT_WINDOW_MULTIPLIER,
        "metric_priority": list(c3_selector_v4_diverse.METRIC_FIELDS),
        "tile_coordinates": list(c3_selector_v4_diverse.TILE_FIELDS),
        "task_orders": task_orders,
        "development_source_task_ids": sorted(grouped),
        "development_source_latency_exposed": True,
        "development_source_can_be_holdout_again": False,
        "performance_labels_read_by_freeze": False,
        "board_contacted": False,
        "performance_model_claim": False,
        "safe_pruning_claim": False,
        "input_manifest_sha256": input_manifest_sha256,
        "claim_boundary": (
            "candidate organization heuristic only; superiority requires same-budget evaluation "
            "on future latency-unseen complete pools"
        ),
    }


def run(args):
    input_dir = Path(args.input_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    manifest_sha256 = verify(input_dir)
    source_contract = read(input_dir / "contract.json")
    if (source_contract.get("performance_labels_collected") is not False
            or source_contract.get("board_contacted") is not False):
        raise ValueError("selector freeze input must predate performance labels")
    rows = read_jsonl(input_dir / "candidates.jsonl")
    policies = read(input_dir / "task_policies.json")
    contract = build_contract(rows, policies, manifest_sha256)

    output.mkdir(parents=True)
    write(output / "selector_v4_contract.json", contract)
    summary = {
        key: contract[key] for key in (
            "schema", "status", "selector_schema", "selector_kind",
            "max_budget", "window_multiplier", "development_source_task_ids",
            "development_source_latency_exposed", "performance_labels_read_by_freeze",
            "board_contacted", "performance_model_claim", "safe_pruning_claim",
        )
    }
    write(output / "summary.json", summary)
    write(output / "artifact_hashes.json", {
        "artifacts": {
            path.name: sha256(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "artifact_hashes.json"
        },
        "source_sha256": sha256(Path(__file__).resolve()),
        "selector_source_sha256": sha256(Path(c3_selector_v4_diverse.__file__).resolve()),
        "input_manifest_sha256": manifest_sha256,
    })
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

