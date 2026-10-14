#!/usr/bin/env python3
"""Freeze the 3-task/27-attempt C3 online campaign before any target label."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from thesis_c3_online_protocol import (
    PHASE_LIMIT_SECONDS,
    freeze_attempt_plan,
    freeze_requested_space,
    random_requested_order,
)


FORBIDDEN = ("outcome", "latency", "measured", "lower_pass", "fsim_pass", "numeric_correct")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_labels(value, path="root"):
    if isinstance(value, dict):
        for key, nested in value.items():
            if any(token in key.lower() for token in FORBIDDEN):
                raise ValueError("target label field is forbidden before freeze: {}.{}".format(path, key))
            reject_labels(nested, "{}.{}".format(path, key))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            reject_labels(nested, "{}[{}]".format(path, index))


def freeze(payload):
    if payload.get("schema") != "thesis_c3_online_source_freeze_input_v1":
        raise ValueError("unexpected source-freeze input schema")
    if payload.get("target_labels_read") is not False:
        raise ValueError("target labels must remain unread")
    reject_labels({key: value for key, value in payload.items() if key != "target_labels_read"})
    tasks = payload.get("tasks", [])
    if [row.get("task_id") for row in tasks] != ["N1", "N2", "N3"]:
        raise ValueError("tasks must be frozen in N1/N2/N3 order")
    spaces = {}
    for task in tasks:
        for field in ("source_model", "workload_signature", "history_exclusion_sha256"):
            if not task.get(field):
                raise ValueError("task lacks {}".format(field))
        identities = freeze_requested_space(task["task_id"], task.get("tile_ids", []))
        spaces[task["task_id"]] = {
            "source_model": task["source_model"],
            "workload_signature": task["workload_signature"],
            "history_exclusion_sha256": task["history_exclusion_sha256"],
            "requested_identities": identities,
            "requested_count": len(identities),
            "random_orders": {
                str(seed): random_requested_order(identities, seed)
                for seed in (57001, 57002, 57003)
            },
        }
    attempts = freeze_attempt_plan()
    return {
        "schema": "thesis_c3_online_campaign_contract_v1",
        "status": "frozen_before_target_labels",
        "target_labels_read": False,
        "tasks": spaces,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "requested_space_per_task": 48,
        "request_budget_per_attempt": 24,
        "online_deadline_seconds": 600,
        "phase_limit_seconds": PHASE_LIMIT_SECONDS,
        "policy_parameters": {
            "xgb": {"trees": 64, "max_depth": 3, "learning_rate": 0.1, "objective": "reg:squarederror", "tree_method": "hist", "n_jobs": 1},
            "xgb_random_until_valid_timing_labels": 4,
            "staged_frontier_width": 4,
            "staged_promotion_width": 2,
        },
        "claim_boundary": "source/space/attempt protocol only; no unseen qualification, execution, correctness, timing, deployment, or oracle result",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    input_path = Path(args.input).resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    contract = freeze(payload)
    contract["input"] = {"path": str(input_path), "sha256": sha256_file(input_path)}
    output.mkdir(parents=True)
    contract_path = output / "online_campaign_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": {contract_path.name: sha256_file(contract_path)}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": contract["status"], "attempt_count": 27, "tasks": 3}, indent=2))


if __name__ == "__main__":
    main()
