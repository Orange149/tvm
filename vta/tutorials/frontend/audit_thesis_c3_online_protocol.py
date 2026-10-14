#!/usr/bin/env python3
"""Audit a host-frozen C3 campaign and paid-information broker trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--broker-trace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    contract = load(args.contract)
    broker = load(args.broker_trace)
    failures = []
    tasks = contract.get("tasks", {})
    if list(tasks) != ["N1", "N2", "N3"]:
        failures.append("task order")
    for task_id, task in tasks.items():
        identities = task.get("requested_identities", [])
        ids = {row.get("requested_id") for row in identities}
        if len(identities) != 48 or len(ids) != 48:
            failures.append("{} requested space".format(task_id))
        by_tile = {}
        for row in identities:
            by_tile.setdefault(row.get("tile_id"), set()).add(row.get("mode"))
        if len(by_tile) != 12 or any(modes != {"O", "I", "W", "IW"} for modes in by_tile.values()):
            failures.append("{} Tile/mode factorial".format(task_id))
        for seed, order in task.get("random_orders", {}).items():
            if len(order) != 48 or set(order) != ids:
                failures.append("{} random order {}".format(task_id, seed))
    attempts = contract.get("attempts", [])
    if len(attempts) != 27 or len({row.get("attempt_id") for row in attempts}) != 27:
        failures.append("attempt cardinality")
    expected_rotations = {
        57001: ["random", "mode_aware_xgb", "staged_pareto"],
        57002: ["mode_aware_xgb", "staged_pareto", "random"],
        57003: ["staged_pareto", "random", "mode_aware_xgb"],
    }
    for task_id in ("N1", "N2", "N3"):
        for seed, expected in expected_rotations.items():
            observed = [
                row["strategy"] for row in attempts
                if row.get("task_id") == task_id and int(row.get("seed", -1)) == seed
            ]
            if observed != expected:
                failures.append("{} seed {} rotation".format(task_id, seed))
    if contract.get("target_labels_read") is not False or contract.get("status") != "frozen_before_target_labels":
        failures.append("label-free freeze status")
    if int(contract.get("request_budget_per_attempt", -1)) != 24 or int(
        contract.get("online_deadline_seconds", -1)
    ) != 600:
        failures.append("online budget")

    if broker.get("proposal_count") != 1 or len(broker.get("requested_ids", [])) != 1:
        failures.append("duplicate proposal accounting")
    events = broker.get("events", [])
    if [event.get("event") for event in events] != [
        "requested",
        "duplicate_request",
        "phase_completed",
        "late_phase_result",
    ]:
        failures.append("broker event sequence")
    candidate = broker.get("candidates", {}).get(broker.get("requested_ids", [None])[0], {})
    known = candidate.get("known", {})
    if "lower_pass" not in known or "fsim_pass" in known:
        failures.append("late result admission")
    if len(broker.get("late_results", [])) != 1:
        failures.append("late result retention")
    result = {
        "schema": "thesis_c3_online_protocol_host_audit_v1",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "tasks_checked": len(tasks),
        "requested_identities_checked": sum(len(task["requested_identities"]) for task in tasks.values()),
        "attempts_checked": len(attempts),
        "late_results_retained": len(broker.get("late_results", [])),
        "hardware_calls": 0,
        "formal_stage05_campaign_frozen": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
