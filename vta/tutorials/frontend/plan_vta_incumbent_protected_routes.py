#!/usr/bin/env python3
"""Fail-closed greedy composition of graph-level VTA residency routes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def lookup(value, dotted_path):
    current = value
    for part in dotted_path.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    return current


def resolve_evidence(path, workspace_root):
    path = Path(path)
    return path if path.is_absolute() else workspace_root / path


def verify_evidence(spec, workspace_root):
    path = resolve_evidence(spec["path"], workspace_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256(path)
    if actual_hash != spec["sha256"]:
        raise RuntimeError("evidence hash mismatch: " + str(path))
    document = read_json(path)
    assertions = []
    for assertion in spec.get("assertions", []):
        actual = lookup(document, assertion["path"])
        expected = assertion["equals"]
        if actual != expected:
            raise RuntimeError(
                "evidence assertion failed at {}: {!r} != {!r}".format(
                    assertion["path"], actual, expected
                )
            )
        assertions.append({"path": assertion["path"], "value": actual})
    return {
        "path": str(path.resolve()),
        "sha256": actual_hash,
        "assertions": assertions,
    }


def run(args):
    contract_path = Path(args.contract)
    contract = read_json(contract_path)
    if contract.get("schema") != "c3_incumbent_protected_route_contract_v1":
        raise ValueError("unexpected contract schema")
    if contract.get("status") != "frozen_before_route_composition":
        raise ValueError("contract is not frozen")
    policy = contract["policy"]
    if policy.get("algorithm") != "greedy_sequential_incumbent_protection":
        raise ValueError("unsupported route composition algorithm")
    minimum_rounds = int(policy["minimum_paired_rounds"])
    minimum_win_fraction = float(policy["minimum_win_fraction"])
    minimum_improvement_ms = float(policy["minimum_improvement_ms"])
    workspace_root = Path(contract.get("workspace_root", ".")).resolve()

    current = list(contract["protected_incumbent"]["route_candidate_ids"])
    decisions = []
    for ordinal, proposal in enumerate(contract["proposals"]):
        if proposal["baseline_route_candidate_ids"] != current:
            raise RuntimeError(
                "proposal {} was not evaluated against the current incumbent route set".format(ordinal)
            )
        expected_proposed = current + [proposal["route"]["candidate_id"]]
        if proposal["proposed_route_candidate_ids"] != expected_proposed:
            raise RuntimeError("proposal {} does not add exactly one route".format(ordinal))
        evidence = [
            verify_evidence(spec, workspace_root) for spec in proposal["evidence"]
        ]
        evaluation = proposal["evaluation"]
        rounds = int(evaluation["paired_rounds"])
        wins = int(evaluation["proposed_paired_wins"])
        delta_ms = float(evaluation["median_paired_delta_ms"])
        checks = {
            "graph_outputs_equal": bool(evaluation["graph_outputs_equal"]),
            "live_memory_fit": bool(evaluation["live_memory_fit"]),
            "minimum_rounds": rounds >= minimum_rounds,
            "minimum_win_fraction": rounds > 0 and wins / rounds >= minimum_win_fraction,
            "minimum_improvement": delta_ms <= -minimum_improvement_ms,
        }
        accepted = all(checks.values())
        before = list(current)
        if accepted:
            current = expected_proposed
        decisions.append({
            "ordinal": ordinal,
            "route": proposal["route"],
            "incumbent_before": before,
            "proposed_route_set": expected_proposed,
            "evaluation": evaluation,
            "checks": checks,
            "accepted": accepted,
            "incumbent_after": list(current),
            "verified_evidence": evidence,
        })

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    result = {
        "schema": "c3_incumbent_protected_route_plan_v1",
        "status": "route_composition_complete",
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": sha256(contract_path),
        "policy": policy,
        "protected_incumbent": contract["protected_incumbent"],
        "decisions": decisions,
        "selected_route_candidate_ids": current,
        "accepted_count": sum(row["accepted"] for row in decisions),
        "rejected_count": sum(not row["accepted"] for row in decisions),
        "claim_boundary": (
            "Greedy sequential marginal admission with exact evidence binding; "
            "no global optimum guarantee over arbitrary route subsets"
        ),
    }
    write_json(output / "selected_routes.json", result)
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "status": result["status"],
        "selected_route_candidate_ids": current,
        "accepted_count": result["accepted_count"],
        "rejected_count": result["rejected_count"],
        "decisions": [
            {
                "candidate_id": row["route"]["candidate_id"],
                "accepted": row["accepted"],
                "checks": row["checks"],
                "median_paired_delta_ms": row["evaluation"]["median_paired_delta_ms"],
            }
            for row in decisions
        ],
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
