#!/usr/bin/env python3
"""Aggregate strict survival-adaptive holdouts without retuning the policy."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import replay_vta_multifidelity_search as search
from run_vta_p7r119_yolo_barrier_pilot import sha256_file, write_json


POLICIES = (
    "adaptive_survival_frontier_service",
    "exhaustive_then_service",
    "hardware_diverse_frontier_service",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def verify(directory):
    directory = Path(directory)
    ledger = read_json(directory / "artifact_hashes.json")
    for name, expected in ledger["artifacts"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: " + str(directory / name))
    return sha256_file(directory / "artifact_hashes.json")


def reduction(value, baseline):
    return 100.0 * (baseline - value) / baseline


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    directories = [Path(path) for path in args.analysis_dir]
    bindings = []
    per_holdout = []
    grouped = defaultdict(list)
    for directory in directories:
        ledger = verify(directory)
        confirmation = read_json(directory / "confirmation.json")
        bindings.append({"path": str(directory.resolve()), "ledger_sha256": ledger})
        per_holdout.append({
            "workload_id": confirmation["workload_id"],
            "registered_candidates": confirmation["registered_candidates"],
            "first_candidate_is_pool_oracle": confirmation["first_candidate_is_pool_oracle"],
            "adaptive_path": confirmation["adaptive_path"],
            "pool_oracle_latency_ms": confirmation["pool_oracle_latency_ms"],
        })
        for row in read_jsonl(directory / "metrics.jsonl"):
            if row["policy"] in POLICIES:
                grouped[(row["policy"], int(row["seed"]))].append(row)

    aggregates = []
    for (policy, seed), rows in sorted(grouped.items()):
        if len(rows) != len(directories):
            raise ValueError("incomplete policy/seed holdout coverage")
        aggregates.append({
            "policy": policy,
            "seed": seed,
            "all_targets_reached": all(row["target_reached"] for row in rows),
            "known_wall_ms": sum(float(row["known_wall_ms"]) for row in rows),
            "gross_candidates_considered": sum(
                int(row["gross_candidates_considered"]) for row in rows
            ),
            "phase_action_counts": {
                phase: sum(int(row["phase_action_counts"][phase]) for row in rows)
                for phase in search.PHASES
            },
            "resource_cost": {
                key: sum(float(row["resource_cost"][key]) for row in rows)
                for key in search.RESOURCE_KEYS
            },
        })

    summary = {}
    for policy in POLICIES:
        rows = [row for row in aggregates if row["policy"] == policy]
        summary[policy] = {
            "runs": len(rows),
            "all_holdouts_success_rate": sum(row["all_targets_reached"] for row in rows) /
            len(rows),
            "wall_ms_to_all_targets": search.distribution(
                row["known_wall_ms"] for row in rows
            ),
            "gross_candidates_to_all_targets": search.distribution(
                row["gross_candidates_considered"] for row in rows
            ),
            "phase_actions_to_all_targets": {
                phase: search.distribution(row["phase_action_counts"][phase] for row in rows)
                for phase in search.PHASES
            },
            "resources_to_all_targets": {
                key: search.distribution(row["resource_cost"][key] for row in rows)
                for key in search.RESOURCE_KEYS
            },
        }
    adaptive_wall = summary[POLICIES[0]]["wall_ms_to_all_targets"]["median"]
    exhaustive_wall = summary[POLICIES[1]]["wall_ms_to_all_targets"]["median"]
    fixed_wall = summary[POLICIES[2]]["wall_ms_to_all_targets"]["median"]
    comparisons = {
        "adaptive_wall_reduction_vs_exhaustive_percent": reduction(
            adaptive_wall, exhaustive_wall
        ),
        "adaptive_wall_reduction_vs_fixed_4_to_2_percent": reduction(
            adaptive_wall, fixed_wall
        ),
        "adaptive_gross_reduction_vs_exhaustive_percent": reduction(
            summary[POLICIES[0]]["gross_candidates_to_all_targets"]["median"],
            summary[POLICIES[1]]["gross_candidates_to_all_targets"]["median"],
        ),
    }
    result = {
        "schema": "c3_vta_strict_adaptive_holdout_aggregate_v1",
        "claim_status": "post_hoc_aggregate_of_individually_preregistered_holdouts_no_retuning",
        "holdouts": per_holdout,
        "bindings": bindings,
        "summary": summary,
        "comparisons": comparisons,
        "claim_boundary": (
            "aggregate only; each target was completed after its own prospective first wave; "
            "logical VTA resources are not physical AXI counters"
        ),
    }
    output.mkdir(parents=True)
    write_json(output / "adaptive_holdout_aggregate.json", result)
    (output / "aggregate_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in aggregates),
        encoding="utf-8",
    )
    write_json(output / "artifact_hashes.json", {
        "artifacts": {
            "adaptive_holdout_aggregate.json": sha256_file(
                output / "adaptive_holdout_aggregate.json"
            ),
            "aggregate_metrics.jsonl": sha256_file(output / "aggregate_metrics.jsonl"),
        },
        "source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__)},
    })
    print(json.dumps({"comparisons": comparisons, "summary": summary},
                     indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
