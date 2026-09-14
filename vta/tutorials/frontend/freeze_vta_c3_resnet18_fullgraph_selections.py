#!/usr/bin/env python3
"""Freeze policy-selected R18 route triplets before any selected full-graph build."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import run_vta_c3_literature_baselines as baseline


POLICIES = (
    "stock_mode_aware_xgb",
    "cheng_minimum_access",
    "ml2tuner_pva",
    "ours_dma_multifidelity",
)
MODE_NUMBERS = {
    "original": 0,
    "input_stationary": 1,
    "weight_resident_barrier": 4,
    "input_weight_resident_barrier": 5,
}
WORKLOADS = ("R18-H1", "R18-H2", "R18-H3")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify(directory):
    directory = Path(directory).resolve()
    ledger = read_json(directory / "artifact_hashes.json")["artifacts"]
    for name, expected in ledger.items():
        if isinstance(expected, str) and baseline.sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: {}".format(directory / name))
    return baseline.sha256_file(directory / "artifact_hashes.json")


def selected_ids(analysis_rows, policy, seed, budget):
    output = {}
    for workload_id in WORKLOADS:
        matched = [
            row
            for row in analysis_rows
            if row["workload_id"] == workload_id
            and row["policy"] == policy
            and int(row["seed"]) == seed
            and int(row["requested_budget"]) == budget
        ]
        if len(matched) != 1:
            raise ValueError("missing unique analysis row for {} {}".format(policy, workload_id))
        candidate_id = matched[0]["target"].get("final_best_candidate_id")
        if not candidate_id:
            raise RuntimeError("{} {} did not find a valid candidate".format(policy, workload_id))
        output[workload_id] = candidate_id
    return output


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    analysis_dir = Path(args.analysis_dir).resolve()
    adapter_dir = Path(args.adapter_dir).resolve()
    source_dir = Path(args.source_contract_dir).resolve()
    input_hashes = {
        "analysis": verify(analysis_dir),
        "adapter": verify(adapter_dir),
        "source_contract": verify(source_dir),
    }
    analysis = read_json(analysis_dir / "summary.json")
    if analysis.get("status") != "complete_six_policy_twenty_seed_pool_analysis":
        raise RuntimeError("complete literature-pool analysis is required")
    source = read_json(source_dir / "contract.json")
    if source.get("status") != "frozen_before_resnet18_fullgraph_build_fpga_or_latency":
        raise RuntimeError("full-graph source is not pristine")
    analysis_rows = read_jsonl(analysis_dir / "results.jsonl")
    candidates = {}
    for workload_id in WORKLOADS:
        contract = read_json(
            adapter_dir / "{}_workload_contract.json".format(workload_id.lower())
        )
        if contract.get("target_labels_present") is not False:
            raise RuntimeError("route contract leaked target outcomes")
        candidates.update({row["candidate_id"]: row for row in contract["candidates"]})

    policy_rows = {}
    for policy in POLICIES:
        ids = selected_ids(analysis_rows, policy, args.seed, args.budget)
        routes = []
        for workload_id in WORKLOADS:
            candidate = candidates.get(ids[workload_id])
            if candidate is None or candidate["workload_id"] != workload_id:
                raise RuntimeError("selected candidate identity/workload mismatch")
            mode = candidate["residence_mode"]
            route = {
                "candidate_id": candidate["candidate_id"],
                "family_id": candidate["family_id"],
                "workload_id": workload_id,
                "public_mode": mode,
                "implementation_mode": MODE_NUMBERS[mode],
                "identity": {
                    "workload": source["verified_target_workloads"][workload_id]["workload"],
                    "complete_config_entity": candidate["complete_config_entity"],
                    "public_mode": mode,
                    "implementation_mode": MODE_NUMBERS[mode],
                },
            }
            if route["identity"]["workload"] != source["verified_target_workloads"][workload_id]["workload"]:
                raise RuntimeError("source workload mismatch")
            routes.append(route)
        policy_rows[policy] = {
            "policy": policy,
            "seed": args.seed,
            "budget": args.budget,
            "routes": routes,
            "route_signature": baseline.digest_value(
                [(row["workload_id"], row["candidate_id"]) for row in routes]
            ),
        }

    aliases = {}
    for policy, row in policy_rows.items():
        aliases.setdefault(row["route_signature"], []).append(policy)
    contract = {
        "schema": "c3_resnet18_fullgraph_selection_contract_v1",
        "status": "frozen_after_complete_operator_pools_before_selected_fullgraph_build_or_label",
        "selection_seed": args.seed,
        "selection_budget": args.budget,
        "policies": list(POLICIES),
        "policy_routes": policy_rows,
        "unique_route_program_count": len(aliases),
        "route_aliases": aliases,
        "stock_reference": "TopHub context; correctness/performance reference, not a search policy",
        "official_empty_history_autotvm": (
            "separate three-clean-start H1 experiment; stock_mode_aware_xgb here is the "
            "same-pool replay baseline and is not relabelled as that experiment"
        ),
        "input_hashes": input_hashes,
        "fullgraph_labels_read": False,
        "board_contacted": False,
    }
    output.mkdir(parents=True)
    write_json(output / "contract.json", contract)
    (output / "results.jsonl").write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in policy_rows.values()),
        encoding="utf-8",
    )
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    write_json(
        output / "summary.json",
        {
            "status": contract["status"],
            "policy_count": len(policy_rows),
            "unique_route_program_count": len(aliases),
            "selection_seed": args.seed,
            "selection_budget": args.budget,
        },
    )
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                path.name: baseline.sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_sha256": baseline.sha256_file(Path(__file__).resolve()),
        },
    )
    print(json.dumps(read_json(output / "summary.json"), indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--source-contract-dir", required=True)
    parser.add_argument("--seed", type=int, default=57001)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
