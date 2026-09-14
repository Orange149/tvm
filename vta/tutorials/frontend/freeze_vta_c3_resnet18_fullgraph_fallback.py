#!/usr/bin/env python3
"""Freeze a performance-label-free fallback after an R18 graph correctness failure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import run_vta_c3_literature_baselines as baseline


MODE_NUMBERS = {
    "original": 0,
    "input_stationary": 1,
    "weight_resident_barrier": 4,
    "input_weight_resident_barrier": 5,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "artifact_hashes.json")
    for name, expected in manifest["artifacts"].items():
        if baseline.sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: {}".format(directory / name))
    return baseline.sha256_file(directory / "artifact_hashes.json")


def program_is_graph_correct(diagnostic, program_id):
    row = diagnostic["programs"].get(program_id)
    return bool(
        row
        and int(row["observations"]) == 6
        and int(row["stock_equal_observations"]) == 6
        and all(int(value) == 0 for value in row["mismatch_counts"])
        and row["stable_hash_by_seed"] is True
    )


def program_is_graph_invalid(diagnostic, program_id):
    row = diagnostic["programs"].get(program_id)
    return bool(
        row
        and int(row["observations"]) == 6
        and int(row["stock_equal_observations"]) == 0
        and all(int(value) > 0 for value in row["mismatch_counts"])
    )


def route_signature(routes):
    return baseline.digest_value(
        [(row["workload_id"], row["candidate_id"]) for row in routes]
    )


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    selection_dir = Path(args.selection_contract_dir).resolve()
    failed_dir = Path(args.failed_board_dir).resolve()
    diagnostic_dir = Path(args.diagnostic_dir).resolve()
    input_hashes = {
        "selection": verify(selection_dir),
        "failed_board": verify(failed_dir),
        "diagnostic": verify(diagnostic_dir),
    }
    selection = read_json(selection_dir / "contract.json")
    failed = read_json(failed_dir / "summary.json")
    failed_contract = read_json(failed_dir / "contract.json")
    diagnostic = read_json(diagnostic_dir / "summary.json")
    if selection.get("status") != "frozen_after_complete_operator_pools_before_selected_fullgraph_build_or_label":
        raise RuntimeError("original selection contract is not complete")
    if failed.get("status") != "invalid_entire_session_fail_closed":
        raise RuntimeError("expected a fail-closed selected-fullgraph session")
    if int(failed.get("timing_calls_completed", -1)) != 0:
        raise RuntimeError("fallback cannot be frozen after selected fullgraph timing")
    if diagnostic.get("status") != "complete_non_spliced_correctness_diagnostic":
        raise RuntimeError("complete mismatch diagnostic is required")
    if diagnostic.get("performance_labels_collected") is not False:
        raise RuntimeError("diagnostic contains performance labels")

    policy_to_program = failed_contract["policy_to_program"]
    source_rows = selection["policy_routes"]
    rejected = args.rejected_candidate_id
    fallback = args.fallback_candidate_id
    rejected_policies = [
        policy
        for policy, row in source_rows.items()
        if any(route["candidate_id"] == rejected for route in row["routes"])
    ]
    if not rejected_policies:
        raise RuntimeError("rejected route is absent from the frozen policy selections")
    for policy in rejected_policies:
        if not program_is_graph_invalid(diagnostic, policy_to_program[policy]):
            raise RuntimeError("rejected route carrier lacks six graph-invalid observations")

    fallback_carriers = [
        (policy, route)
        for policy, row in source_rows.items()
        for route in row["routes"]
        if route["candidate_id"] == fallback
    ]
    if len(fallback_carriers) != 1:
        raise RuntimeError("fallback route must occur exactly once in frozen policy selections")
    fallback_policy, fallback_route = fallback_carriers[0]
    if fallback_route["public_mode"] != "original":
        raise RuntimeError("fallback route is not original")
    if not program_is_graph_correct(diagnostic, policy_to_program[fallback_policy]):
        raise RuntimeError("fallback carrier lacks six stable graph-correct observations")

    replaced_workloads = {
        route["workload_id"]
        for row in source_rows.values()
        for route in row["routes"]
        if route["candidate_id"] == rejected
    }
    if replaced_workloads != {fallback_route["workload_id"]}:
        raise RuntimeError("rejected and fallback route workloads differ")
    policy_rows = {}
    replacements = []
    for policy, row in source_rows.items():
        routes = []
        for route in row["routes"]:
            if route["candidate_id"] != rejected:
                routes.append(route)
                continue
            replacement = json.loads(json.dumps(fallback_route))
            replacement["fallback"] = {
                "reason": "selected route failed final graph-context correctness",
                "rejected_candidate_id": rejected,
                "source_policy": fallback_policy,
                "selection_uses_fullgraph_performance": False,
            }
            routes.append(replacement)
            replacements.append(
                {
                    "policy": policy,
                    "workload_id": route["workload_id"],
                    "rejected_candidate_id": rejected,
                    "fallback_candidate_id": fallback,
                }
            )
        policy_rows[policy] = {
            "policy": policy,
            "seed": row["seed"],
            "budget": row["budget"],
            "routes": routes,
            "route_signature": route_signature(routes),
        }
    aliases = {}
    for policy, row in policy_rows.items():
        aliases.setdefault(row["route_signature"], []).append(policy)
    contract = {
        "schema": "c3_resnet18_fullgraph_correctness_fallback_contract_v1",
        "status": "frozen_after_graph_correctness_failure_before_any_selected_fullgraph_timing",
        "selection_seed": selection["selection_seed"],
        "selection_budget": selection["selection_budget"],
        "policy_routes": policy_rows,
        "route_aliases": aliases,
        "unique_route_program_count": len(aliases),
        "replacements": replacements,
        "fallback_rule": (
            "Reject the exact route that fails final graph-context correctness and use an "
            "already frozen original route from a six-observation graph-correct carrier. "
            "No fullgraph latency or FPS is available or used."
        ),
        "same_tile_fallback_available": False,
        "same_tile_fallback_explanation": (
            "The frozen 214-point pool contains no original identity with the rejected "
            "D0098 ConfigEntity, so a same-tile original must not be invented."
        ),
        "fullgraph_performance_labels_read": False,
        "correctness_labels_read_for_safety": True,
        "board_contacted": False,
        "input_hashes": input_hashes,
    }
    output.mkdir(parents=True)
    write_json(output / "contract.json", contract)
    (output / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in policy_rows.values()),
        encoding="utf-8",
    )
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    write_json(
        output / "summary.json",
        {
            "status": contract["status"],
            "policy_count": len(policy_rows),
            "replacement_count": len(replacements),
            "unique_route_program_count": len(aliases),
            "fullgraph_performance_labels_read": False,
            "board_contacted": False,
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
    parser.add_argument("--selection-contract-dir", required=True)
    parser.add_argument("--failed-board-dir", required=True)
    parser.add_argument("--diagnostic-dir", required=True)
    parser.add_argument("--rejected-candidate-id", required=True)
    parser.add_argument("--fallback-candidate-id", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
