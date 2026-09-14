#!/usr/bin/env python3
"""Replay a forced failed YOLO bundle and its recursive two-leaf split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(directory):
    directory = Path(directory)
    recorded = read_json(directory / "artifact_hashes.json")["artifacts"]
    for relative, expected in recorded.items():
        if sha256(directory / relative) != expected:
            raise RuntimeError("source artifact hash mismatch: " + str(directory / relative))
    return len(recorded)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def edge(rows, left, right):
    rounds = sorted({row["round"] for row in rows})
    deltas = []
    for round_index in rounds:
        values = {
            row["deployment_variant"]: row["latency_ms"]
            for row in rows
            if row["round"] == round_index
        }
        deltas.append(values[right] - values[left])
    return {
        "left": left,
        "right": right,
        "paired_rounds": len(rounds),
        "right_wins": sum(value < 0 for value in deltas),
        "paired_delta_ms": deltas,
        "median_paired_delta_ms": statistics.median(deltas),
        "accepted": len(rounds) >= 7 and all(value < 0 for value in deltas),
    }


def run(args):
    factorial = Path(args.factorial)
    planner = Path(args.greedy_plan)
    factorial_count = verify(factorial)
    planner_count = verify(planner)
    summary = read_json(factorial / "summary.json")
    if summary.get("status") != "yolov3_tiny_two_route_factorial_board_complete":
        raise RuntimeError("factorial source incomplete")
    rows = read_jsonl(factorial / "timing.jsonl")
    if not summary["all_variant_outputs_equal"]:
        raise RuntimeError("factorial outputs differ")

    whole = edge(rows, "stock_all", "y00_input_y02_weight")
    y00 = edge(rows, "stock_all", "y00_input_only")
    y02_after_y00 = edge(rows, "y00_input_only", "y00_input_y02_weight")
    if whole["accepted"] or not y00["accepted"] or y02_after_y00["accepted"]:
        raise RuntimeError("forced negative-control path changed")
    greedy = read_json(planner / "selected_routes.json")
    expected_selected = [summary["factor_definition"]["y00_input"]["candidate_id"]]
    if greedy["selected_route_candidate_ids"] != expected_selected:
        raise RuntimeError("recursive replay and existing greedy selection disagree")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    result = {
        "schema": "c3_vta_hierarchical_bundle_negative_control_v1",
        "status": "forced_bundle_failure_and_recursive_split_replayed",
        "analyzer_source_sha256": sha256(Path(__file__).resolve()),
        "source": {
            "factorial_summary_sha256": sha256(factorial / "summary.json"),
            "factorial_timing_sha256": sha256(factorial / "timing.jsonl"),
            "factorial_artifact_hash_manifest_sha256": sha256(
                factorial / "artifact_hashes.json"
            ),
            "factorial_verified_artifact_count": factorial_count,
            "greedy_plan_sha256": sha256(planner / "selected_routes.json"),
            "greedy_artifact_hash_manifest_sha256": sha256(
                planner / "artifact_hashes.json"
            ),
            "greedy_verified_artifact_count": planner_count,
        },
        "path": [
            {"action": "test_forced_group_y00_y02", **whole},
            {"action": "split_and_test_y00", **y00},
            {"action": "test_y02_against_updated_y00_incumbent", **y02_after_y00},
        ],
        "selected_route_candidate_ids": expected_selected,
        "matches_existing_sequential_planner": True,
        "pair_tests": 3,
        "singleton_greedy_pair_tests": 2,
        "pair_test_overhead_vs_singleton_percent": 50.0,
        "claim_boundary": (
            "Retrospective replay of one existing four-way FPGA run.  Y00 and Y02 use "
            "different workloads and residency modes and would not belong to the normal "
            "same-mode call-site bundle rule; this is an adversarial fallback control, not "
            "a prospective search result.  It also demonstrates that recursive splitting "
            "can cost more than singleton greedy"
        ),
    }
    write_json(output / "summary.json", result)
    (output / "command.txt").write_text(
        " ".join(__import__("sys").argv) + "\n", encoding="utf-8"
    )
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factorial", required=True)
    parser.add_argument("--greedy-plan", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
