#!/usr/bin/env python3
"""Freeze a deterministic R50A pair before any new full-graph call-site latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_vta_resnet50_callsite_composed_graph_board import verify_build_artifacts
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def choose_pair(candidates, correctness, excluded_ids):
    passed = {
        row["candidate_id"]
        for row in correctness
        if row.get("status") == "passed"
        and all(seed.get("correct") for seed in row.get("seeds", []))
    }
    by_family = {}
    family_order = []
    for row in candidates:
        family = row["family_id"]
        if family not in by_family:
            by_family[family] = {}
            family_order.append(family)
        by_family[family][row["public_mode"]] = row
    for family in family_order:
        modes = by_family[family]
        if "original" not in modes or "input_stationary" not in modes:
            continue
        pair = [modes["original"], modes["input_stationary"]]
        ids = {row["candidate_id"] for row in pair}
        if ids & excluded_ids or not ids <= passed:
            continue
        return pair
    raise RuntimeError("no deterministic correctness-qualified unexposed pair")


def run(args):
    registry = Path(args.registry)
    registry_artifacts = verify_build_artifacts(registry)
    board = Path(args.correctness_board)
    board_artifacts = verify_build_artifacts(board)
    candidates_path = registry / "candidates_v2.jsonl"
    correctness_path = board / "correctness.jsonl"
    candidates = read_jsonl(candidates_path)
    correctness = read_jsonl(correctness_path)
    excluded_ids = set(args.exclude_candidate_id)
    selected = choose_pair(candidates, correctness, excluded_ids)
    if selected[0]["identity"]["complete_config_entity"] != selected[1]["identity"][
        "complete_config_entity"
    ]:
        raise RuntimeError("selected pair is not same-tile")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    contract = {
        "schema": "c3_vta_r50a_callsite_latency_holdout_contract_v1",
        "status": "frozen_before_new_full_graph_build_or_callsite_latency",
        "selection_rule": (
            "first family in the pre-existing frozen registry order with three-seed FPGA-"
            "correct original/input modes, excluding all caller-provided candidate IDs"
        ),
        "selected_family_id": selected[0]["family_id"],
        "selected_pair": [
            {
                "candidate_id": row["candidate_id"],
                "public_mode": row["public_mode"],
                "implementation_mode": row["implementation_mode"],
                "knobs": row["knobs"],
                "complete_config_entity": row["identity"]["complete_config_entity"],
            }
            for row in selected
        ],
        "excluded_candidate_ids": sorted(excluded_ids),
        "registry_candidates_sha256": sha256(candidates_path),
        "registry_artifact_hash_manifest_sha256": sha256(registry / "artifact_hashes.json"),
        "verified_registry_artifact_count": len(registry_artifacts),
        "correctness_source_sha256": sha256(correctness_path),
        "correctness_artifact_hash_manifest_sha256": sha256(board / "artifact_hashes.json"),
        "verified_correctness_artifact_count": len(board_artifacts),
        "preexisting_correctness_labels_used": True,
        "operator_latency_labels_used": False,
        "full_graph_latency_labels_used": False,
        "partial_mask_latency_labels_used": False,
        "planned_gate": {
            "correctness_seeds": [0, 20250901, 20260910],
            "paired_rounds": 7,
            "required_candidate_wins": 7,
            "required_median_delta": "strictly_negative",
            "fused_tir_profile": "all ten LOAD/STORE fields exact",
        },
        "board_contacted": False,
        "claim_boundary": (
            "Call-site/full-graph latency holdout only.  The source operator correctness "
            "pool and its historical performance results already exist, but performance "
            "labels are not inputs to this deterministic selection"
        ),
    }
    write_json(output / "contract.json", contract)
    (output / "command.txt").write_text(
        " ".join(__import__("sys").argv) + "\n", encoding="utf-8"
    )
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps(contract, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--correctness-board", required=True)
    parser.add_argument("--exclude-candidate-id", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
