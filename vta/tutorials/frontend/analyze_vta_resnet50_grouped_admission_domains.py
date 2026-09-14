#!/usr/bin/env python3
"""Consolidate two ResNet50 call-site admission development domains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def run(args):
    r50b_dir = Path(args.r50b_greedy)
    r50a_dir = Path(args.r50a_grouped)
    r50b_count = verify(r50b_dir)
    r50a_count = verify(r50a_dir)
    r50b = read_json(r50b_dir / "summary.json")
    r50a = read_json(r50a_dir / "summary.json")
    if r50b.get("status") != "incumbent_protected_four_node_greedy_and_final_validation_complete":
        raise RuntimeError("R50B greedy source incomplete")
    if r50a.get("status") != "singleton_and_whole_group_admission_complete":
        raise RuntimeError("R50A grouped source incomplete")

    r50b_singleton_calls = r50b["proposal_correctness_calls"] + r50b["proposal_timing_calls"]
    # The final validation used three executors.  For the all-original/all-resident
    # decision, retain only their two modes: 3 seeds * 2 + 9 rounds * 2.
    r50b_bundle_relevant_calls = 2 * 3 + 2 * r50b["final_result"]["paired_rounds"]
    r50b_bundle = {
        "singleton_node_count": r50b["proposal_count"],
        "singleton_model_execution_calls": r50b_singleton_calls,
        "singleton_selected_mask": r50b["selected_mask"],
        "singleton_accepted_nodes": r50b["accepted_nodes"],
        "bundle_selected_mask": [1, 1, 1, 1],
        "bundle_paired_rounds": r50b["final_result"]["paired_rounds"],
        "bundle_paired_wins": r50b["final_result"]["full_vs_original_wins"],
        "bundle_delta_ms": r50b["final_result"]["full_vs_original_delta_ms"],
        "bundle_relevant_model_execution_calls": r50b_bundle_relevant_calls,
        "actual_three_way_final_validation_calls": (
            r50b["final_correctness_calls"] + r50b["final_timing_calls"]
        ),
        "outcome": "singleton_false_negative_avoided_by_bundle",
    }
    group = r50a["group_result"]
    r50a_bundle = {
        "singleton_node_count": len(r50a["singleton_results"]),
        "singleton_model_execution_calls": r50a["singleton_model_execution_calls"],
        "singleton_selected_mask": r50a["singleton_selected_mask"],
        "bundle_selected_mask": r50a["grouped_selected_mask"],
        "bundle_model_execution_calls": r50a["group_model_execution_calls"],
        "bundle_paired_rounds": group["paired_rounds"],
        "bundle_paired_wins": group["paired_wins"],
        "bundle_delta_ms": group["paired_delta_ms"],
        "bundle_execution_reduction_vs_singleton_percent": (
            1.0
            - r50a["group_model_execution_calls"]
            / r50a["singleton_model_execution_calls"]
        )
        * 100.0,
        "outcome": "same_selected_bundle_with_fewer_board_executions",
    }
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    summary = {
        "schema": "c3_vta_resnet50_grouped_admission_two_domain_analysis_v1",
        "status": "two_development_domains_support_group_first_admission",
        "analyzer_source_sha256": sha256(Path(__file__).resolve()),
        "sources": {
            "r50b_summary_sha256": sha256(r50b_dir / "summary.json"),
            "r50b_artifact_hash_manifest_sha256": sha256(
                r50b_dir / "artifact_hashes.json"
            ),
            "r50b_verified_artifact_count": r50b_count,
            "r50a_summary_sha256": sha256(r50a_dir / "summary.json"),
            "r50a_artifact_hash_manifest_sha256": sha256(
                r50a_dir / "artifact_hashes.json"
            ),
            "r50a_verified_artifact_count": r50a_count,
        },
        "r50b_weight_residency": r50b_bundle,
        "r50a_input_residency": r50a_bundle,
        "algorithm": {
            "name": "DMA-additivity-guided hierarchical bundle admission",
            "group_rule": (
                "Group sequential graph occurrences whose compiler-derived per-occurrence "
                "DMA deltas are additive and whose residency mode/resource contract matches"
            ),
            "action": (
                "Test incumbent union whole group first; accept only through correctness, "
                "exact-DMA and paired-latency gates; recursively split a failed non-singleton "
                "group and re-evaluate each child against the current incumbent"
            ),
            "best_case_pair_tests": 1,
            "singleton_pair_tests": "number_of_occurrences",
            "worst_case_boundary": (
                "Recursive splitting can require more tests than singleton greedy; no failed "
                "bundle was exercised here, so the fallback is specified but unvalidated"
            ),
        },
        "claim_boundary": (
            "Both are label-exposed development domains on one boot.  They establish a "
            "mechanism and motivating counterexample, not an unseen generalization result, "
            "a globally optimal subset solver, or a physical AXI measurement"
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(
        " ".join(__import__("sys").argv) + "\n", encoding="utf-8"
    )
    write_json(
        output / "artifact_hashes.json",
        {"artifacts": {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "artifact_hashes.json"
        }},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r50b-greedy", required=True)
    parser.add_argument("--r50a-grouped", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
