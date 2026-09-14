#!/usr/bin/env python3
"""Separate grouped-admission development evidence from its first latency holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_vta_resnet50_callsite_composed_graph_board import verify_build_artifacts
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(args):
    development = Path(args.development_analysis)
    holdout = Path(args.holdout_audit)
    development_count = len(verify_build_artifacts(development))
    holdout_count = len(verify_build_artifacts(holdout))
    dev = read_json(development / "summary.json")
    unseen = read_json(holdout / "summary.json")
    if dev.get("status") != "two_development_domains_support_group_first_admission":
        raise RuntimeError("development analysis incomplete")
    if unseen.get("status") != "callsite_and_full_graph_latency_holdout_pass":
        raise RuntimeError("holdout audit incomplete")
    if not unseen["scope"]["full_graph_latency_unseen_before_contract"]:
        raise RuntimeError("holdout full-graph latency scope failed")
    if not unseen["scope"]["partial_mask_latency_unseen_before_contract"]:
        raise RuntimeError("holdout partial-mask latency scope failed")
    if unseen["singleton_selected_mask"] != unseen["grouped_selected_mask"]:
        raise RuntimeError("holdout policies selected different masks")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    result = {
        "schema": "c3_vta_grouped_admission_holdout_confirmation_v1",
        "status": "first_callsite_latency_holdout_confirms_group_first_cost_reduction",
        "method": "DMA-additivity-guided hierarchical bundle admission",
        "development_domains": {
            "R50B_weight_residency": {
                "finding": dev["r50b_weight_residency"]["outcome"],
                "singleton_selected_mask": dev["r50b_weight_residency"][
                    "singleton_selected_mask"
                ],
                "bundle_selected_mask": dev["r50b_weight_residency"][
                    "bundle_selected_mask"
                ],
                "bundle_delta_ms": dev["r50b_weight_residency"]["bundle_delta_ms"],
            },
            "R50A_F06_input_residency": {
                "finding": dev["r50a_input_residency"]["outcome"],
                "singleton_model_execution_calls": dev["r50a_input_residency"][
                    "singleton_model_execution_calls"
                ],
                "bundle_model_execution_calls": dev["r50a_input_residency"][
                    "bundle_model_execution_calls"
                ],
                "bundle_delta_ms": dev["r50a_input_residency"]["bundle_delta_ms"],
            },
        },
        "latency_holdout": {
            "name": "R50A_F00",
            "selection_scope": unseen["scope"],
            "knobs": unseen["knobs"],
            "graph_nodes": unseen["graph_nodes"],
            "prospective_dma_delta": unseen["prospective_full_graph_dma_delta"],
            "singleton_selected_mask": unseen["singleton_selected_mask"],
            "bundle_selected_mask": unseen["grouped_selected_mask"],
            "singleton_model_execution_calls": unseen["singleton_model_execution_calls"],
            "bundle_model_execution_calls": unseen["group_model_execution_calls"],
            "execution_reduction_percent": unseen["group_execution_reduction_percent"],
            "bundle_median_latency_ms": unseen["group_median_latency_ms"],
            "bundle_delta_ms": unseen["group_paired_delta_ms"],
            "bundle_speedup_percent": unseen["group_speedup_percent"],
            "bundle_paired_wins": unseen["group_paired_wins"],
            "bundle_paired_rounds": unseen["group_paired_rounds"],
            "singleton_results": unseen["singleton_results"],
        },
        "supported_claim": (
            "After the group-first rule and 7-of-7 gate were fixed on two development "
            "domains, a deterministically chosen new tile with unseen full-graph and "
            "partial-mask latency reached the same 111 decision as singleton greedy with "
            "20 instead of 60 full-model executions"
        ),
        "source_hashes": {
            "development_summary": sha256(development / "summary.json"),
            "development_artifact_hash_manifest": sha256(
                development / "artifact_hashes.json"
            ),
            "holdout_summary": sha256(holdout / "summary.json"),
            "holdout_artifact_hash_manifest": sha256(holdout / "artifact_hashes.json"),
        },
        "verified_artifact_counts": {
            "development": development_count,
            "holdout": holdout_count,
        },
        "analyzer_source_sha256": sha256(Path(__file__).resolve()),
        "claim_boundary": (
            "The confirmation changes tile but not workload, network or boot, and historical "
            "operator correctness/performance data exist even though they were excluded from "
            "selection.  It is a call-site/full-graph latency holdout, not an independent "
            "model/workload holdout, a natural failed-bundle test, or global-optimum proof"
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
    parser.add_argument("--development-analysis", required=True)
    parser.add_argument("--holdout-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
