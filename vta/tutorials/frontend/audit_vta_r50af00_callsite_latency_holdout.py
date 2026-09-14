#!/usr/bin/env python3
"""Audit the R50AF00 call-site latency holdout without rewriting raw artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_vta_resnet50_callsite_composed_graph_board import verify_build_artifacts
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(args):
    contract_dir = Path(args.holdout_contract)
    fused_dir = Path(args.fused_contract)
    build_dir = Path(args.subset_build)
    board_dir = Path(args.board_run)
    counts = {
        "holdout": len(verify_build_artifacts(contract_dir)),
        "fused": len(verify_build_artifacts(fused_dir)),
        "build": len(verify_build_artifacts(build_dir)),
        "board": len(verify_build_artifacts(board_dir)),
    }
    contract = read_json(contract_dir / "contract.json")
    fused = read_json(fused_dir / "fused_tir_occurrence_delta.json")
    manifest = read_json(build_dir / "manifest.json")
    board = read_json(board_dir / "summary.json")
    if contract.get("status") != "frozen_before_new_full_graph_build_or_callsite_latency":
        raise RuntimeError("holdout was not frozen before target latency")
    if any(
        contract[key]
        for key in (
            "operator_latency_labels_used",
            "full_graph_latency_labels_used",
            "partial_mask_latency_labels_used",
        )
    ):
        raise RuntimeError("holdout selection reports latency leakage")
    if fused.get("status") != "compiler_fused_tir_prediction_frozen_before_fpga":
        raise RuntimeError("fused-TIR prediction was not prospective")
    if fused.get("full_graph_latency_observed") or fused.get("partial_mask_latency_observed"):
        raise RuntimeError("fused-TIR contract reports target latency exposure")
    if manifest.get("status") != "eight_route_subsets_frozen_before_callsite_board_labels":
        raise RuntimeError("eight exact subsets were not frozen")
    if board.get("status") != "singleton_and_whole_group_admission_complete":
        raise RuntimeError("board comparison is incomplete")
    if not board["all_outputs_equal"] or not board["all_outputs_nonzero"]:
        raise RuntimeError("board output gate failed")
    if board["singleton_selected_mask"] != [1, 1, 1]:
        raise RuntimeError("singleton policy outcome changed")
    if board["grouped_selected_mask"] != [1, 1, 1]:
        raise RuntimeError("group policy outcome changed")
    group = board["group_result"]
    if not group["accepted"] or not all(group["checks"].values()):
        raise RuntimeError("whole-group gate failed")
    if group["expected_delta"] != fused["full_graph_fused_tir_delta"]:
        raise RuntimeError("board and prospective fused-TIR contracts differ")
    if group["correctness_profile_delta"] != group["expected_delta"]:
        raise RuntimeError("correctness profile does not match prospective prediction")
    if group["timed_profile_delta"] != group["expected_delta"]:
        raise RuntimeError("timed profile does not match prospective prediction")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    result = {
        "schema": "c3_vta_r50af00_callsite_latency_holdout_audit_v1",
        "status": "callsite_and_full_graph_latency_holdout_pass",
        "scope": {
            "operator_correctness_previously_known": True,
            "operator_latency_used_for_selection": False,
            "full_graph_latency_unseen_before_contract": True,
            "partial_mask_latency_unseen_before_contract": True,
            "new_model_or_workload": False,
            "new_tile": True,
        },
        "selected_family_id": contract["selected_family_id"],
        "candidate_ids": [row["candidate_id"] for row in contract["selected_pair"]],
        "knobs": contract["selected_pair"][0]["knobs"],
        "graph_nodes": manifest["proposal_order"],
        "prospective_full_graph_dma_delta": fused["full_graph_fused_tir_delta"],
        "singleton_selected_mask": board["singleton_selected_mask"],
        "grouped_selected_mask": board["grouped_selected_mask"],
        "singleton_model_execution_calls": board["singleton_model_execution_calls"],
        "group_model_execution_calls": board["group_model_execution_calls"],
        "group_execution_reduction_percent": (
            1.0
            - board["group_model_execution_calls"]
            / board["singleton_model_execution_calls"]
        )
        * 100.0,
        "group_median_latency_ms": group["median_latency_ms"],
        "group_paired_delta_ms": group["paired_delta_ms"],
        "group_speedup_percent": group["speedup_percent"],
        "group_paired_wins": group["paired_wins"],
        "group_paired_rounds": group["paired_rounds"],
        "singleton_results": [
            {
                "graph_node": manifest["proposal_order"][index],
                "paired_delta_ms": row["paired_delta_ms"],
                "paired_wins": row["paired_wins"],
                "accepted": row["accepted"],
            }
            for index, row in enumerate(board["singleton_results"])
        ],
        "source_hashes": {
            "holdout_contract": sha256(contract_dir / "contract.json"),
            "fused_contract": sha256(fused_dir / "fused_tir_occurrence_delta.json"),
            "subset_manifest": sha256(build_dir / "manifest.json"),
            "board_contract": sha256(board_dir / "contract.json"),
            "board_summary": sha256(board_dir / "summary.json"),
        },
        "verified_artifact_counts": counts,
        "auditor_source_sha256": sha256(Path(__file__).resolve()),
        "raw_summary_boundary_correction": (
            "The reused board executor emitted its older hard-coded 'second development "
            "domain/full-route exposed' claim_boundary.  Raw measurements are unchanged; "
            "the pre-board P7R281/P7R284/P7R285 hashes establish the narrower and accurate "
            "call-site/full-graph latency-holdout scope"
        ),
        "claim_boundary": (
            "One additional tile on the same R50A workload, model and boot.  This confirms "
            "call-site/full-graph latency transfer under a prospective contract, not a new "
            "operator workload, cross-boot result, ImageNet accuracy, native AutoTVM "
            "call-site identity, or physical AXI traffic"
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
    parser.add_argument("--holdout-contract", required=True)
    parser.add_argument("--fused-contract", required=True)
    parser.add_argument("--subset-build", required=True)
    parser.add_argument("--board-run", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
