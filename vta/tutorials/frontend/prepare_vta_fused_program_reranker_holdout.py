#!/usr/bin/env python3
"""Freeze a two-program fused-graph service-proxy choice before FPGA timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_vta_fused_program_service_proxy_norefit import (
    service_score,
    total_dma,
    verify_artifacts_compatible,
)
from analyze_vta_resnet50_fused_tir_occurrence_delta import RUNTIME_KEYS
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def aggregate_residency(contract):
    total = {key: 0 for key in RUNTIME_KEYS}
    for occurrence in contract["graph_occurrences"]:
        for key in RUNTIME_KEYS:
            total[key] += int(occurrence["residency_dma"][key])
    return total


def choose(programs):
    return min(
        programs,
        key=lambda row: (row["service_score_byte_equivalent"], row["program_id"]),
    )


def run(args):
    fused_dirs = [Path(args.fused_contract_a), Path(args.fused_contract_b)]
    build_dirs = [Path(args.pair_build_a), Path(args.pair_build_b)]
    holdout_dirs = [Path(args.tile_holdout_a), Path(args.tile_holdout_b)]
    service_dir = Path(args.service_development)
    verified = {
        "fused_contracts": [len(verify_artifacts_compatible(path)) for path in fused_dirs],
        "pair_builds": [len(verify_artifacts_compatible(path)) for path in build_dirs],
        "tile_holdouts": [len(verify_artifacts_compatible(path)) for path in holdout_dirs],
        "service_development": len(verify_artifacts_compatible(service_dir)),
    }

    fused = [read_json(path / "fused_tir_occurrence_delta.json") for path in fused_dirs]
    builds = [read_json(path / "summary.json") for path in build_dirs]
    holdouts = [read_json(path / "contract.json") for path in holdout_dirs]
    service = read_json(service_dir / "analysis.json")
    for row in fused:
        if row.get("status") != "compiler_fused_tir_prediction_frozen_before_fpga":
            raise RuntimeError("fused occurrence contract is incomplete")
        if row.get("full_graph_latency_observed") or row.get("board_contacted"):
            raise RuntimeError("fused occurrence contract has target label exposure")
    for row in builds:
        if row.get("status") != "resnet50_generic_pair_cross_build_dispatch_verified":
            raise RuntimeError("pair build is incomplete")
    for row in holdouts:
        if row.get("status") != "frozen_before_new_full_graph_build_or_callsite_latency":
            raise RuntimeError("tile holdout is incomplete")
        exposure = row.get("label_exposure") or {}
        if exposure.get("full_graph_latency_observed") or exposure.get("partial_mask_latency_observed"):
            raise RuntimeError("target full-graph label was exposed")
        if exposure.get("operator_latency_used_for_selection"):
            raise RuntimeError("operator latency was used for tile selection")
    if service.get("status") != "development_frozen_before_y01_latency_recovery_confirmation":
        raise RuntimeError("service weights are not the frozen development result")

    occurrence_signatures = [
        [(row["graph_node"], row["func_name"]) for row in contract["graph_occurrences"]]
        for contract in fused
    ]
    if occurrence_signatures[0] != occurrence_signatures[1]:
        raise RuntimeError("programs do not cover the same graph occurrences")

    request_cost = int(service["frozen_proxy"]["request_equivalent_bytes"])
    submission_cost = int(service["frozen_proxy"]["extra_submission_equivalent_bytes"])
    programs = []
    for index, (contract, build, holdout) in enumerate(zip(fused, builds, holdouts)):
        if contract["candidate_ids"] != [row["candidate_id"] for row in build["builds"]]:
            raise RuntimeError("fused contract and pair build candidate identities differ")
        if contract["candidate_ids"] != [row["candidate_id"] for row in holdout["selected_pair"]]:
            raise RuntimeError("fused contract and tile holdout candidate identities differ")
        traffic = aggregate_residency(contract)
        dma = total_dma(traffic)
        programs.append({
            "program_id": f"tile_{'a' if index == 0 else 'b'}:input_stationary",
            "candidate_id": contract["candidate_ids"][1],
            "original_candidate_id": contract["candidate_ids"][0],
            "knobs": holdout["selected_pair"][1]["knobs"],
            "traffic": traffic,
            "dma_bytes": dma["bytes"],
            "dma_calls": dma["calls"],
            "extra_residency_submissions": 0,
            "service_score_byte_equivalent": service_score(
                traffic, request_cost, 0, submission_cost
            ),
        })
    if programs[0]["candidate_id"] == programs[1]["candidate_id"]:
        raise RuntimeError("reranker programs must be distinct")

    selected = choose(programs)
    delta_b_minus_a = {
        key: programs[1]["traffic"][key] - programs[0]["traffic"][key]
        for key in RUNTIME_KEYS
    }
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    bound_inputs = {
        "fused_contracts": [{
            "payload_sha256": sha256(path / "fused_tir_occurrence_delta.json"),
            "artifact_manifest_sha256": sha256(path / "artifact_hashes.json"),
        } for path in fused_dirs],
        "pair_builds": [{
            "summary_sha256": sha256(path / "summary.json"),
            "artifact_manifest_sha256": sha256(path / "artifact_hashes.json"),
        } for path in build_dirs],
        "tile_holdouts": [{
            "contract_sha256": sha256(path / "contract.json"),
            "artifact_manifest_sha256": sha256(path / "artifact_hashes.json"),
        } for path in holdout_dirs],
        "service_development": {
            "analysis_sha256": sha256(service_dir / "analysis.json"),
            "artifact_manifest_sha256": sha256(service_dir / "artifact_hashes.json"),
        },
    }
    result = {
        "schema": "c3_vta_fused_program_reranker_holdout_contract_v1",
        "status": "fused_service_choice_frozen_before_fpga_latency",
        "programs": programs,
        "selection_order": [
            row["program_id"] for row in sorted(
                programs,
                key=lambda row: (row["service_score_byte_equivalent"], row["program_id"]),
            )
        ],
        "selected_program": selected["program_id"],
        "selected_candidate_id": selected["candidate_id"],
        "expected_profile_delta_b_minus_a": delta_b_minus_a,
        "exact_profile_fields": list(RUNTIME_KEYS),
        "frozen_formula": service["frozen_proxy"]["formula"],
        "request_equivalent_bytes": request_cost,
        "extra_submission_equivalent_bytes": submission_cost,
        "submission_term_scope": "zero for both input-stationary programs",
        "graph_occurrence_signature": occurrence_signatures[0],
        "target_label_exposure": {
            "full_graph_latency_observed": False,
            "partial_mask_latency_observed": False,
            "operator_latency_used_for_selection": False,
            "operator_correctness_used_for_eligibility": True,
        },
        "bound_inputs": bound_inputs,
        "verified_artifact_counts": verified,
        "preparer_source_sha256": sha256(Path(__file__).resolve()),
        "board_contacted": False,
        "claim_boundary": (
            "Prospective two-program full-graph reranking contract. It reuses fixed P7R166 "
            "byte-equivalent weights without refitting and uses compiler-derived logical VTA "
            "DMA only; it is not a physical AXI model or a new-workload operator holdout."
        ),
    }
    write_json(output / "contract.json", result)
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
    parser.add_argument("--fused-contract-a", required=True)
    parser.add_argument("--fused-contract-b", required=True)
    parser.add_argument("--pair-build-a", required=True)
    parser.add_argument("--pair-build-b", required=True)
    parser.add_argument("--tile-holdout-a", required=True)
    parser.add_argument("--tile-holdout-b", required=True)
    parser.add_argument("--service-development", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
