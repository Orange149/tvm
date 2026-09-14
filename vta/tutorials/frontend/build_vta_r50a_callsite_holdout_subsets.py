#!/usr/bin/env python3
"""Freeze eight R50A call-site subsets from a prospective fused-TIR contract."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import shutil

from build_vta_resnet50_r50a_callsite_subsets import (
    RESIDENCY_FILE,
    WRAPPER_FILE,
    compile_wrapper,
    mask_name,
    patch_mask,
)
from run_vta_resnet50_callsite_composed_graph_board import verify_build_artifacts
from run_vta_resnet50_relay_residency_dispatch_pair_board import (
    semantic_params_hash,
    sha256,
    write_json,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(args):
    pair = Path(args.pair_build)
    pair_artifacts = verify_build_artifacts(pair)
    pair_summary = read_json(pair / "summary.json")
    if pair_summary.get("status") != "resnet50_generic_pair_cross_build_dispatch_verified":
        raise RuntimeError("R50A source pair is not verified")
    original = pair / "original"
    residency = pair / "input_stationary"
    graph = read_json(original / "graph.json")
    if graph != read_json(residency / "graph.json"):
        raise RuntimeError("R50A A/B graph JSON differs")
    original_semantic = semantic_params_hash(original / "params.bin")
    if original_semantic != semantic_params_hash(residency / "params.bin"):
        raise RuntimeError("R50A A/B parameter semantics differ")

    fused_dir = Path(args.fused_contract)
    fused_artifacts = verify_build_artifacts(fused_dir)
    fused = read_json(fused_dir / "fused_tir_occurrence_delta.json")
    if fused.get("status") != "compiler_fused_tir_prediction_frozen_before_fpga":
        raise RuntimeError("prospective fused-TIR contract is not frozen")
    if fused.get("full_graph_latency_observed") or fused.get("partial_mask_latency_observed"):
        raise RuntimeError("prospective contract already exposes target latency")
    if [row["candidate_id"] for row in pair_summary["builds"]] != fused["candidate_ids"]:
        raise RuntimeError("pair and prospective fused contract identities differ")
    routes = []
    for occurrence in fused["graph_occurrences"]:
        node = occurrence["graph_node"]
        symbol = occurrence["func_name"]
        routes.append({
            "graph_node": node,
            "func_name": symbol,
            "alias_symbol": symbol + "_c3_node" + str(node),
            "expected_delta": occurrence["delta"],
        })
    if [row["graph_node"] for row in routes] != [67, 80, 93]:
        raise RuntimeError("R50A graph-node identity changed")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    variants = []
    for mask in product((0, 1), repeat=len(routes)):
        patched = patch_mask(graph, routes, mask)
        name = mask_name(mask) + ".json"
        write_json(output / name, patched)
        variants.append({
            "mask": list(mask),
            "graph_file": name,
            "graph_sha256": sha256(output / name),
            "resident_graph_nodes": [
                row["graph_node"] for bit, row in zip(mask, routes) if bit
            ],
        })
    shutil.copyfile(original / "params.bin", output / "params.bin")
    shutil.copyfile(original / "graphlib.so", output / "incumbent_graphlib.so")
    shutil.copyfile(residency / "graphlib.so", output / RESIDENCY_FILE)
    wrapper_source = Path(args.wrapper_source).resolve()
    wrapper_command = compile_wrapper(
        wrapper_source,
        output / WRAPPER_FILE,
        [row["alias_symbol"] for row in routes],
    )
    manifest = {
        "schema": "c3_vta_r50a_callsite_latency_holdout_subsets_v1",
        "status": "eight_route_subsets_frozen_before_callsite_board_labels",
        "source_pair_summary_sha256": sha256(pair / "summary.json"),
        "source_pair_artifact_hash_manifest_sha256": sha256(pair / "artifact_hashes.json"),
        "verified_source_pair_artifact_count": len(pair_artifacts),
        "fused_contract_sha256": sha256(fused_dir / "fused_tir_occurrence_delta.json"),
        "fused_contract_artifact_hash_manifest_sha256": sha256(
            fused_dir / "artifact_hashes.json"
        ),
        "verified_fused_contract_artifact_count": len(fused_artifacts),
        "builder_source_sha256": sha256(Path(__file__).resolve()),
        "wrapper_source_sha256": sha256(wrapper_source),
        "wrapper_compile_command": wrapper_command,
        "candidate_ids": fused["candidate_ids"],
        "routes": routes,
        "variants": variants,
        "parameter_semantic_sha256": original_semantic[0],
        "parameter_count": len(original_semantic[1]),
        "resident_graphlib": RESIDENCY_FILE,
        "wrapper_file": WRAPPER_FILE,
        "proposal_order": [67, 80, 93],
        "expected_full_graph_delta": fused["full_graph_fused_tir_delta"],
        "gate": {
            "correctness_seeds": [0, 20250901, 20260910],
            "paired_rounds": 7,
            "required_candidate_wins": 7,
            "required_median_delta": "strictly_negative",
        },
        "board_contacted": False,
        "label_exposure": (
            "Operator correctness is historical; no target full-graph or partial-mask "
            "latency label was used before freezing these eight exact graph variants"
        ),
        "planned_comparison": (
            "preregistered singleton greedy versus whole-group-first admission under the "
            "same three-seed, exact-DMA, seven-of-seven gate"
        ),
    }
    write_json(output / "manifest.json", manifest)
    (output / "command.txt").write_text(
        " ".join(__import__("sys").argv) + "\n", encoding="utf-8"
    )
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-build", required=True)
    parser.add_argument("--fused-contract", required=True)
    parser.add_argument("--wrapper-source", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
