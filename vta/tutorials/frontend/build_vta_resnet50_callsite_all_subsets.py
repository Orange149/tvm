#!/usr/bin/env python3
"""Freeze all 16 exact-node ResNet50 residency route subsets."""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil

from build_vta_resnet50_callsite_dose_graphs import read_json, verify_artifacts, write_json
from build_vta_resnet50_callsite_marginal_graphs import mask_name, patch_mask
from run_vta_resnet50_relay_residency_dispatch_pair_board import semantic_params_hash


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    source = Path(args.source_build)
    verify_artifacts(source)
    source_manifest = read_json(source / "manifest.json")
    if source_manifest.get("status") != "five_graph_variants_frozen_before_board":
        raise RuntimeError("source is not the frozen P7R266 dose build")
    nodes = source_manifest["matching_graph_nodes"]
    if nodes != [57, 72, 85, 98]:
        raise RuntimeError("source graph-node identity changed")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    graph = read_json(source / "prefix0_graph.json")
    source_symbol = source_manifest["source_symbol"]
    alias_symbol = source_manifest["alias_symbol"]

    variants = []
    for mask in product((0, 1), repeat=len(nodes)):
        patched = patch_mask(graph, source_symbol, alias_symbol, nodes, mask)
        name = mask_name(mask) + ".json"
        write_json(output / name, patched)
        variants.append({
            "mask": list(mask),
            "graph_file": name,
            "graph_sha256": sha256(output / name),
            "resident_graph_nodes": [node for bit, node in zip(mask, nodes) if bit],
        })

    for name in (
        "params.bin",
        "incumbent_graphlib.so",
        "c3_r50_barrier_graphlib.so",
        "c3_r50_callsite_wrapper.so",
    ):
        shutil.copyfile(source / name, output / name)
    parameter_hash, parameter_inventory = semantic_params_hash(output / "params.bin")
    manifest = {
        "schema": "c3_vta_resnet50_callsite_all_subsets_v1",
        "status": "sixteen_route_subsets_frozen_before_board",
        "source_build_manifest_sha256": sha256(source / "manifest.json"),
        "source_artifact_hash_manifest_sha256": sha256(source / "artifact_hashes.json"),
        "builder_source_sha256": sha256(Path(__file__).resolve()),
        "source_symbol": source_symbol,
        "alias_symbol": alias_symbol,
        "matching_graph_nodes": nodes,
        "parameter_semantic_sha256": parameter_hash,
        "parameter_count": len(parameter_inventory),
        "variants": variants,
        "proposal_order": nodes,
        "proposal_order_rule": "graph execution order; all static per-node DMA deltas tie",
        "acceptance_gate": {
            "correctness_seeds": [0, 20250901, 20260910],
            "minimum_paired_rounds": 7,
            "required_candidate_wins": 7,
            "required_median_delta": "strictly_negative",
            "baseline": "current accepted mask",
            "candidate": "current accepted mask plus exactly one graph node",
        },
        "expected_per_edge_delta": source_manifest["expected_per_resident_node_delta"],
        "board_contacted": False,
        "label_exposure": (
            "P7R269 extreme-context edges are already exposed; this is a deployment-planner "
            "execution test, not a new search holdout"
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
    parser.add_argument("--source-build", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
