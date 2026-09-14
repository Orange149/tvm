#!/usr/bin/env python3
"""Freeze singleton and leave-one-out ResNet50 call-site route graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from build_vta_resnet50_callsite_dose_graphs import read_json, verify_artifacts, write_json
from run_vta_resnet50_relay_residency_dispatch_pair_board import semantic_params_hash


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_mask(graph, source_symbol, alias_symbol, graph_nodes, mask):
    if len(mask) != len(graph_nodes) or any(bit not in (0, 1) for bit in mask):
        raise ValueError("mask must contain one binary value per exact graph node")
    matches = [
        index
        for index, node in enumerate(graph["nodes"])
        if node.get("op") == "tvm_op"
        and node.get("attrs", {}).get("func_name") == source_symbol
    ]
    if matches != graph_nodes:
        raise RuntimeError("exact call-site nodes changed: " + repr(matches))
    result = json.loads(json.dumps(graph))
    for enabled, node_index in zip(mask, graph_nodes):
        if enabled:
            result["nodes"][node_index]["attrs"]["func_name"] = alias_symbol
    return result


def mask_name(mask):
    return "mask_{}".format("".join(str(bit) for bit in mask))


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

    zero = (0, 0, 0, 0)
    full = (1, 1, 1, 1)
    masks = {zero, full}
    pairs = []
    for ordinal, node in enumerate(nodes):
        singleton = tuple(int(index == ordinal) for index in range(len(nodes)))
        leave_one_out = tuple(int(index != ordinal) for index in range(len(nodes)))
        masks.update((singleton, leave_one_out))
        pairs.extend((
            {
                "pair_id": "node{}_isolated_add".format(node),
                "graph_node": node,
                "context": "other_three_original",
                "baseline_mask": list(zero),
                "candidate_mask": list(singleton),
            },
            {
                "pair_id": "node{}_full_context_add".format(node),
                "graph_node": node,
                "context": "other_three_resident",
                "baseline_mask": list(leave_one_out),
                "candidate_mask": list(full),
            },
        ))

    variants = []
    for mask in sorted(masks):
        patched = patch_mask(graph, source_symbol, alias_symbol, nodes, mask)
        name = mask_name(mask) + ".json"
        write_json(output / name, patched)
        changed = [
            index
            for index, (left, right) in enumerate(zip(graph["nodes"], patched["nodes"]))
            if left != right
        ]
        expected_changed = [node for bit, node in zip(mask, nodes) if bit]
        if changed != expected_changed:
            raise RuntimeError("masked graph patch escaped exact nodes")
        variants.append({
            "mask": list(mask),
            "graph_file": name,
            "graph_sha256": sha256(output / name),
            "resident_graph_nodes": expected_changed,
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
        "schema": "c3_vta_resnet50_callsite_marginal_graphs_v1",
        "status": "eight_marginal_pairs_frozen_before_board",
        "source_build_manifest_sha256": sha256(source / "manifest.json"),
        "source_artifact_hash_manifest_sha256": sha256(source / "artifact_hashes.json"),
        "builder_source_sha256": sha256(Path(__file__).resolve()),
        "source_symbol": source_symbol,
        "alias_symbol": alias_symbol,
        "matching_graph_nodes": nodes,
        "parameter_semantic_sha256": parameter_hash,
        "parameter_count": len(parameter_inventory),
        "variants": variants,
        "pairs": pairs,
        "expected_per_edge_delta": source_manifest["expected_per_resident_node_delta"],
        "board_contacted": False,
        "claim_boundary": (
            "Four singleton and four full-context marginal edges only; correctness, "
            "DMA invariance, and latency interactions remain unmeasured"
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
