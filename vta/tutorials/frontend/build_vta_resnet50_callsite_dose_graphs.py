#!/usr/bin/env python3
"""Freeze k=0..4 exact-node ResNet50 residency graph variants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from run_vta_resnet50_relay_residency_dispatch_pair_board import semantic_params_hash


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_artifacts(directory):
    recorded = read_json(Path(directory) / "artifact_hashes.json")["artifacts"]
    bad = {}
    for relative, expected in recorded.items():
        path = Path(directory) / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            bad[relative] = {"expected": expected, "actual": actual}
    if bad:
        raise RuntimeError("source call-site build hash mismatch: " + repr(bad))
    return recorded


def patch_prefix(graph, source_symbol, alias_symbol, expected_nodes, count):
    matches = [
        index
        for index, node in enumerate(graph["nodes"])
        if node.get("op") == "tvm_op"
        and node.get("attrs", {}).get("func_name") == source_symbol
    ]
    if matches != expected_nodes:
        raise RuntimeError("exact call-site nodes changed: " + repr(matches))
    if not 0 <= count <= len(matches):
        raise ValueError("prefix count outside exact call-site domain")
    result = json.loads(json.dumps(graph))
    for node_index in matches[:count]:
        result["nodes"][node_index]["attrs"]["func_name"] = alias_symbol
    return result


def run(args):
    source = Path(args.source_build)
    verify_artifacts(source)
    source_manifest = read_json(source / "manifest.json")
    if source_manifest.get("status") != "cross_built_unmeasured":
        raise RuntimeError("source is not the frozen P7R264 build")
    expected_nodes = source_manifest["matching_graph_nodes"]
    if expected_nodes != [57, 72, 85, 98]:
        raise RuntimeError("source graph-node identity changed")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    graph = read_json(source / "incumbent_graph.json")
    source_symbol = source_manifest["source_symbol"]
    alias_symbol = source_manifest["alias_symbol"]
    variants = []
    for count in range(len(expected_nodes) + 1):
        patched = patch_prefix(graph, source_symbol, alias_symbol, expected_nodes, count)
        name = "prefix{}_graph.json".format(count)
        write_json(output / name, patched)
        changed = [
            index
            for index, (left, right) in enumerate(zip(graph["nodes"], patched["nodes"]))
            if left != right
        ]
        if changed != expected_nodes[:count]:
            raise RuntimeError("graph prefix patch escaped selected nodes")
        variants.append({
            "resident_node_count": count,
            "resident_graph_nodes": expected_nodes[:count],
            "graph_file": name,
            "graph_sha256": sha256(output / name),
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
        "schema": "c3_vta_resnet50_callsite_dose_graphs_v1",
        "status": "five_graph_variants_frozen_before_board",
        "source_build_manifest_sha256": sha256(source / "manifest.json"),
        "source_artifact_hash_manifest_sha256": sha256(source / "artifact_hashes.json"),
        "builder_source_sha256": sha256(Path(__file__).resolve()),
        "source_symbol": source_symbol,
        "alias_symbol": alias_symbol,
        "matching_graph_nodes": expected_nodes,
        "parameter_semantic_sha256": parameter_hash,
        "parameter_count": len(parameter_inventory),
        "variants": variants,
        "expected_per_resident_node_delta": {
            "load_buffer_2d_wgt_bytes": -393216,
            "load_buffer_2d_wgt_calls": -432,
            "synchronize_calls": 16,
            "driver_run_calls": 16,
        },
        "board_contacted": False,
        "claim_boundary": (
            "Prefix graph-node dose domain only; board correctness, exact DMA linearity, "
            "and latency response remain unmeasured"
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
