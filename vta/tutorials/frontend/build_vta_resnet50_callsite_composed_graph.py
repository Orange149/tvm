#!/usr/bin/env python3
"""Compose one ResNet50 graph node from incumbent and barrier VTA modules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from run_vta_resnet50_relay_residency_dispatch_pair_board import semantic_params_hash


SOURCE_SYMBOL = "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_3"
ALIAS_SYMBOL = SOURCE_SYMBOL + "_c3_callsite0"
BARRIER_FILE = "c3_r50_barrier_graphlib.so"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compile_wrapper(source, output):
    compiler = shlex.split(os.environ.get("CXX", "aarch64-xilinx-linux-g++"))
    root = Path(__file__).resolve().parents[3]
    command = [
        *compiler,
        "-shared",
        "-fPIC",
        "-std=c++17",
        "-O2",
        "-I" + str(root / "include"),
        "-I" + str(root / "3rdparty/dlpack/include"),
        str(source),
        "-o",
        str(output),
        "-ldl",
        "-pthread",
    ]
    subprocess.run(command, check=True)
    symbols = subprocess.run(
        ["readelf", "-Ws", str(output)], check=True, text=True, capture_output=True
    ).stdout
    if ALIAS_SYMBOL not in symbols:
        raise RuntimeError("call-site wrapper does not export the graph alias")
    return command


def patch_graph_callsite(graph, source_symbol, alias_symbol, expected_instances, instance):
    matches = [
        index
        for index, node in enumerate(graph["nodes"])
        if node.get("op") == "tvm_op"
        and node.get("attrs", {}).get("func_name") == source_symbol
    ]
    if len(matches) != expected_instances:
        raise RuntimeError("target graph function instance count changed")
    if not 0 <= instance < len(matches):
        raise ValueError("instance is outside the exact graph function occurrences")
    selected_node = matches[instance]
    composed = json.loads(json.dumps(graph))
    composed["nodes"][selected_node]["attrs"]["func_name"] = alias_symbol
    return composed, matches, selected_node


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    pair = Path(args.pair_build)
    original = pair / "original"
    barrier = pair / "weight_resident_barrier"
    original_graph = json.loads((original / "graph.json").read_text(encoding="utf-8"))
    barrier_graph = json.loads((barrier / "graph.json").read_text(encoding="utf-8"))
    if original_graph != barrier_graph:
        raise RuntimeError("source pair graph JSON differs")
    original_semantic = semantic_params_hash(original / "params.bin")
    barrier_semantic = semantic_params_hash(barrier / "params.bin")
    if original_semantic != barrier_semantic:
        raise RuntimeError("source pair parameter semantics differ")

    composed_graph, matches, selected_node = patch_graph_callsite(
        original_graph,
        args.source_symbol,
        args.alias_symbol,
        args.expected_instances,
        args.instance,
    )

    write_json(output / "incumbent_graph.json", original_graph)
    write_json(output / "callsite0_graph.json", composed_graph)
    shutil.copyfile(original / "params.bin", output / "params.bin")
    shutil.copyfile(original / "graphlib.so", output / "incumbent_graphlib.so")
    shutil.copyfile(barrier / "graphlib.so", output / BARRIER_FILE)
    wrapper_source = Path(args.wrapper_source).resolve()
    wrapper_command = compile_wrapper(wrapper_source, output / "c3_r50_callsite_wrapper.so")

    manifest = {
        "schema": "c3_vta_resnet50_callsite_composed_graph_v1",
        "status": "cross_built_unmeasured",
        "source_pair_build": str(pair),
        "source_symbol": args.source_symbol,
        "alias_symbol": args.alias_symbol,
        "matching_graph_nodes": matches,
        "selected_instance": args.instance,
        "selected_graph_node": selected_node,
        "graph_diff": "exactly one tvm_op func_name",
        "semantic_parameter_sha256": original_semantic[0],
        "parameter_count": len(original_semantic[1]),
        "builder_source_sha256": sha256(Path(__file__).resolve()),
        "wrapper_source_sha256": sha256(wrapper_source),
        "wrapper_compile_command": wrapper_command,
        "board_contacted": False,
        "claim_boundary": (
            "Exact graph-node composition of two already qualified full-graph modules; "
            "board correctness and per-call DMA isolation remain unmeasured"
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
    parser.add_argument("--wrapper-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-symbol", default=SOURCE_SYMBOL)
    parser.add_argument("--alias-symbol", default=ALIAS_SYMBOL)
    parser.add_argument("--expected-instances", type=int, default=4)
    parser.add_argument("--instance", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
