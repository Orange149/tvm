#!/usr/bin/env python3
"""Freeze all eight graph-node subsets for the R50A input-stationary route."""

from __future__ import annotations

import argparse
from itertools import product
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from run_vta_resnet50_callsite_composed_graph_board import verify_build_artifacts
from run_vta_resnet50_relay_residency_dispatch_pair_board import (
    semantic_params_hash,
    sha256,
    write_json,
)
from run_vta_resnet50_fused_tir_pair_board_v2 import load_fused_contract


RESIDENCY_FILE = "c3_r50a_input_graphlib.so"
WRAPPER_FILE = "c3_r50a_callsite_wrapper.so"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mask_name(mask):
    return "mask" + "".join(str(int(bit)) for bit in mask)


def compile_wrapper(source, output, aliases):
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
    missing = [alias for alias in aliases if alias not in symbols]
    if missing:
        raise RuntimeError("R50A wrapper aliases missing: " + repr(missing))
    return command


def patch_mask(graph, routes, mask):
    result = json.loads(json.dumps(graph))
    for bit, route in zip(mask, routes):
        node = result["nodes"][route["graph_node"]]
        if node.get("attrs", {}).get("func_name") != route["func_name"]:
            raise RuntimeError("R50A graph-node identity changed")
        if bit:
            node["attrs"]["func_name"] = route["alias_symbol"]
    return result


def run(args):
    pair = Path(args.pair_build)
    pair_artifacts = verify_build_artifacts(pair)
    summary = read_json(pair / "summary.json")
    if summary.get("status") != "resnet50_generic_pair_cross_build_dispatch_verified":
        raise RuntimeError("R50A source pair is not verified")
    if summary.get("public_modes") != ["original", "input_stationary"]:
        raise RuntimeError("R50A source modes changed")
    original = pair / "original"
    residency = pair / "input_stationary"
    graph = read_json(original / "graph.json")
    if graph != read_json(residency / "graph.json"):
        raise RuntimeError("R50A source graph JSON differs")
    original_semantic = semantic_params_hash(original / "params.bin")
    if original_semantic != semantic_params_hash(residency / "params.bin"):
        raise RuntimeError("R50A A/B parameter semantics differ")

    fused_dir = Path(args.fused_analysis)
    fused, expected_full, fused_artifact_count = load_fused_contract(fused_dir)
    routes = []
    for occurrence in fused["graph_occurrences"]:
        node = occurrence["graph_node"]
        symbol = occurrence["func_name"]
        routes.append(
            {
                "graph_node": node,
                "func_name": symbol,
                "alias_symbol": symbol + "_c3_node" + str(node),
                "expected_delta": occurrence["delta"],
            }
        )
    if [row["graph_node"] for row in routes] != [67, 80, 93]:
        raise RuntimeError("frozen R50A graph nodes changed")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    variants = []
    for mask in product((0, 1), repeat=len(routes)):
        patched = patch_mask(graph, routes, mask)
        name = mask_name(mask) + ".json"
        write_json(output / name, patched)
        variants.append(
            {
                "mask": list(mask),
                "graph_file": name,
                "graph_sha256": sha256(output / name),
                "resident_graph_nodes": [
                    route["graph_node"] for bit, route in zip(mask, routes) if bit
                ],
            }
        )
    shutil.copyfile(original / "params.bin", output / "params.bin")
    shutil.copyfile(original / "graphlib.so", output / "incumbent_graphlib.so")
    shutil.copyfile(residency / "graphlib.so", output / RESIDENCY_FILE)
    wrapper_source = Path(args.wrapper_source).resolve()
    wrapper_command = compile_wrapper(
        wrapper_source,
        output / WRAPPER_FILE,
        [row["alias_symbol"] for row in routes],
    )
    if expected_full != {
        key: sum(row["expected_delta"][key] for row in routes)
        for key in expected_full
    }:
        raise RuntimeError("R50A route deltas do not sum to frozen full-graph contract")
    manifest = {
        "schema": "c3_vta_resnet50_r50a_callsite_subsets_v1",
        "status": "eight_route_subsets_frozen_before_callsite_board_labels",
        "source_pair_summary_sha256": sha256(pair / "summary.json"),
        "source_pair_artifact_hash_manifest_sha256": sha256(pair / "artifact_hashes.json"),
        "verified_source_pair_artifact_count": len(pair_artifacts),
        "fused_analysis_sha256": sha256(fused_dir / "fused_tir_occurrence_delta.json"),
        "fused_analysis_artifact_hash_manifest_sha256": sha256(
            fused_dir / "artifact_hashes.json"
        ),
        "verified_fused_analysis_artifact_count": fused_artifact_count,
        "builder_source_sha256": sha256(Path(__file__).resolve()),
        "wrapper_source_sha256": sha256(wrapper_source),
        "wrapper_compile_command": wrapper_command,
        "routes": routes,
        "variants": variants,
        "parameter_semantic_sha256": original_semantic[0],
        "parameter_count": len(original_semantic[1]),
        "resident_graphlib": RESIDENCY_FILE,
        "wrapper_file": WRAPPER_FILE,
        "proposal_order": [67, 80, 93],
        "expected_full_graph_delta": expected_full,
        "gate": {
            "correctness_seeds": [0, 20250901, 20260910],
            "paired_rounds": 7,
            "required_candidate_wins": 7,
            "required_median_delta": "strictly_negative",
        },
        "board_contacted": False,
        "label_exposure": (
            "P7R276 full-route latency is exposed, but no R50A call-site or partial-mask "
            "latency label was read before this contract"
        ),
        "planned_comparison": (
            "strict singleton greedy uses three proposal pairs; grouped admission first tests "
            "000->111 and recursively splits only if the full group fails"
        ),
    }
    write_json(output / "manifest.json", manifest)
    (output / "command.txt").write_text(
        " ".join(__import__("sys").argv) + "\n", encoding="utf-8"
    )
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            }
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-build", required=True)
    parser.add_argument("--fused-analysis", required=True)
    parser.add_argument("--wrapper-source", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
