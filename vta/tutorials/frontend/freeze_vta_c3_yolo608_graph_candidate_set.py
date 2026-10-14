#!/usr/bin/env python3
"""Freeze the real YOLO-608 composable graph set under the generic v3 schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from c3_graph_candidate_set import (
    build_graph_candidate_set,
    inherit_graph_context_certificate,
)
from run_vta_c3_holdout_board_boot_v2 import verify_flat


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_LOCAL = GROUPED / "20260924_c3_yolo608_composable_local_run02"
DEFAULT_BUILD = GROUPED / "20260924_c3_yolo608_composable_build_run01"
DEFAULT_CERTIFICATES = GROUPED / "20260924_c3_composability_certificates_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo608_graph_candidate_set_run01"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_route_scope(graph, workload):
    input_shape = list(workload[1][1])
    weight_shape = list(workload[2][1])
    shapes = graph["attrs"]["shape"][1]
    row_ptr = graph["node_row_ptr"]
    matches = []
    for index, node in enumerate(graph["nodes"]):
        inputs = node.get("inputs") or []
        if len(inputs) < 2 or node.get("op") != "tvm_op":
            continue
        left, right = inputs[0], inputs[1]
        left_shape = shapes[row_ptr[left[0]] + left[1]]
        right_shape = shapes[row_ptr[right[0]] + right[1]]
        if list(left_shape) == input_shape and list(right_shape) == weight_shape:
            function = str((node.get("attrs") or {}).get("func_name", ""))
            matches.append({
                "graph_node_id": "{}:{}".format(index, node.get("name", "")),
                "scoped_function": function + "_compute_",
            })
    if len(matches) != 1:
        raise ValueError("workload maps to {} graph nodes, expected one".format(len(matches)))
    return matches[0]


def candidate_metrics(static, fsim):
    totals = static["dma_features"]["totals"]
    command = fsim["command_features"]["values"]
    capacities = static["arena_capacity_certificate"]["domains"]
    return {
        "inp_request_bytes": totals["load_buffer_2d_inp_bytes"],
        "wgt_request_bytes": totals["load_buffer_2d_wgt_bytes"],
        "acc_request_bytes": None,
        "out_request_bytes": totals["store_buffer_2d_out_bytes"],
        "total_request_bytes": totals["load_buffer_2d_bytes"] + totals["store_buffer_2d_bytes"],
        "inp_request_calls": totals["load_buffer_2d_inp_calls"],
        "wgt_request_calls": totals["load_buffer_2d_wgt_calls"],
        "acc_request_calls": None,
        "out_request_calls": totals["store_buffer_2d_out_calls"],
        "total_request_calls": totals["load_buffer_2d_calls"] + totals["store_buffer_2d_calls"],
        "uop_load_bytes": command["totals"]["uop_bytes"],
        "uop_load_calls": None,
        "instruction_bytes": command["totals"]["insn_bytes"],
        "submissions": command["submissions"],
        "sync_count": static["sync"]["explicit_residency_drains"],
        "inp_arena_fraction": capacities["local.inp_buffer"]["required_upper_bound_bytes"]
        / capacities["local.inp_buffer"]["capacity_bytes"],
        "wgt_arena_fraction": capacities["local.wgt_buffer"]["required_upper_bound_bytes"]
        / capacities["local.wgt_buffer"]["capacity_bytes"],
        "acc_arena_fraction": capacities["local.acc_buffer"]["required_upper_bound_bytes"]
        / capacities["local.acc_buffer"]["capacity_bytes"],
        "uop_arena_fraction": None,
    }


def run(args):
    local = Path(args.local_root).resolve()
    build_root = Path(args.build_root).resolve()
    certificate_root = Path(args.certificate_root).resolve()
    inputs = {
        "local": verify_flat(local),
        "build": verify_flat(build_root),
        "certificates": verify_flat(certificate_root),
    }
    build = read(build_root / "summary.json")
    graph_path = build_root / "selector_v3_composable/graph.json"
    graph = read(graph_path)
    local_candidates = {row["candidate_id"]: row for row in read(local / "candidates.json")}
    static = {row["candidate_id"]: row for row in read_jsonl(local / "static_results.jsonl")}
    fsim = {row["candidate_id"]: row for row in read_jsonl(local / "fsim_results.jsonl")}
    certificates = read_jsonl(certificate_root / "candidate_certificates.jsonl")
    certificates_by_candidate = {
        row["identity"]["candidate_id"]: row for row in certificates
    }
    static_sets = read_jsonl(certificate_root / "static_set_certificates.jsonl")
    legacy_static = next(
        row for row in static_sets if row["identity"]["graph"].get("model") == "yolov3_tiny"
    )
    graph_certificates = read_jsonl(certificate_root / "graph_context_certificates.jsonl")
    legacy_graph = next(
        row for row in graph_certificates
        if row["candidate_set_id"] == legacy_static["candidate_set_id"]
    )
    expected_functions = set(build["debug_scope_audit"]["calls_by_function"])
    rows = []
    for route in build["routes"]:
        candidate_id = route["candidate_id"]
        source = local_candidates[candidate_id]
        scope = graph_route_scope(graph, route["identity"]["workload"])
        if scope["scoped_function"] not in expected_functions:
            raise ValueError("mapped function is outside binary debug scope")
        rows.append({
            **source,
            "candidate_mode": source["residence_mode"],
            "execution_task_id": route["execution_task_id"],
            "graph_node_ids": [scope["graph_node_id"]],
            "scoped_functions": [scope["scoped_function"]],
            "candidate_metrics": candidate_metrics(static[candidate_id], fsim[candidate_id]),
        })
    graph_identity = dict(legacy_static["identity"]["graph"])
    graph_identity["graph_json_sha256"] = sha256(graph_path)
    candidate_set = build_graph_candidate_set(
        graph_identity, rows, certificates_by_candidate
    )
    migrated = inherit_graph_context_certificate(
        candidate_set, legacy_static, legacy_graph
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "contract.json", {
        "schema": "c3_yolo608_graph_candidate_set_freeze_v1",
        "status": "frozen_from_existing_binary_and_correctness_evidence",
        "latency_values_read": False,
        "mapping_rule": "unique graph node whose first two input shapes equal route input/kernel shapes",
        "claim_boundary": "identity migration only; no new correctness or performance experiment",
    })
    write(output / "graph_candidate_set.json", candidate_set)
    write(output / "migrated_graph_context_certificate.json", migrated)
    write(output / "input_hashes.json", inputs)
    summary = {
        "schema": "c3_yolo608_graph_candidate_set_freeze_v1",
        "status": "REAL_GRAPH_CANDIDATE_SET_FROZEN",
        "candidate_set_id": candidate_set["candidate_set_id"],
        "legacy_candidate_set_id": legacy_static["candidate_set_id"],
        "candidate_count": len(candidate_set["candidate_ids"]),
        "graph_node_ids": sorted(
            node for member in candidate_set["members"] for node in member["graph_node_ids"]
        ),
        "scoped_functions": sorted(
            function for member in candidate_set["members"]
            for function in member["scoped_functions"]
        ),
        "graph_correctness_inherited_by_exact_identity_migration": True,
        "latency_values_read": False,
    }
    write(output / "summary.json", summary)
    write(output / "artifact_hashes.json", {
        "artifacts": {
            path.name: sha256(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "artifact_hashes.json"
        },
        "source_sha256": sha256(Path(__file__).resolve()),
    })
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", default=str(DEFAULT_LOCAL))
    parser.add_argument("--build-root", default=str(DEFAULT_BUILD))
    parser.add_argument("--certificate-root", default=str(DEFAULT_CERTIFICATES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
