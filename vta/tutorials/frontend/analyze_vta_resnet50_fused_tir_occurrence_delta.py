#!/usr/bin/env python3
"""Aggregate fused-TIR DMA deltas by exact ResNet50 graph occurrence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_vta_resnet50_callsite_composed_graph_board import verify_build_artifacts, write_json


RUNTIME_KEYS = (
    "load_buffer_2d_bytes",
    "load_buffer_2d_calls",
    "load_buffer_2d_acc_bytes",
    "load_buffer_2d_acc_calls",
    "load_buffer_2d_inp_bytes",
    "load_buffer_2d_inp_calls",
    "load_buffer_2d_wgt_bytes",
    "load_buffer_2d_wgt_calls",
    "store_buffer_2d_bytes",
    "store_buffer_2d_calls",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function_inventory(build):
    result = {}
    for snapshot in build["snapshots"]:
        for function in snapshot["functions"]:
            symbol = function["global_symbol"]
            if symbol in result:
                raise RuntimeError("duplicate fused TIR symbol: " + repr(symbol))
            result[symbol] = function["dma_totals"]
    return result


def vector_delta(left, right):
    return {key: right.get(key, 0) - left.get(key, 0) for key in RUNTIME_KEYS}


def run(args):
    audit = Path(args.tir_audit)
    audit_artifacts = verify_build_artifacts(audit)
    summary = read_json(audit / "summary.json")
    if summary.get("status") != "full_graph_fused_tir_pair_captured":
        raise RuntimeError("fused TIR audit is incomplete")
    builds = summary["builds"]
    modes = [row["public_mode"] for row in builds]
    if modes != ["original", "input_stationary"]:
        raise RuntimeError("unexpected fused TIR modes")
    inventories = {row["public_mode"]: function_inventory(row) for row in builds}

    failed_board = Path(args.failed_board)
    contract = read_json(failed_board / "contract.json")
    correctness = read_json(failed_board / "correctness.json")
    occurrences = contract["graph_occurrences"]
    rows = []
    aggregate = {key: 0 for key in RUNTIME_KEYS}
    for occurrence in occurrences:
        symbol = occurrence["func_name"]
        if any(symbol not in inventory for inventory in inventories.values()):
            raise RuntimeError("graph symbol missing from fused TIR: " + symbol)
        delta = vector_delta(
            inventories[modes[0]][symbol], inventories[modes[1]][symbol]
        )
        for key, value in delta.items():
            aggregate[key] += value
        rows.append({
            "graph_node": occurrence["graph_node"],
            "func_name": symbol,
            "original_dma": {key: inventories[modes[0]][symbol].get(key, 0)
                             for key in RUNTIME_KEYS},
            "residency_dma": {key: inventories[modes[1]][symbol].get(key, 0)
                              for key in RUNTIME_KEYS},
            "delta": delta,
        })

    board_observed = {
        key: correctness[1]["runtime_profile_complete"].get(key, 0)
        - correctness[0]["runtime_profile_complete"].get(key, 0)
        for key in RUNTIME_KEYS
    }
    if aggregate != board_observed:
        raise RuntimeError("fused-TIR occurrence aggregation does not match FPGA profile")
    isolated_prediction = contract["expected_full_graph_delta"]
    residual = {
        key: aggregate.get(key, 0) - isolated_prediction.get(key, 0)
        for key in RUNTIME_KEYS
    }
    per_occurrence = rows[0]["delta"]
    if any(row["delta"] != per_occurrence for row in rows[1:]):
        raise RuntimeError("target fused functions have different per-occurrence deltas")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    result = {
        "schema": "c3_vta_resnet50_fused_tir_occurrence_delta_v1",
        "status": "compiler_fused_tir_prediction_matches_fpga_profile_exactly",
        "modes": modes,
        "graph_occurrence_count": len(occurrences),
        "graph_occurrences": rows,
        "per_occurrence_fused_tir_delta": per_occurrence,
        "full_graph_fused_tir_delta": aggregate,
        "failed_isolated_template_prediction": isolated_prediction,
        "fusion_residual_vs_isolated_template": residual,
        "fpga_observed_delta": board_observed,
        "exact_fields": list(RUNTIME_KEYS),
        "tir_audit_summary_sha256": sha256(audit / "summary.json"),
        "tir_audit_artifact_hash_manifest_sha256": sha256(audit / "artifact_hashes.json"),
        "verified_tir_audit_artifact_count": len(audit_artifacts),
        "failed_board_contract_sha256": sha256(failed_board / "contract.json"),
        "failed_board_correctness_sha256": sha256(failed_board / "correctness.json"),
        "analyzer_source_sha256": sha256(Path(__file__).resolve()),
        "claim_boundary": (
            "Retrospective reconciliation of one failed pre-timing contract; the v2 "
            "compiler prediction must be frozen and rerun before latency collection"
        ),
    }
    write_json(output / "fused_tir_occurrence_delta.json", result)
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
    parser.add_argument("--tir-audit", required=True)
    parser.add_argument("--failed-board", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
