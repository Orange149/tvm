#!/usr/bin/env python3
"""Freeze candidate, static-set and graph-context composability certificates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from c3_composability import (
    candidate_certificate,
    graph_context_certificate,
    static_set_certificate,
)


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_YOLO_LOCAL = GROUPED / "20260924_c3_yolo608_composable_local_run02"
DEFAULT_YOLO_BUILD = GROUPED / "20260924_c3_yolo608_composable_build_run01"
DEFAULT_YOLO_BOARD = GROUPED / "20260924_c3_yolo608_composable_board_run01"
DEFAULT_R18_LOCAL = GROUPED / "20260924_c3_resnet18_h3_composable_local_run01"
DEFAULT_R18_BUILD = GROUPED / "20260924_c3_resnet18_h3_composable_build_run01"
DEFAULT_R18_BOARD = GROUPED / "20260924_c3_resnet18_h3_composable_board_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_composability_certificates_run01"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_evidence(static_set, board):
    return graph_context_certificate(static_set, {
        "candidate_set_id": static_set["candidate_set_id"],
        "all_equal": board["all_equal"],
        "boot_id": board["boot_id"],
        "bitstream_reloaded": board["bitstream_reloaded"],
        "mismatch_count_by_seed_and_output": board["mismatch_count_by_seed_and_output"],
        "complete_graph_run_calls": board["complete_graph_run_calls"],
        "timing_collected": board["timing_collected"],
        "performance_claim": board["performance_claim"],
    })


def yolo_certificates(args):
    local = Path(args.yolo_local).resolve()
    build_root = Path(args.yolo_build).resolve()
    board_root = Path(args.yolo_board).resolve()
    candidates = read(local / "candidates.json")
    static = {row["candidate_id"]: row for row in read_jsonl(local / "static_results.jsonl")}
    fsim = {row["candidate_id"]: row for row in read_jsonl(local / "fsim_results.jsonl")}
    certificates = [candidate_certificate(row, static[row["candidate_id"]], fsim[row["candidate_id"]])
                    for row in candidates]
    build = read(build_root / "summary.json")
    variant = build["variant"]
    static_set = static_set_certificate(certificates, {
        "model": "yolov3_tiny", "input_resolution": 608,
        "graph_sha256": variant["graph_sha256"], "params_sha256": variant["params_sha256"],
        "build_manifest_sha256": sha256(build_root / "artifact_hashes.json"),
    }, build["debug_scope_audit"])
    graph = graph_evidence(static_set, read(board_root / "summary.json"))
    return certificates, static_set, graph


def resnet_certificates(args):
    local = Path(args.resnet_local).resolve()
    build_root = Path(args.resnet_build).resolve()
    board_root = Path(args.resnet_board).resolve()
    candidate = read(local / "candidate.json")
    certificate = candidate_certificate(
        candidate, read(local / "static_result.json"), read(local / "fsim_result.json")
    )
    build = read(build_root / "summary.json")
    built = build["build"]
    static_set = static_set_certificate([certificate], {
        "model": "resnet18", "input_resolution": 224,
        "base_program": "selected_918f902567c608ec",
        "graph_sha256": built["graph_sha256"], "params_sha256": built["params_sha256"],
        "build_manifest_sha256": sha256(build_root / "artifact_hashes.json"),
    }, build["debug_scope_audit"])
    graph = graph_evidence(static_set, read(board_root / "summary.json"))
    return [certificate], static_set, graph


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "contract.json", {
        "schema": "c3_composability_certificate_freeze_v1",
        "status": "frozen_from_existing_correctness_evidence",
        "levels": ["candidate_local", "static_candidate_set", "graph_context"],
        "latency_values_read": False,
        "claim_boundary": (
            "candidate and static-set certificates do not imply graph correctness; "
            "graph certificates bind exact multi-seed board evidence"
        ),
    })
    yolo_candidates, yolo_set, yolo_graph = yolo_certificates(args)
    r18_candidates, r18_set, r18_graph = resnet_certificates(args)
    for row in yolo_candidates + r18_candidates:
        append(output / "candidate_certificates.jsonl", row)
    for row in (yolo_set, r18_set):
        append(output / "static_set_certificates.jsonl", row)
    for row in (yolo_graph, r18_graph):
        append(output / "graph_context_certificates.jsonl", row)
    summary = {
        "schema": "c3_composability_certificate_freeze_v1",
        "status": "COMPOSABILITY_CERTIFICATES_FROZEN",
        "candidate_certificate_count": len(yolo_candidates) + len(r18_candidates),
        "static_set_certificate_count": 2,
        "graph_context_certificate_count": 2,
        "networks": ["resnet18", "yolov3_tiny"],
        "all_graph_sets_timing_eligible": all(row["timing_eligible"] for row in (yolo_graph, r18_graph)),
        "latency_values_read": False,
    }
    write(output / "summary.json", summary)
    write(output / "artifact_hashes.json", {"artifacts": {
        path.name: sha256(path) for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }, "source_sha256": sha256(Path(__file__).resolve())})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-local", default=str(DEFAULT_YOLO_LOCAL))
    parser.add_argument("--yolo-build", default=str(DEFAULT_YOLO_BUILD))
    parser.add_argument("--yolo-board", default=str(DEFAULT_YOLO_BOARD))
    parser.add_argument("--resnet-local", default=str(DEFAULT_R18_LOCAL))
    parser.add_argument("--resnet-build", default=str(DEFAULT_R18_BUILD))
    parser.add_argument("--resnet-board", default=str(DEFAULT_R18_BOARD))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
