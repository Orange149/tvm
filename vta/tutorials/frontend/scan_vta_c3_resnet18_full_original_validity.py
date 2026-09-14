#!/usr/bin/env python3
"""Scan one complete frozen ResNet18 original ConfigSpace with real lowering."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import tvm
from tvm import autotvm
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import canonical_json_bytes, make_failure, normalize_config_entity
from generate_vta_residency_candidates import classify_lower_failure
from run_vta_c3_resnet18_local_qualification import hidden_compiler_features


SCHEMA = "c3_resnet18_full_original_validity_v1"
WORKLOADS = ("R18-H1", "R18-H2", "R18-H3")
DEFAULT_INPUT = Path(__file__).resolve().parent / (
    "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/"
    "20260914_p7r470_resnet18_literature_candidates_run01"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def verify_input(directory, workload_id):
    directory = Path(directory).resolve()
    ledger = read_json(directory / "artifact_hashes.json")["artifacts"]
    for name, expected in ledger.items():
        if sha256_file(directory / name) != expected:
            raise ValueError("P7R470 artifact hash mismatch: {}".format(name))
    path = directory / "{}_complete_original_space.json".format(workload_id.lower())
    payload = read_json(path)
    if payload["workload_id"] != workload_id or payload["role"] != "complete_original_space_for_hw_aware":
        raise ValueError("wrong complete-space contract")
    if len(payload["candidates"]) != payload["complete_config_space_count"]:
        raise ValueError("complete ConfigSpace cardinality mismatch")
    return path, payload, ledger


def scan_one(task, candidate, workload, env):
    start = time.monotonic()
    result = {
        "schema": SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "config_index": int(candidate["debug"]["config_index"]),
        "residence_mode": "original",
        "visible_features": candidate["visible_features"],
        "workload_features": candidate["workload_features"],
        "is_valid": False,
        "status": "invalid",
        "failure": None,
        "performance_label": None,
        "board_contacted": False,
    }
    phase = "config_identity"
    try:
        config = task.config_space.get(result["config_index"])
        if canonical_json_bytes(normalize_config_entity(config.to_json_dict())) != canonical_json_bytes(normalize_config_entity(candidate["complete_config_entity"])):
            raise RuntimeError("ConfigEntity differs from frozen complete-space identity")
        phase = "instantiate"
        with task.target:
            schedule, tensors = task.instantiate(config)
        phase = "lower"
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            module = tvm.lower(schedule, tensors, name="main")
        result.update(
            status="valid",
            is_valid=True,
            lowered_tir_sha256=hashlib.sha256(tvm.ir.save_json(module).encode("utf-8")).hexdigest(),
            hidden_features=hidden_compiler_features(module, {**candidate, "identity": {"workload": workload}}),
        )
    except Exception as error:
        message = str(error) or type(error).__name__
        failure = make_failure("lower", message[:4000], phase=phase)
        failure["exception_type"] = type(error).__name__
        failure["subcategory"] = classify_lower_failure(message, phase)
        result["failure"] = failure
    result["wall_seconds"] = time.monotonic() - start
    return result


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    source_path, payload, ledger = verify_input(args.input_dir, args.workload_id)
    candidates = payload["candidates"]
    output.mkdir(parents=True)
    results_path = output / "results.jsonl"
    timeline_path = output / "timeline.jsonl"
    results_path.write_text("", encoding="utf-8")
    timeline_path.write_text("", encoding="utf-8")
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_full_original_validity_scan",
        "workload_id": args.workload_id,
        "candidate_count": len(candidates),
        "candidate_order": [row["candidate_id"] for row in candidates],
        "input_path": str(source_path),
        "input_sha256": sha256_file(source_path),
        "performance_labels_used": False,
        "board_contacted": False,
    }
    write_json(output / "contract.json", contract)
    env = vta.get_env()
    workload = payload["workload"]
    task = autotvm.task.create(
        "conv2d_packed_residency.vta",
        args=tuple(workload[1:]) + (0,),
        target=env.target,
        target_host=env.target_host,
    )
    if len(task.config_space) != len(candidates):
        raise ValueError("live complete ConfigSpace differs from frozen cardinality")
    started = time.monotonic()
    rows = []
    for position, candidate in enumerate(candidates, 1):
        row = scan_one(task, candidate, workload, env)
        row["compiler_call"] = position
        rows.append(row)
        append_jsonl(results_path, row)
        append_jsonl(timeline_path, {"event": "lower_complete", "compiler_call": position, "candidate_id": candidate["candidate_id"], "status": row["status"], "phase_wall_seconds": row["wall_seconds"], "elapsed_seconds": time.monotonic() - started})
        if position % 50 == 0 or position == len(candidates):
            print("{} lower {}/{} valid={} invalid={}".format(args.workload_id, position, len(candidates), sum(item["is_valid"] for item in rows), sum(not item["is_valid"] for item in rows)), flush=True)
    categories = Counter(row["failure"]["subcategory"] for row in rows if row.get("failure"))
    valid = sum(row["is_valid"] for row in rows)
    summary = {
        "schema": SCHEMA,
        "status": "complete_full_original_validity_scan",
        "workload_id": args.workload_id,
        "compiler_calls": len(rows),
        "valid": valid,
        "invalid": len(rows) - valid,
        "valid_yield": valid / len(rows),
        "invalid_ratio": (len(rows) - valid) / len(rows),
        "failure_subcategories": dict(sorted(categories.items())),
        "wall_seconds": time.monotonic() - started,
        "performance_labels_collected": False,
        "board_contacted": False,
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(" ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n", encoding="utf-8")
    artifacts = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "artifact_hashes.json"}
    write_json(output / "artifact_hashes.json", {"artifacts": artifacts, "input_artifacts": ledger, "source_hashes": {str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve())}})
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--workload-id", required=True, choices=WORKLOADS)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
