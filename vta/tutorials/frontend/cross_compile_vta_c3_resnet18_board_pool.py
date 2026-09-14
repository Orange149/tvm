#!/usr/bin/env python3
"""Cross-compile and retain exact binaries for the frozen P7R482 board pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import tvm
from tvm import autotvm
from tvm.contrib import cc
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import canonical_json_bytes, make_failure, normalize_config_entity


SCHEMA = "c3_resnet18_cross_compile_v1"
MODES = {"original": 0, "input_stationary": 1, "weight_resident_barrier": 4, "input_weight_resident_barrier": 5}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def verify_pool(directory):
    directory = Path(directory).resolve()
    ledger = read_json(directory / "artifact_hashes.json")["artifacts"]
    for name, expected in ledger.items():
        if sha256_file(directory / name) != expected:
            raise ValueError("board-pool artifact hash mismatch: {}".format(name))
    rows = read_jsonl(directory / "candidates.jsonl")
    contract = read_json(directory / "contract.json")
    if [row["candidate_id"] for row in rows] != contract["candidate_order"]:
        raise ValueError("candidate order differs from frozen board-pool contract")
    if contract["board_contacted"] is not False or contract["performance_labels_collected"] is not False:
        raise ValueError("board pool is not label blind")
    return rows, {name: value for name, value in sorted(ledger.items())}


def options():
    sysroot = os.environ.get("SDKTARGETSYSROOT")
    if not sysroot:
        raise RuntimeError("SDKTARGETSYSROOT is unset; source the aarch64 SDK environment")
    return [
        "--sysroot=" + sysroot,
        "-Wl,-rpath-link," + sysroot + "/lib",
        "-Wl,-rpath-link," + sysroot + "/usr/lib",
        "-L" + sysroot + "/lib",
        "-L" + sysroot + "/usr/lib",
    ]


def worker(candidate, binary):
    start = time.monotonic()
    result = {
        "schema": SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "implementation_candidate_id": candidate["implementation_candidate_id"],
        "workload_id": candidate["workload_id"],
        "family_id": candidate["family_id"],
        "residence_mode": candidate["residence_mode"],
        "status": "failed",
        "failure": None,
        "board_contacted": False,
        "performance_label": None,
    }
    phase = "preflight"
    try:
        env = vta.get_env()
        if env.TARGET != "axu5evb":
            raise RuntimeError("expected axu5evb VTA target")
        compile_options = options()
        phase = "identity"
        workload = candidate["identity"]["workload"]
        task = autotvm.task.create(
            "conv2d_packed_residency.vta",
            args=tuple(workload[1:]) + (MODES[candidate["residence_mode"]],),
            target=env.target,
            target_host=env.target_host,
        )
        config = task.config_space.get(int(candidate["debug"]["config_index"]))
        if canonical_json_bytes(normalize_config_entity(config.to_json_dict())) != canonical_json_bytes(normalize_config_entity(candidate["complete_config_entity"])):
            raise RuntimeError("ConfigEntity differs from frozen board-pool identity")
        phase = "build"
        build_start = time.monotonic()
        with task.target:
            schedule, tensors = task.instantiate(config)
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            lowered = tvm.lower(schedule, tensors, name="main")
        tir_hash = hashlib.sha256(tvm.ir.save_json(lowered).encode("utf-8")).hexdigest()
        if tir_hash != candidate["tir_sha256"]:
            raise RuntimeError("cross-compile re-lowering differs from frozen TIR")
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            module = vta.build(schedule, tensors, target=tvm.target.Target(env.target, host=env.target_host), name="main")
        build_seconds = time.monotonic() - build_start
        phase = "export"
        export_start = time.monotonic()
        binary = Path(binary).resolve()
        binary.parent.mkdir(parents=True, exist_ok=True)
        module.export_library(
            str(binary),
            fcompile=cc.cross_compiler("aarch64-xilinx-linux-g++"),
            options=compile_options,
        )
        result.update(
            status="passed",
            relowered_tir_sha256=tir_hash,
            build_seconds=build_seconds,
            export_seconds=time.monotonic() - export_start,
            binary={"path": str(binary), "sha256": sha256_file(binary), "size_bytes": binary.stat().st_size, "retained": True},
        )
    except Exception as error:
        category = "compile" if phase in ("build", "export") else "environment"
        failure = make_failure(category, (str(error) or type(error).__name__)[:4000], phase=phase)
        failure["exception_type"] = type(error).__name__
        result["failure"] = failure
    result["wall_seconds"] = time.monotonic() - start
    return result


def run_main(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    candidates, input_hashes = verify_pool(args.board_pool_dir)
    output.mkdir(parents=True)
    (output / "binaries").mkdir()
    results_path, timeline_path = output / "results.jsonl", output / "timeline.jsonl"
    results_path.write_text("", encoding="utf-8")
    timeline_path.write_text("", encoding="utf-8")
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_cross_compile",
        "candidate_order": [row["candidate_id"] for row in candidates],
        "timeout_seconds_per_candidate": args.timeout,
        "input_artifact_hashes": input_hashes,
        "board_contacted": False,
    }
    write_json(output / "contract.json", contract)
    started = time.monotonic()
    records = []
    for position, candidate in enumerate(candidates, 1):
        binary = output / "binaries" / (candidate["candidate_id"] + ".so")
        with tempfile.TemporaryDirectory(prefix="c3_r18_cross_") as directory:
            result_file = Path(directory) / "result.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-candidate-id", candidate["candidate_id"], "--board-pool-dir", str(Path(args.board_pool_dir).resolve()), "--worker-binary", str(binary), "--worker-output", str(result_file)]
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, check=False, env=os.environ.copy())
                if completed.returncode or not result_file.is_file():
                    result = {
                        "schema": SCHEMA, "candidate_id": candidate["candidate_id"], "workload_id": candidate["workload_id"], "family_id": candidate["family_id"], "residence_mode": candidate["residence_mode"], "status": "failed",
                        "failure": make_failure("environment", "worker exit={} output_present={}".format(completed.returncode, result_file.is_file()), phase="worker_process"), "wall_seconds": None, "board_contacted": False, "performance_label": None,
                    }
                else:
                    result = read_json(result_file)
            except subprocess.TimeoutExpired:
                result = {
                    "schema": SCHEMA, "candidate_id": candidate["candidate_id"], "workload_id": candidate["workload_id"], "family_id": candidate["family_id"], "residence_mode": candidate["residence_mode"], "status": "failed",
                    "failure": make_failure("timeout", "cross compile exceeded {} seconds".format(args.timeout), phase="worker", retryable=True), "wall_seconds": float(args.timeout), "board_contacted": False, "performance_label": None,
                }
        result["position"] = position
        records.append(result)
        append_jsonl(results_path, result)
        append_jsonl(timeline_path, {"event": "cross_compile_complete", "position": position, "candidate_id": candidate["candidate_id"], "status": result["status"], "elapsed_seconds": time.monotonic() - started})
        print("cross {}/{} passed={} failed={}".format(position, len(candidates), sum(row["status"] == "passed" for row in records), sum(row["status"] != "passed" for row in records)), flush=True)
    summary = {
        "schema": SCHEMA,
        "status": "complete_cross_compile_qualification",
        "candidate_count": len(candidates),
        "passed": sum(row["status"] == "passed" for row in records),
        "failed": sum(row["status"] != "passed" for row in records),
        "failure_categories": dict(Counter(row["failure"]["category"] for row in records if row.get("failure"))),
        "wall_seconds": time.monotonic() - started,
        "retained_binary_count": sum(row["status"] == "passed" for row in records),
        "board_contacted": False,
        "performance_labels_collected": False,
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(" ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n", encoding="utf-8")
    artifacts = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "artifact_hashes.json"}
    artifacts["binaries"] = {path.name: sha256_file(path) for path in sorted((output / "binaries").iterdir())}
    write_json(output / "artifact_hashes.json", {"artifacts": artifacts, "source_hashes": {str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve())}})
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_worker(args):
    candidates, _ = verify_pool(args.board_pool_dir)
    matches = [row for row in candidates if row["candidate_id"] == args.worker_candidate_id]
    if len(matches) != 1:
        raise ValueError("worker candidate lookup failed")
    write_json(args.worker_output, worker(matches[0], args.worker_binary))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-pool-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--worker-candidate-id")
    parser.add_argument("--worker-binary")
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    worker_args = (args.worker_candidate_id, args.worker_binary, args.worker_output)
    if any(worker_args) and not all(worker_args):
        parser.error("all worker arguments are required together")
    if not any(worker_args) and not args.output_dir:
        parser.error("--output-dir is required")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker_candidate_id:
        run_worker(parsed)
    else:
        run_main(parsed)
