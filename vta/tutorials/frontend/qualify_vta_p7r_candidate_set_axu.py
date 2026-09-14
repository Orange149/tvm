#!/usr/bin/env python3
"""Cross-compile a small frozen P7R candidate set without contacting a board."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import vta

from run_vta_p7r_joint_correctness import build_module, sha256_file, write_json


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    paths = [Path(value).resolve() for value in args.candidates]
    candidates = [row for path in paths for row in load_jsonl(path)]
    if not candidates or not all(row.get("status") == "ok" for row in candidates):
        raise ValueError("all frozen candidates must be lower-ok")
    identities = [
        (row["candidate_id"], row["residence_mode"],
         int(row.get("config_index", row["debug"]["config_index"])), row["tir_sha256"])
        for row in candidates
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate candidate identity")
    preregistered = {
        "schema": "c3_p7r_small_axu_cross_compile_protocol_v1",
        "status": "frozen_before_cross_compile",
        "candidate_count": len(candidates),
        "identities": identities,
        "inputs": {str(path): sha256_file(path) for path in paths},
        "board_contacted": False,
        "performance_measurement": "not_collected",
        "runner_sha256": sha256_file(__file__),
    }
    write_json(output / "preregistered.json", preregistered)
    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise RuntimeError("VTA TARGET must be axu5evb")
    rows = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="c3_p7r_axu_qual_") as temporary:
        for row in candidates:
            index = int(row.get("config_index", row["debug"]["config_index"]))
            try:
                binary, tir_hash, config = build_module(
                    row["identity"]["workload"], row["residence_mode"], index,
                    env, Path(temporary), row["tir_sha256"]
                )
                result = {
                    "candidate_id": row["candidate_id"],
                    "workload_id": row["workload_id"],
                    "mode": row["residence_mode"],
                    "config_index": index,
                    "tir_sha256": tir_hash,
                    "binary_sha256": sha256_file(binary),
                    "binary_size_bytes": binary.stat().st_size,
                    "complete_config_entity": config,
                    "status": "passed",
                    "board_contacted": False,
                    "performance_measurement": "not_collected",
                }
            except Exception as error:
                result = {
                    "candidate_id": row["candidate_id"],
                    "workload_id": row["workload_id"],
                    "mode": row["residence_mode"],
                    "config_index": index,
                    "status": "failed",
                    "failure": {"type": type(error).__name__, "message": str(error)},
                    "board_contacted": False,
                    "performance_measurement": "not_collected",
                }
            rows.append(result)
    results = output / "results.jsonl"
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {
        "schema": "c3_p7r_small_axu_cross_compile_v1",
        "status": "passed" if all(row["status"] == "passed" for row in rows) else "failed",
        "candidate_count": len(rows),
        "passed": sum(row["status"] == "passed" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "elapsed_seconds": time.monotonic() - started,
        "board_contacted": False,
        "performance_measurement": "not_collected",
    }
    write_json(output / "summary.json", summary)
    (output / "STATUS.md").write_text(
        "# P7R small AXU cross-compile qualification\n\n"
        "- Status: `{}`\n- Passed: {}/{}\n- Board contacted: no\n"
        "- Performance timing: not collected\n".format(
            summary["status"], summary["passed"], summary["candidate_count"]
        ), encoding="utf-8"
    )
    hashes = {
        path.name: sha256_file(path) for path in output.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", {"artifacts": hashes})
    print(json.dumps(summary, indent=2))
    if summary["status"] != "passed":
        raise RuntimeError("one or more AXU cross-compiles failed")


if __name__ == "__main__":
    main()
