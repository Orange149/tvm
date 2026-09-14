#!/usr/bin/env python3
"""Measure one qualified P7 workload in frozen randomized complete blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
import traceback
from pathlib import Path

import numpy as np
from tvm import rpc
import tvm
import vta

from qualify_vta_residency_fsim import reference_data_from_workload
from run_vta_p7_holdout_correctness import (
    build_candidate,
    file_sha256,
    load_contract,
    ssh_preflight,
    write_json,
)


def load_qualified(path):
    path = Path(path).resolve()
    ledger = json.loads((path.parent / "artifact_hashes.json").read_text())
    if ledger["output_sha256"].get(path.name) != file_sha256(path):
        raise ValueError("qualified timing contract hash mismatch")
    value = json.loads(path.read_text())
    if value.get("schema") != "c3_p7_qualified_timing_contract_v1" or value.get(
        "status"
    ) != "frozen_after_correctness_before_timing":
        raise ValueError("unexpected or unfrozen qualified timing contract")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "timing.jsonl"
    if results_path.exists() or (output / "summary.json").exists() or (output / "failure.json").exists():
        raise FileExistsError("refusing to overwrite timing observations")

    contract = load_qualified(args.contract)
    if args.workload not in contract["workloads"]:
        raise ValueError("workload is outside the qualified timing contract")
    workload = contract["workloads"][args.workload]
    entries = {row["candidate_id"]: row for row in workload["gross_candidates"]}
    timed_ids = workload["timed_candidate_ids"]
    if set(timed_ids) != set(entries) - set(workload["invalid_candidate_ids"]):
        raise ValueError("timed pool does not equal the correctness-qualified pool")

    original_p7 = Path(contract["original_p7_contract"]["path"])
    if file_sha256(original_p7) != contract["original_p7_contract"]["sha256"]:
        raise ValueError("original P7 contract changed")
    p7 = load_contract(original_p7)
    env = vta.get_env()
    before = ssh_preflight(p7, args.host)
    remote = rpc.connect(args.host, args.port, session_timeout=300)
    device = remote.ext_dev(0)

    first_entry = entries[timed_ids[0]]
    data, weight, expected = reference_data_from_workload(first_entry["workload"], 0)
    buffers = (
        tvm.nd.array(data, device),
        tvm.nd.array(weight, device),
        tvm.nd.array(np.full(expected.shape, -113, dtype="int8"), device),
    )
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix="c3_p7_timing_") as temporary:
            binary_dir = Path(temporary)
            modules = {}
            timers = {}
            for candidate_id in timed_ids:
                binary, tir_hash = build_candidate(entries[candidate_id], env, binary_dir)
                remote.upload(str(binary))
                module = remote.load_module(binary.name)
                modules[candidate_id] = (module, tir_hash)
                timers[candidate_id] = module.time_evaluator(
                    "main", device, number=1, repeat=1
                )

            for candidate_id in workload["warmup_order"]:
                for _ in range(contract["measurement"]["warmup"]):
                    modules[candidate_id][0]["main"](*buffers)

            sentinel_id = workload["incumbent_candidate_id"]
            with results_path.open("x", encoding="utf-8") as stream:
                for block_index, order in enumerate(workload["timing_block_orders"], 1):
                    preflight = ssh_preflight(p7, args.host)
                    sentinel_before_ms = float(timers[sentinel_id](*buffers).results[0] * 1000.0)
                    sentinel_before = {
                        "schema": "c3_p7_timing_sentinel_v1",
                        "workload_id": args.workload,
                        "block": block_index,
                        "position": "before",
                        "candidate_id": sentinel_id,
                        "latency_ms": sentinel_before_ms,
                        "excluded_from_candidate_samples": True,
                    }
                    stream.write(json.dumps(sentinel_before, sort_keys=True) + "\n")
                    for position, candidate_id in enumerate(order, 1):
                        latency_ms = float(timers[candidate_id](*buffers).results[0] * 1000.0)
                        row = {
                            "schema": "c3_p7_timing_candidate_v1",
                            "workload_id": args.workload,
                            "block": block_index,
                            "position": position,
                            "candidate_id": candidate_id,
                            "residence_mode": entries[candidate_id]["residence_mode"],
                            "config_index": entries[candidate_id]["config_index"],
                            "tir_sha256": modules[candidate_id][1],
                            "latency_ms": latency_ms,
                            "preflight": preflight,
                        }
                        rows.append(row)
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                    sentinel_after_ms = float(timers[sentinel_id](*buffers).results[0] * 1000.0)
                    sentinel_after = {
                        "schema": "c3_p7_timing_sentinel_v1",
                        "workload_id": args.workload,
                        "block": block_index,
                        "position": "after",
                        "candidate_id": sentinel_id,
                        "latency_ms": sentinel_after_ms,
                        "excluded_from_candidate_samples": True,
                    }
                    stream.write(json.dumps(sentinel_after, sort_keys=True) + "\n")
                    stream.flush()
                    print(
                        "block {}/{} complete: {} candidates, sentinel {:.6f}->{:.6f} ms".format(
                            block_index,
                            len(workload["timing_block_orders"]),
                            len(order),
                            sentinel_before_ms,
                            sentinel_after_ms,
                        ),
                        flush=True,
                    )
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "schema": "c3_p7_qualified_timing_failure_v1",
                "workload_id": args.workload,
                "completed_candidate_samples": len(rows),
                "exception_type": type(error).__name__,
                "message": str(error)[:4000],
                "traceback": traceback.format_exc()[-12000:],
            },
        )
        raise

    after = ssh_preflight(p7, args.host)
    samples = {candidate_id: [] for candidate_id in timed_ids}
    for row in rows:
        samples[row["candidate_id"]].append(row["latency_ms"])
    if any(len(values) != contract["measurement"]["blocks"] for values in samples.values()):
        raise RuntimeError("incomplete randomized-block timing coverage")
    summaries = {
        candidate_id: {
            "samples_ms": values,
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
        }
        for candidate_id, values in samples.items()
    }
    write_json(
        output / "summary.json",
        {
            "schema": "c3_p7_qualified_timing_summary_v1",
            "status": "complete",
            "workload_id": args.workload,
            "candidate_count": len(timed_ids),
            "blocks": contract["measurement"]["blocks"],
            "candidate_sample_count": len(rows),
            "candidate_summaries": summaries,
            "timing_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
            "board_before": before,
            "board_after": after,
        },
    )


if __name__ == "__main__":
    main()
