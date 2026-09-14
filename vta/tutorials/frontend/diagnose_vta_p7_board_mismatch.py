#!/usr/bin/env python3
"""Diagnose one P7 board-correctness failure without collecting timing labels."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from tvm import rpc
import tvm
import vta

from qualify_vta_residency_fsim import array_sha256, reference_data_from_workload
from run_vta_p7_holdout_correctness import (
    build_candidate,
    load_contract,
    ssh_preflight,
    validate_workload,
    write_json,
)


def mismatch_sample(actual, expected, limit=16):
    locations = np.argwhere(actual != expected)
    result = []
    for location in locations[:limit]:
        index = tuple(int(value) for value in location)
        result.append(
            {
                "index": list(index),
                "expected": int(expected[index]),
                "actual": int(actual[index]),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--sentinel-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--target-repeats", type=int, default=3)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "diagnostic.jsonl"
    if results_path.exists():
        raise FileExistsError("refusing to overwrite diagnostic observations")

    contract = load_contract(args.contract)
    entries, _ = validate_workload(contract, args.workload)
    for candidate_id in (args.candidate_id, args.sentinel_id):
        if candidate_id not in entries:
            raise ValueError("candidate is not in the frozen workload pool: " + candidate_id)

    before = ssh_preflight(contract, args.host)
    remote = rpc.connect(args.host, args.port, session_timeout=120)
    device = remote.ext_dev(0)
    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise RuntimeError("VTA TARGET must be axu5evb")
    sequence = [("sentinel_before", args.sentinel_id, 1)]
    sequence += [("target_repeat", args.candidate_id, index) for index in range(1, args.target_repeats + 1)]
    sequence += [("sentinel_after", args.sentinel_id, 1)]
    rows = []

    with tempfile.TemporaryDirectory(prefix="c3_p7_mismatch_") as temporary:
        binaries = {}
        for candidate_id in (args.sentinel_id, args.candidate_id):
            binary, tir_hash = build_candidate(entries[candidate_id], env, Path(temporary))
            binaries[candidate_id] = (binary, tir_hash)
        modules = {}
        for candidate_id, (binary, _) in binaries.items():
            remote.upload(str(binary))
            modules[candidate_id] = remote.load_module(binary.name)

        with results_path.open("x", encoding="utf-8") as stream:
            for phase, candidate_id, repeat in sequence:
                entry = entries[candidate_id]
                module = modules[candidate_id]
                preflight = ssh_preflight(contract, args.host)
                seed_rows = []
                for seed in contract["measurement"]["correctness_seeds"]:
                    data, weight, expected = reference_data_from_workload(entry["workload"], int(seed))
                    # Alternating prefill exposes unwritten output locations if present.
                    prefill = -113 if repeat % 2 else 97
                    buffers = [
                        tvm.nd.array(data, device),
                        tvm.nd.array(weight, device),
                        tvm.nd.array(np.full(expected.shape, prefill, dtype="int8"), device),
                    ]
                    module["main"](*buffers)
                    actual = buffers[-1].numpy()
                    mismatch_count = int(np.count_nonzero(actual != expected))
                    seed_rows.append(
                        {
                            "seed": int(seed),
                            "prefill": prefill,
                            "correct": mismatch_count == 0,
                            "mismatch_count": mismatch_count,
                            "expected_sha256": array_sha256(expected),
                            "actual_sha256": array_sha256(actual),
                            "mismatch_sample": mismatch_sample(actual, expected),
                        }
                    )
                row = {
                    "schema": "c3_p7_board_mismatch_diagnostic_v1",
                    "workload_id": args.workload,
                    "phase": phase,
                    "repeat": repeat,
                    "candidate_id": candidate_id,
                    "residence_mode": entry["residence_mode"],
                    "config_index": entry["config_index"],
                    "tir_sha256": binaries[candidate_id][1],
                    "preflight": preflight,
                    "seeds": seed_rows,
                    "correct": all(seed["correct"] for seed in seed_rows),
                    "performance_measurement": "not_collected",
                }
                rows.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                print(
                    "{} repeat={} {} mismatch_counts={}".format(
                        phase,
                        repeat,
                        candidate_id[:12],
                        [seed["mismatch_count"] for seed in seed_rows],
                    ),
                    flush=True,
                )

    after = ssh_preflight(contract, args.host)
    target_rows = [row for row in rows if row["candidate_id"] == args.candidate_id]
    sentinel_rows = [row for row in rows if row["candidate_id"] == args.sentinel_id]
    write_json(
        output / "summary.json",
        {
            "schema": "c3_p7_board_mismatch_diagnostic_summary_v1",
            "workload_id": args.workload,
            "candidate_id": args.candidate_id,
            "sentinel_id": args.sentinel_id,
            "target_repeat_count": args.target_repeats,
            "target_all_correct": all(row["correct"] for row in target_rows),
            "sentinels_all_correct": all(row["correct"] for row in sentinel_rows),
            "board_before": before,
            "board_after": after,
            "performance_measurement": "not_collected",
        },
    )


if __name__ == "__main__":
    main()
