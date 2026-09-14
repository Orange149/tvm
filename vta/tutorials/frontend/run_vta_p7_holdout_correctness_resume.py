#!/usr/bin/env python3
"""Resume an unobserved suffix of a P7 correctness batch with reused RPC buffers."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import tvm
from tvm import rpc
import vta

from qualify_vta_residency_fsim import array_sha256, reference_data_from_workload
from run_vta_p7_holdout_correctness import (
    build_candidate,
    load_contract,
    ssh_preflight,
    validate_workload,
    write_json,
)


RESULT_SCHEMA = "c3_p7_holdout_correctness_resume_candidate_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--start-position", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "correctness.jsonl"
    if results_path.exists() or (output / "summary.json").exists() or (output / "failure.json").exists():
        raise FileExistsError("refusing to overwrite P7 resume observations")

    contract = load_contract(args.contract)
    entries, order = validate_workload(contract, args.workload)
    if args.start_position < 1 or args.start_position > len(order):
        raise ValueError("start position is outside the frozen correctness order")
    selected = list(enumerate(order[args.start_position - 1 :], args.start_position))
    seeds = [int(seed) for seed in contract["measurement"]["correctness_seeds"]]

    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise RuntimeError("VTA TARGET must be axu5evb")
    before = ssh_preflight(contract, args.host)
    remote = rpc.connect(args.host, args.port, session_timeout=120)
    device = remote.ext_dev(0)

    first_entry = entries[selected[0][1]]
    first_data, first_weight, first_expected = reference_data_from_workload(
        first_entry["workload"], seeds[0]
    )
    remote_data = tvm.nd.array(first_data, device)
    remote_weight = tvm.nd.array(first_weight, device)
    remote_output = tvm.nd.array(
        np.full(first_expected.shape, -113, dtype="int8"), device
    )

    rows = []
    batch_start = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="c3_p7_correctness_resume_") as temporary:
            binary_dir = Path(temporary)
            with results_path.open("x", encoding="utf-8") as stream:
                for position, candidate_id in selected:
                    entry = entries[candidate_id]
                    preflight = ssh_preflight(contract, args.host)
                    started = time.monotonic()
                    build_start = time.monotonic()
                    binary, tir_hash = build_candidate(entry, env, binary_dir)
                    build_seconds = time.monotonic() - build_start
                    upload_start = time.monotonic()
                    remote.upload(str(binary))
                    module = remote.load_module(binary.name)
                    upload_seconds = time.monotonic() - upload_start
                    seed_rows = []
                    for seed in seeds:
                        data, weight, expected = reference_data_from_workload(entry["workload"], seed)
                        if data.shape != first_data.shape or weight.shape != first_weight.shape:
                            raise RuntimeError("workload shape changed within the frozen candidate pool")
                        remote_data.copyfrom(data)
                        remote_weight.copyfrom(weight)
                        remote_output.copyfrom(np.full(expected.shape, -113, dtype="int8"))
                        module["main"](remote_data, remote_weight, remote_output)
                        actual = remote_output.numpy()
                        mismatch_count = int(np.count_nonzero(actual != expected))
                        seed_rows.append(
                            {
                                "seed": seed,
                                "correct": mismatch_count == 0,
                                "mismatch_count": mismatch_count,
                                "expected_sha256": array_sha256(expected),
                                "actual_sha256": array_sha256(actual),
                            }
                        )
                        if mismatch_count:
                            raise AssertionError(
                                "wrong answer: {} seed {}".format(candidate_id, seed)
                            )
                    row = {
                        "schema": RESULT_SCHEMA,
                        "workload_id": args.workload,
                        "correctness_position": position,
                        "candidate_id": candidate_id,
                        "residence_mode": entry["residence_mode"],
                        "config_index": entry["config_index"],
                        "tir_sha256": tir_hash,
                        "build_seconds": build_seconds,
                        "upload_load_seconds": upload_seconds,
                        "candidate_wall_seconds": time.monotonic() - started,
                        "preflight": preflight,
                        "seeds": seed_rows,
                        "overall_status": "passed",
                        "performance_measurement": "not_collected",
                        "remote_buffer_policy": "one_shape_compatible_set_reused",
                    }
                    rows.append(row)
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    print(
                        "{}/{} {} mode={} config={} exact=3/3".format(
                            position,
                            len(order),
                            candidate_id[:12],
                            entry["residence_mode"],
                            entry["config_index"],
                        ),
                        flush=True,
                    )
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "schema": "c3_p7_holdout_correctness_resume_failure_v1",
                "workload_id": args.workload,
                "start_position": args.start_position,
                "completed_candidates": len(rows),
                "status": "failed_stop_batch",
                "exception_type": type(error).__name__,
                "message": str(error)[:4000],
                "traceback": traceback.format_exc()[-12000:],
            },
        )
        raise

    after = ssh_preflight(contract, args.host)
    digest = hashlib.sha256(results_path.read_bytes()).hexdigest()
    write_json(
        output / "summary.json",
        {
            "schema": "c3_p7_holdout_correctness_resume_summary_v1",
            "status": "passed",
            "workload_id": args.workload,
            "start_position": args.start_position,
            "end_position": selected[-1][0],
            "candidate_count": len(rows),
            "seed_checks": len(rows) * len(seeds),
            "failed_candidates": 0,
            "correctness_sha256": digest,
            "batch_wall_seconds": time.monotonic() - batch_start,
            "board_before": before,
            "board_after": after,
            "performance_measurement": "not_collected",
            "remote_buffer_policy": "one_shape_compatible_set_reused",
            "next_step": "merge only with the immutable prefix from the failed infrastructure run",
        },
    )
    print(
        "P7 {} correctness resume passed: {} candidates, {} exact seed checks".format(
            args.workload, len(rows), len(rows) * len(seeds)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
