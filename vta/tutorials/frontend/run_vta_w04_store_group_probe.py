#!/usr/bin/env python3
"""Run the frozen W04 output-store grouping correctness probe on a VTA board."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import traceback
from pathlib import Path

import numpy as np
from tvm import rpc
import tvm
import vta

from qualify_vta_residency_fsim import array_sha256, reference_data_from_workload
from run_vta_p7_holdout_correctness import (
    build_candidate,
    file_sha256,
    load_contract,
    ssh_preflight,
    write_json,
)


def load_probe(path):
    path = Path(path).resolve()
    ledger = json.loads((path.parent / "artifact_hashes.json").read_text())
    if ledger["output_sha256"].get(path.name) != file_sha256(path):
        raise ValueError("probe contract hash mismatch")
    probe = json.loads(path.read_text())
    if probe.get("schema") != "c3_w04_store_group_probe_contract_v1":
        raise ValueError("unexpected probe schema")
    if probe.get("status") != "frozen_before_new_probe_labels":
        raise ValueError("probe is not frozen")
    return probe


def run_one(module, buffers, entry, seed, device):
    data, weight, expected = reference_data_from_workload(entry["workload"], int(seed))
    remote_data, remote_weight, remote_output = buffers
    remote_data.copyfrom(data)
    remote_weight.copyfrom(weight)
    remote_output.copyfrom(np.full(expected.shape, -113, dtype="int8"))
    module["main"](remote_data, remote_weight, remote_output)
    actual = remote_output.numpy()
    mismatch_count = int(np.count_nonzero(actual != expected))
    return {
        "seed": int(seed),
        "correct": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "expected_sha256": array_sha256(expected),
        "actual_sha256": array_sha256(actual),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    if results_path.exists() or (output / "summary.json").exists() or (output / "failure.json").exists():
        raise FileExistsError("refusing to overwrite store-group probe observations")

    probe = load_probe(args.probe_contract)
    p7_path = Path(probe["p7_contract"]["path"])
    if file_sha256(p7_path) != probe["p7_contract"]["sha256"]:
        raise ValueError("referenced P7 contract changed")
    p7 = load_contract(p7_path)
    candidates = {row["candidate_id"]: row for row in probe["candidates"]}
    if set(probe["board_order"]) != {
        row["candidate_id"] for row in probe["candidates"] if row["label_source"] == "new_board_probe"
    }:
        raise ValueError("board order does not cover exactly the new candidates")

    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise RuntimeError("VTA TARGET must be axu5evb")
    before = ssh_preflight(p7, args.host)
    remote = rpc.connect(args.host, args.port, session_timeout=120)
    device = remote.ext_dev(0)
    first = candidates[probe["board_order"][0]]
    data, weight, expected = reference_data_from_workload(first["workload"], probe["correctness_seeds"][0])
    buffers = (
        tvm.nd.array(data, device),
        tvm.nd.array(weight, device),
        tvm.nd.array(np.full(expected.shape, -113, dtype="int8"), device),
    )

    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix="c3_w04_store_group_") as temporary:
            binary_dir = Path(temporary)
            all_entries = [probe["sentinel"]] + [candidates[cid] for cid in probe["board_order"]]
            modules = {}
            for entry in all_entries:
                binary, tir_hash = build_candidate(entry, env, binary_dir)
                remote.upload(str(binary))
                modules[entry["candidate_id"]] = (remote.load_module(binary.name), tir_hash)

            sentinel = probe["sentinel"]
            with results_path.open("x", encoding="utf-8") as stream:
                for position, candidate_id in enumerate(probe["board_order"], 1):
                    entry = candidates[candidate_id]
                    preflight = ssh_preflight(p7, args.host)
                    module, tir_hash = modules[candidate_id]
                    seed_rows = [
                        run_one(module, buffers, entry, seed, device)
                        for seed in probe["correctness_seeds"]
                    ]
                    sentinel_module, _ = modules[sentinel["candidate_id"]]
                    sentinel_row = run_one(
                        sentinel_module,
                        buffers,
                        sentinel,
                        probe["correctness_seeds"][0],
                        device,
                    )
                    if not sentinel_row["correct"]:
                        raise RuntimeError("post-candidate sentinel failed")
                    row = {
                        "schema": "c3_w04_store_group_probe_result_v1",
                        "probe_position": position,
                        "candidate_id": candidate_id,
                        "config_index": entry["config_index"],
                        "tile_co": entry["split_knobs"]["tile_co"],
                        "tir_sha256": tir_hash,
                        "store_sites": entry["store_sites"],
                        "uop_kernel_explicit_push_counts": entry[
                            "uop_kernel_explicit_push_counts"
                        ],
                        "seed_results": seed_rows,
                        "all_correct": all(row["correct"] for row in seed_rows),
                        "sentinel_seed0": sentinel_row,
                        "preflight": preflight,
                        "performance_measurement": "not_collected",
                    }
                    rows.append(row)
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    print(
                        "{}/{} config={} tile_co={} mismatches={} sentinel=exact".format(
                            position,
                            len(probe["board_order"]),
                            entry["config_index"],
                            entry["split_knobs"]["tile_co"],
                            [seed["mismatch_count"] for seed in seed_rows],
                        ),
                        flush=True,
                    )
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "schema": "c3_w04_store_group_probe_failure_v1",
                "completed_candidates": len(rows),
                "exception_type": type(error).__name__,
                "message": str(error)[:4000],
                "traceback": traceback.format_exc()[-12000:],
            },
        )
        raise

    after = ssh_preflight(p7, args.host)
    inherited = probe["inherited_failure"]
    combined = [
        {
            "config_index": row["config_index"],
            "tile_co": row["tile_co"],
            "all_correct": row["all_correct"],
            "evidence": "new_board_probe",
        }
        for row in rows
    ]
    inherited_entry = next(
        row for row in probe["candidates"] if row["config_index"] == inherited["config_index"]
    )
    combined.append(
        {
            "config_index": inherited["config_index"],
            "tile_co": inherited_entry["split_knobs"]["tile_co"],
            "all_correct": inherited["target_all_correct"],
            "evidence": "inherited_p7c1_three_repeats",
        }
    )
    combined.sort(key=lambda row: row["tile_co"])
    write_json(
        output / "summary.json",
        {
            "schema": "c3_w04_store_group_probe_summary_v1",
            "status": "complete",
            "new_candidates": len(rows),
            "new_correct_candidates": sum(row["all_correct"] for row in rows),
            "new_wrong_candidates": sum(not row["all_correct"] for row in rows),
            "combined_tile_co_sweep": combined,
            "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
            "board_before": before,
            "board_after": after,
            "performance_measurement": "not_collected",
        },
    )


if __name__ == "__main__":
    main()
