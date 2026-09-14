#!/usr/bin/env python3
"""Run the frozen W04 tile-width correctness boundary probe."""

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

from qualify_vta_residency_fsim import reference_data_from_workload
from run_vta_p7_holdout_correctness import build_candidate, file_sha256, load_contract, ssh_preflight, write_json
from run_vta_w04_store_group_probe import run_one


def load_probe(path):
    path = Path(path).resolve()
    ledger = json.loads((path.parent / "artifact_hashes.json").read_text())
    if ledger["output_sha256"].get(path.name) != file_sha256(path):
        raise ValueError("width-probe contract hash mismatch")
    probe = json.loads(path.read_text())
    if probe.get("schema") != "c3_w04_width_probe_contract_v1" or probe.get("status") != "frozen_before_new_probe_labels":
        raise ValueError("unexpected or unfrozen width-probe contract")
    return probe


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
        raise FileExistsError("refusing to overwrite width-probe observations")
    probe = load_probe(args.probe_contract)
    p7_path = Path(probe["p7_contract"]["path"])
    if file_sha256(p7_path) != probe["p7_contract"]["sha256"]:
        raise ValueError("referenced P7 contract changed")
    p7 = load_contract(p7_path)
    candidates = {row["candidate_id"]: row for row in probe["candidates"]}
    env = vta.get_env()
    before = ssh_preflight(p7, args.host)
    remote = rpc.connect(args.host, args.port, session_timeout=120)
    device = remote.ext_dev(0)
    first = candidates[probe["board_order"][0]]
    data, weight, expected = reference_data_from_workload(first["workload"], probe["correctness_seeds"][0])
    buffers = (tvm.nd.array(data, device), tvm.nd.array(weight, device),
               tvm.nd.array(np.full(expected.shape, -113, dtype="int8"), device))
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix="c3_w04_width_") as temporary:
            binary_dir = Path(temporary)
            sentinel = probe["sentinel"]
            all_entries = [sentinel] + [candidates[cid] for cid in probe["board_order"]]
            modules = {}
            for entry in all_entries:
                binary, tir_hash = build_candidate(entry, env, binary_dir)
                remote.upload(str(binary))
                modules[entry["candidate_id"]] = (remote.load_module(binary.name), tir_hash)
            with results_path.open("x", encoding="utf-8") as stream:
                for position, candidate_id in enumerate(probe["board_order"], 1):
                    entry = candidates[candidate_id]
                    preflight = ssh_preflight(p7, args.host)
                    module, tir_hash = modules[candidate_id]
                    seed_rows = [run_one(module, buffers, entry, seed, device) for seed in probe["correctness_seeds"]]
                    sentinel_row = run_one(modules[sentinel["candidate_id"]][0], buffers, sentinel,
                                           probe["correctness_seeds"][0], device)
                    if not sentinel_row["correct"]:
                        raise RuntimeError("post-candidate sentinel failed")
                    row = {"schema": "c3_w04_width_probe_result_v1", "probe_position": position,
                           "candidate_id": candidate_id, "config_index": entry["config_index"],
                           "tile_w": entry["split_knobs"]["tile_w"],
                           "accumulator_vectors_per_residency_region": entry["accumulator_vectors_per_residency_region"],
                           "tir_sha256": tir_hash, "store_sites": entry["store_sites"],
                           "uop_kernel_explicit_push_counts": entry["uop_kernel_explicit_push_counts"],
                           "seed_results": seed_rows, "all_correct": all(seed["correct"] for seed in seed_rows),
                           "sentinel_seed0": sentinel_row, "preflight": preflight,
                           "performance_measurement": "not_collected"}
                    rows.append(row)
                    stream.write(json.dumps(row, sort_keys=True) + "\n"); stream.flush()
                    print("{}/{} config={} tile_w={} acc_vectors={} mismatches={} sentinel=exact".format(
                        position, len(probe["board_order"]), entry["config_index"], entry["split_knobs"]["tile_w"],
                        entry["accumulator_vectors_per_residency_region"], [seed["mismatch_count"] for seed in seed_rows]), flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"schema": "c3_w04_width_probe_failure_v1",
                   "completed_candidates": len(rows), "exception_type": type(error).__name__,
                   "message": str(error)[:4000], "traceback": traceback.format_exc()[-12000:]})
        raise
    after = ssh_preflight(p7, args.host)
    combined = [{"config_index": row["config_index"], "tile_w": row["tile_w"],
                 "accumulator_vectors": row["accumulator_vectors_per_residency_region"],
                 "outcome": "correct" if row["all_correct"] else "wrong_answer",
                 "evidence": "new_board_probe"} for row in rows]
    combined += [{"config_index": 139, "tile_w": 7, "accumulator_vectors": 1568,
                  "outcome": "wrong_answer", "evidence": "inherited_p7c1_three_repeats"},
                 {"config_index": 143, "tile_w": 14, "accumulator_vectors": 3136,
                  "outcome": "static_capacity_reject", "evidence": "contract"}]
    combined.sort(key=lambda row: row["tile_w"])
    write_json(output / "summary.json", {"schema": "c3_w04_width_probe_summary_v1", "status": "complete",
               "new_candidates": len(rows), "combined_width_sweep": combined,
               "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
               "board_before": before, "board_after": after, "performance_measurement": "not_collected"})


if __name__ == "__main__":
    main()
