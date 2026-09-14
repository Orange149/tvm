#!/usr/bin/env python3
"""Board-qualify and time an already cross-built Relay residency pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import time

import numpy as np
import tvm
from tvm import rpc
from tvm.contrib import graph_executor

from qualify_vta_residency_fsim import array_sha256, reference_data_from_workload
from run_vta_p7r132_y00_search_confirmation import CleanStartBoard


SEEDS = (0, 20250901, 20260910)
ROUNDS = 7
DEFAULT_RUNTIME = "/var/volatile/vta_c3_ram/runtime"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_workload(candidates_path, candidate_id):
    for line in Path(candidates_path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["candidate_id"] == candidate_id:
            return row["identity"]["workload"]
    raise KeyError(candidate_id)


def runtime_functions(remote):
    names = ("vta.runtime.profiler_clear", "vta.runtime.profiler_status")
    return {name: remote.get_function(name) for name in names}


def load_graph_module(remote, pair_dir, mode, device, temporary):
    local = Path(pair_dir) / mode
    remote_name = mode + "_relay_graphlib.so"
    upload_path = Path(temporary) / remote_name
    shutil.copyfile(local / "graphlib.so", upload_path)
    remote.upload(str(upload_path))
    loaded = remote.load_module(remote_name)
    executor = graph_executor.create(
        (local / "graph.json").read_text(encoding="utf-8"),
        loaded,
        [device, remote.cpu(0)],
    )
    executor.load_params((local / "params.bin").read_bytes())
    return executor


def invoke(executor, workload, seed, functions, timed=False, device=None):
    data, weight, expected = reference_data_from_workload(workload, seed)
    executor.set_input("data", data)
    executor.set_input("weight", weight)
    functions["vta.runtime.profiler_clear"]()
    started = time.perf_counter()
    if timed:
        result = executor.module.time_evaluator("run", device, number=1, repeat=1)()
        latency_ms = float(result.mean * 1000.0)
    else:
        executor.run()
        latency_ms = None
    host_wall_ms = (time.perf_counter() - started) * 1000.0
    actual = executor.get_output(0).numpy()
    mismatch = int(np.count_nonzero(actual != expected))
    profile = json.loads(functions["vta.runtime.profiler_status"]())
    return {
        "seed": int(seed),
        "correct": mismatch == 0,
        "mismatch_count": mismatch,
        "expected_sha256": array_sha256(expected),
        "actual_sha256": array_sha256(actual),
        "latency_ms": latency_ms,
        "host_wall_ms": host_wall_ms,
        "runtime_profile_complete": profile,
    }


def run(args):
    pair_dir = Path(args.pair_dir)
    pair = read_json(pair_dir / "summary.json")
    if pair["status"] != "relay_cross_build_dispatch_verified":
        raise ValueError("pair build is not verified")
    builds = pair["builds"]
    modes = [row["public_mode"] for row in builds]
    if modes[0] != "original" or modes[1] not in ("input_stationary", "weight_resident_barrier"):
        raise ValueError("expected original/residency build pair")
    workload = load_workload(args.candidates, builds[0]["candidate_id"])
    if workload != load_workload(args.candidates, builds[1]["candidate_id"]):
        raise ValueError("pair workload mismatch")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    write_json(output / "input_pair_summary.json", pair)

    board = CleanStartBoard(args.host, args.ssh_known_hosts)
    before = board.state()
    if before.get("FPGA") != "operating" or before.get("UDMABUF") != "201326592":
        raise RuntimeError("FPGA/u-dma-buf preflight failed")
    if before.get("STORAGE_ERRORS"):
        raise RuntimeError("storage errors present before run")
    prior_rpc = board.rpc_state()
    if prior_rpc.get("CWD") != args.default_runtime:
        raise RuntimeError("unexpected RPC directory: " + repr(prior_rpc))
    board.stop_rpc(prior_rpc)
    reload_result = board.reload_frozen_bitstream()
    fresh_rpc = board.start_rpc(args.default_runtime, "p7r228_relay_dispatch.log")
    clean_start = {
        "before": before,
        "stopped_rpc": prior_rpc,
        "bitstream_reload": reload_result,
        "fresh_rpc": fresh_rpc,
    }
    write_json(output / "clean_board_start.json", clean_start)

    remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
    device = remote.ext_dev(0)
    functions = runtime_functions(remote)
    correctness = []
    timings = []
    with tempfile.TemporaryDirectory(prefix="c3_relay_dispatch_") as temporary:
        executors = {
            mode: load_graph_module(remote, pair_dir, mode, device, temporary) for mode in modes
        }
        for mode in modes:
            for seed in SEEDS:
                row = invoke(executors[mode], workload, seed, functions)
                row["public_mode"] = mode
                correctness.append(row)
                if not row["correct"]:
                    write_json(output / "correctness.json", correctness)
                    raise RuntimeError("Relay board correctness failed for " + mode)
        for round_index in range(ROUNDS):
            order = modes if round_index % 2 == 0 else list(reversed(modes))
            for position, mode in enumerate(order):
                row = invoke(executors[mode], workload, 0, functions, timed=True, device=device)
                if not row["correct"]:
                    raise RuntimeError("timed Relay board invocation failed for " + mode)
                row.update(round=round_index, position=position, public_mode=mode)
                timings.append(row)
            print("timing round {}/{}".format(round_index + 1, ROUNDS), flush=True)

    write_json(output / "correctness.json", correctness)
    (output / "timing.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in timings), encoding="utf-8"
    )
    medians = {
        mode: statistics.median(
            row["latency_ms"] for row in timings if row["public_mode"] == mode
        )
        for mode in modes
    }
    residency = modes[1]
    paired_wins = sum(
        1
        for round_index in range(ROUNDS)
        if next(row["latency_ms"] for row in timings if row["round"] == round_index and row["public_mode"] == residency)
        < next(row["latency_ms"] for row in timings if row["round"] == round_index and row["public_mode"] == "original")
    )
    after = board.state()
    if after.get("STORAGE_ERRORS"):
        raise RuntimeError("storage errors appeared during run")
    summary = {
        "schema": "c3_vta_relay_residency_dispatch_board_pair_v1",
        "status": "completed_relay_dispatch_correctness_and_paired_timing",
        "boot_id": after.get("BOOT"),
        "modes": modes,
        "same_workload_and_config": pair["same_workload_and_config"],
        "all_correctness_calls_correct": all(row["correct"] for row in correctness),
        "correctness_calls": len(correctness),
        "all_timing_calls_correct": all(row["correct"] for row in timings),
        "timing_calls": len(timings),
        "median_latency_ms": medians,
        "residency_speedup_percent": (medians["original"] / medians[residency] - 1.0) * 100.0,
        "residency_paired_wins": paired_wins,
        "paired_rounds": ROUNDS,
        "board_after": after,
        "claim_boundary": (
            "Minimal packed Relay operator integration; not a full ResNet50 stage or end-to-end model"
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    artifacts = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", {"artifacts": artifacts})
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=180)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default=DEFAULT_RUNTIME)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
