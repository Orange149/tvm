#!/usr/bin/env python3
"""Run one non-spliceable clean-start FPGA session for the frozen R18 pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tvm
from tvm import rpc
import vta

from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
from run_vta_p7r132_y00_search_confirmation import (
    CleanStartBoard,
    EXPECTED_DEFAULT_RUNTIME_HASHES,
    allocate_reusable_buffers,
    profiled_reused_call,
    timed_reused_call,
    timing_orders,
)


SCHEMA = "c3_resnet18_board_pool_v1"
SEEDS = (0, 20250901, 20260910)
ROUNDS = 5
POOL_RPC_SESSION_TIMEOUT_SECONDS = 7200


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
        stream.flush()


def verify_flat_run(directory):
    directory = Path(directory).resolve()
    ledger = read_json(directory / "artifact_hashes.json")["artifacts"]
    for name, expected in ledger.items():
        if isinstance(expected, str) and sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: {}".format(directory / name))
    return ledger


def verify_inputs(board_pool_dir, cross_compile_dir):
    board_pool_dir = Path(board_pool_dir).resolve()
    cross_compile_dir = Path(cross_compile_dir).resolve()
    board_hashes = verify_flat_run(board_pool_dir)
    cross_hashes = verify_flat_run(cross_compile_dir)
    binary_hashes = read_json(cross_compile_dir / "artifact_hashes.json")["artifacts"].get("binaries", {})
    for name, expected in binary_hashes.items():
        if sha256_file(cross_compile_dir / "binaries" / name) != expected:
            raise ValueError("binary artifact hash mismatch: {}".format(name))
    candidates = read_jsonl(board_pool_dir / "candidates.jsonl")
    contract = read_json(board_pool_dir / "contract.json")
    cross = {row["candidate_id"]: row for row in read_jsonl(cross_compile_dir / "results.jsonl")}
    if [row["candidate_id"] for row in candidates] != contract["candidate_order"]:
        raise ValueError("board candidate order changed")
    for candidate in candidates:
        certificate = cross.get(candidate["candidate_id"])
        if certificate is None or certificate["status"] != "passed":
            raise ValueError("candidate lacks passing cross-compile certificate")
        binary = cross_compile_dir / "binaries" / (candidate["candidate_id"] + ".so")
        if certificate["binary"]["sha256"] != sha256_file(binary):
            raise ValueError("candidate binary differs from certificate")
    return candidates, cross, {"board_pool": board_hashes, "cross_compile": cross_hashes, "binaries": binary_hashes}


def profile_cost(rows):
    total = Counter()
    for row in rows:
        profile = row.get("runtime_profile_complete") or {}
        total["logical_load_bytes"] += int(profile.get("load_buffer_2d_bytes", 0))
        total["logical_store_bytes"] += int(profile.get("store_buffer_2d_bytes", 0))
        total["logical_dma_calls"] += int(profile.get("load_buffer_2d_calls", 0)) + int(profile.get("store_buffer_2d_calls", 0))
        total["fpga_kernel_invocations"] += int(profile.get("driver_run_calls", 1))
    return dict(total)


def summarize_workload_timing(rows, candidate_ids):
    by_id = defaultdict(list)
    for row in rows:
        by_id[row["candidate_id"]].append(float(row["latency_ms"]))
    if set(by_id) != set(candidate_ids) or any(len(values) != ROUNDS for values in by_id.values()):
        raise ValueError("oracle forbidden: workload timing pool is incomplete")
    stats = {
        candidate_id: {
            "values_ms": values,
            "median_ms": statistics.median(values),
            "q1_ms": float(np.percentile(values, 25)),
            "q3_ms": float(np.percentile(values, 75)),
        }
        for candidate_id, values in by_id.items()
    }
    oracle = min(stats, key=lambda candidate_id: (stats[candidate_id]["median_ms"], candidate_id))
    return {"candidate_stats": stats, "pool_oracle_candidate_id": oracle, "pool_oracle_latency_ms": stats[oracle]["median_ms"]}


def finalize(output):
    artifacts = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "artifact_hashes.json"}
    write_json(output / "artifact_hashes.json", {"artifacts": artifacts, "source_hashes": {str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve())}})


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable board output {}".format(output))
    candidates, cross, input_hashes = verify_inputs(args.board_pool_dir, args.cross_compile_dir)
    by_id = {row["candidate_id"]: row for row in candidates}
    output.mkdir(parents=True)
    w0_wall = time.time()
    w0_mono = time.monotonic()
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_current_board_observation",
        "candidate_order": [row["candidate_id"] for row in candidates],
        "seeds": list(SEEDS),
        "timing_rounds": ROUNDS,
        "input_hashes": input_hashes,
        "boundaries": {"W0": "host process begins clean-board preparation", "T0": "clean bitstream and fresh RPC ready", "T1": "complete correctness and timing pool committed"},
        "session_policy": "any power/RPC/execution interruption invalidates this whole session; no timing splice",
        "buffer_policy": "one output-data-weight allocation per workload; exact order is contractual",
        "board_contacted": False,
        "performance_labels_collected": False,
    }
    write_json(output / "contract.json", contract)
    write_json(output / "pre_observation_hashes.json", {"contract.json": sha256_file(output / "contract.json")})
    timeline_path = output / "timeline.jsonl"
    results_path = output / "results.jsonl"
    timeline_path.write_text("", encoding="utf-8")
    results_path.write_text("", encoding="utf-8")
    append_jsonl(
        timeline_path,
        {
            "event": "W0",
            "phase": "host_preflight",
            "wall_unix": w0_wall,
            "elapsed_from_W0_seconds": 0.0,
        },
    )
    correctness, timings = [], []
    try:
        env = vta.get_env()
        if env.TARGET != "axu5evb":
            raise RuntimeError("board execution requires VTA TARGET=axu5evb")
        board = CleanStartBoard(args.host, args.ssh_known_hosts)
        before = board.state()
        if before.get("FPGA") != "operating" or before.get("UDMABUF") != "201326592":
            raise RuntimeError("FPGA/u-dma-buf clean-start preflight failed")
        if before.get("STORAGE_ERRORS"):
            raise RuntimeError("storage errors present in current boot: {}".format(before["STORAGE_ERRORS"][-3:]))
        if board.runtime_hashes(args.default_runtime) != EXPECTED_DEFAULT_RUNTIME_HASHES:
            raise RuntimeError("default runtime hash mismatch")
        prior_rpc = board.rpc_state()
        if not prior_rpc.get("PID") or prior_rpc.get("CWD") != args.default_runtime:
            raise RuntimeError("expected default RPC/runtime is not running")
        board.stop_rpc(prior_rpc)
        reload_result = board.reload_frozen_bitstream()
        fresh_rpc = board.start_rpc(args.default_runtime, "p7r484_resnet18_board.log")
        remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
        inventory, functions = runtime_inventory(remote)
        t0_wall = time.time()
        t0_mono = time.monotonic()
        write_json(output / "clean_start.json", {"before": before, "stopped_rpc": prior_rpc, "bitstream": reload_result, "fresh_rpc": fresh_rpc, "rpc_inventory": inventory, "W0_wall_unix": w0_wall, "T0_wall_unix": t0_wall, "W0_to_T0_seconds": t0_mono - w0_mono})
        append_jsonl(
            timeline_path,
            {
                "event": "T0",
                "phase": "clean_board_ready",
                "wall_unix": t0_wall,
                "elapsed_from_W0_seconds": t0_mono - w0_mono,
                "elapsed_from_T0_seconds": 0.0,
            },
        )
        device = remote.ext_dev(0)
        modules = {}
        binary_dir = Path(args.cross_compile_dir).resolve() / "binaries"
        for position, candidate in enumerate(candidates, 1):
            modules[candidate["candidate_id"]] = load_ephemeral(remote, binary_dir / (candidate["candidate_id"] + ".so"))
            print("load {}/{}".format(position, len(candidates)), flush=True)
        workloads = {}
        for candidate in candidates:
            workloads.setdefault(candidate["workload_id"], candidate["identity"]["workload"])
        buffers = {workload_id: allocate_reusable_buffers(device, workload) for workload_id, workload in workloads.items()}
        correctness_path = output / "correctness.jsonl"
        correctness_path.write_text("", encoding="utf-8")
        for position, candidate in enumerate(candidates, 1):
            rows = []
            for seed in SEEDS:
                started = time.monotonic()
                row = profiled_reused_call(modules[candidate["candidate_id"]]["main"], buffers[candidate["workload_id"]], candidate["identity"]["workload"], seed, functions)
                row["host_wall_seconds"] = time.monotonic() - started
                rows.append(row)
                if row.get("status") == "execution_failed":
                    raise RuntimeError("RPC/device execution interruption at candidate {}".format(candidate["candidate_id"]))
                if not row.get("correct"):
                    break
            record = {
                "candidate_id": candidate["candidate_id"], "implementation_candidate_id": candidate["implementation_candidate_id"], "workload_id": candidate["workload_id"], "family_id": candidate["family_id"], "residence_mode": candidate["residence_mode"], "position": position,
                "status": "passed" if len(rows) == len(SEEDS) and all(row.get("correct") for row in rows) else "failed", "seeds": rows,
            }
            correctness.append(record)
            append_jsonl(correctness_path, record)
            append_jsonl(results_path, {"record_type": "correctness", **record})
            append_jsonl(
                timeline_path,
                {
                    "event": "candidate_correctness_complete",
                    "phase": "fpga_correctness",
                    "candidate_id": candidate["candidate_id"],
                    "workload_id": candidate["workload_id"],
                    "position": position,
                    "status": record["status"],
                    "fpga_invocations": len(rows),
                    "host_wall_seconds": sum(float(row["host_wall_seconds"]) for row in rows),
                    "elapsed_from_T0_seconds": time.monotonic() - t0_mono,
                },
            )
            print("correctness {}/{} {}".format(position, len(candidates), record["status"]), flush=True)
        correct_by_workload = {
            workload_id: [row["candidate_id"] for row in correctness if row["workload_id"] == workload_id and row["status"] == "passed"]
            for workload_id in workloads
        }
        if any(not values for values in correct_by_workload.values()):
            raise RuntimeError("at least one workload has no FPGA-correct candidate; timing forbidden")
        orders = {workload_id: timing_orders(values, rounds=ROUNDS) for workload_id, values in correct_by_workload.items()}
        write_json(output / "timing_orders.json", orders)
        timing_path = output / "timing.jsonl"
        timing_path.write_text("", encoding="utf-8")
        for round_index in range(ROUNDS):
            for workload_id in sorted(workloads):
                for position, candidate_id in enumerate(orders[workload_id][round_index]):
                    candidate = by_id[candidate_id]
                    started = time.monotonic()
                    row = timed_reused_call(modules[candidate_id], device, buffers[workload_id], candidate["identity"]["workload"], SEEDS[round_index % len(SEEDS)], functions)
                    row.update(candidate_id=candidate_id, workload_id=workload_id, round=round_index, position=position, host_wall_seconds=time.monotonic() - started)
                    timings.append(row)
                    append_jsonl(timing_path, row)
                    append_jsonl(results_path, {"record_type": "timing", **row})
                    append_jsonl(
                        timeline_path,
                        {
                            "event": "candidate_timing_complete",
                            "phase": "timing",
                            "candidate_id": candidate_id,
                            "workload_id": workload_id,
                            "round": round_index,
                            "position": position,
                            "host_wall_seconds": row["host_wall_seconds"],
                            "elapsed_from_T0_seconds": time.monotonic() - t0_mono,
                        },
                    )
            print("timing round {}/{}".format(round_index + 1, ROUNDS), flush=True)
        timing_summary = {}
        for workload_id, candidate_ids in correct_by_workload.items():
            workload_rows = [row for row in timings if row["workload_id"] == workload_id]
            timing_summary[workload_id] = summarize_workload_timing(workload_rows, candidate_ids)
        write_json(output / "timing_summary.json", timing_summary)
        t1_wall = time.time()
        t1_mono = time.monotonic()
        append_jsonl(
            timeline_path,
            {
                "event": "T1",
                "phase": "complete_pool_committed",
                "wall_unix": t1_wall,
                "elapsed_from_W0_seconds": t1_mono - w0_mono,
                "elapsed_from_T0_seconds": t1_mono - t0_mono,
            },
        )
        all_runtime_rows = [seed for row in correctness for seed in row["seeds"]] + timings
        summary = {
            "schema": SCHEMA,
            "status": "complete_non_spliced_board_pool",
            "boot_id": before["BOOT"],
            "candidate_count": len(candidates),
            "fpga_correct_count": sum(row["status"] == "passed" for row in correctness),
            "correctness_invocations": sum(len(row["seeds"]) for row in correctness),
            "timing_samples": len(timings),
            "workload_oracles": {workload_id: {"candidate_id": value["pool_oracle_candidate_id"], "latency_ms": value["pool_oracle_latency_ms"]} for workload_id, value in timing_summary.items()},
            "cost": {"W0_to_T1_seconds": t1_mono - w0_mono, "T0_to_T1_seconds": t1_mono - t0_mono, **profile_cost(all_runtime_rows)},
            "W0_wall_unix": w0_wall, "T0_wall_unix": t0_wall, "T1_wall_unix": t1_wall,
            "logical_dma_scope": "VTA runtime LOAD/STORE counters; not physical AXI traffic",
            "session_spliced": False,
        }
        write_json(output / "summary.json", summary)
        (output / "command.txt").write_text(" ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n", encoding="utf-8")
        finalize(output)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except Exception as error:
        invalid = {
            "schema": SCHEMA, "status": "invalid_entire_session_fail_closed", "exception_type": type(error).__name__, "message": (str(error) or type(error).__name__)[:4000], "traceback": traceback.format_exc()[-12000:],
            "W0_elapsed_seconds": time.monotonic() - w0_mono, "correctness_records": len(correctness), "timing_records": len(timings), "may_splice_with_other_session": False,
        }
        write_json(output / "invalid_session.json", invalid)
        write_json(
            output / "summary.json",
            {
                "schema": SCHEMA,
                "status": invalid["status"],
                "session_spliced": False,
                "candidate_count": len(candidates),
                "correctness_records": len(correctness),
                "timing_records": len(timings),
                "W0_elapsed_seconds": invalid["W0_elapsed_seconds"],
                "failure": {
                    "exception_type": invalid["exception_type"],
                    "message": invalid["message"],
                },
            },
        )
        append_jsonl(
            timeline_path,
            {
                "event": "session_invalidated",
                "phase": "failure",
                "exception_type": invalid["exception_type"],
                "message": invalid["message"],
                "elapsed_from_W0_seconds": invalid["W0_elapsed_seconds"],
                "may_splice_with_other_session": False,
            },
        )
        (output / "command.txt").write_text(" ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n", encoding="utf-8")
        finalize(output)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-pool-dir", required=True)
    parser.add_argument("--cross-compile-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument(
        "--session-timeout",
        type=int,
        default=POOL_RPC_SESSION_TIMEOUT_SECONDS,
        help="whole-session RPC lifetime; the 214-point non-spliced pool needs hours, not 120s",
    )
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
