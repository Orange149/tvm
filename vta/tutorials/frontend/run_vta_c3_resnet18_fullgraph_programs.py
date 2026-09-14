#!/usr/bin/env python3
"""Validate and time frozen policy-selected ResNet18 programs in one clean session."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
from tvm import rpc

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from run_vta_p7r132_y00_search_confirmation import (
    CleanStartBoard,
    EXPECTED_DEFAULT_RUNTIME_HASHES,
)
from run_vta_resnet50_fused_program_pareto_board import (
    assert_board_state,
    invoke_model,
    load_executor,
    model_input_for_seed,
    model_mismatch_count,
)
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


SEEDS = (0, 20250901, 20260910)
ROUNDS = 7


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def balanced_orders(ids, rounds):
    ids = list(ids)
    output = []
    for round_index in range(rounds):
        order = ids[round_index % len(ids) :] + ids[: round_index % len(ids)]
        if round_index % 2:
            order = list(reversed(order))
        output.append(order)
    return output


def profile_cost(rows):
    total = Counter()
    for row in rows:
        profile = row.get("runtime_profile_complete") or {}
        total["logical_load_bytes"] += int(profile.get("load_buffer_2d_bytes", 0))
        total["logical_store_bytes"] += int(profile.get("store_buffer_2d_bytes", 0))
        total["logical_dma_calls"] += int(profile.get("load_buffer_2d_calls", 0))
        total["logical_dma_calls"] += int(profile.get("store_buffer_2d_calls", 0))
        total["fpga_kernel_invocations"] += int(profile.get("driver_run_calls", 0))
    return dict(total)


def finish(output):
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_sha256": sha256(Path(__file__).resolve()),
        },
    )


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    build_dir = Path(args.build_dir).resolve()
    verify_artifacts_compatible(build_dir)
    built = read_json(build_dir / "summary.json")
    if built.get("status") != "selected_fullgraphs_built_once_before_board":
        raise RuntimeError("selected full-graph builds are incomplete")
    program_rows = built["programs"]
    program_ids = ["stock_reference"] + [row["program_id"] for row in program_rows]
    if len(program_ids) != len(set(program_ids)):
        raise ValueError("duplicate unique full-graph program ID")
    local_dirs = {"stock_reference": build_dir / "stock_reference"}
    for row in program_rows:
        local_dirs[row["program_id"]] = build_dir / row["relative_dir"]
    for program_id, local in local_dirs.items():
        if not all((local / name).is_file() for name in ("graph.json", "params.bin", "graphlib.so")):
            raise RuntimeError("incomplete build for {}".format(program_id))

    output.mkdir(parents=True)
    w0_wall = time.time()
    w0_mono = time.monotonic()
    contract = {
        "schema": "c3_resnet18_selected_fullgraph_board_contract_v1",
        "status": "frozen_before_current_board_observation",
        "build_manifest_sha256": sha256(build_dir / "artifact_hashes.json"),
        "program_ids": program_ids,
        "policy_to_program": built["policy_to_program"],
        "correctness_seeds": list(SEEDS),
        "timing_rounds": ROUNDS,
        "timing_orders": balanced_orders(program_ids, ROUNDS),
        "boundaries": {
            "W0": "host process starts before board preflight/bitstream/RPC preparation",
            "T0": "clean frozen bitstream and fresh default RPC are ready",
            "T1": "all programs pass three inputs and finish seven balanced rounds",
        },
        "session_policy": "any RPC, device or power interruption invalidates the entire session; clocks cannot be spliced",
        "correctness_scope": "three deterministic random inputs, every graph output, exact equality to stock; not ImageNet accuracy",
        "board_contacted": False,
        "performance_labels_collected": False,
    }
    write_json(output / "contract.json", contract)
    write_json(output / "pre_observation_hashes.json", {"contract.json": sha256(output / "contract.json")})
    invocation_path = output / "invocations.jsonl"
    invocation_path.write_text("", encoding="utf-8")
    timeline_path = output / "timeline.jsonl"
    timeline_path.write_text("", encoding="utf-8")
    results_path = output / "results.jsonl"
    results_path.write_text("", encoding="utf-8")
    append_jsonl(
        timeline_path,
        {"phase": "contract_frozen", "W0_elapsed_seconds": time.monotonic() - w0_mono},
    )
    correctness, timings = [], []
    remote = None
    try:
        board = CleanStartBoard(args.host, args.ssh_known_hosts)
        before = board.state()
        assert_board_state(before)
        if board.runtime_hashes(args.default_runtime) != EXPECTED_DEFAULT_RUNTIME_HASHES:
            raise RuntimeError("default runtime hash mismatch")
        prior = board.rpc_state()
        if not prior.get("PID") or prior.get("CWD") != args.default_runtime:
            raise RuntimeError("expected healthy default RPC before clean start")
        board.stop_rpc(prior)
        bitstream = board.reload_frozen_bitstream()
        fresh = board.start_rpc(args.default_runtime, "p7r_resnet18_selected_fullgraph.log")
        t0_wall = time.time()
        t0_mono = time.monotonic()
        append_jsonl(
            timeline_path,
            {
                "phase": "T0_clean_bitstream_and_rpc_ready",
                "W0_elapsed_seconds": t0_mono - w0_mono,
            },
        )
        write_json(
            output / "clean_start.json",
            {
                "before": before,
                "stopped_rpc": prior,
                "bitstream": bitstream,
                "fresh_rpc": fresh,
                "W0_wall_unix": w0_wall,
                "T0_wall_unix": t0_wall,
                "W0_to_T0_seconds": t0_mono - w0_mono,
            },
        )
        remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
        device = remote.ext_dev(0)
        contexts = [device, remote.cpu(0)]
        functions = {
            "clear": remote.get_function("vta.runtime.profiler_clear"),
            "status": remote.get_function("vta.runtime.profiler_status"),
        }
        input_spec = {
            "shape": [1, 3, 224, 224],
            "uniform_range": [0.0, 1.0],
            "all_outputs": True,
        }
        with tempfile.TemporaryDirectory(prefix="c3_r18_fullgraph_") as temporary:
            executors = {
                program_id: load_executor(
                    remote,
                    local_dirs[program_id],
                    "r18_{}_{:02d}.so".format(program_id[:20], index),
                    contexts,
                    temporary,
                )
                for index, program_id in enumerate(program_ids)
            }
            for seed_index, seed in enumerate(SEEDS):
                order = balanced_orders(program_ids, len(SEEDS))[seed_index]
                outputs = {}
                seed_rows = []
                data = model_input_for_seed(seed, input_spec)
                for position, program_id in enumerate(order):
                    actual, row = invoke_model(
                        executors[program_id], data, functions, device, False, True
                    )
                    outputs[program_id] = actual
                    row.update(
                        seed=int(seed), position=position, program_id=program_id
                    )
                    correctness.append(row)
                    seed_rows.append(row)
                    append_jsonl(
                        invocation_path,
                        {"phase": "correctness", **row},
                    )
                    append_jsonl(
                        results_path,
                        {"record_type": "correctness_invocation", **row},
                    )
                for row in seed_rows:
                    mismatch = model_mismatch_count(
                        outputs["stock_reference"], outputs[row["program_id"]], True
                    )
                    row["stock_mismatch_count"] = mismatch
                    row["stock_equal"] = mismatch == 0
                if any(not row["stock_equal"] for row in seed_rows):
                    raise RuntimeError("selected full-graph output mismatch")
                if any(row["output_nonzero"] == 0 for row in seed_rows):
                    raise RuntimeError("selected full-graph produced all-zero output")

            orders = balanced_orders(program_ids, ROUNDS)
            write_json(output / "timing_orders.json", orders)
            data = model_input_for_seed(0, input_spec)
            for round_index, order in enumerate(orders):
                round_outputs = {}
                round_rows = []
                for position, program_id in enumerate(order):
                    actual, row = invoke_model(
                        executors[program_id], data, functions, device, True, True
                    )
                    round_outputs[program_id] = actual
                    row.update(
                        round=round_index, position=position, program_id=program_id
                    )
                    timings.append(row)
                    round_rows.append(row)
                    append_jsonl(invocation_path, {"phase": "timing", **row})
                    append_jsonl(
                        results_path,
                        {"record_type": "timing_invocation", **row},
                    )
                for row in round_rows:
                    mismatch = model_mismatch_count(
                        round_outputs["stock_reference"],
                        round_outputs[row["program_id"]],
                        True,
                    )
                    row["stock_mismatch_count"] = mismatch
                    row["stock_equal"] = mismatch == 0
                if any(not row["stock_equal"] for row in round_rows):
                    raise RuntimeError("timed selected full-graph output mismatch")
                print("fullgraph round {}/{}".format(round_index + 1, ROUNDS), flush=True)

        (output / "correctness.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in correctness),
            encoding="utf-8",
        )
        (output / "timing.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in timings),
            encoding="utf-8",
        )
        statistics_by_program = {}
        for program_id in program_ids:
            values = [
                float(row["latency_ms"])
                for row in timings
                if row["program_id"] == program_id
            ]
            if len(values) != ROUNDS:
                raise RuntimeError("incomplete timing values for {}".format(program_id))
            statistics_by_program[program_id] = {
                "values_ms": values,
                "median_ms": statistics.median(values),
                "q1_ms": float(np.percentile(values, 25)),
                "q3_ms": float(np.percentile(values, 75)),
            }
        policy_results = {
            policy: {
                "program_id": program_id,
                **statistics_by_program[program_id],
            }
            for policy, program_id in built["policy_to_program"].items()
        }
        t1_wall = time.time()
        t1_mono = time.monotonic()
        append_jsonl(
            timeline_path,
            {
                "phase": "T1_all_correctness_and_timing_complete",
                "W0_elapsed_seconds": t1_mono - w0_mono,
                "T0_elapsed_seconds": t1_mono - t0_mono,
            },
        )
        summary = {
            "schema": "c3_resnet18_selected_fullgraph_board_result_v1",
            "status": "complete_non_spliced_selected_fullgraph_comparison",
            "boot_id": before["BOOT"],
            "unique_program_count": len(program_ids),
            "policy_count": len(built["policy_to_program"]),
            "correctness_inputs": len(SEEDS),
            "all_outputs_equal": True,
            "timing_rounds": ROUNDS,
            "program_statistics": statistics_by_program,
            "policy_results": policy_results,
            "stock_reference": statistics_by_program["stock_reference"],
            "cost": {
                "W0_to_T1_seconds": t1_mono - w0_mono,
                "T0_to_T1_seconds": t1_mono - t0_mono,
                **profile_cost(correctness + timings),
            },
            "W0_wall_unix": w0_wall,
            "T0_wall_unix": t0_wall,
            "T1_wall_unix": t1_wall,
            "session_spliced": False,
            "logical_dma_scope": "VTA runtime logical LOAD/STORE; not physical AXI traffic",
            "accuracy_scope": "deterministic random-input schedule equivalence; not ImageNet accuracy",
        }
        write_json(output / "summary.json", summary)
        for policy, value in policy_results.items():
            append_jsonl(
                results_path,
                {"record_type": "policy_summary", "policy": policy, **value},
            )
        (output / "command.txt").write_text(
            " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
            encoding="utf-8",
        )
        finish(output)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except Exception as error:
        invalid = {
            "schema": "c3_resnet18_selected_fullgraph_board_result_v1",
            "status": "invalid_entire_session_fail_closed",
            "exception_type": type(error).__name__,
            "message": (str(error) or type(error).__name__)[:4000],
            "traceback": traceback.format_exc()[-12000:],
            "W0_elapsed_seconds": time.monotonic() - w0_mono,
            "correctness_calls_completed": len(correctness),
            "timing_calls_completed": len(timings),
            "may_splice_with_other_session": False,
        }
        write_json(
            output / "invalid_session.json",
            invalid,
        )
        write_json(
            output / "summary.json",
            {
                "schema": invalid["schema"],
                "status": invalid["status"],
                "session_spliced": False,
                "program_count": len(program_ids),
                "correctness_calls_completed": len(correctness),
                "timing_calls_completed": len(timings),
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
                "phase": "session_invalidated",
                "W0_elapsed_seconds": invalid["W0_elapsed_seconds"],
                "exception_type": invalid["exception_type"],
                "message": invalid["message"],
                "may_splice_with_other_session": False,
            },
        )
        (output / "command.txt").write_text(
            " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
            encoding="utf-8",
        )
        finish(output)
        raise
    finally:
        if remote is not None:
            del remote
            gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
