#!/usr/bin/env python3
"""Clean-wave timing screen for remaining YOLO-224 Input/mode11 pairs.

The board pool and same-Tile pairs are frozen before this runner observes any
latency.  Each program gets a fresh RPC and bitstream reload, three correctness
calls, three explicit warmups, and six three-seed timed rounds.  This is a
single-boot opportunity screen; it does not by itself authorize graph routing.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import shlex
import sys
import time
import traceback
from pathlib import Path

from tvm import rpc

import run_vta_c3_reuse_board as reuse_board
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_fpga_correctness_waves import append, read, seal, sha256_file, write
from run_vta_c3_holdout_board_boot_v2 import validate_profile
from run_vta_c3_resnet18_board_pool import verify_inputs
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
from run_vta_p7r132_y00_search_confirmation import allocate_reusable_buffers


SCHEMA = "c3_yolo224_remaining_full_fit_clean_timing_v1"
SEEDS = (0, 20250901, 20260910)
EVALUATION_ROUNDS = 6
ORDER_SEED = 20260927
TIMED_MODES = frozenset({"input_stationary", "input_weight_resident_transition_safe"})


def verify_flat(directory):
    directory = Path(directory).resolve()
    ledger = read(directory / "artifact_hashes.json")
    for name, expected in ledger.get("artifacts", {}).items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("artifact hash mismatch: {}".format(path))
    return sha256_file(directory / "artifact_hashes.json")


def connect(host, port, timeout):
    for _ in range(30):
        try:
            return rpc.connect(host, port, session_timeout=timeout)
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("RPC unavailable after clean timing start")


def frozen_pair_order(candidates, task_records):
    by_program = {row["program_id"]: row for row in candidates}
    pairs = []
    for task_id, task in sorted(task_records.items()):
        for pair in task["same_tile_input_mode11_pairs"]:
            record = dict(pair)
            record["execution_task_id"] = task_id
            pairs.append(record)
    if len(pairs) != 12:
        raise ValueError("expected twelve frozen same-Tile pairs")
    random.Random(ORDER_SEED).shuffle(pairs)
    order = []
    for index, pair in enumerate(pairs):
        programs = [pair["input_program_id"], pair["mode11_program_id"]]
        if index % 2:
            programs.reverse()
        for program_id in programs:
            candidate = by_program.get(program_id)
            if candidate is None or candidate["residence_mode"] not in TIMED_MODES:
                raise ValueError("frozen pair references an unavailable timing program")
            order.append(program_id)
    if len(order) != 24 or len(set(order)) != 24:
        raise ValueError("pair timing order does not cover 24 distinct programs")
    return pairs, order


def run(args):
    pool_dir = Path(args.board_pool_dir).resolve()
    cross_dir = Path(args.cross_compile_dir).resolve()
    correctness_dir = Path(args.correctness_dir).resolve()
    helper_dir = Path(args.helper_build_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))

    candidates, _, pool_hashes = verify_inputs(pool_dir, cross_dir)
    by_program = {row["program_id"]: row for row in candidates}
    if len(candidates) != 34 or len(by_program) != 34:
        raise ValueError("requires the frozen 34-program remaining-task pool")
    task_records = read(pool_dir / "task_records.json")
    pairs, order = frozen_pair_order(candidates, task_records)

    correctness_manifest = verify_flat(correctness_dir)
    correctness = read(correctness_dir / "summary.json")
    if (correctness.get("status") != "FPGA_CORRECTNESS_QUALIFICATION_COMPLETE"
            or correctness.get("candidate_count") != 34
            or correctness.get("fpga_correct_count") != 34
            or correctness.get("latency_values_collected") is not False):
        raise ValueError("requires complete latency-free 34/34 FPGA correctness")

    helper_manifest = verify_flat(helper_dir)
    helper_contract = read(helper_dir / "contract.json")
    helper = Path(helper_contract["binary"])
    if sha256_file(helper) != helper_contract["binary_sha256"]:
        raise ValueError("timing helper changed")
    runtime_contract = read(args.runtime_contract)
    runtime_paths = runtime_contract["runtime_paths"]
    runtime_hashes = runtime_contract["runtime_sha256"]
    for name, path in runtime_paths.items():
        if sha256_file(path) != runtime_hashes[name]:
            raise ValueError("runtime changed: {}".format(name))

    output.mkdir(parents=True)
    for name in ("calls.jsonl", "qualification.jsonl", "warmups.jsonl",
                 "evaluation_timing.jsonl", "waves.jsonl"):
        (output / name).write_text("", encoding="utf-8")
    contract = {
        "schema": SCHEMA,
        "status": "FROZEN_BEFORE_FIRST_REMAINING_TASK_LATENCY",
        "input_resolution": 224,
        "task_count": 4,
        "pair_count": len(pairs),
        "candidate_count": len(order),
        "candidate_program_order": order,
        "pair_order": pairs,
        "seeds": list(SEEDS),
        "evaluation_rounds": EVALUATION_ROUNDS,
        "clean_start_per_candidate": True,
        "bitstream_reload_per_candidate": True,
        "fresh_rpc_per_candidate": True,
        "qualification_calls_per_candidate": len(SEEDS),
        "explicit_warmup_calls_per_candidate": len(SEEDS),
        "evaluation_calls_per_candidate": EVALUATION_ROUNDS * len(SEEDS),
        "primary_comparison": "same-Tile Input versus mode11",
        "timing_boundary": "board steady_clock helper around exactly one synchronous operator call",
        "pool_input_hashes": pool_hashes,
        "correctness_manifest_sha256": correctness_manifest,
        "correctness_boot_id": correctness["boot_id"],
        "helper_manifest_sha256": helper_manifest,
        "helper_binary_sha256": helper_contract["binary_sha256"],
        "runtime_sha256": runtime_hashes,
        "same_boot_opportunity_screen": True,
        "cross_boot_stability_claim": False,
        "full_graph_claim": False,
        "latency_values_collected": False,
        "claim_boundary": (
            "single-boot isolated-operator opportunity screen; no cross-boot or graph-routing claim"
        ),
    }
    write(output / "contract.json", contract)
    write(output / "pre_observation_hashes.json", {
        "contract.json": sha256_file(output / "contract.json"),
        "correctness_manifest_sha256": correctness_manifest,
    })

    reuse_board.RUNTIME = runtime_hashes
    binary_dir = cross_dir / "binaries"
    calls, qualifications, warmups, evaluations = [], [], [], []
    boot_id = None
    initial_governor = None
    started = time.monotonic()
    try:
        for wave_index, program_id in enumerate(order, start=1):
            candidate = by_program[program_id]
            wave_port = args.port + 2 * (wave_index - 1)
            remote_runtime = "/var/volatile/{}_wave_{:02d}".format(output.name, wave_index)
            board = ExperimentBoard(args, remote_runtime, wave_port)
            remote = None
            retained = []
            try:
                initial = board.state()
                reuse_board.validate_state(initial, exclusive=True)
                environment = board.environment()
                if boot_id is None:
                    boot_id = initial["boot_id"]
                    initial_governor = environment["governor"]
                    if boot_id != correctness["boot_id"]:
                        raise RuntimeError("timing boot differs from preceding correctness boot")
                elif initial["boot_id"] != boot_id:
                    raise RuntimeError("kernel boot changed between timing candidates")
                if environment["governor"] != initial_governor:
                    raise RuntimeError("CPU governor changed between timing candidates")
                write(output / "wave_{:02d}_preflight.json".format(wave_index), {
                    "state": initial, "environment": environment,
                })
                write(output / "wave_{:02d}_runtime_attestation.json".format(wave_index),
                      board.deploy(runtime_paths))
                write(output / "wave_{:02d}_bitstream_reload.json".format(wave_index),
                      board.reload())
                board.start()
                remote = connect(args.host, wave_port, args.session_timeout)
                inventory, functions = runtime_inventory(remote)
                write(output / "wave_{:02d}_runtime_inventory.json".format(wave_index), inventory)
                retained.append(load_ephemeral(remote, helper))
                timer = remote.get_function("c3.v5.operator_timed_once")
                reset_counts = remote.get_function("c3.v5.reset_counts")
                helper_counts = remote.get_function("c3.v5.counts")
                reset_counts()
                module = load_ephemeral(
                    remote, binary_dir / (candidate["candidate_id"] + ".so")
                )
                retained.append(module)
                buffers = allocate_reusable_buffers(
                    remote.ext_dev(0), candidate["identity"]["workload"]
                )
                write(output / "wave_{:02d}_affinity.json".format(wave_index),
                      board.environment(require_affinity=True))

                def call(seed, phase, timed, round_index=None, seed_position=None):
                    base = {
                        "invocation_index": len(calls),
                        "wave": wave_index,
                        "program_id": program_id,
                        "candidate_id": candidate["candidate_id"],
                        "execution_task_id": candidate["execution_task_id"],
                        "workload_id": candidate["workload_id"],
                        "family_id": candidate["family_id"],
                        "residence_mode": candidate["residence_mode"],
                        "phase": phase,
                        "seed": seed,
                        "round": round_index,
                        "seed_position": seed_position,
                    }
                    result = reuse_board.invoke(
                        module, buffers, candidate["identity"]["workload"], seed,
                        functions, timer if timed else None, timed=timed,
                    )
                    if result["correct"]:
                        validate_profile(candidate, result)
                    row = {**base, **result}
                    calls.append(row)
                    append(output / "calls.jsonl", row)
                    if not row["correct"]:
                        raise RuntimeError("wrong output during {}".format(phase))
                    return row

                for seed in SEEDS:
                    row = call(seed, "qualification", False)
                    qualifications.append(row)
                    append(output / "qualification.jsonl", row)
                for seed in SEEDS:
                    row = call(seed, "explicit_warmup", False)
                    warmups.append(row)
                    append(output / "warmups.jsonl", row)
                for round_index in range(EVALUATION_ROUNDS):
                    shift = round_index % len(SEEDS)
                    seed_order = SEEDS[shift:] + SEEDS[:shift]
                    for seed_position, seed in enumerate(seed_order):
                        row = call(seed, "evaluation_timing", True, round_index, seed_position)
                        evaluations.append(row)
                        append(output / "evaluation_timing.jsonl", row)

                counts = json.loads(str(helper_counts()))
                if counts != {
                    "operator_calls": EVALUATION_ROUNDS * len(SEEDS),
                    "graph_calls": 0,
                    "explicit_sync_calls": 0,
                }:
                    raise RuntimeError("timing helper call-count drift: {}".format(counts))
                final = board.state()
                if final["boot_id"] != boot_id:
                    raise RuntimeError("kernel boot changed during timing candidate")
                append(output / "waves.jsonl", {
                    "wave": wave_index,
                    "program_id": program_id,
                    "candidate_id": candidate["candidate_id"],
                    "execution_task_id": candidate["execution_task_id"],
                    "family_id": candidate["family_id"],
                    "residence_mode": candidate["residence_mode"],
                    "rpc_port": wave_port,
                    "boot_id": boot_id,
                    "helper_counts": counts,
                    "status": "complete",
                })
            finally:
                retained.clear()
                remote = None
                gc.collect()
                if board.pid is not None:
                    board.stop()
                    board.pid = None
                    time.sleep(0.25)
                if board.runtime != remote_runtime:
                    raise RuntimeError("refusing unexpected timing cleanup target")
                target = shlex.quote(remote_runtime)
                board.ssh("rm -rf " + target + " && test ! -e " + target)
            print("timing {}/{} {} complete".format(
                wave_index, len(order), program_id[:12]), flush=True)

        expected = {
            "qualification": len(order) * len(SEEDS),
            "warmups": len(order) * len(SEEDS),
            "evaluation": len(order) * EVALUATION_ROUNDS * len(SEEDS),
        }
        observed = {
            "qualification": len(qualifications),
            "warmups": len(warmups),
            "evaluation": len(evaluations),
        }
        if observed != expected:
            raise RuntimeError("timing call-count drift: {} != {}".format(observed, expected))
        summary = {
            "schema": SCHEMA,
            "status": "REMAINING_FULL_FIT_SINGLE_BOOT_CLEAN_TIMING_COMPLETE",
            "boot_id": boot_id,
            "candidate_count": len(order),
            "pair_count": len(pairs),
            "all_outputs_correct": True,
            "call_counts": observed,
            "all_logged_operator_calls": len(calls),
            "cross_boot_stability_claim": False,
            "full_graph_claim": False,
            "wall_seconds": time.monotonic() - started,
        }
        write(output / "summary.json", summary)
        (output / "command.txt").write_text(
            " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        write(output / "summary.json", {
            "schema": SCHEMA,
            "status": "INVALID_NON_SPLICEABLE_REMAINING_TASK_TIMING",
            "message": str(error),
            "type": type(error).__name__,
            "traceback": traceback.format_exc()[-16000:],
            "logged_operator_calls": len(calls),
            "all_timing_invalid": True,
            "may_splice": False,
        })
        raise
    finally:
        seal(output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-pool-dir", type=Path, required=True)
    parser.add_argument("--cross-compile-dir", type=Path, required=True)
    parser.add_argument("--correctness-dir", type=Path, required=True)
    parser.add_argument("--helper-build-dir", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=7200)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
