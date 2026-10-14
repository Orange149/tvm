#!/usr/bin/env python3
"""One-boot clean-start production-pool timing for frozen YOLO-224 C3 tasks.

Each program receives its own bitstream reload and fresh RPC.  Three timed
selection calls are kept separate from six three-seed evaluation rounds.  The
runner is a first-screen experiment: it does not claim cross-boot stability or
full-graph deployment performance.
"""

from __future__ import annotations

import argparse
import gc
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
from freeze_vta_c3_selector_v4_production_timing import verify as verify_flat
from run_vta_c3_fpga_correctness_waves import (
    append,
    read,
    seal,
    select_pool_scope,
    sha256_file,
    write,
)
from run_vta_c3_holdout_board_boot_v2 import validate_profile
from run_vta_c3_resnet18_board_pool import verify_inputs
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
from run_vta_p7r132_y00_search_confirmation import allocate_reusable_buffers


SCHEMA = "c3_yolo224_clean_timing_v1"
SEEDS = (0, 20250901, 20260910)
EVALUATION_ROUNDS = 6
ORDER_SEED = 20260926


def connect(host, port, timeout):
    for _ in range(30):
        try:
            return rpc.connect(host, port, session_timeout=timeout)
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("RPC unavailable after clean timing start")


def frozen_orders(candidates, source_policies):
    by_program = {row["program_id"]: row for row in candidates}
    policies = {}
    for task_id, source in sorted(source_policies.items()):
        incumbent = source["incumbent_program_id"]
        production = list(source["production_program_pool"])
        new_programs = [program for program in production if program != incumbent]
        selector = [program for program in source["selector_v4_order"]
                    if program in new_programs]
        if len(production) != 9 or len(new_programs) != 8:
            raise ValueError("unexpected YOLO-224 production cardinality")
        if set(selector) != set(new_programs):
            raise ValueError("selector order does not cover each new production program")

        def metric_order(fields):
            return sorted(new_programs, key=lambda program: tuple(
                int(by_program[program]["timing_freeze_metrics"][field]) for field in fields
            ) + (program,))

        task_seed = ORDER_SEED ^ int(task_id[:16], 16)
        fixed_random = list(new_programs)
        random.Random(task_seed).shuffle(fixed_random)
        policies[task_id] = {
            "execution_task_id": task_id,
            "workload_id": source["workload_id"],
            "incumbent_program_id": incumbent,
            "production_program_pool": production,
            "orders": {
                "selector_v4": selector,
                "bytes_only": metric_order(("total_request_bytes",)),
                "calls_only": metric_order(("total_request_calls",)),
                "command_only": metric_order((
                    "submissions", "uop_load_bytes", "instruction_bytes",
                    "total_request_bytes", "total_request_calls",
                )),
                "fixed_random": fixed_random,
            },
        }
    return policies


def session_order(policies):
    programs = [program for task_id in sorted(policies)
                for program in policies[task_id]["production_program_pool"]]
    random.Random(ORDER_SEED).shuffle(programs)
    return programs


def run(args):
    freeze_dir = Path(args.freeze_dir).resolve()
    pool_dir = Path(args.board_pool_dir).resolve()
    cross_dir = Path(args.cross_compile_dir).resolve()
    helper_dir = Path(args.helper_build_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))

    freeze_manifest = verify_flat(freeze_dir)
    helper_manifest = verify_flat(helper_dir)
    candidates, _, pool_hashes = verify_inputs(pool_dir, cross_dir)
    candidates = select_pool_scope(candidates, pool_dir, "production")
    by_program = {row["program_id"]: row for row in candidates}
    if len(candidates) != 27 or len(by_program) != 27:
        raise ValueError("clean timing requires 27 distinct production programs")

    freeze = read(freeze_dir / "contract.json")
    if (freeze.get("status") != "FROZEN_AFTER_FPGA_CORRECTNESS_BEFORE_FIRST_LATENCY"
            or freeze.get("input_resolution") != 224
            or freeze.get("correctness_candidate_count") != 27
            or freeze.get("latency_values_collected") is not False):
        raise ValueError("requires the frozen latency-unseen YOLO-224 timing contract")
    source_policies = read(pool_dir / "task_policies.json")
    policies = frozen_orders(candidates, source_policies)
    order = session_order(policies)
    if set(order) != set(by_program):
        raise ValueError("frozen session order does not cover the production pool")

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
                 "selection_timing.jsonl", "evaluation_timing.jsonl", "waves.jsonl"):
        (output / name).write_text("", encoding="utf-8")
    contract = {
        "schema": SCHEMA,
        "status": "FROZEN_BEFORE_FIRST_YOLO224_LATENCY",
        "input_resolution": 224,
        "task_count": len(policies),
        "candidate_count": len(candidates),
        "candidate_session_order": order,
        "task_policies": policies,
        "seeds": list(SEEDS),
        "evaluation_rounds": EVALUATION_ROUNDS,
        "clean_start_per_candidate": True,
        "bitstream_reload_per_candidate": True,
        "fresh_rpc_per_candidate": True,
        "qualification_calls_per_candidate": len(SEEDS),
        "explicit_warmup_calls_per_candidate": len(SEEDS),
        "selection_calls_per_candidate": len(SEEDS),
        "evaluation_calls_per_candidate": EVALUATION_ROUNDS * len(SEEDS),
        "selection_and_evaluation_rows_disjoint": True,
        "primary_comparison": {"method": "selector_v4", "baseline": "bytes_only", "budget": 3},
        "timing_boundary": "board steady_clock helper around exactly one synchronous operator call",
        "pool_input_hashes": pool_hashes,
        "freeze_manifest_sha256": freeze_manifest,
        "helper_manifest_sha256": helper_manifest,
        "helper_binary_sha256": helper_contract["binary_sha256"],
        "runtime_sha256": runtime_hashes,
        "correctness_boot_id": freeze["correctness_boot_id"],
        "same_boot_first_screen_allowed": True,
        "cross_boot_stability_claim": False,
        "full_graph_claim": False,
        "latency_values_collected": False,
        "claim_boundary": (
            "single-boot clean-candidate operator screen; selection uses only three dedicated "
            "samples and oracle evaluation uses only six disjoint three-seed rounds"
        ),
    }
    write(output / "contract.json", contract)
    write(output / "pre_observation_hashes.json", {
        "contract.json": sha256_file(output / "contract.json"),
        "freeze_manifest_sha256": freeze_manifest,
    })

    reuse_board.RUNTIME = runtime_hashes
    binary_dir = cross_dir / "binaries"
    calls = []
    qualification = []
    warmups = []
    selection = []
    evaluation = []
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
                    qualification.append(row)
                    append(output / "qualification.jsonl", row)
                for seed in SEEDS:
                    row = call(seed, "explicit_warmup", False)
                    warmups.append(row)
                    append(output / "warmups.jsonl", row)
                for round_index, seed in enumerate(SEEDS):
                    row = call(seed, "selection_timing", True, round_index, 0)
                    selection.append(row)
                    append(output / "selection_timing.jsonl", row)
                for round_index in range(EVALUATION_ROUNDS):
                    seed_order = SEEDS[round_index % len(SEEDS):] + SEEDS[:round_index % len(SEEDS)]
                    for seed_position, seed in enumerate(seed_order):
                        row = call(seed, "evaluation_timing", True, round_index, seed_position)
                        evaluation.append(row)
                        append(output / "evaluation_timing.jsonl", row)

                final = board.state()
                if final["boot_id"] != boot_id:
                    raise RuntimeError("kernel boot changed during timing candidate")
                append(output / "waves.jsonl", {
                    "wave": wave_index,
                    "program_id": program_id,
                    "candidate_id": candidate["candidate_id"],
                    "execution_task_id": candidate["execution_task_id"],
                    "residence_mode": candidate["residence_mode"],
                    "rpc_port": wave_port,
                    "boot_id": boot_id,
                    "helper_counts": json.loads(str(helper_counts())),
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
            print("timing {}/27 {} complete".format(wave_index, program_id[:12]), flush=True)

        expected = {
            "qualification": len(candidates) * len(SEEDS),
            "warmups": len(candidates) * len(SEEDS),
            "selection": len(candidates) * len(SEEDS),
            "evaluation": len(candidates) * EVALUATION_ROUNDS * len(SEEDS),
        }
        observed = {
            "qualification": len(qualification), "warmups": len(warmups),
            "selection": len(selection), "evaluation": len(evaluation),
        }
        if observed != expected:
            raise RuntimeError("timing call-count drift: {} != {}".format(observed, expected))
        summary = {
            "schema": SCHEMA,
            "status": "YOLO224_SINGLE_BOOT_CLEAN_TIMING_COMPLETE",
            "boot_id": boot_id,
            "candidate_count": len(candidates),
            "all_outputs_correct": True,
            "call_counts": observed,
            "all_logged_operator_calls": len(calls),
            "selection_and_evaluation_rows_disjoint": True,
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
            "status": "INVALID_NON_SPLICEABLE_YOLO224_TIMING",
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
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--board-pool-dir", type=Path, required=True)
    parser.add_argument("--cross-compile-dir", type=Path, required=True)
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
