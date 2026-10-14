#!/usr/bin/env python3
"""FPGA correctness qualification with fail-isolating bitstream waves.

No latency is collected.  Candidates retain their frozen order.  A wave runs
until the first wrong output, records that candidate as FPGA-invalid, then
stops the experiment RPC and reloads the frozen bitstream before continuing.
This prevents one wrong program from contaminating correctness labels for all
later candidates.  Invalid candidates consume their original positions and no
replacement candidate is introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import time
import traceback
from pathlib import Path

from tvm import rpc

import run_vta_c3_reuse_board as reuse_board
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_holdout_board_boot_v2 import validate_profile
from run_vta_c3_resnet18_board_pool import verify_inputs
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
from run_vta_p7r132_y00_search_confirmation import allocate_reusable_buffers


SCHEMA = "c3_fpga_correctness_waves_v1"
SEEDS = (0, 20250901, 20260910)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def select_pool_scope(candidates, board_pool_dir, pool_scope):
    """Restrict qualification to an explicitly frozen pool without reordering it."""
    if pool_scope == "all":
        return candidates
    policies_path = Path(board_pool_dir).resolve() / "task_policies.json"
    if not policies_path.is_file():
        raise FileNotFoundError("production scope requires {}".format(policies_path))
    policies = read(policies_path)
    allowed = set()
    for task_id, policy in policies.items():
        programs = policy.get("production_program_pool")
        if not programs:
            raise ValueError("task {} has no frozen production_program_pool".format(task_id))
        overlap = allowed.intersection(programs)
        if overlap:
            raise ValueError(
                "production program appears in multiple tasks: {}".format(sorted(overlap))
            )
        allowed.update(programs)
    selected = [row for row in candidates if row["program_id"] in allowed]
    observed = {row["program_id"] for row in selected}
    if observed != allowed:
        raise ValueError(
            "production pool does not match candidates: missing={} extra={}".format(
                sorted(allowed - observed), sorted(observed - allowed)
            )
        )
    return selected


def seal(output):
    write(output / "artifact_hashes.json", {
        "artifacts": {
            path.name: sha256_file(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "artifact_hashes.json"
        },
        "source": sha256_file(Path(__file__).resolve()),
    })


def connect(host, port, timeout):
    for _ in range(20):
        try:
            return rpc.connect(host, port, session_timeout=timeout)
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("RPC unavailable after clean wave start")


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    candidates, _, pool_hashes = verify_inputs(args.board_pool_dir, args.cross_compile_dir)
    declared_candidate_count = len(candidates)
    candidates = select_pool_scope(candidates, args.board_pool_dir, args.pool_scope)
    runtime_contract = read(args.runtime_contract)
    runtime_paths = runtime_contract["runtime_paths"]
    runtime_hashes = runtime_contract["runtime_sha256"]
    for name, path in runtime_paths.items():
        if sha256_file(path) != runtime_hashes[name]:
            raise ValueError("runtime changed: {}".format(name))
    output.mkdir(parents=True)
    for name in ("waves.jsonl", "calls.jsonl", "fpga_correctness.jsonl", "timeline.jsonl"):
        (output / name).write_text("", encoding="utf-8")
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_fpga_correctness_observation",
        "pool_scope": args.pool_scope,
        "declared_candidate_count": declared_candidate_count,
        "qualified_candidate_count": len(candidates),
        "candidate_order": [row["candidate_id"] for row in candidates],
        "program_order": [row["program_id"] for row in candidates],
        "seeds": list(SEEDS),
        "pool_input_hashes": pool_hashes,
        "wave_policy": (
            "reload frozen bitstream before every candidate"
            if args.wave_policy == "every_candidate"
            else "stop after first wrong output, reload frozen bitstream, continue at next identity"
        ),
        "invalid_candidate_policy": "consume frozen identity/position; no replacement; never time",
        "latency_collection_forbidden": True,
        "invocations_per_call": 1,
        "implicit_warmup_calls": 0,
        "cpu_affinity": 3,
        "rpc_port_policy": "wave N uses base_port + 2*(N-1), reserving a disjoint two-port range",
        "remote_runtime_cleanup": "after each stopped wave, delete only that wave's exact tmpfs directory",
        "board_contacted": False,
        "latency_values_collected": False,
    }
    write(output / "contract.json", contract)
    write(output / "pre_observation_hashes.json", {"contract.json": sha256_file(output / "contract.json")})

    reuse_board.RUNTIME = runtime_hashes
    binary_dir = Path(args.cross_compile_dir).resolve() / "binaries"
    cursor = 0
    wave_index = 0
    boot_id = None
    records = []
    calls = []
    started = time.monotonic()
    try:
        while cursor < len(candidates):
            wave_index += 1
            wave_port = args.port + 2 * (wave_index - 1)
            board = ExperimentBoard(
                args, "/var/volatile/{}_wave_{:02d}".format(output.name, wave_index), wave_port
            )
            remote = None
            wave_start = cursor
            wave_wrong = None
            try:
                initial = board.state()
                reuse_board.validate_state(initial, exclusive=True)
                environment = board.environment()
                if boot_id is None:
                    boot_id = initial["boot_id"]
                elif initial["boot_id"] != boot_id:
                    raise RuntimeError("kernel boot changed between correctness waves")
                write(output / "wave_{:02d}_preflight.json".format(wave_index), {
                    "state": initial, "environment": environment,
                })
                write(output / "wave_{:02d}_runtime_attestation.json".format(wave_index),
                      board.deploy(runtime_paths))
                write(output / "wave_{:02d}_bitstream_reload.json".format(wave_index), board.reload())
                board.start()
                remote = connect(args.host, wave_port, args.session_timeout)
                inventory, functions = runtime_inventory(remote)
                write(output / "wave_{:02d}_runtime_inventory.json".format(wave_index), inventory)
                device = remote.ext_dev(0)
                modules, buffers = {}, {}
                write(output / "wave_{:02d}_affinity.json".format(wave_index),
                      board.environment(require_affinity=True))

                while cursor < len(candidates):
                    candidate = candidates[cursor]
                    candidate_id = candidate["candidate_id"]
                    workload_id = candidate["workload_id"]
                    if candidate_id not in modules:
                        modules[candidate_id] = load_ephemeral(
                            remote, binary_dir / (candidate_id + ".so")
                        )
                    if workload_id not in buffers:
                        buffers[workload_id] = allocate_reusable_buffers(
                            device, candidate["identity"]["workload"]
                        )
                    seed_rows = []
                    for seed in SEEDS:
                        base = {
                            "wave": wave_index,
                            "candidate_position": cursor,
                            "candidate_id": candidate_id,
                            "program_id": candidate["program_id"],
                            "execution_task_id": candidate["execution_task_id"],
                            "workload_id": workload_id,
                            "residence_mode": candidate["residence_mode"],
                            "seed": seed,
                            "invocation_index": len(calls),
                        }
                        append(output / "timeline.jsonl", {**base, "event": "begin"})
                        result = reuse_board.invoke(
                            modules[candidate_id], buffers[workload_id],
                            candidate["identity"]["workload"], seed, functions, None, timed=False,
                        )
                        if result.get("latency_ms") is not None:
                            raise RuntimeError("correctness-only call unexpectedly produced latency")
                        row = {**base, **result}
                        calls.append(row)
                        seed_rows.append(row)
                        append(output / "calls.jsonl", row)
                        append(output / "timeline.jsonl", {**base, "event": "end",
                                                           "correct": row["correct"]})
                        if row["correct"]:
                            validate_profile(candidate, row)
                        else:
                            break
                    status = (
                        "passed" if len(seed_rows) == len(SEEDS)
                        and all(row["correct"] for row in seed_rows) else "fpga_rejected"
                    )
                    record = {
                        "candidate_position": cursor,
                        "candidate_id": candidate_id,
                        "program_id": candidate["program_id"],
                        "execution_task_id": candidate["execution_task_id"],
                        "workload_id": workload_id,
                        "residence_mode": candidate["residence_mode"],
                        "wave": wave_index,
                        "status": status,
                        "seed_checks": len(seed_rows),
                        "first_failure": next((row for row in seed_rows if not row["correct"]), None),
                    }
                    records.append(record)
                    append(output / "fpga_correctness.jsonl", record)
                    print("wave {} correctness {}/{} {}".format(
                        wave_index, cursor + 1, len(candidates), status), flush=True)
                    cursor += 1
                    if status != "passed":
                        wave_wrong = candidate_id
                        break
                    if args.wave_policy == "every_candidate":
                        break
                final = board.state()
                if final["boot_id"] != boot_id:
                    raise RuntimeError("kernel boot changed during correctness wave")
                append(output / "waves.jsonl", {
                    "wave": wave_index,
                    "rpc_port": wave_port,
                    "boot_id": boot_id,
                    "start_position": wave_start,
                    "end_position_exclusive": cursor,
                    "first_wrong_candidate_id": wave_wrong,
                    "status": (
                        "ended_after_first_wrong" if wave_wrong
                        else "completed_candidate_isolation"
                        if args.wave_policy == "every_candidate"
                        else "completed_remaining_pool"
                    ),
                })
            finally:
                remote = None
                if board.pid is not None:
                    board.stop()
                    time.sleep(0.25)
                expected_runtime = "/var/volatile/{}_wave_{:02d}".format(
                    output.name, wave_index
                )
                if board.runtime != expected_runtime:
                    raise RuntimeError("refusing unexpected remote cleanup target")
                target = shlex.quote(expected_runtime)
                board.ssh("rm -rf " + target + " && test ! -e " + target)

        summary = {
            "schema": SCHEMA,
            "status": "FPGA_CORRECTNESS_QUALIFICATION_COMPLETE",
            "boot_id": boot_id,
            "candidate_count": len(candidates),
            "fpga_correct_count": sum(row["status"] == "passed" for row in records),
            "fpga_rejected_count": sum(row["status"] != "passed" for row in records),
            "wave_count": wave_index,
            "operator_calls": len(calls),
            "latency_values_collected": False,
            "invalid_candidates_replaced": 0,
            "wall_seconds": time.monotonic() - started,
        }
        write(output / "summary.json", summary)
        (output / "command.txt").write_text(
            " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
            encoding="utf-8",
        )
        seal(output)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except Exception as error:
        invalid = {
            "schema": SCHEMA,
            "status": "INVALID_CORRECTNESS_QUALIFICATION",
            "exception_type": type(error).__name__,
            "message": (str(error) or type(error).__name__)[:4000],
            "traceback": traceback.format_exc()[-12000:],
            "classified_candidates": len(records),
            "operator_calls": len(calls),
            "latency_values_collected": False,
            "may_splice": False,
        }
        write(output / "summary.json", invalid)
        seal(output)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-pool-dir", type=Path, required=True)
    parser.add_argument("--cross-compile-dir", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=7200)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    parser.add_argument(
        "--wave-policy",
        choices=("wrong_only", "every_candidate"),
        default="wrong_only",
    )
    parser.add_argument(
        "--pool-scope",
        choices=("all", "production"),
        default="all",
        help="qualify every declared candidate or only each task's frozen production pool",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
