#!/usr/bin/env python3
"""Test empty-sync and CommandQueue-reset boundaries before mode-10 programs.

Every case uses a fresh RPC process and bitstream reload.  The experiment is
correctness-only: predecessor, boundary action, and successor are recorded
separately, and no latency API is loaded or called.
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


SCHEMA = "c3_transition_boundary_diagnostic_v1"
A3 = "8e650d7a04984d025b74990330e76355753cf4a9324ed2d4d95dc84275d60c81"
ORIGINAL = "f9d0c3343e7ee3978edf6b572ac35d83cb49ceaa9cb5b582dd6865f327888aec"
B1 = "97af0b2e9101ff69614ba13deacfef97c513a3579fc7a4d37c46783b380d9014"
B2 = "3acb5b1b7b9cb5fc03c404781cec062f6d4c941e07673f35d8d2775c071d64d2"


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
    raise RuntimeError("RPC unavailable after clean diagnostic start")


def frozen_cases():
    return [
        {"case_id": "a3_to_b1_no_boundary", "predecessor": A3, "action": "none", "successor": B1},
        {"case_id": "a3_to_b1_empty_sync", "predecessor": A3, "action": "empty_sync", "successor": B1},
        {"case_id": "a3_to_b1_shutdown_queue", "predecessor": A3, "action": "shutdown_queue", "successor": B1},
        {"case_id": "a3_to_b2_no_boundary", "predecessor": A3, "action": "none", "successor": B2},
        {"case_id": "a3_to_b2_empty_sync", "predecessor": A3, "action": "empty_sync", "successor": B2},
        {"case_id": "a3_to_b2_shutdown_queue", "predecessor": A3, "action": "shutdown_queue", "successor": B2},
        {"case_id": "original_to_b1_no_boundary", "predecessor": ORIGINAL, "action": "none", "successor": B1},
        {"case_id": "original_to_b1_empty_sync", "predecessor": ORIGINAL, "action": "empty_sync", "successor": B1},
        {"case_id": "original_to_b1_shutdown_queue", "predecessor": ORIGINAL, "action": "shutdown_queue", "successor": B1},
    ]


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    candidates, _, pool_hashes = verify_inputs(args.board_pool_dir, args.cross_compile_dir)
    by_candidate = {row["candidate_id"]: row for row in candidates}
    required = {A3, ORIGINAL, B1, B2}
    if not required.issubset(by_candidate):
        raise ValueError("boundary candidate missing: {}".format(sorted(required - set(by_candidate))))
    if len({by_candidate[x]["execution_task_id"] for x in required}) != 1:
        raise ValueError("boundary diagnostic candidates do not share an execution task")

    helper_contract = read(Path(args.helper).parent / "contract.json")
    if sha256_file(args.helper) != helper_contract["binary_sha256"]:
        raise ValueError("transition boundary helper changed")
    runtime_contract = read(args.runtime_contract)
    runtime_paths = runtime_contract["runtime_paths"]
    runtime_hashes = runtime_contract["runtime_sha256"]
    for name, path in runtime_paths.items():
        if sha256_file(path) != runtime_hashes[name]:
            raise ValueError("runtime changed: {}".format(name))

    cases = frozen_cases()
    output.mkdir(parents=True)
    for name in ("cases.jsonl", "calls.jsonl", "actions.jsonl", "timeline.jsonl"):
        (output / name).write_text("", encoding="utf-8")
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_board_observation",
        "pool_input_hashes": pool_hashes,
        "helper_sha256": sha256_file(args.helper),
        "runtime_sha256": runtime_hashes,
        "cases": cases,
        "seed": 0,
        "latency_collection_forbidden": True,
        "fresh_rpc_and_bitstream_reload_per_case": True,
        "cpu_affinity": 3,
        "board_contacted": False,
    }
    write(output / "contract.json", contract)
    write(output / "pre_observation_hashes.json", {"contract.json": sha256_file(output / "contract.json")})

    reuse_board.RUNTIME = runtime_hashes
    binary_dir = Path(args.cross_compile_dir).resolve() / "binaries"
    boot_id = None
    records = []
    started = time.monotonic()
    try:
        for case_index, case in enumerate(cases):
            port = args.port + 2 * case_index
            board = ExperimentBoard(
                args, "/var/volatile/{}_case_{:02d}".format(output.name, case_index + 1), port
            )
            remote = None
            try:
                initial = board.state()
                reuse_board.validate_state(initial, exclusive=True)
                environment = board.environment()
                if boot_id is None:
                    boot_id = initial["boot_id"]
                elif initial["boot_id"] != boot_id:
                    raise RuntimeError("kernel boot changed between boundary cases")
                write(output / "case_{:02d}_preflight.json".format(case_index + 1), {
                    "state": initial, "environment": environment,
                })
                write(output / "case_{:02d}_runtime_attestation.json".format(case_index + 1),
                      board.deploy(runtime_paths))
                write(output / "case_{:02d}_bitstream_reload.json".format(case_index + 1), board.reload())
                board.start()
                remote = connect(args.host, port, args.session_timeout)
                inventory, functions = runtime_inventory(remote)
                write(output / "case_{:02d}_runtime_inventory.json".format(case_index + 1), inventory)
                retained = load_ephemeral(remote, args.helper)
                action_functions = {
                    "empty_sync": remote.get_function("c3.transition.empty_sync"),
                    "shutdown_queue": remote.get_function("c3.transition.shutdown_queue"),
                }
                modules = {
                    candidate_id: load_ephemeral(remote, binary_dir / (candidate_id + ".so"))
                    for candidate_id in (case["predecessor"], case["successor"])
                }
                workload = by_candidate[B1]["identity"]["workload"]
                buffers = allocate_reusable_buffers(remote.ext_dev(0), workload)
                write(output / "case_{:02d}_affinity.json".format(case_index + 1),
                      board.environment(require_affinity=True))

                calls = []
                for role in ("predecessor", "successor"):
                    if role == "successor" and case["action"] != "none":
                        functions["vta.runtime.profiler_clear"]()
                        action_start = time.monotonic()
                        action_functions[case["action"]]()
                        action_record = {
                            "case_index": case_index,
                            "case_id": case["case_id"],
                            "action": case["action"],
                            "wall_seconds": time.monotonic() - action_start,
                            "runtime_profile_complete": json.loads(
                                functions["vta.runtime.profiler_status"]()
                            ),
                        }
                        append(output / "actions.jsonl", action_record)
                    candidate_id = case[role]
                    base = {
                        "case_index": case_index,
                        "case_id": case["case_id"],
                        "role": role,
                        "candidate_id": candidate_id,
                        "program_id": by_candidate[candidate_id]["program_id"],
                        "seed": 0,
                    }
                    append(output / "timeline.jsonl", {**base, "event": "begin"})
                    result = reuse_board.invoke(
                        modules[candidate_id], buffers, workload, 0, functions, None, timed=False,
                    )
                    if result.get("latency_ms") is not None:
                        raise RuntimeError("boundary diagnostic unexpectedly produced latency")
                    row = {**base, **result}
                    calls.append(row)
                    append(output / "calls.jsonl", row)
                    append(output / "timeline.jsonl", {**base, "event": "end", "correct": row["correct"]})
                    if row["correct"]:
                        validate_profile(by_candidate[candidate_id], row)

                record = {
                    **case,
                    "case_index": case_index,
                    "predecessor_correct": calls[0]["correct"],
                    "successor_correct": calls[1]["correct"],
                    "successor_mismatch_count": calls[1]["mismatch_count"],
                }
                records.append(record)
                append(output / "cases.jsonl", record)
                print("case {}/{} {} successor_correct={}".format(
                    case_index + 1, len(cases), case["case_id"], record["successor_correct"]
                ), flush=True)
                retained = None
                final = board.state()
                if final["boot_id"] != boot_id:
                    raise RuntimeError("kernel boot changed during boundary case")
            finally:
                remote = None
                if board.pid is not None:
                    board.stop()
                    time.sleep(0.25)
                board.ssh("rm -rf -- {}".format(shlex.quote(board.runtime)))

        by_action = {}
        for action in ("none", "empty_sync", "shutdown_queue"):
            subset = [row for row in records if row["action"] == action]
            by_action[action] = {
                "cases": len(subset),
                "successor_correct": sum(row["successor_correct"] for row in subset),
                "successor_wrong": sum(not row["successor_correct"] for row in subset),
            }
        summary = {
            "schema": SCHEMA,
            "status": "TRANSITION_BOUNDARY_DIAGNOSTIC_COMPLETE",
            "boot_id": boot_id,
            "case_count": len(records),
            "by_action": by_action,
            "latency_values_collected": False,
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
        write(output / "summary.json", {
            "schema": SCHEMA,
            "status": "INVALID_TRANSITION_BOUNDARY_DIAGNOSTIC",
            "exception_type": type(error).__name__,
            "message": (str(error) or type(error).__name__)[:4000],
            "traceback": traceback.format_exc()[-12000:],
            "completed_cases": len(records),
            "latency_values_collected": False,
            "may_splice": False,
        })
        seal(output)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-pool-dir", type=Path, required=True)
    parser.add_argument("--cross-compile-dir", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9420)
    parser.add_argument("--session-timeout", type=int, default=7200)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
