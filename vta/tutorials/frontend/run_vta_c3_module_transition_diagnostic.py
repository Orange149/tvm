#!/usr/bin/env python3
"""Separate VTA module-load effects from cross-program execution effects.

This is a correctness-only diagnostic.  Every frozen case runs in a fresh RPC
process after reloading the same bitstream.  No latency API is requested.  The
cases vary only module preload order and the programs executed before a known
context-sensitive target program.
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


SCHEMA = "c3_module_transition_diagnostic_v1"
SEEDS = (0, 20250901, 20260910)

# The first three programs in the failed formal session and the following
# context-sensitive mode-10 program.  These are frozen implementation IDs,
# not labels selected after this diagnostic observes an output.
A1 = "16e26e089d8e7f46ea4d3ea7c96bdf6af58894ab16dcb92d1e831ade289fb005"
A2 = "5843ea33cb51cde159a5b8f8b95293efb487d395ad3af1b785e9f96392a6c8fd"
A3 = "8e650d7a04984d025b74990330e76355753cf4a9324ed2d4d95dc84275d60c81"
TARGET = "97af0b2e9101ff69614ba13deacfef97c513a3579fc7a4d37c46783b380d9014"


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


def frozen_cases(all_ids):
    first_four = [A1, A2, A3, TARGET]
    return [
        {
            "case_id": "target_only_repeated",
            "preload": [TARGET],
            "calls": [(TARGET, seed) for seed in SEEDS] + [(TARGET, 0)],
            "question": "Can the target survive repeated calls in isolation?",
        },
        {
            "case_id": "preload_a1_target_run_target",
            "preload": [A1, TARGET],
            "calls": [(TARGET, seed) for seed in SEEDS],
            "question": "Does loading A1 before the target fail without executing A1?",
        },
        {
            "case_id": "preload_target_a1_run_target",
            "preload": [TARGET, A1],
            "calls": [(TARGET, seed) for seed in SEEDS],
            "question": "Is a preload-only effect sensitive to DSO load order?",
        },
        {
            "case_id": "lazy_a1_execute_then_target",
            "preload": [A1],
            "lazy_load_before_call": {1: TARGET},
            "calls": [(A1, 0), (TARGET, 0)],
            "question": "Does executing A1 contaminate a target loaded afterwards?",
        },
        {
            "case_id": "preload_a1_target_execute_a1_then_target",
            "preload": [A1, TARGET],
            "calls": [(A1, 0), (TARGET, 0)],
            "question": "Does A1 execution matter when both modules were preloaded?",
        },
        {
            "case_id": "preload_first_four_run_target",
            "preload": first_four,
            "calls": [(TARGET, seed) for seed in SEEDS],
            "question": "Does preloading the exact first-four module set alone break the target?",
        },
        {
            "case_id": "preload_first_four_replay_prefix",
            "preload": first_four,
            "calls": (
                [(candidate_id, seed) for candidate_id in (A1, A2, A3) for seed in SEEDS]
                + [(TARGET, 0)]
            ),
            "question": "Does the exact failed formal correctness prefix reproduce the target failure?",
        },
        {
            "case_id": "preload_complete_pool_run_target",
            "preload": list(all_ids),
            "calls": [(TARGET, 0)],
            "question": "Does complete-pool preload alone break the target?",
        },
    ]


def classify(results):
    by_case = {row["case_id"]: row for row in results}
    def target_ok(case_id):
        return all(call["correct"] for call in by_case[case_id]["calls"]
                   if call["candidate_id"] == TARGET)
    return {
        "target_isolated_reentrant": target_ok("target_only_repeated"),
        "a1_then_target_preload_only_correct": target_ok("preload_a1_target_run_target"),
        "target_then_a1_preload_only_correct": target_ok("preload_target_a1_run_target"),
        "a1_execution_then_lazy_target_correct": target_ok("lazy_a1_execute_then_target"),
        "a1_execution_with_preload_correct": target_ok("preload_a1_target_execute_a1_then_target"),
        "first_four_preload_only_correct": target_ok("preload_first_four_run_target"),
        "formal_prefix_replay_target_correct": target_ok("preload_first_four_replay_prefix"),
        "complete_pool_preload_only_correct": target_ok("preload_complete_pool_run_target"),
        "interpretation_rule": (
            "preload-only failure supports module-load/linkage interaction; execution-only failure "
            "supports runtime/hardware state contamination; both may be present"
        ),
    }


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    candidates, _, pool_hashes = verify_inputs(args.board_pool_dir, args.cross_compile_dir)
    by_candidate = {row["candidate_id"]: row for row in candidates}
    required = {A1, A2, A3, TARGET}
    if not required.issubset(by_candidate):
        raise ValueError("frozen diagnostic candidate missing: {}".format(sorted(required - set(by_candidate))))
    task_ids = {by_candidate[candidate_id]["execution_task_id"] for candidate_id in required}
    workload_ids = {by_candidate[candidate_id]["workload_id"] for candidate_id in required}
    if len(task_ids) != 1 or len(workload_ids) != 1:
        raise ValueError("transition diagnostic requires a single execution task/workload")

    runtime_contract = read(args.runtime_contract)
    runtime_paths = runtime_contract["runtime_paths"]
    runtime_hashes = runtime_contract["runtime_sha256"]
    for name, path in runtime_paths.items():
        if sha256_file(path) != runtime_hashes[name]:
            raise ValueError("runtime changed: {}".format(name))

    cases = frozen_cases([row["candidate_id"] for row in candidates])
    binary_dir = Path(args.cross_compile_dir).resolve() / "binaries"
    output.mkdir(parents=True)
    for name in ("cases.jsonl", "calls.jsonl", "timeline.jsonl"):
        (output / name).write_text("", encoding="utf-8")
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_board_observation",
        "pool_input_hashes": pool_hashes,
        "candidate_order": [row["candidate_id"] for row in candidates],
        "cases": cases,
        "target_candidate_id": TARGET,
        "latency_collection_forbidden": True,
        "invocations_per_call": 1,
        "implicit_warmup_calls": 0,
        "fresh_rpc_and_bitstream_reload_per_case": True,
        "cpu_affinity": 3,
        "board_contacted": False,
    }
    write(output / "contract.json", contract)
    write(output / "pre_observation_hashes.json", {"contract.json": sha256_file(output / "contract.json")})

    reuse_board.RUNTIME = runtime_hashes
    boot_id = None
    case_results = []
    all_calls = []
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
                    raise RuntimeError("kernel boot changed between diagnostic cases")
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
                device = remote.ext_dev(0)
                modules = {}
                for candidate_id in case["preload"]:
                    modules[candidate_id] = load_ephemeral(
                        remote, binary_dir / (candidate_id + ".so")
                    )
                workload = by_candidate[TARGET]["identity"]["workload"]
                buffers = allocate_reusable_buffers(device, workload)
                write(output / "case_{:02d}_affinity.json".format(case_index + 1),
                      board.environment(require_affinity=True))

                rows = []
                lazy = {int(index): candidate_id
                        for index, candidate_id in case.get("lazy_load_before_call", {}).items()}
                for call_index, (candidate_id, seed) in enumerate(case["calls"]):
                    if call_index in lazy:
                        lazy_id = lazy[call_index]
                        if lazy_id != candidate_id:
                            raise ValueError("lazy-load identity does not match call")
                        modules[lazy_id] = load_ephemeral(
                            remote, binary_dir / (lazy_id + ".so")
                        )
                    base = {
                        "case_index": case_index,
                        "case_id": case["case_id"],
                        "call_index": call_index,
                        "global_invocation_index": len(all_calls),
                        "candidate_id": candidate_id,
                        "program_id": by_candidate[candidate_id]["program_id"],
                        "seed": seed,
                    }
                    append(output / "timeline.jsonl", {**base, "event": "begin"})
                    result = reuse_board.invoke(
                        modules[candidate_id], buffers, workload, seed, functions, None, timed=False,
                    )
                    if result.get("latency_ms") is not None:
                        raise RuntimeError("correctness-only diagnostic unexpectedly produced latency")
                    row = {**base, **result}
                    rows.append(row)
                    all_calls.append(row)
                    append(output / "calls.jsonl", row)
                    append(output / "timeline.jsonl", {**base, "event": "end", "correct": row["correct"]})
                    if row["correct"]:
                        validate_profile(by_candidate[candidate_id], row)

                final = board.state()
                if final["boot_id"] != boot_id:
                    raise RuntimeError("kernel boot changed during diagnostic case")
                record = {
                    "case_index": case_index,
                    "case_id": case["case_id"],
                    "question": case["question"],
                    "preload": case["preload"],
                    "calls": rows,
                    "all_outputs_correct": all(row["correct"] for row in rows),
                    "target_outputs_correct": all(row["correct"] for row in rows
                                                  if row["candidate_id"] == TARGET),
                }
                case_results.append(record)
                append(output / "cases.jsonl", record)
                print("case {}/{} {} target_correct={}".format(
                    case_index + 1, len(cases), case["case_id"], record["target_outputs_correct"]
                ), flush=True)
            finally:
                remote = None
                if board.pid is not None:
                    board.stop()
                    time.sleep(0.25)
                # Each case intentionally uses an independent RPC directory.
                # Remove only that case's own transient deployment after the
                # process has stopped so a long diagnostic cannot exhaust the
                # board's tmpfs.  Host-side evidence remains immutable.
                board.ssh("rm -rf -- {}".format(shlex.quote(board.runtime)))

        classification = classify(case_results)
        write(output / "classification.json", classification)
        summary = {
            "schema": SCHEMA,
            "status": "MODULE_TRANSITION_DIAGNOSTIC_COMPLETE",
            "boot_id": boot_id,
            "case_count": len(case_results),
            "operator_calls": len(all_calls),
            "target_correct_cases": sum(row["target_outputs_correct"] for row in case_results),
            "target_wrong_cases": sum(not row["target_outputs_correct"] for row in case_results),
            "latency_values_collected": False,
            "classification": classification,
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
            "status": "INVALID_MODULE_TRANSITION_DIAGNOSTIC",
            "exception_type": type(error).__name__,
            "message": (str(error) or type(error).__name__)[:4000],
            "traceback": traceback.format_exc()[-12000:],
            "completed_cases": len(case_results),
            "operator_calls": len(all_calls),
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
    parser.add_argument("--port", type=int, default=9300)
    parser.add_argument("--session-timeout", type=int, default=7200)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
