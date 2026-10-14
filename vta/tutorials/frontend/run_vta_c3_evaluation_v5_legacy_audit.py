#!/usr/bin/env python3
"""Correctness-only predecessor/initial-state audit for C3 v5 legacy programs."""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np
from tvm import rpc

import run_vta_c3_reuse_board as reuse_board
from c3_evaluation_v5 import (
    SEEDS,
    append_jsonl,
    read_json,
    read_jsonl,
    seal_flat,
    sha256_file,
    verify_flat,
    write_json,
)
from c3_thread_sync_board import ExperimentBoard
from qualify_vta_residency_fsim import array_sha256
from run_vta_c3_resnet18_board_pool import verify_inputs
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
from run_vta_p7r132_y00_search_confirmation import (
    allocate_reusable_buffers,
    prepare_reused_buffers,
)


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_FREEZE = GROUPED / "20260926_c3_yolo512_evaluation_v5_freeze_run01"
DEFAULT_POOL = GROUPED / "20260924_c3_yolo512_selector_v4_production_board_pool_freeze_run02"
DEFAULT_CROSS = GROUPED / "20260924_c3_yolo512_selector_v4_production_cross_compile_run01"
DEFAULT_OUTPUT = GROUPED / "20260926_c3_yolo512_evaluation_v5_legacy_audit_run02"


def connect(host, port, timeout):
    for _ in range(30):
        try:
            return rpc.connect(host, port, session_timeout=timeout)
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("RPC unavailable after clean legacy-audit start")


def profile_request_totals(profile):
    return {
        "load_buffer_2d_bytes": int(profile.get("load_buffer_2d_bytes", 0)),
        "load_buffer_2d_calls": int(profile.get("load_buffer_2d_calls", 0)),
        "store_buffer_2d_bytes": int(profile.get("store_buffer_2d_bytes", 0)),
        "store_buffer_2d_calls": int(profile.get("store_buffer_2d_calls", 0)),
    }


def expected_request_totals(candidate):
    totals = candidate["dma_features"]["totals"]
    return {name: int(totals[name]) for name in (
        "load_buffer_2d_bytes", "load_buffer_2d_calls",
        "store_buffer_2d_bytes", "store_buffer_2d_calls",
    )}


def run(args):
    freeze_dir = Path(args.freeze_dir).resolve()
    pool_dir = Path(args.board_pool_dir).resolve()
    cross_dir = Path(args.cross_compile_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    freeze_manifest = verify_flat(freeze_dir)
    candidates, _, input_hashes = verify_inputs(pool_dir, cross_dir)
    freeze = read_json(freeze_dir / "contract.json")
    if freeze.get("status") != "PRE_LATENCY_FROZEN":
        raise ValueError("v5 freeze is not pre-latency")
    policies = read_json(freeze_dir / "task_policies.json")
    helper = Path(freeze["helper_binary"])
    if sha256_file(helper) != freeze["helper_sha256"]:
        raise ValueError("v5 helper changed")
    runtime = read_json(freeze["runtime_contract"])
    for name, path in runtime["runtime_paths"].items():
        if sha256_file(path) != runtime["runtime_sha256"][name]:
            raise ValueError("runtime changed: {}".format(name))

    by_program = {row["program_id"]: row for row in candidates}
    legacy = [program for task in sorted(policies)
              for program in policies[task]["legacy_controlled_program_pool"]]
    production_predecessor = {
        task: policies[task]["incumbent_program_id"] for task in policies
    }
    output.mkdir(parents=True)
    (output / "calls.jsonl").write_text("", encoding="utf-8")
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    contract = {
        "schema": "c3_evaluation_v5_legacy_transition_audit_v1",
        "status": "FROZEN_BEFORE_CORRECTNESS_ONLY_AUDIT",
        "freeze_manifest_sha256": freeze_manifest,
        "pool_input_hashes": input_hashes,
        "legacy_program_ids": legacy,
        "predecessor_classes": ["same_task_incumbent", "cyclic_legacy"],
        "seeds": list(SEEDS),
        "latency_collection_forbidden": True,
        "timer_function_forbidden": True,
        "boundary": (
            "legacy helper performs pre-sync, clears profiler, invokes main once, and performs "
            "post-sync; no elapsed value is produced"
        ),
        "cpu_affinity": 3,
        "correctness_boot_id_must_change": freeze["correctness_boot_id"],
    }
    write_json(output / "contract.json", contract)
    write_json(output / "pre_observation_hashes.json", {
        "contract.json": sha256_file(output / "contract.json"),
        "freeze_manifest_sha256": freeze_manifest,
    })

    board = ExperimentBoard(args, "/var/volatile/" + output.name, args.port)
    remote = None
    calls = []
    initial = None
    started = time.monotonic()
    try:
        initial = board.state()
        reuse_board.validate_state(initial, exclusive=True)
        if initial["boot_id"] == freeze["correctness_boot_id"]:
            raise RuntimeError("legacy audit requires a boot after the correctness-only campaign")
        environment = board.environment()
        write_json(output / "preflight.json", {"state": initial, "environment": environment})
        reuse_board.RUNTIME = runtime["runtime_sha256"]
        write_json(output / "runtime_attestation.json", board.deploy(runtime["runtime_paths"]))
        write_json(output / "bitstream_reload.json", board.reload())
        board.start()
        remote = connect(args.host, args.port, args.session_timeout)
        inventory, functions = runtime_inventory(remote)
        write_json(output / "runtime_inventory.json", inventory)
        retained = [load_ephemeral(remote, helper)]
        audit_once = remote.get_function("c3.v5.legacy_audit_once")
        symbol_info = json.loads(str(remote.get_function("c3.v5.runtime_symbols")()))
        write_json(output / "runtime_symbols.json", symbol_info)
        binary_dir = cross_dir / "binaries"
        needed = set(legacy)
        needed.update(production_predecessor.values())
        modules = {
            program: load_ephemeral(remote, binary_dir / (by_program[program]["candidate_id"] + ".so"))
            for program in sorted(needed)
        }
        workloads = {
            row["workload_id"]: row["identity"]["workload"]
            for row in candidates if row["program_id"] in needed
        }
        buffers = {
            workload_id: allocate_reusable_buffers(remote.ext_dev(0), workload)
            for workload_id, workload in workloads.items()
        }
        write_json(output / "affinity.json", board.environment(require_affinity=True))

        def normal_call(program_id, seed, role):
            candidate = by_program[program_id]
            result = reuse_board.invoke(
                modules[program_id], buffers[candidate["workload_id"]],
                candidate["identity"]["workload"], seed, functions, None, timed=False,
            )
            row = {
                **result, "invocation_index": len(calls), "program_id": program_id,
                "candidate_id": candidate["candidate_id"], "execution_task_id": candidate["execution_task_id"],
                "role": role, "wrapper": "native_production_call", "latency_ms": None,
            }
            calls.append(row)
            append_jsonl(output / "calls.jsonl", row)
            return row

        def legacy_call(program_id, seed, role):
            candidate = by_program[program_id]
            local_buffers = buffers[candidate["workload_id"]]
            expected = prepare_reused_buffers(local_buffers, candidate["identity"]["workload"], seed)
            started_call = time.monotonic()
            audit_once(modules[program_id], local_buffers["data"], local_buffers["weight"],
                       local_buffers["output"])
            wall = time.monotonic() - started_call
            actual = local_buffers["output"].numpy()
            mismatch = int(np.count_nonzero(actual != expected))
            profile = json.loads(functions["vta.runtime.profiler_status"]())
            row = {
                "invocation_index": len(calls), "program_id": program_id,
                "candidate_id": candidate["candidate_id"], "execution_task_id": candidate["execution_task_id"],
                "role": role, "wrapper": "v5_legacy_audit_once", "seed": seed,
                "correct": mismatch == 0, "mismatch_count": mismatch,
                "expected_sha256": array_sha256(expected), "actual_sha256": array_sha256(actual),
                "runtime_profile_complete": profile, "latency_ms": None,
                "rpc_call_wall_seconds": wall, "complete_operator_calls": 1,
                "request_totals_match_static": (
                    profile_request_totals(profile) == expected_request_totals(candidate)
                ),
                "post_completion_observed": int(profile.get("synchronize_calls", 0)) >= 2,
                "driver_timeout_calls": int(profile.get("driver_timeout_calls", 0)),
            }
            calls.append(row)
            append_jsonl(output / "calls.jsonl", row)
            return row

        audit_rows = []
        for index, target in enumerate(legacy):
            task_id = by_program[target]["execution_task_id"]
            predecessors = (
                ("same_task_incumbent", production_predecessor[task_id]),
                ("cyclic_legacy", legacy[index - 1]),
            )
            for predecessor_class, predecessor in predecessors:
                for seed in SEEDS:
                    append_jsonl(output / "timeline.jsonl", {
                        "event": "transition_begin", "target_program_id": target,
                        "predecessor_program_id": predecessor,
                        "predecessor_class": predecessor_class, "seed": seed,
                    })
                    if predecessor_class == "same_task_incumbent":
                        before = normal_call(predecessor, seed, "predecessor")
                    else:
                        before = legacy_call(predecessor, seed, "predecessor")
                    target_row = legacy_call(target, seed, "target")
                    passed = bool(
                        before["correct"] and target_row["correct"]
                        and target_row["request_totals_match_static"]
                        and target_row["post_completion_observed"]
                        and target_row["driver_timeout_calls"] == 0
                    )
                    audit = {
                        "target_program_id": target, "predecessor_program_id": predecessor,
                        "predecessor_class": predecessor_class, "seed": seed, "passed": passed,
                    }
                    audit_rows.append(audit)
                    append_jsonl(output / "timeline.jsonl", {"event": "transition_end", **audit})
            print("legacy audit {}/{}".format(index + 1, len(legacy)), flush=True)

        running_final = board.state()
        reuse_board.validate_state(running_final, exclusive=False)
        final_environment = board.environment(require_affinity=True)
        if running_final["boot_id"] != initial["boot_id"]:
            raise RuntimeError("boot changed during legacy transition audit")
        if final_environment["governor"] != environment["governor"]:
            raise RuntimeError("governor changed during legacy transition audit")
        # Demand exclusivity only after stopping this experiment own RPC.
        remote = None
        gc.collect()
        board.stop()
        board.pid = None
        final = board.state()
        reuse_board.validate_state(final, exclusive=True)
        passed = all(row["passed"] for row in audit_rows)
        certificates = []
        for program in legacy:
            rows = [row for row in audit_rows if row["target_program_id"] == program]
            certificates.append({
                "program_id": program,
                "candidate_id": by_program[program]["candidate_id"],
                "status": "passed" if len(rows) == 6 and all(row["passed"] for row in rows) else "failed",
                "transition_checks": len(rows),
                "predecessor_classes": sorted({row["predecessor_class"] for row in rows}),
                "seeds": sorted({row["seed"] for row in rows}),
                "timing_allowed": len(rows) == 6 and all(row["passed"] for row in rows),
                "timing_wrapper": "pre-sync outside timing; main plus post-sync inside timing",
            })
        write_json(output / "legacy_certificates.json", certificates)
        summary = {
            "schema": contract["schema"],
            "status": "LEGACY_TRANSITION_AUDIT_PASSED" if passed else "LEGACY_TRANSITION_AUDIT_FAILED",
            "boot_id": initial["boot_id"], "legacy_programs": len(legacy),
            "transition_checks": len(audit_rows), "operator_calls": len(calls),
            "latency_values_collected": False, "all_transitions_passed": passed,
            "legacy_reference_timing_allowed": passed,
            "wall_seconds": time.monotonic() - started,
        }
        write_json(output / "summary.json", summary)
        write_json(output / "postflight.json", {
            "running_state": running_final,
            "running_environment": final_environment,
            "stopped_exclusive_state": final,
        })
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        write_json(output / "summary.json", {
            "schema": contract["schema"], "status": "INVALID_LEGACY_TRANSITION_AUDIT",
            "message": str(error), "type": type(error).__name__,
            "traceback": traceback.format_exc()[-12000:], "operator_calls": len(calls),
            "latency_values_collected": False, "legacy_reference_timing_allowed": False,
        })
        raise
    finally:
        remote = None
        gc.collect()
        try:
            board.stop()
        except Exception:
            pass
        seal_flat(output, (Path(__file__).resolve(),))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--board-pool-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--cross-compile-dir", type=Path, default=DEFAULT_CROSS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=7200)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
