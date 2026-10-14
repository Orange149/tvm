#!/usr/bin/env python3
"""Run one non-spliceable C3 v5 operator boot.

Boot 1 collects the shared selection panel and atomically seals every
method/budget winner before confirmation or reference timing.  Boot 2 imports
those exact decisions and never reselects.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from tvm import rpc

import run_vta_c3_reuse_board as reuse_board
from c3_evaluation_v5 import (
    CONFIRMATION_ROUNDS,
    REFERENCE_ROUNDS,
    SEEDS,
    append_jsonl,
    confirmation_schedule,
    qualification_schedule,
    read_json,
    reference_schedule,
    round_statistics,
    seal_flat,
    select_winners,
    selection_schedule,
    sha256_file,
    transition_state,
    v4_b3_winners,
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
DEFAULT_LEGACY_AUDIT = GROUPED / "20260926_c3_yolo512_evaluation_v5_legacy_audit_run02"


def connect(host, port, timeout):
    for _ in range(30):
        try:
            return rpc.connect(host, port, session_timeout=timeout)
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("RPC unavailable after clean C3 v5 start")


def atomic_write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    os.replace(str(temporary), str(path))


def expected_request_totals(candidate):
    totals = candidate["dma_features"]["totals"]
    return {name: int(totals[name]) for name in (
        "load_buffer_2d_bytes", "load_buffer_2d_calls",
        "store_buffer_2d_bytes", "store_buffer_2d_calls",
    )}


def observed_request_totals(profile):
    return {name: int(profile.get(name, 0)) for name in (
        "load_buffer_2d_bytes", "load_buffer_2d_calls",
        "store_buffer_2d_bytes", "store_buffer_2d_calls",
    )}


def summarize_programs(rows, rounds):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["execution_task_id"], row["program_id"])].append(row)
    return {
        task_id: {
            program_id: round_statistics(program_rows, rounds)
            for (candidate_task, program_id), program_rows in grouped.items()
            if candidate_task == task_id
        }
        for task_id in sorted({row["execution_task_id"] for row in rows})
    }


def run(args):
    freeze_dir = Path(args.freeze_dir).resolve()
    pool_dir = Path(args.board_pool_dir).resolve()
    cross_dir = Path(args.cross_compile_dir).resolve()
    legacy_dir = Path(args.legacy_audit_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    freeze_manifest = verify_flat(freeze_dir)
    candidates, _, input_hashes = verify_inputs(pool_dir, cross_dir)
    freeze = read_json(freeze_dir / "contract.json")
    if freeze.get("status") != "PRE_LATENCY_FROZEN":
        raise ValueError("v5 freeze is not pre-latency")
    policies = read_json(freeze_dir / "task_policies.json")
    legacy_manifest = verify_flat(legacy_dir)
    legacy_summary = read_json(legacy_dir / "summary.json")
    legacy_enabled = legacy_summary.get("status") == "LEGACY_TRANSITION_AUDIT_PASSED"
    helper = Path(freeze["helper_binary"])
    if sha256_file(helper) != freeze["helper_sha256"]:
        raise ValueError("v5 helper changed")
    runtime = read_json(freeze["runtime_contract"])
    for name, path in runtime["runtime_paths"].items():
        if sha256_file(path) != runtime["runtime_sha256"][name]:
            raise ValueError("runtime changed: {}".format(name))

    previous = None
    decisions = None
    if args.boot_index == 2:
        if args.previous_boot_dir is None:
            raise ValueError("boot 2 requires --previous-boot-dir")
        previous_dir = Path(args.previous_boot_dir).resolve()
        previous_manifest = verify_flat(previous_dir)
        previous = read_json(previous_dir / "summary.json")
        if previous.get("status") != "C3_EVALUATION_V5_OPERATOR_BOOT1_COMPLETE":
            raise ValueError("previous directory is not a complete v5 boot 1")
        decisions = read_json(previous_dir / "selection_decisions_frozen.json")
        if decisions.get("state") != "WINNERS_SEALED":
            raise ValueError("boot-1 winners were not atomically sealed")
    else:
        if args.previous_boot_dir is not None:
            raise ValueError("boot 1 cannot import a previous boot")
        previous_manifest = None

    by_program = {row["program_id"]: row for row in candidates}
    valid_programs = {
        program for policy in policies.values()
        for field in ("production_program_pool", "legacy_controlled_program_pool")
        for program in policy[field]
    }
    if set(by_program) != valid_programs | {
            program for policy in policies.values() for program in policy["fpga_invalid_programs"]}:
        raise ValueError("freeze/pool identity mismatch")

    output.mkdir(parents=True)
    for name in (
        "calls.jsonl", "production_correctness.jsonl", "legacy_correctness.jsonl",
        "selection_timing.jsonl", "confirmation_timing.jsonl",
        "production_reference_timing.jsonl", "legacy_reference_timing.jsonl",
        "timeline.jsonl",
    ):
        (output / name).write_text("", encoding="utf-8")
    state = "PRE_LATENCY_FROZEN"
    contract = {
        "schema": "c3_evaluation_v5_operator_boot_v1",
        "status": state,
        "attempt_id": args.attempt_id,
        "boot_index": args.boot_index,
        "freeze_manifest_sha256": freeze_manifest,
        "legacy_audit_manifest_sha256": legacy_manifest,
        "legacy_controlled_reference_enabled": legacy_enabled,
        "previous_boot_dir": str(Path(args.previous_boot_dir).resolve()) if previous else None,
        "previous_boot_manifest_sha256": previous_manifest,
        "pool_input_hashes": input_hashes,
        "production_correctness_calls": 48,
        "legacy_correctness_calls": 36 if legacy_enabled else 0,
        "shared_selection_calls": 48 if args.boot_index == 1 else 0,
        "production_reference_calls": 288,
        "legacy_reference_calls": 216 if legacy_enabled else 0,
        "legacy_warmup_calls_outside_prescribed_budget": 36 if legacy_enabled else 0,
        "confirmation_calls": "derived from sealed winners; at most 72",
        "session_policy": (
            "production wrong output, interruption, identity/boot/affinity drift, or any timed "
            "wrong output invalidates the whole boot; legacy correctness failure disables the "
            "entire legacy reference branch without changing production results"
        ),
        "winner_retry_policy": (
            "after WINNERS_SEALED no failure can trigger reselection; failed rows are never spliced"
        ),
        "cpu_affinity": 3,
        "implicit_warmup_calls": 0,
        "invocations_per_timing_sample": 1,
    }
    write_json(output / "contract.json", contract)
    write_json(output / "campaign_state.json", {"state": state, "attempt_id": args.attempt_id})
    write_json(output / "pre_observation_hashes.json", {
        "contract.json": sha256_file(output / "contract.json"),
        "freeze_manifest_sha256": freeze_manifest,
    })

    board = ExperimentBoard(args, "/var/volatile/" + output.name, args.port)
    remote = None
    calls = []
    production_correctness = []
    legacy_correctness = []
    selection_rows = []
    confirmation_rows = []
    production_reference_rows = []
    legacy_reference_rows = []
    winners_sealed = False
    initial = None
    started = time.monotonic()
    try:
        initial = board.state()
        reuse_board.validate_state(initial, exclusive=True)
        if initial["boot_id"] == freeze["correctness_boot_id"]:
            raise RuntimeError("formal timing requires a boot after correctness qualification")
        if previous and initial["boot_id"] == previous["boot_id"]:
            raise RuntimeError("boot 2 requires a changed boot ID")
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
        native_timer = remote.get_function("c3.v5.operator_timed_once")
        legacy_timer = remote.get_function("c3.v5.legacy_timed_once")
        legacy_audit = remote.get_function("c3.v5.legacy_audit_once")
        reset_counts = remote.get_function("c3.v5.reset_counts")
        helper_counts = remote.get_function("c3.v5.counts")
        reset_counts()
        write_json(output / "runtime_symbols.json", json.loads(str(
            remote.get_function("c3.v5.runtime_symbols")()
        )))
        binary_dir = cross_dir / "binaries"
        modules = {
            program: load_ephemeral(remote, binary_dir / (by_program[program]["candidate_id"] + ".so"))
            for program in sorted(valid_programs)
        }
        workloads = {
            row["workload_id"]: row["identity"]["workload"]
            for row in candidates if row["program_id"] in valid_programs
        }
        buffers = {
            workload_id: allocate_reusable_buffers(remote.ext_dev(0), workload)
            for workload_id, workload in workloads.items()
        }
        write_json(output / "affinity_after_load.json", board.environment(require_affinity=True))

        def record(program_id, seed, phase, timer=None, spec=None, legacy_wrapper=False):
            candidate = by_program[program_id]
            base = {
                "invocation_index": len(calls), "program_id": program_id,
                "candidate_id": candidate["candidate_id"],
                "execution_task_id": candidate["execution_task_id"],
                "workload_id": candidate["workload_id"], "residence_mode": candidate["residence_mode"],
                "phase": phase, "seed": seed,
            }
            if spec:
                base.update({key: value for key, value in spec.items()
                             if key not in ("phase", "execution_task_id", "program_id", "seed")})
            append_jsonl(output / "timeline.jsonl", {**base, "event": "begin"})
            if timer is not None:
                result = reuse_board.invoke(
                    modules[program_id], buffers[candidate["workload_id"]],
                    candidate["identity"]["workload"], seed, functions, timer, timed=True,
                )
            elif legacy_wrapper:
                local_buffers = buffers[candidate["workload_id"]]
                expected = prepare_reused_buffers(local_buffers, candidate["identity"]["workload"], seed)
                legacy_audit(modules[program_id], local_buffers["data"], local_buffers["weight"],
                             local_buffers["output"])
                actual = local_buffers["output"].numpy()
                mismatch = int(np.count_nonzero(actual != expected))
                result = {
                    "correct": mismatch == 0, "mismatch_count": mismatch, "latency_ms": None,
                    "expected_sha256": array_sha256(expected), "actual_sha256": array_sha256(actual),
                    "runtime_profile_complete": json.loads(functions["vta.runtime.profiler_status"]()),
                    "complete_operator_calls": 1, "hidden_warmup_calls": 0,
                }
            else:
                result = reuse_board.invoke(
                    modules[program_id], buffers[candidate["workload_id"]],
                    candidate["identity"]["workload"], seed, functions, None, timed=False,
                )
            profile = result["runtime_profile_complete"]
            result["request_totals_match_static"] = (
                observed_request_totals(profile) == expected_request_totals(candidate)
            )
            if not result["request_totals_match_static"]:
                raise RuntimeError("runtime/static request drift for {}".format(program_id))
            if int(profile.get("driver_timeout_calls", 0)) != 0:
                raise RuntimeError("driver timeout for {}".format(program_id))
            row = {**base, **result}
            calls.append(row)
            append_jsonl(output / "calls.jsonl", row)
            append_jsonl(output / "timeline.jsonl", {**base, "event": "end", "correct": row["correct"]})
            return row

        # Production identities must remain fully correct because the shared
        # panel and production oracle require all eight programs per task.
        for spec in qualification_schedule(policies, "production"):
            row = record(spec["program_id"], spec["seed"], spec["phase"], spec=spec)
            production_correctness.append(row)
            append_jsonl(output / "production_correctness.jsonl", row)
            if not row["correct"]:
                raise RuntimeError("production correctness failure before latency")
        if len(production_correctness) != 48:
            raise RuntimeError("production correctness call-count drift")

        # Legacy is a separate optional analysis branch.  One failure disables
        # all legacy timing rather than contaminating or shrinking its oracle.
        if legacy_enabled:
            for spec in qualification_schedule(policies, "legacy"):
                row = record(spec["program_id"], spec["seed"], spec["phase"],
                             spec=spec, legacy_wrapper=True)
                legacy_correctness.append(row)
                append_jsonl(output / "legacy_correctness.jsonl", row)
            if len(legacy_correctness) != 36:
                raise RuntimeError("legacy correctness call-count drift")
            if not all(row["correct"] for row in legacy_correctness):
                legacy_enabled = False
                write_json(output / "legacy_branch_disabled.json", {
                    "reason": "per_boot_controlled_correctness_failure",
                    "legacy_timing_rows_allowed": 0,
                })

        if args.boot_index == 1:
            state = transition_state(state, "SELECTION_RUNNING")
            atomic_write(output / "campaign_state.json", {"state": state, "attempt_id": args.attempt_id})
            for spec in selection_schedule(policies):
                row = record(spec["program_id"], spec["seed"], spec["phase"],
                             native_timer, spec=spec)
                if not row["correct"]:
                    raise RuntimeError("wrong output in shared selection panel")
                selection_rows.append(row)
                append_jsonl(output / "selection_timing.jsonl", row)
            if len(selection_rows) != 48:
                raise RuntimeError("shared selection panel must contain exactly 48 calls")
            decisions_payload = {
                "schema": "c3_evaluation_v5_selection_decisions_v1",
                "state": "WINNERS_SEALED",
                "attempt_id": args.attempt_id,
                "boot_id": initial["boot_id"],
                "selection_rows_sha256": sha256_file(output / "selection_timing.jsonl"),
                "decisions": select_winners(policies, selection_rows),
                "sealed_before_confirmation_or_reference": True,
                "reselection_after_seal_forbidden": True,
            }
            atomic_write(output / "selection_decisions_frozen.json", decisions_payload)
            decisions = decisions_payload
            state = transition_state(state, "WINNERS_SEALED")
            atomic_write(output / "campaign_state.json", {
                "state": state, "attempt_id": args.attempt_id,
                "selection_decisions_sha256": sha256_file(output / "selection_decisions_frozen.json"),
            })
            winners_sealed = True
        else:
            winners_sealed = True
            state = "WINNERS_SEALED"
            write_json(output / "selection_decisions_frozen.json", decisions)
            write_json(output / "campaign_state.json", {
                "state": state, "attempt_id": args.attempt_id,
                "selection_source": str(Path(args.previous_boot_dir).resolve()),
            })

        v4_winners = v4_b3_winners(decisions["decisions"])
        write_json(output / "v4_b3_winners.json", {
            task: {
                "search_winner": winner,
                "incumbent": policies[task]["incumbent_program_id"],
                "residence_mode": by_program[winner]["residence_mode"],
                "mode11_deployment_candidate": (
                    by_program[winner]["residence_mode"] == "input_weight_resident_transition_safe"
                ),
            }
            for task, winner in v4_winners.items()
        })

        state = transition_state(state, "CONFIRMATION")
        atomic_write(output / "campaign_state.json", {"state": state, "attempt_id": args.attempt_id})
        confirm_plan = confirmation_schedule(policies, v4_winners)
        write_json(output / "confirmation_schedule.json", confirm_plan)
        for spec in confirm_plan:
            row = record(spec["program_id"], spec["seed"], spec["phase"], native_timer, spec=spec)
            if not row["correct"]:
                raise RuntimeError("wrong output in frozen-winner confirmation")
            confirmation_rows.append(row)
            append_jsonl(output / "confirmation_timing.jsonl", row)

        state = transition_state(state, "REFERENCE")
        atomic_write(output / "campaign_state.json", {"state": state, "attempt_id": args.attempt_id})
        production_plan = reference_schedule(policies, "production")
        write_json(output / "production_reference_schedule.json", production_plan)
        for spec in production_plan:
            row = record(spec["program_id"], spec["seed"], spec["phase"], native_timer, spec=spec)
            if not row["correct"]:
                raise RuntimeError("wrong output in production reference")
            production_reference_rows.append(row)
            append_jsonl(output / "production_reference_timing.jsonl", row)
            if len(production_reference_rows) % 48 == 0:
                write_json(output / "environment_reference_{:03d}.json".format(
                    len(production_reference_rows)), board.environment(require_affinity=True))
        if len(production_reference_rows) != 288:
            raise RuntimeError("production reference call-count drift")

        if legacy_enabled:
            # Explicit wrapped warmups are outside the prescribed formal count.
            for spec in qualification_schedule(policies, "legacy"):
                row = record(spec["program_id"], spec["seed"], "legacy_reference_warmup",
                             legacy_wrapper=True)
                if not row["correct"]:
                    raise RuntimeError("legacy wrapped warmup failed")
            legacy_plan = reference_schedule(policies, "legacy")
            write_json(output / "legacy_reference_schedule.json", legacy_plan)
            for spec in legacy_plan:
                row = record(spec["program_id"], spec["seed"], spec["phase"],
                             legacy_timer, spec=spec)
                if not row["correct"]:
                    raise RuntimeError("wrong output in legacy controlled reference")
                legacy_reference_rows.append(row)
                append_jsonl(output / "legacy_reference_timing.jsonl", row)
            if len(legacy_reference_rows) != 216:
                raise RuntimeError("legacy reference call-count drift")

        running_final = board.state()
        reuse_board.validate_state(running_final, exclusive=False)
        final_environment = board.environment(require_affinity=True)
        if running_final["boot_id"] != initial["boot_id"]:
            raise RuntimeError("boot changed during formal C3 v5 operator session")
        if final_environment["governor"] != environment["governor"]:
            raise RuntimeError("governor changed during formal C3 v5 operator session")
        helper_count_snapshot = json.loads(str(helper_counts()))
        # Stop this experiment recorded RPC before demanding exclusive ownership.
        modules.clear()
        remote = None
        gc.collect()
        board.stop()
        board.pid = None
        final = board.state()
        reuse_board.validate_state(final, exclusive=True)
        state = transition_state(state, "COMPLETE")
        atomic_write(output / "campaign_state.json", {"state": state, "attempt_id": args.attempt_id})
        confirmation_stats = summarize_programs(confirmation_rows, CONFIRMATION_ROUNDS)
        production_stats = summarize_programs(production_reference_rows, REFERENCE_ROUNDS)
        legacy_stats = summarize_programs(legacy_reference_rows, REFERENCE_ROUNDS) if legacy_enabled else {}
        write_json(output / "confirmation_summary.json", confirmation_stats)
        write_json(output / "production_reference_summary.json", production_stats)
        write_json(output / "legacy_reference_summary.json", legacy_stats)
        write_json(output / "postflight.json", {
            "running_state": running_final,
            "running_environment": final_environment,
            "stopped_exclusive_state": final,
        })
        summary = {
            "schema": contract["schema"],
            "status": "C3_EVALUATION_V5_OPERATOR_BOOT{}_COMPLETE".format(args.boot_index),
            "state": state, "attempt_id": args.attempt_id, "boot_id": initial["boot_id"],
            "production_correctness_calls": len(production_correctness),
            "legacy_correctness_calls": len(legacy_correctness),
            "selection_calls": len(selection_rows),
            "confirmation_calls": len(confirmation_rows),
            "production_reference_calls": len(production_reference_rows),
            "legacy_reference_calls": len(legacy_reference_rows),
            "all_logged_operator_calls": len(calls),
            "helper_counts": helper_count_snapshot,
            "legacy_controlled_reference_complete": legacy_enabled,
            "selection_decisions_sha256": sha256_file(output / "selection_decisions_frozen.json"),
            "winners_reselected": False,
            "timing_rows_spliced": False,
            "wall_seconds": time.monotonic() - started,
        }
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        write_json(output / "invalid_session.json", {
            "schema": contract["schema"], "status": "INVALID_NON_SPLICEABLE_OPERATOR_BOOT",
            "attempt_id": args.attempt_id, "boot_index": args.boot_index,
            "message": str(error), "type": type(error).__name__,
            "traceback": traceback.format_exc()[-16000:],
            "winner_sealed_before_failure": winners_sealed,
            "sealed_winner_must_persist": winners_sealed,
            "logged_operator_calls": len(calls), "all_timing_invalid": True,
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
    parser.add_argument("--boot-index", type=int, choices=(1, 2), required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--previous-boot-dir", type=Path)
    parser.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--board-pool-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--cross-compile-dir", type=Path, default=DEFAULT_CROSS)
    parser.add_argument("--legacy-audit-dir", type=Path, default=DEFAULT_LEGACY_AUDIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=7200)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
