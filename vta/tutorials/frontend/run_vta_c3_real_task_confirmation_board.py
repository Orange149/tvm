#!/usr/bin/env python3
"""Run the immutable later-boot confirmation and second full-pool reference."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

from tvm import rpc

import run_vta_c3_reuse_board as reuse_board
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_real_task_board_boot import (
    DEFAULT_CROSS,
    DEFAULT_HELPER,
    DEFAULT_POOL,
    DEFAULT_RUNTIME_CONTRACT,
    DEFAULT_SELECTOR,
    GROUPED,
    SEEDS,
    append_jsonl,
    round_statistics,
    sha256_file,
    verify_flat,
    write_json,
)
from run_vta_c3_resnet18_board_pool import verify_inputs
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
from run_vta_p7r132_y00_search_confirmation import (
    allocate_reusable_buffers,
    timing_orders,
)


DEFAULT_CONFIRMATION = GROUPED / "20260924_c3_real_task_confirmation_freeze_run01"
SCHEMA = "c3_real_task_confirmation_board_v1"
CONFIRMATION_ROUNDS = 8
REFERENCE_ROUNDS = 6


def finalize(output):
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_sha256": sha256_file(Path(__file__).resolve()),
        },
    )


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    candidates, _, input_hashes = verify_inputs(args.board_pool_dir, args.cross_compile_dir)
    confirmation_manifest = verify_flat(args.confirmation_dir)
    confirmation = json.loads(
        (Path(args.confirmation_dir) / "contract.json").read_text(encoding="utf-8")
    )
    if confirmation.get("status") != "frozen_after_boot1_before_boot2":
        raise ValueError("confirmation identities are not frozen")
    if args.previous_boot_id != confirmation["boot1_boot_id"]:
        raise ValueError("previous boot ID must exactly match the frozen boot1 ID")
    boot1 = Path(confirmation["boot1_dir"])
    bound = {
        "summary.json": confirmation["boot1_summary_sha256"],
        "selection_decisions_frozen.json": confirmation["boot1_decisions_sha256"],
        "reference_summary.json": confirmation["boot1_reference_sha256"],
        "fpga_correctness_summary.json": confirmation["boot1_correctness_sha256"],
    }
    for name, expected in bound.items():
        if sha256_file(boot1 / name) != expected:
            raise ValueError("boot1 bound artifact changed: {}".format(name))
    if verify_flat(args.selector_dir) != confirmation["selector_manifest_sha256"]:
        raise ValueError("selector manifest changed")
    if verify_flat(args.board_pool_dir) != confirmation["pool_manifest_sha256"]:
        raise ValueError("board pool manifest changed")
    selector = json.loads(
        (Path(args.selector_dir) / "contract.json").read_text(encoding="utf-8")
    )
    runtime_contract = json.loads(Path(args.runtime_contract).read_text(encoding="utf-8"))
    runtime_paths = runtime_contract["runtime_paths"]
    runtime_hashes = runtime_contract["runtime_sha256"]
    for name, path in runtime_paths.items():
        if sha256_file(path) != runtime_hashes[name]:
            raise ValueError("runtime changed: {}".format(name))
    if sha256_file(args.timer_helper) != runtime_contract["helper_sha256"]:
        raise ValueError("single-call timer helper changed")
    program_by_candidate = {}
    candidate_by_program = {}
    for workload in selector["workloads"].values():
        for program in workload["programs"]:
            program_by_candidate[program["candidate_id"]] = program["program_id"]
            candidate_by_program[program["program_id"]] = program["candidate_id"]
    if set(program_by_candidate) != {row["candidate_id"] for row in candidates}:
        raise ValueError("selector/full-pool identity set differs")

    output.mkdir(parents=True)
    for name in (
        "calls.jsonl",
        "fpga_correctness.jsonl",
        "confirmation_timing.jsonl",
        "reference_timing.jsonl",
        "timeline.jsonl",
    ):
        (output / name).write_text("", encoding="utf-8")
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_boot2_observation",
        "confirmation_manifest_sha256": confirmation_manifest,
        "pool_input_hashes": input_hashes,
        "candidate_order": [row["candidate_id"] for row in candidates],
        "confirmation_program_ids": confirmation["confirmation_program_ids"],
        "confirmation_rounds": CONFIRMATION_ROUNDS,
        "reference_rounds": REFERENCE_ROUNDS,
        "seeds": list(SEEDS),
        "previous_boot_id": args.previous_boot_id,
        "requires_changed_boot_id": True,
        "selected_identities_may_change": False,
        "qualification_gate": (
            "all frozen local candidates; three seeds fail-fast; wrong output rejects only that "
            "candidate before timing"
        ),
        "cross_boot_pool_scope": "intersection of boot1 and boot2 FPGA-correct candidates",
        "session_policy": (
            "any interruption, post-qualification wrong output, boot or affinity change "
            "invalidates the whole non-spliceable session"
        ),
        "single_call_helper_sha256": runtime_contract["helper_sha256"],
        "invocations_per_timing_sample": 1,
        "implicit_warmup_calls": 0,
        "cpu_affinity": 3,
        "board_contacted": False,
        "performance_labels_collected": False,
    }
    write_json(output / "contract.json", contract)
    write_json(output / "pre_observation_hashes.json", {"contract.json": sha256_file(output / "contract.json")})
    started = time.monotonic()
    calls, correctness, confirmation_rows, reference_rows = [], [], [], []
    remote = None
    board = ExperimentBoard(args, "/var/volatile/" + output.name + "_timing", args.port)
    try:
        initial = board.state()
        reuse_board.validate_state(initial, exclusive=True)
        if initial["boot_id"] == args.previous_boot_id:
            raise RuntimeError("later-boot confirmation requires a changed boot ID")
        environment = board.environment()
        write_json(output / "preflight.json", {"state": initial, "environment": environment})
        reuse_board.RUNTIME = runtime_hashes
        write_json(output / "runtime_attestation.json", board.deploy(runtime_paths))
        write_json(output / "bitstream_reload.json", board.reload())
        board.start()
        for _ in range(20):
            try:
                remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
                break
            except Exception:
                time.sleep(0.25)
        if remote is None:
            raise RuntimeError("RPC unavailable")
        inventory, functions = runtime_inventory(remote)
        write_json(output / "runtime_inventory.json", inventory)
        retained = [load_ephemeral(remote, args.timer_helper)]
        timer = remote.get_function("c3.reuse.timed_once")
        binary_dir = Path(args.cross_compile_dir).resolve() / "binaries"
        modules = {}
        by_id = {row["candidate_id"]: row for row in candidates}
        for position, candidate in enumerate(candidates, 1):
            modules[candidate["candidate_id"]] = load_ephemeral(
                remote, binary_dir / (candidate["candidate_id"] + ".so")
            )
            print("load {}/{}".format(position, len(candidates)), flush=True)
        workloads = {}
        for candidate in candidates:
            workloads.setdefault(candidate["workload_id"], candidate["identity"]["workload"])
        buffers = {
            workload_id: allocate_reusable_buffers(remote.ext_dev(0), workload)
            for workload_id, workload in workloads.items()
        }
        write_json(output / "affinity_after_load.json", board.environment(require_affinity=True))

        def invoke(candidate_id, seed, phase, round_index=None, position=None, reject_wrong=True):
            candidate = by_id[candidate_id]
            workload_id = candidate["workload_id"]
            base = {
                "invocation_index": len(calls),
                "candidate_id": candidate_id,
                "program_id": program_by_candidate[candidate_id],
                "workload_id": workload_id,
                "seed": seed,
                "phase": phase,
                "round": round_index,
                "position": position,
            }
            append_jsonl(output / "timeline.jsonl", {**base, "event": "begin"})
            result = reuse_board.invoke(
                modules[candidate_id],
                buffers[workload_id],
                candidate["identity"]["workload"],
                seed,
                functions,
                timer,
                timed=phase.endswith("timing"),
            )
            record = {**base, **result}
            calls.append(record)
            append_jsonl(output / "calls.jsonl", record)
            append_jsonl(output / "timeline.jsonl", {**base, "event": "end"})
            if reject_wrong and not record["correct"]:
                raise RuntimeError("wrong output at invocation {}".format(base["invocation_index"]))
            return record

        for position, candidate in enumerate(candidates):
            rows = []
            for seed in SEEDS:
                row = invoke(
                    candidate["candidate_id"], seed, "fpga_correctness", position=position,
                    reject_wrong=False,
                )
                rows.append(row)
                if not row["correct"]:
                    break
            record = {
                "candidate_id": candidate["candidate_id"],
                "program_id": program_by_candidate[candidate["candidate_id"]],
                "workload_id": candidate["workload_id"],
                "position": position,
                "status": (
                    "passed" if len(rows) == len(SEEDS) and all(row["correct"] for row in rows)
                    else "fpga_rejected"
                ),
                "seed_checks": len(rows),
                "first_failure": next((row for row in rows if not row["correct"]), None),
            }
            correctness.append(record)
            append_jsonl(output / "fpga_correctness.jsonl", record)
            print("correctness {}/{} {}".format(position + 1, len(candidates), record["status"]), flush=True)
        correct_candidate_ids = {
            row["candidate_id"] for row in correctness if row["status"] == "passed"
        }
        correct_program_ids = {program_by_candidate[value] for value in correct_candidate_ids}
        boot1_correct = json.loads((boot1 / "fpga_correctness_summary.json").read_text(encoding="utf-8"))
        boot1_correct_programs = {
            program_by_candidate[value] for value in boot1_correct["fpga_correct_candidate_ids"]
        }
        intersection = correct_program_ids & boot1_correct_programs
        write_json(
            output / "fpga_correctness_summary.json",
            {
                "candidate_count": len(candidates),
                "fpga_correct_count": len(correct_candidate_ids),
                "fpga_rejected_count": len(candidates) - len(correct_candidate_ids),
                "fpga_correct_candidate_ids": sorted(correct_candidate_ids),
                "fpga_rejected_candidate_ids": sorted(set(by_id) - correct_candidate_ids),
                "cross_boot_correct_program_intersection": sorted(intersection),
            },
        )

        confirmation_plan = {}
        for workload_id, program_ids in confirmation["confirmation_program_ids"].items():
            confirmation_plan[workload_id] = [
                program_id for program_id in program_ids if program_id in correct_program_ids
            ]
        write_json(output / "confirmation_timing_plan.json", confirmation_plan)
        for round_index in range(CONFIRMATION_ROUNDS):
            for workload_id in sorted(confirmation_plan):
                programs = confirmation_plan[workload_id]
                if not programs:
                    continue
                order = timing_orders(programs, rounds=CONFIRMATION_ROUNDS)[round_index]
                for position, program_id in enumerate(order):
                    offset = (round_index + position) % len(SEEDS)
                    seed_order = SEEDS[offset:] + SEEDS[:offset]
                    for seed in seed_order:
                        row = invoke(
                            candidate_by_program[program_id], seed, "confirmation_timing",
                            round_index, position,
                        )
                        confirmation_rows.append(row)
                        append_jsonl(output / "confirmation_timing.jsonl", row)
            print("confirmation round {}/{}".format(round_index + 1, CONFIRMATION_ROUNDS), flush=True)
        confirmation_summary = {}
        for workload_id, program_ids in confirmation_plan.items():
            confirmation_summary[workload_id] = {
                "frozen_program_ids": confirmation["confirmation_program_ids"][workload_id],
                "fpga_correct_program_ids": program_ids,
                "program_statistics": {
                    program_id: round_statistics(
                        [
                            row for row in confirmation_rows
                            if row["workload_id"] == workload_id and row["program_id"] == program_id
                        ],
                        CONFIRMATION_ROUNDS,
                    )
                    for program_id in program_ids
                },
            }
        write_json(output / "confirmation_summary.json", confirmation_summary)

        correct_candidates = [
            candidate for candidate in candidates if candidate["candidate_id"] in correct_candidate_ids
        ]
        for position, candidate in enumerate(correct_candidates):
            for seed in SEEDS:
                invoke(candidate["candidate_id"], seed, "reference_warmup", position=position)
        reference_orders = {}
        for workload_id in sorted(workloads):
            ids = sorted(
                program_by_candidate[row["candidate_id"]]
                for row in correct_candidates if row["workload_id"] == workload_id
            )
            reference_orders[workload_id] = timing_orders(ids, rounds=REFERENCE_ROUNDS)
        write_json(output / "reference_orders.json", reference_orders)
        for round_index in range(REFERENCE_ROUNDS):
            for workload_id in sorted(workloads):
                for position, program_id in enumerate(reference_orders[workload_id][round_index]):
                    offset = (round_index + position) % len(SEEDS)
                    seed_order = SEEDS[offset:] + SEEDS[:offset]
                    for seed in seed_order:
                        row = invoke(
                            candidate_by_program[program_id], seed, "reference_timing",
                            round_index, position,
                        )
                        reference_rows.append(row)
                        append_jsonl(output / "reference_timing.jsonl", row)
            write_json(
                output / "environment_round_{:02d}.json".format(round_index),
                board.environment(require_affinity=True),
            )
            print("reference round {}/{}".format(round_index + 1, REFERENCE_ROUNDS), flush=True)
        reference_summary = {}
        for workload_id in sorted(workloads):
            programs = reference_orders[workload_id][0]
            stats = {
                program_id: round_statistics(
                    [
                        row for row in reference_rows
                        if row["workload_id"] == workload_id and row["program_id"] == program_id
                    ],
                    REFERENCE_ROUNDS,
                )
                for program_id in programs
            }
            oracle = min(
                stats,
                key=lambda program_id: (stats[program_id]["median_of_round_medians_ms"], program_id),
            )
            reference_summary[workload_id] = {
                "oracle_scope": confirmation["oracle_scope"],
                "program_statistics": stats,
                "pool_oracle_program_id": oracle,
                "pool_oracle_latency_ms": stats[oracle]["median_of_round_medians_ms"],
                "cross_boot_correct_program_ids": sorted(
                    program_id for program_id in programs if program_id in intersection
                ),
            }
        write_json(output / "reference_summary.json", reference_summary)
        final = board.state()
        reuse_board.validate_state(final)
        final_environment = board.environment(require_affinity=True)
        if final["boot_id"] != initial["boot_id"]:
            raise RuntimeError("boot changed during session")
        if final_environment["governor"] != environment["governor"]:
            raise RuntimeError("CPU governor changed during session")
        write_json(output / "postflight.json", {"state": final, "environment": final_environment})
        summary = {
            "schema": SCHEMA,
            "status": "BOOT2_COMPLETE",
            "boot_id": initial["boot_id"],
            "previous_boot_id": args.previous_boot_id,
            "governor": environment["governor"],
            "candidate_count": len(candidates),
            "fpga_correct_count": len(correct_candidate_ids),
            "fpga_rejected_count": len(candidates) - len(correct_candidate_ids),
            "correctness_seed_checks": sum(row["seed_checks"] for row in correctness),
            "confirmation_timing_samples": len(confirmation_rows),
            "reference_timing_samples": len(reference_rows),
            "complete_operator_calls": len(calls),
            "implicit_extra_calls": 0,
            "all_post_qualification_outputs_correct": True,
            "session_spliced": False,
            "wall_seconds": time.monotonic() - started,
            "next_step": "cross-boot analysis only; selected identities remain immutable",
        }
        write_json(output / "summary.json", summary)
        (output / "command.txt").write_text(
            " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
            encoding="utf-8",
        )
        finalize(output)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        modules.clear()
        retained.clear()
    except Exception as error:
        invalid = {
            "schema": SCHEMA,
            "status": "invalid_entire_session_fail_closed",
            "exception_type": type(error).__name__,
            "message": (str(error) or type(error).__name__)[:4000],
            "traceback": traceback.format_exc()[-12000:],
            "completed_operator_calls": len(calls),
            "correctness_records": len(correctness),
            "confirmation_timing_records": len(confirmation_rows),
            "reference_timing_records": len(reference_rows),
            "may_splice": False,
        }
        write_json(output / "invalid_session.json", invalid)
        write_json(output / "summary.json", invalid)
        finalize(output)
        raise
    finally:
        remote = None
        if board.pid is not None:
            try:
                board.stop()
            except Exception as error:
                write_json(output / "cleanup_warning.json", {"message": str(error)})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-dir", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--board-pool-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--cross-compile-dir", type=Path, default=DEFAULT_CROSS)
    parser.add_argument("--selector-dir", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--runtime-contract", type=Path, default=DEFAULT_RUNTIME_CONTRACT)
    parser.add_argument("--timer-helper", type=Path, default=DEFAULT_HELPER)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--previous-boot-id", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=7200)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
