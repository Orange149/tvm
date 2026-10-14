#!/usr/bin/env python3
"""Run one non-spliceable FPGA boot for the frozen real-task reference pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from tvm import rpc

import run_vta_c3_reuse_board as reuse_board
import run_vta_c3_reuse_supplement as supplement
from c3_thread_sync_board import ExperimentBoard
from qualify_vta_residency_fsim import array_sha256
from run_vta_c3_resnet18_board_pool import verify_inputs
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
from run_vta_p7r132_y00_search_confirmation import (
    allocate_reusable_buffers,
    prepare_reused_buffers,
    timing_orders,
)


HERE = Path(__file__).resolve().parent
C3 = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
GROUPED = C3 / "07_grouped_holdout"
DEFAULT_POOL = GROUPED / "20260924_c3_real_task_board_pool_freeze_run01"
DEFAULT_CROSS = GROUPED / "20260924_c3_real_task_cross_compile_run01"
DEFAULT_SELECTOR = GROUPED / "20260924_c3_real_task_selector_freeze_run01"
DEFAULT_RUNTIME_CONTRACT = GROUPED / "20260922_c3_thread_sync_freeze_run04/contract.json"
DEFAULT_HELPER = GROUPED / "20260922_c3_thread_sync_prepare_run07/binaries/c3_thread_sync_helper.so"
SCHEMA = "c3_real_task_board_boot_v2"
SEEDS = (0, 20250901, 20260910)
SELECTION_ROUNDS = 2
REFERENCE_ROUNDS = 6


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def verify_flat(directory):
    directory = Path(directory).resolve()
    ledger = json.loads((directory / "artifact_hashes.json").read_text(encoding="utf-8"))
    for name, expected in ledger["artifacts"].items():
        if isinstance(expected, str) and sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: {}".format(name))
    return sha256_file(directory / "artifact_hashes.json")


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


def round_statistics(rows, rounds):
    per_round = []
    for round_index in range(rounds):
        values = [float(row["latency_ms"]) for row in rows if row["round"] == round_index]
        if len(values) != len(SEEDS):
            raise ValueError("candidate lacks exact three-seed round")
        per_round.append(statistics.median(values))
    return {
        "round_medians_ms": per_round,
        "median_of_round_medians_ms": statistics.median(per_round),
        "all_single_call_values_ms": [float(row["latency_ms"]) for row in rows],
    }


def selection_plan(selector):
    union = {}
    for workload_id, frozen in selector["workloads"].items():
        ids = {frozen["incumbent_program_id"]}
        for order in frozen["orders"].values():
            ids.update(order[:3])
        union[workload_id] = sorted(ids)
    return union


def selection_timing_plan(selector, fpga_correct_program_ids):
    """Return the frozen selection union after the FPGA correctness hard gate."""
    correct = set(fpga_correct_program_ids)
    return {
        workload_id: [program_id for program_id in programs if program_id in correct]
        for workload_id, programs in selection_plan(selector).items()
    }


def choose_programs(selector, observations, fpga_correct_program_ids):
    """Apply frozen budget orders; rejected programs consume budget but have no latency."""
    correct = set(fpga_correct_program_ids)
    by_program = defaultdict(list)
    for row in observations:
        by_program[(row["workload_id"], row["program_id"])].append(row)
    decisions = {}
    for workload_id, frozen in selector["workloads"].items():
        planned = selection_plan(selector)[workload_id]
        stats = {
            program_id: round_statistics(by_program[workload_id, program_id], SELECTION_ROUNDS)
            for program_id in planned
            if program_id in correct
        }
        incumbent = frozen["incumbent_program_id"]
        if incumbent not in correct:
            decisions[workload_id] = {
                "status": "incumbent_fpga_rejected_no_selector_evaluation",
                "incumbent_program_id": incumbent,
                "program_statistics": stats,
                "budget_decisions": {},
                "decision_frozen_before_full_pool_reference": True,
            }
            continue
        budgets = {}
        for method, order in frozen["orders"].items():
            budgets[method] = {}
            for budget in (1, 2, 3):
                attempted = order[:budget]
                eligible = [incumbent] + [program_id for program_id in attempted if program_id in correct]
                winner = min(
                    eligible,
                    key=lambda program_id: (
                        stats[program_id]["median_of_round_medians_ms"], program_id
                    ),
                )
                budgets[method][str(budget)] = {
                    "attempted_new_program_ids": attempted,
                    "fpga_rejected_attempted_program_ids": [
                        program_id for program_id in attempted if program_id not in correct
                    ],
                    "eligible_program_ids": eligible,
                    "selected_program_id": winner,
                    "selection_latency_ms": stats[winner]["median_of_round_medians_ms"],
                }
        decisions[workload_id] = {
            "status": "selected_from_fpga_correct_budget_prefix",
            "program_statistics": stats,
            "budget_decisions": budgets,
            "decision_frozen_before_full_pool_reference": True,
        }
    return decisions


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    candidates, _, input_hashes = verify_inputs(args.board_pool_dir, args.cross_compile_dir)
    selector_manifest = verify_flat(args.selector_dir)
    selector = json.loads((Path(args.selector_dir) / "contract.json").read_text(encoding="utf-8"))
    if selector.get("status") != "frozen_before_current_board_latency":
        raise ValueError("selector is not latency blind")
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
    for name in ("calls.jsonl", "selection_timing.jsonl", "reference_timing.jsonl", "timeline.jsonl"):
        (output / name).write_text("", encoding="utf-8")
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_this_boot_observation",
        "boot_index": args.boot_index,
        "pool_input_hashes": input_hashes,
        "selector_manifest_sha256": selector_manifest,
        "candidate_order": [row["candidate_id"] for row in candidates],
        "selection_program_union": selection_plan(selector),
        "qualification_order": [row["candidate_id"] for row in candidates],
        "qualification_gate": (
            "all candidates before any timing; three seeds fail-fast; wrong output rejects only "
            "that candidate; execution interruption invalidates the session"
        ),
        "selection_rounds": SELECTION_ROUNDS,
        "reference_rounds": REFERENCE_ROUNDS,
        "seeds": list(SEEDS),
        "single_call_helper_sha256": runtime_contract["helper_sha256"],
        "invocations_per_timing_sample": 1,
        "implicit_warmup_calls": 0,
        "buffer_allocation_order": ["output", "data", "weight"],
        "cpu_affinity": 3,
        "governor_policy": "record_only",
        "oracle_scope": "locally qualified and FPGA-correct candidates in the frozen pool",
        "session_policy": (
            "candidate wrong output during the pre-timing qualification gate rejects that "
            "candidate; any interruption, wrong warm-up/timed output after qualification, boot "
            "or affinity change invalidates the whole session"
        ),
        "prior_invalid_session_disclosure": None,
        "board_contacted": False,
        "performance_labels_collected": False,
    }
    if args.prior_invalid_session_dir is not None:
        prior = Path(args.prior_invalid_session_dir).resolve()
        prior_summary = json.loads((prior / "summary.json").read_text(encoding="utf-8"))
        if prior_summary.get("status") != "invalid_entire_session_fail_closed":
            raise ValueError("prior disclosed session is not invalid")
        contract["prior_invalid_session_disclosure"] = {
            "path": str(prior),
            "summary_sha256": sha256_file(prior / "summary.json"),
            "selection_decisions_sha256": (
                sha256_file(prior / "selection_decisions_frozen.json")
                if (prior / "selection_decisions_frozen.json").exists()
                else None
            ),
            "policy": (
                "no rows or decisions are reused; tasks are disclosed as latency-exposed by an "
                "invalid non-spliceable attempt; frozen selector identities and rules are unchanged"
            ),
        }
    write_json(output / "contract.json", contract)
    write_json(output / "pre_observation_hashes.json", {"contract.json": sha256_file(output / "contract.json")})
    started = time.monotonic()
    calls, correctness, selection_rows, reference_rows = [], [], [], []
    remote = None
    board = ExperimentBoard(args, "/var/volatile/" + output.name + "_timing", args.port)
    try:
        initial = board.state()
        reuse_board.validate_state(initial, exclusive=True)
        environment = board.environment()
        if args.previous_boot_id and initial["boot_id"] == args.previous_boot_id:
            raise ValueError("second boot requires a changed boot ID")
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

        def invoke(
            candidate_id,
            seed,
            phase,
            round_index=None,
            position=None,
            reject_wrong_output=True,
        ):
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
            if reject_wrong_output and not record["correct"]:
                raise RuntimeError("wrong output at invocation {}".format(base["invocation_index"]))
            return record

        # Hard gate first: local qualification does not replace real FPGA correctness.
        correctness_path = output / "fpga_correctness.jsonl"
        correctness_path.write_text("", encoding="utf-8")
        for position, candidate in enumerate(candidates):
            rows = []
            for seed in SEEDS:
                row = invoke(
                    candidate["candidate_id"],
                    seed,
                    "fpga_correctness",
                    position=position,
                    reject_wrong_output=False,
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
                    "passed"
                    if len(rows) == len(SEEDS) and all(row["correct"] for row in rows)
                    else "fpga_rejected"
                ),
                "seed_checks": len(rows),
                "first_failure": next((row for row in rows if not row["correct"]), None),
            }
            correctness.append(record)
            append_jsonl(correctness_path, record)
            print(
                "correctness {}/{} {}".format(position + 1, len(candidates), record["status"]),
                flush=True,
            )
        fpga_correct_candidate_ids = {
            row["candidate_id"] for row in correctness if row["status"] == "passed"
        }
        fpga_correct_program_ids = {
            program_by_candidate[candidate_id] for candidate_id in fpga_correct_candidate_ids
        }
        write_json(
            output / "fpga_correctness_summary.json",
            {
                "candidate_count": len(candidates),
                "fpga_correct_count": len(fpga_correct_candidate_ids),
                "fpga_rejected_count": len(candidates) - len(fpga_correct_candidate_ids),
                "fpga_correct_candidate_ids": sorted(fpga_correct_candidate_ids),
                "fpga_rejected_candidate_ids": sorted(set(by_id) - fpga_correct_candidate_ids),
                "oracle_scope": contract["oracle_scope"],
            },
        )

        # Qualification calls are the common three-seed selection warm-up.
        selected_plan = selection_timing_plan(selector, fpga_correct_program_ids)
        write_json(output / "selection_timing_plan.json", selected_plan)
        for workload_id, programs in selected_plan.items():
            if not programs:
                continue
            order = timing_orders(programs, rounds=SELECTION_ROUNDS)
            for round_index in range(SELECTION_ROUNDS):
                for position, program_id in enumerate(order[round_index]):
                    candidate_id = candidate_by_program[program_id]
                    seed_order = SEEDS[round_index % len(SEEDS):] + SEEDS[:round_index % len(SEEDS)]
                    for seed in seed_order:
                        row = invoke(candidate_id, seed, "selection_timing", round_index, position)
                        selection_rows.append(row)
                        append_jsonl(output / "selection_timing.jsonl", row)
        decisions = choose_programs(selector, selection_rows, fpga_correct_program_ids)
        write_json(output / "selection_decisions_frozen.json", decisions)

        # Uniform reference warm-up, collected only after selection decisions are sealed.
        correct_candidates = [
            candidate
            for candidate in candidates
            if candidate["candidate_id"] in fpga_correct_candidate_ids
        ]
        for position, candidate in enumerate(correct_candidates):
            for seed in SEEDS:
                invoke(candidate["candidate_id"], seed, "reference_warmup", position=position)
        reference_orders = {}
        for workload_id in sorted(workloads):
            ids = sorted(
                program_by_candidate[row["candidate_id"]]
                for row in correct_candidates
                if row["workload_id"] == workload_id
            )
            reference_orders[workload_id] = timing_orders(ids, rounds=REFERENCE_ROUNDS)
        write_json(output / "reference_orders.json", reference_orders)
        for round_index in range(REFERENCE_ROUNDS):
            for workload_id in sorted(workloads):
                for position, program_id in enumerate(reference_orders[workload_id][round_index]):
                    candidate_id = candidate_by_program[program_id]
                    offset = (round_index + position) % len(SEEDS)
                    seed_order = SEEDS[offset:] + SEEDS[:offset]
                    for seed in seed_order:
                        row = invoke(candidate_id, seed, "reference_timing", round_index, position)
                        reference_rows.append(row)
                        append_jsonl(output / "reference_timing.jsonl", row)
            write_json(
                output / "environment_round_{:02d}.json".format(round_index),
                board.environment(require_affinity=True),
            )
            print("reference round {}/{}".format(round_index + 1, REFERENCE_ROUNDS), flush=True)

        reference_summary = {}
        for workload_id in sorted(workloads):
            stats = {}
            correct_programs = [
                program_id
                for program_id in selector["workloads"][workload_id]["full_pool_program_ids"]
                if program_id in fpga_correct_program_ids
            ]
            for program_id in correct_programs:
                rows = [
                    row
                    for row in reference_rows
                    if row["workload_id"] == workload_id and row["program_id"] == program_id
                ]
                stats[program_id] = round_statistics(rows, REFERENCE_ROUNDS)
            oracle = min(
                stats,
                key=lambda program_id: (stats[program_id]["median_of_round_medians_ms"], program_id),
            )
            reference_summary[workload_id] = {
                "oracle_scope": contract["oracle_scope"],
                "local_full_pool_program_ids": selector["workloads"][workload_id][
                    "full_pool_program_ids"
                ],
                "fpga_correct_program_ids": correct_programs,
                "fpga_rejected_program_ids": [
                    program_id
                    for program_id in selector["workloads"][workload_id]["full_pool_program_ids"]
                    if program_id not in fpga_correct_program_ids
                ],
                "program_statistics": stats,
                "pool_oracle_program_id": oracle,
                "pool_oracle_latency_ms": stats[oracle]["median_of_round_medians_ms"],
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
            "status": "BOOT{}_COMPLETE".format(args.boot_index),
            "boot_id": initial["boot_id"],
            "governor": environment["governor"],
            "candidate_count": len(candidates),
            "fpga_correct_count": len(fpga_correct_candidate_ids),
            "fpga_rejected_count": len(candidates) - len(fpga_correct_candidate_ids),
            "correctness_seed_checks": sum(row["seed_checks"] for row in correctness),
            "selection_timing_samples": len(selection_rows),
            "reference_timing_samples": len(reference_rows),
            "complete_operator_calls": len(calls),
            "implicit_extra_calls": 0,
            "all_post_qualification_outputs_correct": True,
            "latency_exposure_status": (
                "exposed_by_prior_invalid_attempt_nonadaptive_rerun"
                if args.prior_invalid_session_dir is not None
                else "first_attempt"
            ),
            "session_spliced": False,
            "wall_seconds": time.monotonic() - started,
            "next_step": "independent later-boot confirmation; selected identities may not change",
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
            "selection_timing_records": len(selection_rows),
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
    parser.add_argument("--board-pool-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--cross-compile-dir", type=Path, default=DEFAULT_CROSS)
    parser.add_argument("--selector-dir", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--runtime-contract", type=Path, default=DEFAULT_RUNTIME_CONTRACT)
    parser.add_argument("--timer-helper", type=Path, default=DEFAULT_HELPER)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--boot-index", type=int, choices=(1, 2), default=1)
    parser.add_argument("--previous-boot-id")
    parser.add_argument("--prior-invalid-session-dir", type=Path)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=7200)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
