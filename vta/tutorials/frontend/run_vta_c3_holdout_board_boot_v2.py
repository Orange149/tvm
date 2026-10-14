#!/usr/bin/env python3
"""Run one non-spliceable C3 selector-v2 FPGA boot.

Boot 1 performs the frozen limited-budget selection before any confirmation or
full-pool timing.  Boot 2 accepts only the already frozen winners.  Both boots
run an independent FPGA correctness gate, six-round confirmation, and the two
complete-pool references.
"""

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

from tvm import rpc

import run_vta_c3_reuse_board as reuse_board
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_resnet18_board_pool import verify_inputs
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
from run_vta_p7r132_y00_search_confirmation import (
    allocate_reusable_buffers,
    timing_orders,
)


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_POOL = GROUPED / "20260924_c3_holdout_board_pool_freeze_v2_run01"
DEFAULT_CROSS = GROUPED / "20260924_c3_holdout_cross_compile_v2_run01"
DEFAULT_TASKS = GROUPED / "20260924_c3_holdout_task_freeze_v2_run01"
DEFAULT_SHORTLISTS = GROUPED / "20260924_c3_holdout_shortlists_v2_run02"
DEFAULT_RUNTIME_CONTRACT = GROUPED / "20260922_c3_thread_sync_freeze_run04/contract.json"
DEFAULT_HELPER = GROUPED / "20260922_c3_thread_sync_prepare_run07/binaries/c3_thread_sync_helper.so"
SCHEMA = "c3_holdout_board_boot_v2"
SEEDS = (0, 20250901, 20260910)
SELECTION_ROUNDS = 3
FORMAL_ROUNDS = 6


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


def verify_flat(directory):
    directory = Path(directory).resolve()
    ledger_path = directory / "artifact_hashes.json"
    ledger = read(ledger_path)
    artifacts = ledger.get("artifacts", ledger)
    for name, expected in artifacts.items():
        if not isinstance(expected, str) or name.startswith("source."):
            continue
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("artifact hash mismatch: {}".format(path))
    return sha256_file(ledger_path)


def seal(output):
    write(output / "artifact_hashes.json", {
        "artifacts": {
            path.name: sha256_file(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "artifact_hashes.json"
        },
        "source": sha256_file(Path(__file__).resolve()),
    })


def validate_profile(candidate, result):
    observed = result.get("runtime_profile_complete") or {}
    expected = candidate["dma_features"]["totals"]
    keys = [key for key in expected if key.startswith("load_buffer_2d_") or key.startswith("store_buffer_2d_")]
    mismatched = {key: {"expected": expected[key], "observed": observed.get(key)}
                  for key in keys if observed.get(key) != expected[key]}
    expected_submissions = candidate["command_features"]["values"]["submissions"]
    if observed.get("driver_run_calls") != expected_submissions:
        mismatched["driver_run_calls"] = {
            "expected": expected_submissions, "observed": observed.get("driver_run_calls")
        }
    if mismatched:
        raise ValueError("board/local profile mismatch for {}: {}".format(candidate["candidate_id"], mismatched))


def three_seed_round_statistics(rows, program_id, phase):
    selected = [row for row in rows if row["program_id"] == program_id and row["phase"] == phase]
    medians = []
    for round_index in range(FORMAL_ROUNDS):
        values = [float(row["latency_ms"]) for row in selected if row["round"] == round_index]
        if len(values) != len(SEEDS):
            raise ValueError("{} lacks three seed values in round {}".format(program_id, round_index))
        medians.append(float(statistics.median(values)))
    return {
        "round_medians_ms": medians,
        "median_of_round_medians_ms": float(statistics.median(medians)),
        "all_single_call_values_ms": [float(row["latency_ms"]) for row in selected],
    }


def selection_statistics(rows, program_id):
    selected = [row for row in rows if row["program_id"] == program_id and row["phase"] == "selection_timing"]
    if len(selected) != SELECTION_ROUNDS or sorted(row["round"] for row in selected) != list(range(SELECTION_ROUNDS)):
        raise ValueError("selection program lacks exact three round/single-seed samples")
    values = [float(row["latency_ms"]) for row in selected]
    return {"values_ms": values, "median_ms": float(statistics.median(values))}


def paired_improvement(rows, incumbent, candidate):
    left = three_seed_round_statistics(rows, incumbent, "confirmation_timing")["round_medians_ms"]
    right = three_seed_round_statistics(rows, candidate, "confirmation_timing")["round_medians_ms"]
    gains = [100.0 * (1.0 - test / base) for base, test in zip(left, right)]
    return {
        "round_improvements_pct": gains,
        "median_improvement_pct": float(statistics.median(gains)),
        "wins": sum(value > 0.0 for value in gains),
    }


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    candidates, _, pool_hashes = verify_inputs(args.board_pool_dir, args.cross_compile_dir)
    task_manifest = verify_flat(args.task_dir)
    shortlist_manifest = verify_flat(args.shortlist_dir)
    policies = read(Path(args.shortlist_dir) / "task_policies.json")
    full_tasks = {row["task_id"] for row in read(Path(args.task_dir) / "full_pool_tasks.json")}
    limited_tasks = {row["task_id"] for row in read(Path(args.task_dir) / "limited_budget_tasks.json")}
    runtime_contract = read(args.runtime_contract)
    runtime_paths = runtime_contract["runtime_paths"]
    runtime_hashes = runtime_contract["runtime_sha256"]
    for name, path in runtime_paths.items():
        if sha256_file(path) != runtime_hashes[name]:
            raise ValueError("runtime changed: {}".format(name))
    if sha256_file(args.timer_helper) != runtime_contract["helper_sha256"]:
        raise ValueError("single-call timer helper changed")
    by_candidate = {row["candidate_id"]: row for row in candidates}
    by_program = {row["program_id"]: row for row in candidates}
    if len(by_candidate) != len(candidates) or len(by_program) != len(candidates):
        raise ValueError("board identity is not one-to-one")
    for task_id, policy in policies.items():
        available = {row["program_id"] for row in candidates if row["execution_task_id"] == task_id}
        expected = set(policy["complete_local_program_pool"] if task_id in full_tasks
                       else policy["shortlist"]["shortlist_candidate_ids"])
        if task_id in limited_tasks:
            expected.update(policy["shortlist"]["shortlist_candidate_ids"])
        if available != expected:
            raise ValueError("frozen board/task pool mismatch for {}".format(task_id))

    previous = None
    boot_index = 1
    if args.previous_boot_dir:
        previous_dir = Path(args.previous_boot_dir).resolve()
        verify_flat(previous_dir)
        previous = read(previous_dir / "summary.json")
        if previous.get("status") != "BOOT1_COMPLETE":
            raise ValueError("previous session is not a complete boot 1")
        previous_contract = read(previous_dir / "contract.json")
        if previous_contract["pool_manifest_sha256"] != sha256_file(Path(args.board_pool_dir) / "artifact_hashes.json"):
            raise ValueError("previous session used another board pool")
        boot_index = 2
    output.mkdir(parents=True)
    for name in ("calls.jsonl", "fpga_correctness.jsonl", "selection_timing.jsonl",
                 "confirmation_timing.jsonl", "reference_timing.jsonl", "timeline.jsonl"):
        (output / name).write_text("", encoding="utf-8")
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_this_boot_observation",
        "boot_index": boot_index,
        "previous_boot_dir": str(Path(args.previous_boot_dir).resolve()) if previous else None,
        "pool_manifest_sha256": sha256_file(Path(args.board_pool_dir) / "artifact_hashes.json"),
        "task_manifest_sha256": task_manifest,
        "shortlist_manifest_sha256": shortlist_manifest,
        "pool_input_hashes": pool_hashes,
        "candidate_order": [row["candidate_id"] for row in candidates],
        "program_order": [row["program_id"] for row in candidates],
        "full_pool_task_ids": sorted(full_tasks),
        "limited_budget_task_ids": sorted(limited_tasks),
        "selection_rounds": SELECTION_ROUNDS if boot_index == 1 else 0,
        "selection_seed_by_round": list(SEEDS) if boot_index == 1 else [],
        "formal_rounds": FORMAL_ROUNDS,
        "formal_seeds": list(SEEDS),
        "single_call_helper_sha256": runtime_contract["helper_sha256"],
        "invocations_per_timing_sample": 1,
        "implicit_warmup_calls": 0,
        "qualification_calls_are_common_explicit_warmup": True,
        "buffer_allocation_order": ["output", "data", "weight"],
        "cpu_affinity": 3,
        "governor_policy": "record_only",
        "invalid_candidate_policy": "consumes frozen budget position; no replacement; no timing",
        "session_policy": "interruption, timed wrong output, identity or environment change invalidates whole session; no splicing",
        "board_contacted": False,
        "performance_labels_collected": False,
    }
    write(output / "contract.json", contract)
    write(output / "pre_observation_hashes.json", {"contract.json": sha256_file(output / "contract.json")})

    started = time.monotonic()
    calls, correctness = [], []
    selection_rows, confirmation_rows, reference_rows = [], [], []
    remote = None
    board = ExperimentBoard(args, "/var/volatile/" + output.name + "_timing", args.port)
    try:
        initial = board.state()
        reuse_board.validate_state(initial, exclusive=True)
        environment = board.environment()
        if previous and initial["boot_id"] == previous["boot_id"]:
            raise ValueError("boot 2 requires a changed boot ID")
        write(output / "preflight.json", {"state": initial, "environment": environment})
        reuse_board.RUNTIME = runtime_hashes
        write(output / "runtime_attestation.json", board.deploy(runtime_paths))
        write(output / "bitstream_reload.json", board.reload())
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
        write(output / "runtime_inventory.json", inventory)
        retained = [load_ephemeral(remote, args.timer_helper)]
        timer = remote.get_function("c3.reuse.timed_once")
        binary_dir = Path(args.cross_compile_dir).resolve() / "binaries"
        modules = {}
        for position, candidate in enumerate(candidates, 1):
            modules[candidate["candidate_id"]] = load_ephemeral(
                remote, binary_dir / (candidate["candidate_id"] + ".so")
            )
            print("load {}/{}".format(position, len(candidates)), flush=True)
        workloads = {}
        for candidate in candidates:
            workloads.setdefault(candidate["workload_id"], candidate["identity"]["workload"])
        buffers = {key: allocate_reusable_buffers(remote.ext_dev(0), workload)
                   for key, workload in workloads.items()}
        write(output / "affinity_after_load.json", board.environment(require_affinity=True))

        def invoke(program_id, seed, phase, round_index=None, position=None, task_id=None,
                   reject_wrong=True):
            candidate = by_program[program_id]
            base = {
                "invocation_index": len(calls), "candidate_id": candidate["candidate_id"],
                "program_id": program_id, "execution_task_id": candidate["execution_task_id"],
                "workload_id": candidate["workload_id"], "seed": seed, "phase": phase,
                "round": round_index, "position": position,
            }
            append(output / "timeline.jsonl", {**base, "event": "begin"})
            result = reuse_board.invoke(
                modules[candidate["candidate_id"]], buffers[candidate["workload_id"]],
                candidate["identity"]["workload"], seed, functions, timer,
                timed=phase.endswith("timing"),
            )
            record = {**base, **result}
            calls.append(record)
            append(output / "calls.jsonl", record)
            append(output / "timeline.jsonl", {**base, "event": "end"})
            if record["correct"]:
                validate_profile(candidate, record)
            elif reject_wrong:
                raise RuntimeError("wrong timed output at invocation {}".format(base["invocation_index"]))
            return record

        # Common hard gate.  Invalid candidates remain in the budget history
        # but never receive a timing sample and are never replaced.
        for position, candidate in enumerate(candidates):
            seed_rows = []
            for seed in SEEDS:
                row = invoke(candidate["program_id"], seed, "fpga_correctness", position=position,
                             reject_wrong=False)
                seed_rows.append(row)
                if not row["correct"]:
                    break
            status = "passed" if len(seed_rows) == len(SEEDS) and all(row["correct"] for row in seed_rows) else "fpga_rejected"
            record = {
                "candidate_id": candidate["candidate_id"], "program_id": candidate["program_id"],
                "execution_task_id": candidate["execution_task_id"], "status": status,
                "seed_checks": len(seed_rows),
                "first_failure": next((row for row in seed_rows if not row["correct"]), None),
            }
            correctness.append(record)
            append(output / "fpga_correctness.jsonl", record)
            print("correctness {}/{} {}".format(position + 1, len(candidates), status), flush=True)
        correct = {row["program_id"] for row in correctness if row["status"] == "passed"}
        write(output / "fpga_correctness_summary.json", {
            "program_count": len(candidates), "fpga_correct_count": len(correct),
            "fpga_rejected_count": len(candidates) - len(correct),
            "fpga_correct_program_ids": sorted(correct),
            "fpga_rejected_program_ids": sorted(set(by_program) - correct),
        })

        if boot_index == 1:
            decisions = {}
            for task_id in sorted(limited_tasks):
                policy = policies[task_id]
                planned = policy["shortlist"]["shortlist_candidate_ids"]
                timed = [program_id for program_id in planned if program_id in correct]
                order = timing_orders(timed, rounds=SELECTION_ROUNDS)
                for round_index in range(SELECTION_ROUNDS):
                    for position, program_id in enumerate(order[round_index]):
                        row = invoke(program_id, SEEDS[round_index], "selection_timing",
                                     round_index, position, task_id)
                        selection_rows.append(row)
                        append(output / "selection_timing.jsonl", row)
                stats = {program_id: selection_statistics(selection_rows, program_id) for program_id in timed}
                incumbent = policy["incumbent_program_id"]
                if incumbent not in correct:
                    decisions[task_id] = {
                        "status": "incumbent_fpga_rejected_no_deployment_decision",
                        "incumbent_program_id": incumbent,
                        "attempted_new_program_ids": policy["B1_B2_B3"]["B3"],
                        "fpga_rejected_attempted_program_ids": [x for x in policy["B1_B2_B3"]["B3"] if x not in correct],
                        "program_statistics": stats,
                    }
                    continue
                winner = min(timed, key=lambda program_id: (stats[program_id]["median_ms"], program_id))
                decisions[task_id] = {
                    "status": "winner_frozen_before_confirmation_and_reference",
                    "incumbent_program_id": incumbent,
                    "attempted_new_program_ids": policy["B1_B2_B3"]["B3"],
                    "fpga_rejected_attempted_program_ids": [x for x in policy["B1_B2_B3"]["B3"] if x not in correct],
                    "timing_eligible_program_ids": timed,
                    "selected_program_id": winner,
                    "program_statistics": stats,
                }
            write(output / "selection_decisions_frozen.json", decisions)
        else:
            decisions = read(Path(args.previous_boot_dir) / "selection_decisions_frozen.json")
            write(output / "selection_decisions_reused.json", {
                "source": str(Path(args.previous_boot_dir).resolve()),
                "sha256": sha256_file(Path(args.previous_boot_dir) / "selection_decisions_frozen.json"),
                "decisions": decisions,
            })

        # Formal confirmation uses no selection observation.  Qualification
        # calls above are the uniform explicit warm-up for this boot.
        confirmation_pairs = {}
        for task_id in sorted(limited_tasks):
            decision = decisions[task_id]
            if decision["status"] != "winner_frozen_before_confirmation_and_reference":
                continue
            incumbent, selected = decision["incumbent_program_id"], decision["selected_program_id"]
            ids = list(dict.fromkeys((incumbent, selected)))
            ids = [program_id for program_id in ids if program_id in correct]
            confirmation_pairs[task_id] = ids
            orders = timing_orders(ids, rounds=FORMAL_ROUNDS)
            for round_index in range(FORMAL_ROUNDS):
                for position, program_id in enumerate(orders[round_index]):
                    offset = (round_index + position) % len(SEEDS)
                    seed_order = SEEDS[offset:] + SEEDS[:offset]
                    for seed in seed_order:
                        row = invoke(program_id, seed, "confirmation_timing", round_index, position, task_id)
                        confirmation_rows.append(row)
                        append(output / "confirmation_timing.jsonl", row)
        write(output / "confirmation_pairs.json", confirmation_pairs)

        # Complete-pool reference is restricted to the two pre-frozen tasks.
        reference_programs = {}
        for task_id in sorted(full_tasks):
            ids = [row["program_id"] for row in candidates
                   if row["execution_task_id"] == task_id and row["program_id"] in correct]
            reference_programs[task_id] = sorted(ids)
            orders = timing_orders(sorted(ids), rounds=FORMAL_ROUNDS)
            for round_index in range(FORMAL_ROUNDS):
                for position, program_id in enumerate(orders[round_index]):
                    offset = (round_index + position) % len(SEEDS)
                    seed_order = SEEDS[offset:] + SEEDS[:offset]
                    for seed in seed_order:
                        row = invoke(program_id, seed, "reference_timing", round_index, position, task_id)
                        reference_rows.append(row)
                        append(output / "reference_timing.jsonl", row)
                write(output / "environment_reference_{}_round_{:02d}.json".format(task_id[:12], round_index),
                      board.environment(require_affinity=True))
                print("reference {} round {}/{}".format(task_id[:12], round_index + 1, FORMAL_ROUNDS), flush=True)

        confirmation_summary = {}
        for task_id, ids in confirmation_pairs.items():
            incumbent = decisions[task_id]["incumbent_program_id"]
            selected = decisions[task_id]["selected_program_id"]
            stats = {program_id: three_seed_round_statistics(confirmation_rows, program_id, "confirmation_timing")
                     for program_id in ids}
            confirmation_summary[task_id] = {
                "incumbent_program_id": incumbent, "selected_program_id": selected,
                "program_statistics": stats,
                "paired": paired_improvement(confirmation_rows, incumbent, selected),
            }
        write(output / "confirmation_summary.json", confirmation_summary)
        reference_summary = {}
        for task_id, ids in reference_programs.items():
            stats = {program_id: three_seed_round_statistics(reference_rows, program_id, "reference_timing")
                     for program_id in ids}
            oracle = min(stats, key=lambda program_id: (stats[program_id]["median_of_round_medians_ms"], program_id))
            reference_summary[task_id] = {
                "fpga_correct_program_ids": ids,
                "fpga_rejected_program_ids": sorted(
                    set(policies[task_id]["complete_local_program_pool"]) - set(ids)
                ),
                "program_statistics": stats,
                "pool_oracle_program_id": oracle,
                "pool_oracle_latency_ms": stats[oracle]["median_of_round_medians_ms"],
            }
        write(output / "reference_summary.json", reference_summary)

        final = board.state()
        reuse_board.validate_state(final)
        final_environment = board.environment(require_affinity=True)
        if final["boot_id"] != initial["boot_id"]:
            raise RuntimeError("boot changed during session")
        if final_environment["governor"] != environment["governor"]:
            raise RuntimeError("CPU governor changed during session")
        write(output / "postflight.json", {"state": final, "environment": final_environment})
        summary = {
            "schema": SCHEMA, "status": "BOOT{}_COMPLETE".format(boot_index),
            "boot_id": initial["boot_id"], "governor": environment["governor"],
            "candidate_count": len(candidates), "fpga_correct_count": len(correct),
            "fpga_rejected_count": len(candidates) - len(correct),
            "correctness_seed_checks": sum(row["seed_checks"] for row in correctness),
            "selection_timing_samples": len(selection_rows),
            "confirmation_timing_samples": len(confirmation_rows),
            "reference_timing_samples": len(reference_rows),
            "complete_operator_calls": len(calls), "implicit_extra_calls": 0,
            "all_post_qualification_outputs_correct": True,
            "session_spliced": False, "wall_seconds": time.monotonic() - started,
            "next_step": "user reboot then boot 2" if boot_index == 1 else "cross-boot analysis",
        }
        write(output / "summary.json", summary)
        (output / "command.txt").write_text(
            " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
            encoding="utf-8",
        )
        seal(output)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        modules.clear()
        retained.clear()
    except Exception as error:
        invalid = {
            "schema": SCHEMA, "status": "invalid_entire_session_fail_closed",
            "exception_type": type(error).__name__, "message": (str(error) or type(error).__name__)[:4000],
            "traceback": traceback.format_exc()[-12000:], "completed_operator_calls": len(calls),
            "correctness_records": len(correctness), "selection_timing_records": len(selection_rows),
            "confirmation_timing_records": len(confirmation_rows), "reference_timing_records": len(reference_rows),
            "may_splice": False,
        }
        write(output / "invalid_session.json", invalid)
        write(output / "summary.json", invalid)
        seal(output)
        raise
    finally:
        remote = None
        if board.pid is not None:
            try:
                board.stop()
            except Exception as error:
                write(output / "cleanup_warning.json", {"message": str(error)})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-pool-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--cross-compile-dir", type=Path, default=DEFAULT_CROSS)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--shortlist-dir", type=Path, default=DEFAULT_SHORTLISTS)
    parser.add_argument("--runtime-contract", type=Path, default=DEFAULT_RUNTIME_CONTRACT)
    parser.add_argument("--timer-helper", type=Path, default=DEFAULT_HELPER)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--previous-boot-dir", type=Path)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=7200)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
