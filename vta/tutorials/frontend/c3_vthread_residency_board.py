"""One-boot development measurement for the three frozen vthread variants."""

from __future__ import annotations

import json
from pathlib import Path
import time

import c3_thread_sync_audit as instruction_audit
import c3_vthread_residency_protocol as protocol
import run_vta_c3_reuse_board as board_common
import run_vta_c3_reuse_supplement as s
from run_vta_c3_thread_sync_ablation import SCHEMA as OLD_SCHEMA  # identity guard only
from c3_thread_sync_board import ExperimentBoard


SCHEMA = "c3_vthread_weight_residency_v1"


def _validate_profile(row, result, require_warmed_signature):
    observed = result["runtime_profile_complete"]
    expected = row["local_qualification"]["semantics"]["totals"]
    if any(observed.get(key) != value for key, value in expected.items()):
        raise ValueError("board/static data requests differ: " + row["candidate_id"])
    if require_warmed_signature:
        local = row["command_qualification"]["calls"][-1]["profile"]
        if protocol.logical_signature(observed) != protocol.logical_signature(local):
            raise ValueError("board/local warmed command signature differs")
    expected_submits = row["command_qualification"]["instruction_audits"]["0"][
        "submission_count"
    ]
    if observed["driver_run_calls"] != expected_submits:
        raise ValueError("board/local submission count differs")


def run(args, out):
    if OLD_SCHEMA == SCHEMA:
        raise ValueError("new pilot must not alias the sealed eight-version schema")
    if not args.execute_board:
        raise ValueError("board phase requires --execute-board")
    import numpy as np
    from tvm import rpc
    from qualify_vta_residency_fsim import array_sha256
    from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
    from run_vta_p7r132_y00_search_confirmation import (
        allocate_reusable_buffers,
        prepare_reused_buffers,
    )

    frozen = Path(args.contract).resolve().parent
    contract = s.verify(frozen)
    if contract.get("phase") != "freeze" or contract.get("schema") != SCHEMA:
        raise ValueError("board requires the frozen vthread-residency contract")
    s.check_bindings(contract["source_bindings"])
    prepared = Path(contract["prepared_directory"])
    s.verify(prepared)
    if s.file_hash(prepared / "artifact_hashes.json") != contract["prepared_manifest_sha256"]:
        raise ValueError("prepared artifacts changed")
    for name, path in contract["runtime_paths"].items():
        if s.file_hash(path) != contract["runtime_sha256"][name]:
            raise ValueError("qualified runtime changed: " + name)
    rows = {row["candidate_id"]: row for row in s.read(prepared / "candidates.json")}
    selected = list(contract["accepted"])
    prefix = protocol.candidate_prefix(selected)
    baseline_id = prefix + "_" + protocol.VARIANTS[0]
    if selected != [prefix + "_" + variant for variant in protocol.VARIANTS]:
        raise ValueError("frozen candidate set changed")

    s.write(
        out / "contract.json",
        {
            "schema": SCHEMA,
            "phase": "board",
            "frozen_directory": str(frozen),
            "frozen_manifest_sha256": s.file_hash(frozen / "artifact_hashes.json"),
            "source_bindings": contract["source_bindings"],
            "development_session_only": True,
            "single_boot": True,
            "hidden_extra_calls": 0,
            "invocation_per_sample": 1,
            "no_transparent_retry": True,
            "software_policy": {
                "cpu": 3,
                "threads": 1,
                "governor": "record_only",
                "replay": "disabled",
                "session_ports": {"diagnostic": 9090, "timing": 9091},
            },
        },
    )
    (out / "diagnostic_dumps").mkdir()
    boot_id, governor = None, None
    diagnostic_signatures = {}
    calls, timing = [], []
    board_common.RUNTIME = contract["runtime_sha256"]

    for session in ("diagnostic", "timing"):
        port = {"diagnostic": 9090, "timing": 9091}[session]
        board = ExperimentBoard(args, "/var/volatile/" + out.name + "_" + session, port)
        remote, modules, retained = None, {}, []
        try:
            state = board.state()
            board_common.validate_state(state, exclusive=True)
            environment = board.environment()
            s.write(out / (session + "_preflight.json"), {"state": state, "environment": environment})
            if boot_id is not None and state["boot_id"] != boot_id:
                raise ValueError("boot changed between sessions")
            boot_id = state["boot_id"]
            if governor is not None and environment["governor"] != governor:
                raise ValueError("CPU governor changed between sessions")
            governor = environment["governor"]
            board.deploy(contract["runtime_paths"])
            s.write(out / (session + "_programming.json"), board.reload())
            board.start()
            for _ in range(20):
                try:
                    remote = rpc.connect(args.host, port, session_timeout=7200)
                    break
                except Exception:  # connection retry never invokes an operator
                    time.sleep(0.25)
            if remote is None:
                raise RuntimeError("RPC unavailable")
            inventory, functions = runtime_inventory(remote)
            s.write(out / (session + "_inventory.json"), inventory)
            helper = prepared / "binaries/c3_thread_sync_helper.so"
            if s.file_hash(helper) != contract["helper_sha256"]:
                raise ValueError("timer helper identity mismatch")
            retained.append(load_ephemeral(remote, helper))
            timer = remote.get_function("c3.reuse.timed_once")
            diagnostic = remote.get_function("c3.thread_sync.diagnostic_once")
            buffers = allocate_reusable_buffers(
                remote.ext_dev(0), rows[baseline_id]["identity"]["workload"]
            )
            s.write(out / (session + "_affinity.json"), board.environment(require_affinity=True))

            def call(candidate_id, seed, phase, spec=None):
                row = rows[candidate_id]
                if candidate_id not in modules:
                    binary = prepared / "binaries" / (candidate_id + ".so")
                    if s.file_hash(binary) != row["cross_qualification"]["binary_sha256"]:
                        raise ValueError("operator binary changed")
                    modules[candidate_id] = load_ephemeral(remote, binary)
                record = {
                    "candidate_id": candidate_id,
                    "variant": row["variant"],
                    "seed": seed,
                    "phase": phase,
                    "session": session,
                    "invocation_index": len(calls),
                }
                s.append(out / "dispatch.jsonl", {**record, "event": "begin"})
                if session == "diagnostic":
                    expected = prepare_reused_buffers(
                        buffers, row["identity"]["workload"], seed
                    )
                    functions["vta.runtime.profiler_clear"]()
                    dump = str(
                        diagnostic(
                            modules[candidate_id],
                            buffers["data"],
                            buffers["weight"],
                            buffers["output"],
                        )
                    )
                    actual = buffers["output"].numpy()
                    profile = json.loads(functions["vta.runtime.profiler_status"]())
                    result = {
                        "correct": bool(np.array_equal(actual, expected)),
                        "mismatch_count": int(np.count_nonzero(actual != expected)),
                        "actual_sha256": array_sha256(actual),
                        "expected_sha256": array_sha256(expected),
                        "runtime_profile_complete": profile,
                        "complete_operator_calls": 1,
                        "hidden_warmup_calls": 0,
                    }
                    name = "{:04d}_{}_{}_{}.txt".format(
                        len(calls), candidate_id, phase, seed
                    )
                    (out / "diagnostic_dumps" / name).write_text(dump)
                    result["instruction_audit"] = instruction_audit.instruction_dump(dump)
                    if result["instruction_audit"]["submission_count"] != profile[
                        "driver_run_calls"
                    ]:
                        raise ValueError("board dump/submission mismatch")
                    result["dump_file"] = "diagnostic_dumps/" + name
                else:
                    result = board_common.invoke(
                        modules[candidate_id],
                        buffers,
                        row["identity"]["workload"],
                        seed,
                        functions,
                        timer,
                        timed=phase == "timing",
                    )
                record.update(result)
                if spec:
                    record.update(spec)
                calls.append(record)
                s.append(out / "calls.jsonl", record)
                s.append(
                    out / "dispatch.jsonl",
                    {"invocation_index": record["invocation_index"], "event": "end"},
                )
                if not record["correct"]:
                    return record
                _validate_profile(row, record, phase == "diagnostic")
                if session == "diagnostic" and phase == "diagnostic":
                    diagnostic_signatures[candidate_id, seed] = protocol.logical_signature(
                        record["runtime_profile_complete"]
                    )
                if phase == "timing":
                    record["diagnostic_signature_matches"] = diagnostic_signatures.get(
                        (candidate_id, seed)
                    ) == protocol.logical_signature(record["runtime_profile_complete"])
                    timing.append(record)
                    s.append(out / "timing.jsonl", record)
                return record

            for seed in protocol.SEEDS:
                if not call(baseline_id, seed, "health")["correct"]:
                    raise ValueError(baseline_id + " health output incorrect")
            if session == "diagnostic":
                for candidate_id in selected:
                    for seed in protocol.SEEDS:
                        if not call(candidate_id, seed, "correctness_warmup")["correct"]:
                            raise ValueError(candidate_id + " board correctness failed")
                    for seed in protocol.SEEDS:
                        if not call(candidate_id, seed, "diagnostic")["correct"]:
                            raise ValueError(candidate_id + " diagnostic correctness failed")
                    print("board qualification", candidate_id, True, flush=True)
            else:
                for candidate_id in selected:
                    for seed in protocol.SEEDS:
                        if not call(candidate_id, seed, "warmup")["correct"]:
                            raise ValueError(candidate_id + " timing warmup failed")
                for spec in protocol.schedule(selected):
                    if not call(spec["candidate_id"], spec["seed"], "timing", spec)[
                        "correct"
                    ]:
                        raise ValueError("timed output incorrect; whole session invalid")
                protocol.validate_timing(timing, selected)
            final = board.state()
            board_common.validate_state(final)
            final_environment = board.environment(require_affinity=True)
            if final["boot_id"] != boot_id or final_environment["governor"] != governor:
                raise ValueError("boot/environment changed")
            s.write(
                out / (session + "_postflight.json"),
                {"state": final, "environment": final_environment},
            )
            s.check_bindings(contract["source_bindings"])
        finally:
            modules.clear()
            retained.clear()
            remote = None
            if board.pid is not None:
                board.stop()

    s.write(
        out / "summary.json",
        {
            "status": "DEVELOPMENT_BOOT_COMPLETE",
            "timing_valid": True,
            "boot_id": boot_id,
            "governor": governor,
            "accepted": selected,
            "timing_samples": len(timing),
            "timed_operator_calls": len(timing),
            "formal_warmup_calls": sum(
                row["session"] == "timing" and row["phase"] == "warmup" for row in calls
            ),
            "all_completed_operator_calls": len(calls),
            "hidden_extra_calls": 0,
            "all_timing_correct": True,
            "all_diagnostic_signatures_match": all(
                row["diagnostic_signature_matches"] for row in timing
            ),
            "cross_boot_stability_claim": False,
        },
    )
