#!/usr/bin/env python3
"""Bounded correctness diagnosis, not a restarted/stitched confirmation session.

Reuses the sealed binaries/runtime, compares immediate output with a delayed
read without re-executing the operator. No performance labels are promoted.
"""
import argparse
import json
from pathlib import Path
import shlex
import time

import numpy as np
import tvm
from tvm import rpc

from run_vta_c3_reuse_supplement import read, write, append, seal, verify, check_bindings, file_hash, SEEDS
from run_vta_c3_reuse_board import Board, validate_state, invoke
from run_vta_p7r120_yolo_rpc_board import runtime_inventory, load_ephemeral
from run_vta_p7r132_y00_search_confirmation import allocate_reusable_buffers
from qualify_vta_residency_fsim import reference_data_from_workload, array_sha256


class DiagnosticBoard(Board):
    def stop(self):
        # cpp_rpc has a timeout child surviving the listening parent's exit.
        # Resolve cwd AND executable for each PID; never kill a foreign session.
        path = shlex.quote(self.runtime)
        executable = shlex.quote(self.runtime + "/tvm_rpc")
        return self.ssh("for c3_rpc_pid in $(pidof tvm_rpc); do "
                        "case $c3_rpc_pid in ''|*[!0-9]*) exit 9;; esac; "
                        "if test \"$(readlink /proc/$c3_rpc_pid/cwd)\" = " + path
                        + " && test \"$(readlink /proc/$c3_rpc_pid/exe)\" = " + executable
                        + "; then kill $c3_rpc_pid; fi; done")


def run(args):
    started = time.monotonic()
    prepared_dir = Path(args.contract).resolve().parent
    prepared = verify(prepared_dir)
    check_bindings(prepared["source_bindings"])
    parent = Path(prepared["parent_directory"])
    verify(parent)
    candidates = read(parent / "candidates.json")
    # Selection uses only the failed identity/family, never their latency rank.
    families = ("M2D0140", "M2D0078", "M2D0071")
    lookup = {(r["family_id"], r["residence_mode"]): r for r in candidates}
    order = ("original", "input_weight_resident_barrier", "weight_resident_barrier", "input_weight_resident_barrier")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    board = DiagnosticBoard(args, "/var/volatile/" + output.name + "/runtime")
    results, remote = [], None
    contract = {"phase": "completion_diagnosis", "parent_directory": str(parent),
                "prepared_directory": str(prepared_dir),
                "parent_manifest_sha256": file_hash(parent / "artifact_hashes.json"),
                "source_bindings": {**prepared["source_bindings"], str(Path(__file__).resolve()): file_hash(__file__)},
                "families": list(families), "order": list(order), "seeds": list(SEEDS),
                "delayed_read_seconds": .01, "delayed_read_reexecutes_operator": False,
                "maximum_complete_operator_calls": 42, "formal_performance_labels": False,
                "selection": "three M2 combined correctness/late-timing failure families; no latency-based selection"}
    write(output / "contract.json", contract)
    append(output / "timeline.jsonl", {"event": "W0", "wall_unix": time.time()})
    try:
        initial = board.state()
        validate_state(initial, exclusive=True)
        write(output / "preflight.json", initial)
        board.deploy(prepared["runtime_paths"])
        board.reload()
        board.start()
        for _ in range(20):
            try:
                remote = rpc.connect(args.host, 9090, session_timeout=120)
                break
            except Exception:
                time.sleep(.25)
        if remote is None:
            raise RuntimeError("diagnostic RPC unavailable")
        inventory, functions = runtime_inventory(remote)
        write(output / "runtime_inventory.json", inventory)
        device = remote.ext_dev(0)
        buffers, modules = {}, {}

        def call(row, seed, phase, **extra):
            workload = row["identity"]["workload"]
            key, cid = row["workload_id"], row["candidate_id"]
            if key not in buffers:
                buffers[key] = allocate_reusable_buffers(device, workload)
            if cid not in modules:
                directory = prepared_dir if key == "HEALTH" else parent
                binary = directory / "binaries" / (cid + ".so")
                if file_hash(binary) != row["cross_qualification"]["binary_sha256"]:
                    raise ValueError("binary changed")
                modules[cid] = load_ephemeral(remote, binary)
            append(output / "timeline.jsonl", {"event": "dispatch_begin", "candidate_id": cid, "seed": seed, "phase": phase})
            result = invoke(modules[cid], buffers[key], workload, seed, functions, None, False)
            # No second execution: check whether the shared output changed later.
            time.sleep(.01)
            delayed = buffers[key]["output"].numpy()
            _, _, expected = reference_data_from_workload(workload, seed)
            result.update(candidate_id=cid, family_id=row["family_id"], residence_mode=row["residence_mode"],
                          phase=phase, delayed_correct=bool(np.array_equal(delayed, expected)),
                          delayed_mismatch_count=int(np.count_nonzero(delayed != expected)),
                          delayed_sha256=array_sha256(delayed),
                          changed_after_return=array_sha256(delayed) != result["actual_sha256"], **extra)
            results.append(result)
            append(output / "results.jsonl", result)
            print("{} {} seed={} immediate={} delayed={} driver_us={}".format(
                row["family_id"], row["residence_mode"], seed, result["correct"], result["delayed_correct"],
                result["runtime_profile_complete"]["driver_run_total_us"]), flush=True)
            return result

        for seed in SEEDS:
            if not call(prepared["health"], seed, "health_before")["correct"]:
                raise RuntimeError("diagnostic initial health failed")
        for family in families:
            for seed in SEEDS:
                for position, mode in enumerate(order):
                    call(lookup[family, mode], seed, "diagnosis", sequence_position=position)
        for seed in SEEDS:
            if not call(prepared["health"], seed, "health_after")["correct"]:
                raise RuntimeError("diagnostic final health failed")
        final = board.state()
        validate_state(final)
        if initial["boot_id"] != final["boot_id"]:
            raise RuntimeError("boot changed")
        write(output / "final_state.json", final)
        wrong = [r for r in results if not r["correct"]]
        write(output / "summary.json", {"status": "CORRECTNESS_DIAGNOSIS_COMPLETE_NO_PERFORMANCE_CLAIM",
               "boot_id": initial["boot_id"], "completed_operator_calls": len(results),
               "immediate_wrong": len(wrong), "wrong_became_correct_without_reexecution": sum(r["delayed_correct"] for r in wrong),
               "output_changed_after_return": sum(r["changed_after_return"] for r in results),
               "wall_seconds": time.monotonic() - started,
               "does_not_repair_or_resume_invalid_C1_session": True})
    except BaseException as error:
        write(output / "invalid_session.json", {"message": str(error), "completed_calls": len(results),
                                                "wall_seconds": time.monotonic() - started})
        raise
    finally:
        if remote is not None:
            del remote
        try:
            board.stop()
        except Exception as error:
            write(output / "cleanup_warning.json", {"message": str(error)})
        append(output / "timeline.jsonl", {"event": "T1", "elapsed_seconds": time.monotonic() - started})
        seal(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    run(parser.parse_args())
