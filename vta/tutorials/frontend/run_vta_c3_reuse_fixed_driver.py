#!/usr/bin/env python3
"""C1/C2 adapter for the qualified isolated driver, not an original-DSO result."""
import argparse
from pathlib import Path
import shutil
import time

from run_vta_c3_reuse_supplement import read, write, verify, seal, file_hash, check_bindings, digest


def prepare(args):
    original_dir = Path(args.contract).resolve().parent
    original = verify(original_dir)
    runtime_dir = Path(args.runtime_contract).resolve().parent
    runtime = verify(runtime_dir)
    qualification = Path(args.qualification_dir).resolve()
    qualified = verify(qualification)
    result = read(qualification / "summary.json")
    if qualified["phase"] != "handshake_qualification_analysis" or (qualification / "invalid_session.json").exists():
        raise ValueError("handshake qualification incomplete")
    if (result["status"] != "HANDSHAKE_MICRO_QUALIFICATION_PASS" or not result["completion_decision_changed"]
            or result["immediate_wrong"] or result["changed_after_return"] or result["driver_submissions_verified"] != 90
            or result["complete_operator_calls"] != 42):
        raise ValueError("requires 42 correct immediate calls and 90 fresh-event returns")
    if qualified["overlay_manifest_sha256"] != file_hash(runtime_dir / "artifact_hashes.json"):
        raise ValueError("qualification does not match runtime contract")
    if runtime["handshake_mode"] != "fresh_valid" or not runtime["completion_decision_changed"]:
        raise ValueError("observer is not a fixed runtime")
    if runtime["parent_manifest_sha256"] != original["parent_manifest_sha256"]:
        raise ValueError("frozen candidate pool differs")
    check_bindings(runtime["source_bindings"])
    check_bindings(original["source_bindings"])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "binaries").mkdir()
    for name in (original["health"]["candidate_id"] + ".so", "c3_reuse_timed_once.so"):
        shutil.copyfile(original_dir / "binaries" / name, output / "binaries" / name)
    bindings = {**runtime["source_bindings"], str(Path(__file__).resolve()): file_hash(__file__)}
    context = {"runtime_sha256": runtime["runtime_sha256"], "parent_manifest_sha256": original["parent_manifest_sha256"],
               "completion_policy": "fresh_valid", "runtime_trace": False, "known_register_map": "empirically qualified output-valid, no generated map attestation"}
    contract = {**original, "source_bindings": bindings, "runtime_paths": runtime["runtime_paths"],
                "runtime_sha256": runtime["runtime_sha256"], "diagnosis_only": False,
                "driver_context": context, "dispatch_context_sha256": digest(context),
                "micro_qualification_directory": str(qualification),
                "micro_qualification_manifest_sha256": file_hash(qualification / "artifact_hashes.json"),
                "generated_register_map_attested": False,
                "trace_during_formal_timing": False,
                "all_modes_same_experimental_driver": True,
                "old_invalid_session_timing_reused": False,
                "candidate_pool_and_order_changed": False,
                "comparison_scope": "within new isolated-runtime C1/C2; not mixed with old-runtime absolute latency"}
    write(output / "contract.json", contract)
    write(output / "summary.json", {"status": "FIXED_DRIVER_C1_READY", "board_contacted": False,
                                    "candidate_pool_and_order_changed": False})
    check_bindings(bindings)
    seal(output)


def run_board(args):
    if not args.execute_board:
        raise ValueError("requires --execute-board")
    directory = Path(args.contract).resolve().parent
    prepared = verify(directory)
    if not prepared.get("all_modes_same_experimental_driver") or prepared.get("trace_during_formal_timing"):
        raise ValueError("requires explicit trace-off fixed-driver protocol")
    qualification = Path(prepared["micro_qualification_directory"])
    verify(qualification)
    if file_hash(qualification / "artifact_hashes.json") != prepared["micro_qualification_manifest_sha256"]:
        raise ValueError("qualification changed")
    check_bindings(prepared["source_bindings"])
    import run_vta_c3_reuse_board as original
    from diagnose_vta_c3_reuse_completion import DiagnosticBoard
    original.RUNTIME = dict(prepared["runtime_sha256"])
    original.Board = DiagnosticBoard  # Own parent and timeout child cleanup.
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    from run_vta_c3_reuse_supplement import append
    append(output / "timeline.jsonl", {"event": "W0", "wall_unix": time.time()})
    try:
        original.run_board(args, output, started)
        append(output / "timeline.jsonl", {"event": "T1", "elapsed_seconds": time.monotonic() - started})
    except BaseException as error:
        if not (output / "invalid_session.json").exists():
            write(output / "invalid_session.json", {"message": str(error), "all_timing_invalid": True})
        raise
    finally:
        seal(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "board"), required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--runtime-contract")
    parser.add_argument("--qualification-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute-board", action="store_true")
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    parser.add_argument("--previous-boot")
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else run_board)(args)
