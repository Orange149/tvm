#!/usr/bin/env python3
"""Recover SDK-only failures without rerunning or altering local qualification.

This emits a new immutable local-protocol directory. Original phases/costs are
linked, not presented as one continuous session or rewritten to remove failures.
"""
import argparse
import os
from pathlib import Path
import shutil
import time

from run_vta_c3_reuse_supplement import verify, read, write, append, seal, file_hash, check_bindings, cross


def sdk_environment():
    """SDK requires a clean linker environment; affect child processes only."""
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    return environment


def run(args):
    start = time.monotonic()
    parent = Path(args.contract).resolve().parent
    contract = verify(parent)
    if contract["phase"] != "local":
        raise ValueError("SDK recovery requires completed local evidence")
    original_summary = read(parent / "summary.json")
    if not original_summary["positive_combination_gate"]:
        raise ValueError("do not recover a failed local mechanism gate")
    check_bindings(contract["source_bindings"])
    candidates, controls = read(parent / "candidates.json"), read(parent / "controls.json")
    for row in candidates + controls:
        if row["cross_qualification"]["status"] != "failed" or row["cross_qualification"].get("failure", {}).get("message") != "cross worker exit 1":
            raise ValueError("this recovery is only for SDK setup exit 1")
        if row["local_qualification"]["status"] != "passed" or row["fsim_qualification"]["status"] != "passed":
            raise ValueError("cannot recover unqualified identity")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    bindings = {**contract["source_bindings"], str(Path(__file__).resolve()): file_hash(Path(__file__))}
    write(output / "contract.json", {**contract, "source_bindings": bindings,
                                    "qualification_source_bindings": contract["source_bindings"],
                                    "sdk_recovery_directory": str(parent),
                                    "sdk_recovery_input_manifest_sha256": file_hash(parent / "artifact_hashes.json"),
                                    "recovery": "remove inherited LD_LIBRARY_PATH only for SDK child; same qualified identity/TIR",
                                    "cost_boundary": "new recovery phase; original qualification cost remains separately linked"})
    append(output / "timeline.jsonl", {"event": "W0", "wall_unix": time.time(), "T0": None})
    results = []
    prior_ld = os.environ.get("LD_LIBRARY_PATH")
    try:
        os.environ.pop("LD_LIBRARY_PATH", None)
        for position, row in enumerate(candidates + controls):
            # All retained storage and final TIR artifacts are copied unchanged.
            for suffix, hash_key in ((".json", "tir_sha256"), (".storage.json", "storage_tir_sha256")):
                source = parent / "tir" / (row["candidate_id"] + suffix)
                if file_hash(source) != row["local_qualification"][hash_key]:
                    raise ValueError("qualified TIR artifact changed")
                destination = output / "tir" / source.name
                destination.parent.mkdir(exist_ok=True)
                shutil.copyfile(source, destination)
            result = cross(row, output, args)
            result["recovery_position"] = position
            result["previous_cross_qualification"] = row["cross_qualification"]
            row["cross_qualification"] = result
            results.append(result)
            append(output / "results.jsonl", result)
            append(output / "timeline.jsonl", {"event": "cross_complete", "candidate_id": row["candidate_id"],
                                                "status": result["status"], "elapsed_seconds": time.monotonic() - start})
            print("SDK recovery {}/{} {} {}".format(position + 1, len(candidates + controls), row["family_id"], result["status"]), flush=True)
    except BaseException as error:
        write(output / "invalid_session.json", {"type": type(error).__name__, "message": str(error),
                                               "elapsed_seconds": time.monotonic() - start, "board_contacted": False})
        seal(output)
        raise
    finally:
        if prior_ld is not None:
            os.environ["LD_LIBRARY_PATH"] = prior_ld
    write(output / "candidates.json", candidates)
    write(output / "controls.json", controls)
    write(output / "prior_local_summary.json", original_summary)
    shutil.copyfile(parent / "families.jsonl", output / "families.jsonl")
    elapsed = time.monotonic() - start
    summary = {**original_summary, "status": "LOCAL_GATE_PASS_READY_TO_FREEZE" if all(row["status"] == "passed" for row in results) else "CROSS_RECOVERY_FAILED_PAUSED",
               "cross_attempts": len(results), "cross_passed": sum(row["status"] == "passed" for row in results),
               "wall_seconds": elapsed, "qualification_wall_seconds": original_summary["wall_seconds"],
               "prior_sdk_setup_failures": original_summary["cross_attempts"],
               "stage_host_wall_seconds": {**original_summary["stage_host_wall_seconds"], "cross": sum(row["host_wall_seconds"] for row in results)},
               "cost_scope": "separate completed phases, not one continuous session; prior setup failures retained",
               "total_valid_local_phase_investment_seconds": elapsed + original_summary["wall_seconds"]}
    write(output / "summary.json", summary)
    verify(parent)
    check_bindings(bindings)
    append(output / "timeline.jsonl", {"event": "T1", "elapsed_seconds": time.monotonic() - start})
    seal(output)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sdk-environment", default="/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux")
    run(parser.parse_args())
