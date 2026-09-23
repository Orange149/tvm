#!/usr/bin/env python3
"""Isolated trace-runtime overlay and bounded FPGA completion diagnosis."""
import argparse
import json
from pathlib import Path
import shlex
import shutil

from run_vta_c3_reuse_supplement import read, write, verify, seal, file_hash, check_bindings


def prepare(args):
    base = Path(args.contract).resolve().parent
    original = verify(base)
    driver = Path(args.driver_build).resolve()
    build = verify(driver)
    if build["phase"] != "driver_probe_build" or (driver / "invalid_session.json").exists():
        raise ValueError("requires completed isolated driver build")
    check_bindings(original["source_bindings"])
    check_bindings(build["source_bindings"])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    bindings = {**original["source_bindings"], **build["source_bindings"],
                str(Path(__file__).resolve()): file_hash(__file__),
                str(Path(__file__).with_name("diagnose_vta_c3_reuse_completion.py")): file_hash(Path(__file__).with_name("diagnose_vta_c3_reuse_completion.py")),
                str(driver / "libvta.so"): file_hash(driver / "libvta.so")}
    binaries = output / "binaries"
    binaries.mkdir()
    cid = original["health"]["candidate_id"]
    shutil.copyfile(base / "binaries" / (cid + ".so"), binaries / (cid + ".so"))
    hashes = {**original["runtime_sha256"], "libvta.so": file_hash(driver / "libvta.so")}
    overlay = {**original, "source_bindings": bindings,
               "runtime_paths": {**original["runtime_paths"], "libvta.so": str(driver / "libvta.so")},
               "runtime_sha256": hashes, "experimental_driver": str(driver),
               "experimental_driver_manifest_sha256": file_hash(driver / "artifact_hashes.json"),
               "original_prepared_directory": str(base),
               "original_prepared_manifest_sha256": file_hash(base / "artifact_hashes.json"),
               "permission": "user approved isolated driver modification",
               "diagnosis_only": True, "completion_trace_enabled": True,
               "completion_decision_changed": False, "formal_performance_labels": False}
    write(output / "contract.json", overlay)
    write(output / "summary.json", {"status": "ISOLATED_DRIVER_OVERLAY_READY", "board_contacted": False,
                                    "runtime_sha256": hashes})
    seal(output)


def analyze_trace(log, results):
    records = []
    for line in log.splitlines():
        if "[C3_COMPLETION] " in line:
            records.append(json.loads(line.split("[C3_COMPLETION] ", 1)[1]))
    if len(records) != sum(r["runtime_profile_complete"]["driver_run_calls"] for r in results):
        raise ValueError("missing, extra or unbound driver trace")
    offset, matched = 0, []
    for result in results:
        profile = result["runtime_profile_complete"]
        count = profile["driver_run_calls"]
        subset = records[offset:offset+count]
        offset += count
        if sum(r["insn_count"] for r in subset) != profile["driver_run_insns"]:
            raise ValueError("trace/profile instruction count mismatch")
        for position, record in enumerate(subset):
            matched.append({**record, "candidate_id": result["candidate_id"], "family_id": result["family_id"],
                            "mode": result["residence_mode"], "phase": result["phase"], "seed": result["seed"],
                            "operator_sequence_position": result.get("sequence_position"),
                            "submit_position": position + 1, "immediate_correct": result["correct"],
                            "delayed_correct": result["delayed_correct"]})
    return matched


def board_run(args):
    overlay = verify(Path(args.contract).resolve().parent)
    if not overlay.get("diagnosis_only") or not overlay.get("completion_trace_enabled"):
        raise ValueError("not an isolated trace-runtime contract")
    check_bindings(overlay["source_bindings"])
    # Only changes this process's expected attestation map. Sealed source and
    # original runtime stay unchanged; the overlay binds the alternative DSO.
    import run_vta_c3_reuse_board as board_module
    import diagnose_vta_c3_reuse_completion as diagnostic
    board_module.RUNTIME = dict(overlay["runtime_sha256"])
    observed_board = []

    class TraceBoard(diagnostic.DiagnosticBoard):
        def __init__(self, *values):
            super().__init__(*values)
            observed_board.append(self)

        def start(self):
            path = shlex.quote(self.runtime)
            command = (
                "(cd " + path + " && exec env LD_LIBRARY_PATH=" + path + " LD_PRELOAD=" + path + "/libvta.so "
                "C3_COMPLETION_TRACE=1 TVM_NUM_THREADS=4 VTA_REPLAY_POLICY=disabled "
                "VTA_QUEUE_DIAGNOSTICS=0 VTA_RUNTIME_TRACE=0 "
                "./tvm_rpc server --host=0.0.0.0 --port=9090 --port-end=9091) >" + path + "/rpc.log 2>&1 < /dev/null & echo $!"
            )
            self.pid = int(self.ssh(command).strip())
            return self.pid

    diagnostic.DiagnosticBoard = TraceBoard
    diagnostic.run(args)
    output = Path(args.output_dir).resolve()
    completed = verify(output)
    log = observed_board[0].ssh("cat " + shlex.quote(observed_board[0].runtime + "/rpc.log"))
    results = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    matched = analyze_trace(log, results)
    analysis = Path(str(output) + "_trace_analysis")
    analysis.mkdir(exist_ok=False)
    write(analysis / "contract.json", {"phase": "driver_trace_analysis", "input_directory": str(output),
          "input_manifest_sha256": file_hash(output / "artifact_hashes.json"),
          "overlay_manifest_sha256": file_hash(Path(args.contract).resolve().parent / "artifact_hashes.json"),
          "source_sha256": file_hash(__file__), "formal_latency_labels": False})
    (analysis / "rpc.log").write_text(log)
    write(analysis / "submit_trace.json", matched)
    wrong = [r for r in results if not r["correct"]]
    report = {"status": "ISOLATED_TRACE_COMPLETE_NO_FIX_CLAIM", "complete_operator_calls": len(results),
              "driver_submissions_verified": len(matched), "immediate_wrong": len(wrong),
              "wrong_became_correct_without_reexecution": sum(r["delayed_correct"] for r in wrong),
              "wrong_call_submit_patterns": [r for r in matched if not r["immediate_correct"]],
              "completion_decision_changed": False, "formal_performance_labels": False,
              "valid_register_inspected": False, "original_driver_preserved": True}
    write(analysis / "summary.json", report)
    seal(analysis)
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "board"), required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--driver-build")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else board_run)(args)
