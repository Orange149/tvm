#!/usr/bin/env python3
"""Qualification only for the isolated, explicitly hypothetical valid handshake."""
import argparse
import json
from pathlib import Path
import shlex
import shutil

from run_vta_c3_reuse_supplement import read, write, verify, seal, file_hash, check_bindings
from run_vta_c3_completion_probe import analyze_trace


def prepare(args):
    base = Path(args.contract).resolve().parent
    original = verify(base)
    driver = Path(args.driver_build).resolve()
    build = verify(driver)
    if build["phase"] != "completion_handshake_build" or (driver / "invalid_session.json").exists():
        raise ValueError("invalid handshake build")
    check_bindings(original["source_bindings"])
    check_bindings(build["source_bindings"])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("run_vta_c3_completion_probe.py"),
             Path(__file__).with_name("diagnose_vta_c3_reuse_completion.py"), driver / "libvta.so"]
    bindings = {**original["source_bindings"], **build["source_bindings"],
                **{str(p.resolve()): file_hash(p) for p in paths}}
    (output / "binaries").mkdir()
    name = original["health"]["candidate_id"] + ".so"
    shutil.copyfile(base / "binaries" / name, output / "binaries" / name)
    write(output / "contract.json", {**original, "source_bindings": bindings,
          "runtime_paths": {**original["runtime_paths"], "libvta.so": str(driver / "libvta.so")},
          "runtime_sha256": {**original["runtime_sha256"], "libvta.so": file_hash(driver / "libvta.so")},
          "driver_build": str(driver), "driver_manifest_sha256": file_hash(driver / "artifact_hashes.json"),
          "diagnosis_only": True, "handshake_mode": build["mode"],
          "completion_decision_changed": build["require_fresh_event"],
          "register_hypothesis": build["register_hypothesis"], "generated_register_map_attested": False,
          "formal_performance_labels": False, "fixed_completion_sleep_added": False,
          "permission": "user explicitly approved isolated driver modification"})
    write(output / "summary.json", {"status": "HANDSHAKE_QUALIFICATION_READY", "board_contacted": False})
    seal(output)


def board_run(args):
    if not args.execute_board:
        raise ValueError("requires --execute-board")
    overlay = verify(Path(args.contract).resolve().parent)
    if not overlay.get("diagnosis_only") or "handshake_mode" not in overlay:
        raise ValueError("not a handshake qualification overlay")
    check_bindings(overlay["source_bindings"])
    import run_vta_c3_reuse_board as board_module
    import diagnose_vta_c3_reuse_completion as diagnostic
    board_module.RUNTIME = dict(overlay["runtime_sha256"])
    instances = []

    class HandshakeBoard(diagnostic.DiagnosticBoard):
        def __init__(self, *values):
            super().__init__(*values)
            instances.append(self)

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

    diagnostic.DiagnosticBoard = HandshakeBoard
    failure = None
    try:
        diagnostic.run(args)
    except BaseException as error:
        failure = error
    output = Path(args.output_dir).resolve()
    analysis = Path(str(output) + "_trace_analysis")
    analysis.mkdir(exist_ok=False)
    write(analysis / "contract.json", {"phase": "handshake_qualification_analysis",
          "input_directory": str(output), "input_manifest_sha256": file_hash(output / "artifact_hashes.json"),
          "overlay_manifest_sha256": file_hash(Path(args.contract).resolve().parent / "artifact_hashes.json"),
          "source_sha256": file_hash(__file__), "handshake_mode": overlay["handshake_mode"],
          "register_hypothesis": overlay["register_hypothesis"], "formal_performance_labels": False})
    try:
        log = instances[0].ssh("cat " + shlex.quote(instances[0].runtime + "/rpc.log"))
        (analysis / "rpc.log").write_text(log)
        results = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
        if failure is not None:
            write(analysis / "invalid_session.json", {"message": str(failure), "raw_log_preserved": True})
            raise failure
        matched = analyze_trace(log, results)
        write(analysis / "submit_trace.json", matched)
        fresh = overlay["completion_decision_changed"]
        if fresh and any(r["return_valid"] != 1 or r["return_done"] != 1 or r["timeout"] for r in matched):
            raise ValueError("fresh-valid completion gate did not attest every return")
        if any(r["after_clear_valid"] != 0 for r in matched):
            raise ValueError("old valid event did not clear before start")
        wrong = [r for r in results if not r["correct"]]
        report = {"status": "HANDSHAKE_MICRO_QUALIFICATION_PASS" if not wrong else "HANDSHAKE_MICRO_QUALIFICATION_WRONG_OUTPUT",
                  "complete_operator_calls": len(results), "driver_submissions_verified": len(matched),
                  "immediate_wrong": len(wrong), "changed_after_return": sum(r["changed_after_return"] for r in results),
                  "all_delayed_correct": all(r["delayed_correct"] for r in results),
                  "prestart_pending_valid_events": sum(r["before_valid"] == 1 for r in matched),
                  "all_prestart_events_clear": True, "completion_decision_changed": fresh,
                  "generated_register_map_attested": False, "full_48_identity_pool_qualified": False,
                  "formal_performance_labels": False, "fixed_completion_sleep_added": False,
                  "wrong_call_submit_patterns": [r for r in matched if not r["immediate_correct"]]}
        write(analysis / "summary.json", report)
        print(report, flush=True)
    except BaseException as error:
        if not (analysis / "invalid_session.json").exists():
            write(analysis / "invalid_session.json", {"message": str(error)})
        raise
    finally:
        seal(analysis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "board"), required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--driver-build")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute-board", action="store_true")
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else board_run)(args)
