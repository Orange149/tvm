#!/usr/bin/env python3
"""Audit and freeze the board-side completion boundary for C3 v5 YOLO-512."""
from __future__ import annotations

import argparse
import gc
import json
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from tvm import rpc

import run_vta_c3_reuse_board as reuse_board
from c3_evaluation_v5 import read_json, seal_flat, sha256_file, verify_flat, write_json
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_yolo608_graph_boot import load_executor, mismatch_counts, output_summary, random_input
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory

HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_BUILD = GROUPED / "20260926_c3_yolo512_evaluation_v5_graph_build_run01"
DEFAULT_FREEZE = GROUPED / "20260926_c3_yolo512_evaluation_v5_freeze_run01"
DEFAULT_OUTPUT = GROUPED / "20260926_c3_yolo512_evaluation_v5_completion_audit_run01"


def connect(host, port, timeout):
    for _ in range(30):
        try:
            return rpc.connect(host, port, session_timeout=timeout)
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("RPC unavailable for completion audit")


def run(args):
    build_dir = Path(args.build_dir).resolve()
    freeze_dir = Path(args.freeze_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    build_manifest = verify_flat(build_dir)
    freeze_manifest = verify_flat(freeze_dir)
    build = read_json(build_dir / "summary.json")
    freeze = read_json(freeze_dir / "contract.json")
    if build.get("status") != "C3_EVALUATION_V5_THREE_YOLO512_DEPLOYMENTS_BUILT":
        raise ValueError("graph build is incomplete")
    helper = Path(freeze["helper_binary"])
    if sha256_file(helper) != freeze["helper_sha256"]:
        raise ValueError("v5 helper changed")
    runtime = read_json(freeze["runtime_contract"])
    for name, path in runtime["runtime_paths"].items():
        if sha256_file(path) != runtime["runtime_sha256"][name]:
            raise ValueError("runtime changed: " + name)

    output.mkdir(parents=True)
    contract = {
        "schema": "c3_evaluation_v5_completion_audit_v1",
        "status": "FROZEN_BEFORE_COMPLETION_OBSERVATION",
        "build_manifest_sha256": build_manifest,
        "freeze_manifest_sha256": freeze_manifest,
        "diagnostics": ["run_only", "run_plus_sync", "completed_sync_only"],
        "same_rpc_process_required": True,
        "helper_static_vta_link_forbidden": True,
        "one_graph_run_per_run_diagnostic": True,
        "decision_rule": (
            "run_only iff it completes correct output and run+sync/sync-only each add an otherwise "
            "unnecessary device submission; otherwise run+sync only when it is required for completion"
        ),
    }
    write_json(output / "contract.json", contract)
    board = ExperimentBoard(args, "/var/volatile/" + output.name, args.port)
    remote = None
    initial = None
    records = []
    try:
        initial = board.state()
        reuse_board.validate_state(initial, exclusive=True)
        environment = board.environment()
        reuse_board.RUNTIME = runtime["runtime_sha256"]
        write_json(output / "runtime_attestation.json", board.deploy(runtime["runtime_paths"]))
        write_json(output / "bitstream_reload.json", board.reload())
        board.start()
        remote = connect(args.host, args.port, args.session_timeout)
        inventory, funcs = runtime_inventory(remote)
        write_json(output / "runtime_inventory.json", inventory)
        retained = [load_ephemeral(remote, helper)]
        symbols = json.loads(str(remote.get_function("c3.v5.runtime_symbols")()))
        write_json(output / "runtime_symbols.json", symbols)
        for name, row in symbols.items():
            if not row["dso"].endswith("/libvta.so"):
                raise RuntimeError(name + " did not resolve from the experiment libvta.so")
            if board.runtime not in row["dso"]:
                raise RuntimeError(name + " resolved from another runtime directory")
        clear = funcs["vta.runtime.profiler_clear"]
        status = funcs["vta.runtime.profiler_status"]
        reset = remote.get_function("c3.v5.reset_counts")
        counts = remote.get_function("c3.v5.counts")
        run_only = remote.get_function("c3.v5.graph_run_only_timed_once")
        run_sync = remote.get_function("c3.v5.graph_run_sync_timed_once")
        empty_sync = remote.get_function("c3.v5.empty_sync")
        reset()
        with tempfile.TemporaryDirectory(prefix="c3_v5_completion_") as temporary:
            executor = load_executor(remote, build_dir, "stock", [remote.ext_dev(0), remote.cpu(0)], temporary)
            data = random_input(0, resolution=512)

            def graph_case(name, function):
                executor.set_input("data", data)
                clear()
                before = json.loads(str(counts()))
                latency = float(function(executor.module))
                profile = json.loads(status())
                after = json.loads(str(counts()))
                outputs = [executor.get_output(i).numpy() for i in range(executor.get_num_outputs())]
                row = {
                    "diagnostic": name, "latency_ms_diagnostic_only": latency,
                    "counts_before": before, "counts_after": after,
                    "runtime_profile": profile, "outputs": output_summary(outputs),
                    "output_arrays": outputs,
                }
                return row

            a = graph_case("run_only", run_only)
            b = graph_case("run_plus_sync", run_sync)
            clear()
            before = json.loads(str(counts()))
            empty_sync()
            sync_profile = json.loads(status())
            after = json.loads(str(counts()))
            c = {
                "diagnostic": "completed_sync_only", "counts_before": before,
                "counts_after": after, "runtime_profile": sync_profile,
            }
            if any(mismatch_counts(a["output_arrays"], b["output_arrays"])):
                raise RuntimeError("run-only and run+sync outputs differ")
            if not all(item["nonzero"] > 0 for item in a["outputs"] + b["outputs"]):
                raise RuntimeError("completion diagnostic produced an all-zero output")
            for row in (a, b):
                row.pop("output_arrays")
                records.append(row)
            records.append(c)
            del executor

        run_only_driver = int(a["runtime_profile"].get("driver_run_calls", 0))
        run_sync_driver = int(b["runtime_profile"].get("driver_run_calls", 0))
        sync_only_driver = int(c["runtime_profile"].get("driver_run_calls", 0))
        counters = json.loads(str(counts()))
        one_each = (a["counts_after"]["graph_calls"] - a["counts_before"]["graph_calls"] == 1
                    and b["counts_after"]["graph_calls"] - b["counts_before"]["graph_calls"] == 1)
        extra_sync_submit = run_sync_driver == run_only_driver + 1 and sync_only_driver == 1
        if one_each and run_only_driver > 0 and extra_sync_submit:
            mode = "run_only"
            reason = "graph run already completed; explicit sync adds FINISH/device submission"
        elif one_each and run_only_driver == 0 and run_sync_driver > 0:
            mode = "run_plus_sync"
            reason = "graph run requires explicit final submit/wait"
        else:
            raise RuntimeError("completion boundary remains ambiguous")
        if counters["graph_calls"] != 2:
            raise RuntimeError("completion audit did not execute exactly two graphs")
        write_json(output / "diagnostics.json", records)
        write_json(output / "completion_policy.json", {
            "schema": "c3_evaluation_v5_completion_policy_v1",
            "status": "COMPLETION_POLICY_FROZEN",
            "mode": mode, "reason": reason,
            "helper_binary": str(helper), "helper_sha256": freeze["helper_sha256"],
            "runtime_symbols": symbols,
            "driver_run_calls": {"run_only": run_only_driver, "run_plus_sync": run_sync_driver,
                                 "completed_sync_only": sync_only_driver},
            "exact_graph_calls": 2,
        })
        running = board.state()
        reuse_board.validate_state(running, exclusive=False)
        running_env = board.environment(require_affinity=True)
        if running["boot_id"] != initial["boot_id"] or running_env["governor"] != environment["governor"]:
            raise RuntimeError("board identity changed during completion audit")
        remote = None
        gc.collect()
        board.stop()
        board.pid = None
        stopped = board.state()
        reuse_board.validate_state(stopped, exclusive=True)
        summary = {
            "schema": contract["schema"], "status": "COMPLETION_AUDIT_COMPLETE",
            "boot_id": initial["boot_id"], "completion_mode": mode,
            "graph_calls": counters["graph_calls"], "explicit_sync_calls": counters["explicit_sync_calls"],
            "outputs_equal": True, "same_runtime_queue_proven": True,
        }
        write_json(output / "postflight.json", {"running": running, "environment": running_env,
                                                 "stopped": stopped})
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        write_json(output / "invalid_audit.json", {"status": "INVALID_COMPLETION_AUDIT",
                   "message": str(error), "type": type(error).__name__,
                   "traceback": traceback.format_exc()[-12000:], "formal_graph_timing_allowed": False})
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    p.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--host", default="192.168.1.247")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--session-timeout", type=int, default=1800)
    p.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
