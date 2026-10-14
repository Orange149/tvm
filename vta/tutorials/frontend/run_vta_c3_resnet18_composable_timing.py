#!/usr/bin/env python3
"""Single-boot development timing for safe-original versus composable ResNet18 H3."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import tempfile
import time
import traceback

from tvm import rpc

import run_vta_c3_reuse_board as reuse_board
from build_vta_c3_resnet18_composable_h3 import VARIANT as SCOPED_ARTIFACT
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_holdout_board_boot_v2 import sha256_file, verify_flat
from run_vta_c3_resnet18_fullgraph_programs import SEEDS
from run_vta_c3_yolo608_graph_boot import read, seal, write
from run_vta_p7r120_yolo_rpc_board import runtime_inventory
from run_vta_resnet50_fused_program_pareto_board import (
    invoke_model,
    load_executor,
    model_input_for_seed,
)


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_REFERENCE = GROUPED / "20260914_p7r515_resnet18_selected_fullgraph_build_run01"
DEFAULT_SCOPED = GROUPED / "20260924_c3_resnet18_h3_composable_build_run01"
DEFAULT_RUNTIME = GROUPED / "20260922_c3_thread_sync_freeze_run04/contract.json"
SAFE_PROGRAM = "selected_e53bbdbb71050431"
VARIANTS = ("stock", "safe_h3_original", "h3_composable")
ORDERS = (
    ("stock", "safe_h3_original", "h3_composable"),
    ("safe_h3_original", "h3_composable", "stock"),
    ("h3_composable", "stock", "safe_h3_original"),
)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def connect(host, port):
    for _ in range(30):
        try:
            return rpc.connect(host, port, session_timeout=1200)
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("RPC unavailable on port {}".format(port))


def mismatches(left, right):
    return sum(int((a != b).sum()) for a, b in zip(left, right))


def paired(rows, baseline, candidate):
    values = {(row["wave"], row["seed"], row["variant"]): row["latency_ms"] for row in rows}
    gains = [100.0 * (1.0 - values[w, s, candidate] / values[w, s, baseline])
             for w in range(len(ORDERS)) for s in SEEDS]
    return {"values_pct": gains, "median_pct": float(statistics.median(gains)),
            "wins": sum(value > 0 for value in gains), "pairs": len(gains)}


def run(args):
    reference = Path(args.reference_build).resolve()
    scoped = Path(args.scoped_build).resolve()
    verify_flat(scoped)
    if read(scoped / "summary.json")["status"] != "RESNET18_H3_COMPOSABLE_BUILT_AND_SCOPED":
        raise ValueError("scoped build identity mismatch")
    roots = {
        "stock": reference / "stock_reference",
        "safe_h3_original": reference / SAFE_PROGRAM,
        "h3_composable": scoped / SCOPED_ARTIFACT,
    }
    runtime = read(args.runtime_contract)
    for name, path in runtime["runtime_paths"].items():
        if sha256_file(path) != runtime["runtime_sha256"][name]:
            raise ValueError("runtime changed: " + name)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "calls.jsonl").write_text("", encoding="utf-8")
    contract = {
        "schema": "c3_resnet18_h3_composable_timing_v1",
        "status": "frozen_before_latency_observation",
        "variants": list(VARIANTS), "session_orders": [list(row) for row in ORDERS],
        "seeds": list(SEEDS), "fresh_rpc_per_variant_per_wave": True,
        "explicit_warmups_per_session": 1, "timed_calls_per_session": len(SEEDS),
        "timing_boundary": "TVM time_evaluator run, number=1, repeat=1",
        "board_reboot_requested": False, "bitstream_reload_forbidden": True,
        "claim_boundary": "single-boot development evidence; not two-boot confirmation",
    }
    write(output / "contract.json", contract)
    board = ExperimentBoard(args, "/var/volatile/" + output.name + "_runtime", args.port)
    rows, references = [], {}
    session = 0
    started = time.monotonic()
    try:
        initial = board.state()
        reuse_board.validate_state(initial, exclusive=True)
        initial_environment = board.environment()
        reuse_board.RUNTIME = runtime["runtime_sha256"]
        write(output / "runtime_attestation.json", board.deploy(runtime["runtime_paths"]))
        with tempfile.TemporaryDirectory(prefix="c3_r18_composable_timing_") as temporary:
            for wave, order in enumerate(ORDERS):
                for position, variant in enumerate(order):
                    board.port = args.port + 2 * session
                    board.pid = None
                    remote = None
                    try:
                        board.start()
                        remote = connect(args.host, board.port)
                        _, funcs = runtime_inventory(remote)
                        functions = {"clear": funcs["vta.runtime.profiler_clear"],
                                     "status": funcs["vta.runtime.profiler_status"]}
                        device = remote.ext_dev(0)
                        executor = load_executor(remote, roots[variant], variant + ".so",
                                                 [device, remote.cpu(0)], temporary)
                        environment = board.environment(require_affinity=True)
                        if environment["governor"] != initial_environment["governor"]:
                            raise RuntimeError("governor changed")
                        warm_data = model_input_for_seed(
                            SEEDS[0], {"shape": [1, 3, 224, 224], "uniform_range": [0.0, 1.0]}
                        )
                        _, warm = invoke_model(executor, warm_data, functions, device, False, True)
                        warm.update(phase="explicit_warmup", variant=variant, wave=wave,
                                    position=position, session=session, seed=int(SEEDS[0]))
                        append(output / "calls.jsonl", warm)
                        for seed in SEEDS:
                            data = model_input_for_seed(
                                seed, {"shape": [1, 3, 224, 224], "uniform_range": [0.0, 1.0]}
                            )
                            actual, row = invoke_model(executor, data, functions, device, True, True)
                            if not math.isfinite(row["latency_ms"]) or row["latency_ms"] <= 0:
                                raise RuntimeError("invalid latency")
                            profile = row["runtime_profile_complete"]
                            if profile.get("driver_timeout_calls") != 0 or profile.get("driver_run_calls", 0) <= 0:
                                raise RuntimeError("invalid runtime completion profile")
                            if variant == "stock" and seed not in references:
                                references[seed] = actual
                            elif seed in references and mismatches(references[seed], actual):
                                raise RuntimeError("full graph output mismatch")
                            row.update(phase="timing", variant=variant, wave=wave,
                                       position=position, session=session, seed=int(seed))
                            rows.append(row)
                            append(output / "calls.jsonl", row)
                        del executor
                        del remote
                        remote = None
                        gc.collect()
                    finally:
                        if remote is not None:
                            del remote
                            gc.collect()
                        board.stop()
                        board.pid = None
                    print("wave {} position {} {} complete".format(wave, position, variant), flush=True)
                    session += 1
        stats = {}
        for variant in VARIANTS:
            values = [float(row["latency_ms"]) for row in rows if row["variant"] == variant]
            if len(values) != len(ORDERS) * len(SEEDS):
                raise RuntimeError("incomplete timing coverage")
            stats[variant] = {"values_ms": values, "median_ms": float(statistics.median(values)),
                              "minimum_ms": min(values), "maximum_ms": max(values)}
        final = board.state()
        reuse_board.validate_state(final, exclusive=True)
        if final["boot_id"] != initial["boot_id"]:
            raise RuntimeError("boot changed")
        summary = {
            "schema": contract["schema"], "status": "SINGLE_BOOT_DEVELOPMENT_TIMING_COMPLETE",
            "boot_id": initial["boot_id"], "statistics": stats,
            "paired_improvement": {
                "composable_vs_stock": paired(rows, "stock", "h3_composable"),
                "safe_original_vs_stock": paired(rows, "stock", "safe_h3_original"),
                "composable_vs_safe_original": paired(rows, "safe_h3_original", "h3_composable"),
            },
            "all_outputs_equal": True, "sessions": session, "timing_calls": len(rows),
            "complete_graph_run_calls": session * (1 + len(SEEDS)), "implicit_extra_calls": 0,
            "board_reboot_requested": False, "bitstream_reloaded": False,
            "claim_boundary": contract["claim_boundary"],
            "wall_seconds": time.monotonic() - started,
        }
        write(output / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        write(output / "invalid_session.json", {"message": str(error),
              "traceback": traceback.format_exc()[-10000:], "all_timing_invalid": True,
              "sessions_completed": session, "timing_calls_completed": len(rows)})
        raise
    finally:
        try:
            board.stop()
        except Exception:
            pass
        seal(output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-build", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--scoped-build", default=str(DEFAULT_SCOPED))
    parser.add_argument("--runtime-contract", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9110)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
