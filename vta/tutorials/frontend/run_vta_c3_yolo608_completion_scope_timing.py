#!/usr/bin/env python3
"""Single-boot development timing for global versus scoped completion safety."""

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
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_holdout_board_boot_v2 import sha256_file, verify_flat
from run_vta_c3_yolo608_graph_boot import (
    SEEDS,
    invoke_once,
    load_executor,
    mismatch_counts,
    random_input,
    read,
    seal,
    write,
)
from run_vta_p7r120_yolo_rpc_board import runtime_inventory


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_REFERENCE = GROUPED / "20260924_c3_yolo608_graph_deployments_build_run02"
DEFAULT_GLOBAL = GROUPED / "20260924_c3_yolo608_force_serial_diagnostic_build_run01"
DEFAULT_SCOPED = GROUPED / "20260924_c3_yolo608_composable_build_run01"
DEFAULT_RUNTIME = GROUPED / "20260922_c3_thread_sync_freeze_run04/contract.json"
VARIANTS = ("stock", "global_force_serial", "scoped_composable")
SESSION_ORDERS = (
    ("stock", "global_force_serial", "scoped_composable"),
    ("global_force_serial", "scoped_composable", "stock"),
    ("scoped_composable", "stock", "global_force_serial"),
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


def paired_gain(rows, baseline, candidate):
    by_key = {(row["wave"], row["seed"], row["variant"]): row["latency_ms"] for row in rows}
    values = []
    for wave in range(len(SESSION_ORDERS)):
        for seed in SEEDS:
            left = by_key[wave, seed, baseline]
            right = by_key[wave, seed, candidate]
            values.append(100.0 * (1.0 - right / left))
    return {"values_pct": values, "median_pct": float(statistics.median(values)),
            "wins": sum(value > 0 for value in values), "pairs": len(values)}


def run(args):
    roots = {
        "stock": (Path(args.reference_build).resolve(), "stock_production"),
        "global_force_serial": (Path(args.global_build).resolve(), "selector_v2_force_serial"),
        "scoped_composable": (Path(args.scoped_build).resolve(), "selector_v3_composable"),
    }
    manifests = {name: verify_flat(root) for name, (root, _) in roots.items()}
    if read(roots["global_force_serial"][0] / "summary.json")["status"] != "FORCE_SERIAL_TWO_ROUTE_DIAGNOSTIC_BUILT":
        raise ValueError("global force-serial build identity mismatch")
    scoped = read(roots["scoped_composable"][0] / "summary.json")
    if scoped["status"] != "COMPOSABLE_TWO_ROUTE_GRAPH_BUILT_AND_SCOPED":
        raise ValueError("scoped composable build identity mismatch")
    runtime = read(args.runtime_contract)
    for name, path in runtime["runtime_paths"].items():
        if sha256_file(path) != runtime["runtime_sha256"][name]:
            raise ValueError("runtime changed: " + name)

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "calls.jsonl").write_text("", encoding="utf-8")
    contract = {
        "schema": "c3_yolo608_completion_scope_timing_v1",
        "status": "frozen_before_latency_observation",
        "purpose": "development comparison of global versus candidate-scoped completion safety",
        "variants": list(VARIANTS),
        "session_orders": [list(row) for row in SESSION_ORDERS],
        "seeds_per_session": list(SEEDS),
        "fresh_rpc_per_variant_per_wave": True,
        "explicit_warmups_per_session": 1,
        "timed_calls_per_session": len(SEEDS),
        "expected_sessions": len(SESSION_ORDERS) * len(VARIANTS),
        "expected_complete_graph_calls": len(SESSION_ORDERS) * len(VARIANTS) * (1 + len(SEEDS)),
        "timing_boundary": "host steady_clock around one synchronous GraphExecutor run RPC",
        "build_manifests": manifests,
        "board_reboot_requested": False,
        "bitstream_reload_forbidden": True,
        "claim_boundary": "single-boot development evidence; not independent two-boot confirmation",
    }
    write(output / "contract.json", contract)

    board = ExperimentBoard(args, "/var/volatile/" + output.name + "_runtime", args.port)
    started = time.monotonic()
    rows, references = [], {}
    session_index = 0
    initial = None
    try:
        initial = board.state()
        reuse_board.validate_state(initial, exclusive=True)
        environment = board.environment()
        reuse_board.RUNTIME = runtime["runtime_sha256"]
        write(output / "runtime_attestation.json", board.deploy(runtime["runtime_paths"]))
        with tempfile.TemporaryDirectory(prefix="c3_yolo608_scope_timing_") as temporary:
            for wave, order in enumerate(SESSION_ORDERS):
                for position, variant in enumerate(order):
                    root, artifact_variant = roots[variant]
                    board.port = args.port + 2 * session_index
                    board.pid = None
                    remote = None
                    try:
                        state = board.state()
                        reuse_board.validate_state(state, exclusive=True)
                        if state["boot_id"] != initial["boot_id"]:
                            raise RuntimeError("boot changed before session")
                        board.start()
                        remote = connect(args.host, board.port)
                        _, funcs = runtime_inventory(remote)
                        functions = {
                            "clear": funcs["vta.runtime.profiler_clear"],
                            "status": funcs["vta.runtime.profiler_status"],
                        }
                        executor = load_executor(
                            remote, root, artifact_variant,
                            [remote.ext_dev(0), remote.cpu(0)], temporary,
                        )
                        current = board.environment(require_affinity=True)
                        if current["governor"] != environment["governor"]:
                            raise RuntimeError("governor changed")
                        warm_outputs, warm = invoke_once(
                            executor, random_input(SEEDS[0]), functions, timed=False
                        )
                        warm.update(phase="explicit_warmup", wave=wave, position=position,
                                    variant=variant, session=session_index, seed=SEEDS[0])
                        append(output / "calls.jsonl", warm)
                        for seed in SEEDS:
                            outputs, row = invoke_once(executor, random_input(seed), functions, timed=True)
                            if not math.isfinite(row["latency_ms"]) or row["latency_ms"] <= 0:
                                raise RuntimeError("invalid latency")
                            profile = row["runtime_profile_complete"]
                            if profile.get("driver_timeout_calls") != 0 or profile.get("driver_run_calls", 0) <= 0:
                                raise RuntimeError("invalid runtime completion profile")
                            if variant == "stock" and seed not in references:
                                references[seed] = outputs
                            elif seed in references and any(mismatch_counts(references[seed], outputs)):
                                raise RuntimeError("full graph output mismatch")
                            row.update(phase="timing", wave=wave, position=position,
                                       variant=variant, session=session_index, seed=seed)
                            rows.append(row)
                            append(output / "calls.jsonl", row)
                        del warm_outputs
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
                    session_index += 1

        # A stock session occurs in every wave; validate any rows observed before
        # their wave's stock session now that all canonical outputs are present.
        if set(references) != set(SEEDS):
            raise RuntimeError("stock reference coverage incomplete")
        statistics_by_variant = {}
        for variant in VARIANTS:
            values = [float(row["latency_ms"]) for row in rows if row["variant"] == variant]
            if len(values) != len(SESSION_ORDERS) * len(SEEDS):
                raise RuntimeError("incomplete timing coverage for " + variant)
            statistics_by_variant[variant] = {
                "values_ms": values,
                "median_ms": float(statistics.median(values)),
                "minimum_ms": min(values),
                "maximum_ms": max(values),
            }
        final = board.state()
        reuse_board.validate_state(final, exclusive=True)
        if final["boot_id"] != initial["boot_id"]:
            raise RuntimeError("boot changed during experiment")
        summary = {
            "schema": contract["schema"],
            "status": "SINGLE_BOOT_DEVELOPMENT_TIMING_COMPLETE",
            "boot_id": initial["boot_id"],
            "statistics": statistics_by_variant,
            "paired_improvement": {
                "scoped_vs_stock": paired_gain(rows, "stock", "scoped_composable"),
                "global_vs_stock": paired_gain(rows, "stock", "global_force_serial"),
                "scoped_vs_global": paired_gain(rows, "global_force_serial", "scoped_composable"),
            },
            "all_outputs_equal": True,
            "sessions": session_index,
            "timing_calls": len(rows),
            "complete_graph_run_calls": session_index * (1 + len(SEEDS)),
            "implicit_extra_calls": 0,
            "board_reboot_requested": False,
            "bitstream_reloaded": False,
            "claim_boundary": contract["claim_boundary"],
            "wall_seconds": time.monotonic() - started,
        }
        write(output / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        write(output / "invalid_session.json", {
            "message": str(error), "type": type(error).__name__,
            "traceback": traceback.format_exc()[-10000:],
            "sessions_completed": session_index,
            "timing_calls_completed": len(rows),
            "all_timing_invalid": True,
        })
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
    parser.add_argument("--global-build", default=str(DEFAULT_GLOBAL))
    parser.add_argument("--scoped-build", default=str(DEFAULT_SCOPED))
    parser.add_argument("--runtime-contract", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9110)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
