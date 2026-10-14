#!/usr/bin/env python3
"""Test whether force-serial restores correctness of the frozen two-route graph."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import tempfile
import time
import traceback

from tvm import rpc

import run_vta_c3_reuse_board as reuse_board
from build_vta_c3_yolo608_force_serial_diagnostic import VARIANT
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
DEFAULT_BUILD = GROUPED / "20260924_c3_yolo608_force_serial_diagnostic_build_run01"
DEFAULT_REFERENCE = GROUPED / "20260924_c3_yolo608_graph_deployments_build_run02"
DEFAULT_RUNTIME = GROUPED / "20260922_c3_thread_sync_freeze_run04/contract.json"


def connect(remote_host, port):
    for _ in range(30):
        try:
            return rpc.connect(remote_host, port, session_timeout=1200)
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("RPC unavailable on port {}".format(port))


def run(args):
    build_dir = Path(args.build_dir).resolve()
    reference_dir = Path(args.reference_build).resolve()
    verify_flat(build_dir)
    verify_flat(reference_dir)
    build = read(build_dir / "summary.json")
    if build["variant"]["deployment_variant"] != VARIANT:
        raise ValueError("unexpected force-serial variant")
    runtime = read(args.runtime_contract)
    for name, path in runtime["runtime_paths"].items():
        if sha256_file(path) != runtime["runtime_sha256"][name]:
            raise ValueError("runtime changed: " + name)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "contract.json", {
        "schema": "c3_yolo608_force_serial_diagnostic_v1",
        "variants": ["stock_production", VARIANT],
        "seeds": list(SEEDS),
        "timing_collected": False,
        "performance_claim": False,
        "one_executor_per_fresh_rpc": True,
        "board_reboot_requested": False,
    })
    board = ExperimentBoard(args, "/var/volatile/" + output.name + "_runtime", args.port)
    observed = {}
    started = time.monotonic()
    initial = None
    try:
        initial = board.state()
        reuse_board.validate_state(initial, exclusive=True)
        reuse_board.RUNTIME = runtime["runtime_sha256"]
        write(output / "runtime_attestation.json", board.deploy(runtime["runtime_paths"]))
        if args.reload_bitstream:
            write(output / "bitstream_reload.json", board.reload())
        variants = (
            ("stock_production", reference_dir),
            (VARIANT, build_dir),
        )
        with tempfile.TemporaryDirectory(prefix="c3_yolo608_force_serial_") as temporary:
            for index, (variant, variant_dir) in enumerate(variants):
                board.port = args.port + 2 * index
                board.pid = None
                remote = None
                try:
                    board.start()
                    remote = connect(args.host, board.port)
                    _, funcs = runtime_inventory(remote)
                    functions = {
                        "clear": funcs["vta.runtime.profiler_clear"],
                        "status": funcs["vta.runtime.profiler_status"],
                    }
                    executor = load_executor(
                        remote,
                        variant_dir,
                        variant,
                        [remote.ext_dev(0), remote.cpu(0)],
                        temporary,
                    )
                    board.environment(require_affinity=True)
                    for seed in SEEDS:
                        values, row = invoke_once(
                            executor, random_input(seed), functions, timed=False
                        )
                        observed[variant, seed] = values
                        write(output / (variant + "_seed_" + str(seed) + ".json"), row)
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
                print("correctness {} complete".format(variant), flush=True)
        per_seed = {
            str(seed): mismatch_counts(
                observed["stock_production", seed], observed[VARIANT, seed]
            )
            for seed in SEEDS
        }
        all_equal = all(not any(values) for values in per_seed.values())
        final = board.state()
        reuse_board.validate_state(final, exclusive=True)
        if final["boot_id"] != initial["boot_id"]:
            raise RuntimeError("boot changed")
        summary = {
            "schema": "c3_yolo608_force_serial_diagnostic_v1",
            "status": (
                "FORCE_SERIAL_RESTORES_CORRECTNESS"
                if all_equal else "FORCE_SERIAL_DOES_NOT_RESTORE_CORRECTNESS"
            ),
            "boot_id": initial["boot_id"],
            "mismatch_count_by_seed_and_output": per_seed,
            "all_equal": all_equal,
            "complete_graph_run_calls": 2 * len(SEEDS),
            "timing_collected": False,
            "performance_claim": False,
            "board_reboot_requested": False,
            "bitstream_reloaded": bool(args.reload_bitstream),
            "wall_seconds": time.monotonic() - started,
        }
        write(output / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        write(output / "invalid_session.json", {
            "message": str(error),
            "traceback": traceback.format_exc()[-8000:],
            "performance_claim": False,
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
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD))
    parser.add_argument("--reference-build", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--runtime-contract", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9110)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    parser.add_argument("--reload-bitstream", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
