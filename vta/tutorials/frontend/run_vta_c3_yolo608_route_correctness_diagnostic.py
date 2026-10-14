#!/usr/bin/env python3
"""Compare stock and each one-route YOLO-608 graph without collecting latency."""

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
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_holdout_board_boot_v2 import sha256_file, verify_flat
from run_vta_c3_yolo608_graph_boot import (
    SEEDS,
    load_executor,
    mismatch_counts,
    random_input,
    read,
    seal,
    write,
    invoke_once,
)
from run_vta_p7r120_yolo_rpc_board import runtime_inventory


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_BUILD = GROUPED / "20260924_c3_yolo608_route_correctness_diagnostic_build_run01"
DEFAULT_REFERENCE = GROUPED / "20260924_c3_yolo608_graph_deployments_build_run02"
DEFAULT_RUNTIME = GROUPED / "20260922_c3_thread_sync_freeze_run04/contract.json"


def run(args):
    build_dir = Path(args.build_dir).resolve()
    reference_dir = Path(args.reference_build).resolve()
    verify_flat(build_dir)
    verify_flat(reference_dir)
    build = read(build_dir / "summary.json")
    variants = ["stock_production"] + [row["deployment_variant"] for row in build["variants"]]
    variant_dirs = {"stock_production": reference_dir}
    variant_dirs.update({name: build_dir for name in variants[1:]})
    runtime = read(args.runtime_contract)
    for name, path in runtime["runtime_paths"].items():
        if sha256_file(path) != runtime["runtime_sha256"][name]:
            raise ValueError("runtime changed: " + name)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "contract.json", {
        "schema": "c3_yolo608_route_correctness_diagnostic_v1",
        "variants": variants, "seeds": list(SEEDS),
        "timing_collected": False, "performance_claim": False,
        "one_executor_per_fresh_rpc": True,
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
        write(output / "bitstream_reload.json", board.reload())
        with tempfile.TemporaryDirectory(prefix="c3_yolo608_route_diag_") as temporary:
            for index, variant in enumerate(variants):
                board.port = args.port + 2 * index
                board.pid = None
                remote = None
                try:
                    board.start()
                    for _ in range(30):
                        try:
                            remote = rpc.connect(args.host, board.port, session_timeout=1200)
                            break
                        except Exception:
                            time.sleep(0.25)
                    if remote is None:
                        raise RuntimeError("RPC unavailable")
                    _, funcs = runtime_inventory(remote)
                    functions = {"clear": funcs["vta.runtime.profiler_clear"],
                                 "status": funcs["vta.runtime.profiler_status"]}
                    executor = load_executor(remote, variant_dirs[variant], variant,
                                             [remote.ext_dev(0), remote.cpu(0)], temporary)
                    board.environment(require_affinity=True)
                    for seed in SEEDS:
                        values, row = invoke_once(executor, random_input(seed), functions, False)
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
        comparisons = {}
        for variant in variants[1:]:
            per_seed = {str(seed): mismatch_counts(observed["stock_production", seed],
                                                   observed[variant, seed]) for seed in SEEDS}
            comparisons[variant] = {
                "mismatch_count_by_seed_and_output": per_seed,
                "all_equal": all(not any(values) for values in per_seed.values()),
            }
        final = board.state()
        reuse_board.validate_state(final, exclusive=True)
        if final["boot_id"] != initial["boot_id"]:
            raise RuntimeError("boot changed")
        summary = {
            "schema": "c3_yolo608_route_correctness_diagnostic_v1",
            "status": "ONE_ROUTE_CORRECTNESS_DIAGNOSTIC_COMPLETE",
            "boot_id": initial["boot_id"],
            "comparisons": comparisons,
            "complete_graph_run_calls": len(variants) * len(SEEDS),
            "timing_collected": False,
            "performance_claim": False,
            "wall_seconds": time.monotonic() - started,
        }
        write(output / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        write(output / "invalid_session.json", {"message": str(error),
              "traceback": traceback.format_exc()[-8000:], "performance_claim": False})
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
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
