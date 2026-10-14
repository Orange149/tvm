#!/usr/bin/env python3
"""No-timing full-graph correctness gate for composable ResNet18 H3."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import time
import traceback

from tvm import rpc

import run_vta_c3_reuse_board as reuse_board
from build_vta_c3_resnet18_composable_h3 import VARIANT
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_holdout_board_boot_v2 import sha256_file, verify_flat
from run_vta_c3_resnet18_fullgraph_programs import SEEDS
from run_vta_p7r120_yolo_rpc_board import runtime_inventory
from run_vta_resnet50_fused_program_pareto_board import (
    invoke_model,
    load_executor,
    model_input_for_seed,
)
from run_vta_c3_yolo608_graph_boot import read, seal, write


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_BUILD = GROUPED / "20260924_c3_resnet18_h3_composable_build_run01"
DEFAULT_REFERENCE = GROUPED / "20260914_p7r515_resnet18_selected_fullgraph_build_run01"
DEFAULT_RUNTIME = GROUPED / "20260922_c3_thread_sync_freeze_run04/contract.json"


def connect(host, port):
    for _ in range(30):
        try:
            return rpc.connect(host, port, session_timeout=1200)
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("RPC unavailable on port {}".format(port))


def hashes(outputs):
    return [hashlib.sha256(value.tobytes(order="C")).hexdigest() for value in outputs]


def mismatch(left, right):
    if len(left) != len(right):
        raise ValueError("output count differs")
    return [int((a != b).sum()) for a, b in zip(left, right)]


def run(args):
    build_root = Path(args.build_dir).resolve()
    reference_root = Path(args.reference_build).resolve()
    verify_flat(build_root)
    built = read(build_root / "summary.json")
    if built["status"] != "RESNET18_H3_COMPOSABLE_BUILT_AND_SCOPED":
        raise ValueError("unexpected composable ResNet18 build")
    runtime = read(args.runtime_contract)
    for name, path in runtime["runtime_paths"].items():
        if sha256_file(path) != runtime["runtime_sha256"][name]:
            raise ValueError("runtime changed: " + name)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "contract.json", {
        "schema": "c3_resnet18_h3_composable_diagnostic_v1",
        "status": "frozen_before_board_observation",
        "variants": ["stock_reference", VARIANT],
        "seeds": list(SEEDS), "timing_collected": False,
        "performance_claim": False, "board_reboot_requested": False,
        "bitstream_reload_forbidden": True, "fresh_rpc_per_variant": True,
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
        variants = (
            ("stock_reference", reference_root / "stock_reference"),
            (VARIANT, build_root / VARIANT),
        )
        with tempfile.TemporaryDirectory(prefix="c3_r18_composable_") as temporary:
            for index, (variant, local) in enumerate(variants):
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
                    device = remote.ext_dev(0)
                    executor = load_executor(
                        remote, local, variant + ".so", [device, remote.cpu(0)], temporary
                    )
                    board.environment(require_affinity=True)
                    for seed in SEEDS:
                        data = model_input_for_seed(
                            seed, {"shape": [1, 3, 224, 224], "uniform_range": [0.0, 1.0]}
                        )
                        actual, invocation = invoke_model(
                            executor, data, functions, device, False, True
                        )
                        observed[variant, seed] = actual
                        write(output / (variant + "_seed_" + str(seed) + ".json"), {
                            "seed": int(seed), "output_sha256": hashes(actual), **invocation,
                        })
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
        differences = {
            str(seed): mismatch(observed["stock_reference", seed], observed[VARIANT, seed])
            for seed in SEEDS
        }
        all_equal = all(not any(row) for row in differences.values())
        final = board.state()
        reuse_board.validate_state(final, exclusive=True)
        if final["boot_id"] != initial["boot_id"]:
            raise RuntimeError("boot changed")
        summary = {
            "schema": "c3_resnet18_h3_composable_diagnostic_v1",
            "status": (
                "RESNET18_H3_COMPOSABLE_RESTORES_GRAPH_CORRECTNESS"
                if all_equal else "RESNET18_H3_COMPOSABLE_GRAPH_INCORRECT"
            ),
            "boot_id": initial["boot_id"], "all_equal": all_equal,
            "mismatch_count_by_seed_and_output": differences,
            "complete_graph_run_calls": 2 * len(SEEDS),
            "timing_collected": False, "performance_claim": False,
            "board_reboot_requested": False, "bitstream_reloaded": False,
            "wall_seconds": time.monotonic() - started,
        }
        write(output / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        if not all_equal:
            raise RuntimeError("ResNet18 composable route remains graph incorrect")
    except BaseException as error:
        write(output / "invalid_session.json", {
            "message": str(error), "traceback": traceback.format_exc()[-8000:],
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
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
