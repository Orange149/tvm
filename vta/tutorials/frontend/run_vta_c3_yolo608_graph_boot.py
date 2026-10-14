#!/usr/bin/env python3
"""Run one non-spliceable boot of the three frozen YOLO-608 deployments."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import shutil
import statistics
import tempfile
import time
import traceback

import numpy as np
from tvm import rpc
from tvm.contrib import graph_executor
from tvm.relay.testing import darknet

import run_vta_c3_reuse_board as reuse_board
from build_vta_c3_yolo608_graph_deployments import VARIANTS
from c3_thread_sync_board import ExperimentBoard
from qualify_vta_residency_fsim import array_sha256
from run_vta_c3_holdout_board_boot_v2 import sha256_file, verify_flat
from run_vta_p7r120_yolo_rpc_board import runtime_inventory


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_BUILD = GROUPED / "20260924_c3_yolo608_graph_deployments_build_run02"
DEFAULT_RUNTIME_CONTRACT = GROUPED / "20260922_c3_thread_sync_freeze_run04/contract.json"
SEEDS = (0, 20250901, 20260910)
TIMING_REPEATS = 10


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def seal(output):
    write(output / "artifact_hashes.json", {"artifacts": {
        path.name: sha256_file(path) for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }, "source": sha256_file(Path(__file__).resolve())})


def random_input(seed, resolution=608):
    rng = np.random.default_rng(int(seed))
    return rng.uniform(0.0, 1.0, size=(1, 3, resolution, resolution)).astype("float32")


def image_input(path, resolution=608):
    data = darknet.load_image(str(path), resolution, resolution).transpose(1, 2, 0)
    return np.ascontiguousarray(data.transpose(2, 0, 1)[None, :].astype("float32"))


def output_summary(outputs):
    return [{
        "sha256": array_sha256(array), "shape": list(array.shape), "dtype": str(array.dtype),
        "nonzero": int(np.count_nonzero(array)), "min": float(array.min()),
        "max": float(array.max()),
    } for array in outputs]


def mismatch_counts(lhs, rhs):
    if len(lhs) != len(rhs):
        raise ValueError("graph output count differs")
    return [int(np.count_nonzero(a != b)) for a, b in zip(lhs, rhs)]


def load_executor(remote, build_dir, variant, contexts, temporary):
    local = Path(build_dir) / variant
    upload = Path(temporary) / (variant + ".so")
    shutil.copyfile(local / "graphlib.so", upload)
    remote.upload(str(upload))
    module = remote.load_module(upload.name)
    executor = graph_executor.create(
        (local / "graph.json").read_text(encoding="utf-8"), module, contexts
    )
    executor.load_params((local / "params.bin").read_bytes())
    return executor


def invoke_once(executor, data, functions, timed):
    executor.set_input("data", data)
    functions["clear"]()
    started = time.perf_counter()
    executor.run()
    latency_ms = (time.perf_counter() - started) * 1000.0 if timed else None
    outputs = [executor.get_output(index).numpy() for index in range(executor.get_num_outputs())]
    return outputs, {
        "latency_ms": latency_ms,
        "outputs": output_summary(outputs),
        "runtime_profile_complete": json.loads(functions["status"]()),
        "complete_graph_run_calls": 1,
        "implicit_warmup_calls": 0,
        "timing_boundary": "host steady_clock around one synchronous GraphExecutor run RPC",
    }


def variant_order(boot_index):
    return list(VARIANTS if boot_index == 1 else reversed(VARIANTS))


def run(args):
    build_dir = Path(args.build_dir).resolve()
    build_manifest = verify_flat(build_dir)
    build = read(build_dir / "summary.json")
    if build.get("status") != "THREE_YOLO608_DEPLOYMENTS_CROSS_BUILT":
        raise ValueError("build is not the frozen three-deployment YOLO-608 artifact")
    if tuple(row["deployment_variant"] for row in build["variants"]) != VARIANTS:
        raise ValueError("unexpected deployment variant order")
    if build["input_shape"] != [1, 3, 608, 608]:
        raise ValueError("graph is not the frozen 608 input contract")
    runtime_contract = read(args.runtime_contract)
    runtime_paths = runtime_contract["runtime_paths"]
    runtime_hashes = runtime_contract["runtime_sha256"]
    for name, path in runtime_paths.items():
        if sha256_file(path) != runtime_hashes[name]:
            raise ValueError("runtime changed: " + name)
    previous = None
    boot_index = 1
    if args.previous_boot_dir:
        previous_dir = Path(args.previous_boot_dir).resolve()
        verify_flat(previous_dir)
        previous = read(previous_dir / "summary.json")
        if previous.get("status") != "YOLO608_GRAPH_BOOT1_COMPLETE":
            raise ValueError("previous directory is not a complete graph boot 1")
        if read(previous_dir / "contract.json")["build_manifest_sha256"] != build_manifest:
            raise ValueError("previous graph boot used another build")
        boot_index = 2

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "calls.jsonl").write_text("", encoding="utf-8")
    contract = {
        "schema": "c3_yolo608_graph_boot_v1",
        "status": "frozen_before_board_observation",
        "boot_index": boot_index,
        "previous_boot_dir": str(Path(args.previous_boot_dir).resolve()) if previous else None,
        "build_dir": str(build_dir),
        "build_manifest_sha256": build_manifest,
        "variants": list(VARIANTS),
        "variant_session_order": variant_order(boot_index),
        "variant_session_ports": {
            variant: args.port + 2 * index
            for index, variant in enumerate(variant_order(boot_index))
        },
        "correctness_seeds": list(SEEDS),
        "explicit_person_image_warmups_per_variant": 1,
        "timed_graph_runs_per_variant": TIMING_REPEATS,
        "timing_invocations_per_observation": 1,
        "implicit_warmup_calls": 0,
        "executor_residency": "one variant per fresh RPC process",
        "cpu_affinity": 3,
        "governor_policy": "record_only",
        "timing_boundary": "host steady_clock around one synchronous GraphExecutor run RPC",
        "excludes": ["RPC startup", "module upload", "executor creation", "params load",
                     "set_input", "get_output", "output comparison"],
        "session_policy": "any wrong output, boot drift, identity drift or interruption invalidates the whole boot",
        "frozen_task_incumbent_warning": (
            "known_incumbent is the task protocol mode-0 fallback, not a tuned graph baseline claim"
        ),
    }
    write(output / "contract.json", contract)
    write(output / "pre_observation_hashes.json", {
        "contract.json": sha256_file(output / "contract.json"),
        "build_manifest_sha256": build_manifest,
    })

    image = image_input(args.image)
    started = time.monotonic()
    board = ExperimentBoard(args, "/var/volatile/" + output.name + "_runtime", args.port)
    correctness_outputs, timed_outputs = {}, {}
    rows = []
    initial = None
    governor = None
    try:
        initial = board.state()
        reuse_board.validate_state(initial, exclusive=True)
        environment = board.environment()
        governor = environment["governor"]
        if previous and initial["boot_id"] == previous["boot_id"]:
            raise ValueError("graph boot 2 requires a changed boot ID")
        write(output / "preflight.json", {"state": initial, "environment": environment})
        reuse_board.RUNTIME = runtime_hashes
        write(output / "runtime_attestation.json", board.deploy(runtime_paths))
        write(output / "bitstream_reload.json", board.reload())

        with tempfile.TemporaryDirectory(prefix="c3_yolo608_graph_") as temporary:
            for session_index, variant in enumerate(variant_order(boot_index)):
                remote = None
                board.pid = None
                session_port = args.port + 2 * session_index
                board.port = session_port
                try:
                    state = board.state()
                    reuse_board.validate_state(state, exclusive=True)
                    if state["boot_id"] != initial["boot_id"]:
                        raise ValueError("boot changed before variant session")
                    board.start()
                    for _ in range(30):
                        try:
                            remote = rpc.connect(args.host, session_port,
                                                 session_timeout=args.session_timeout)
                            break
                        except Exception:
                            time.sleep(0.25)
                    if remote is None:
                        raise RuntimeError("RPC unavailable for " + variant)
                    inventory, runtime_functions = runtime_inventory(remote)
                    if session_index == 0:
                        write(output / "runtime_inventory.json", inventory)
                    functions = {
                        "clear": runtime_functions["vta.runtime.profiler_clear"],
                        "status": runtime_functions["vta.runtime.profiler_status"],
                    }
                    device = remote.ext_dev(0)
                    executor = load_executor(remote, build_dir, variant,
                                             [device, remote.cpu(0)], temporary)
                    affinity = board.environment(require_affinity=True)
                    if affinity["governor"] != governor:
                        raise ValueError("governor changed")

                    for seed in SEEDS:
                        actual, row = invoke_once(executor, random_input(seed), functions, False)
                        correctness_outputs[variant, seed] = actual
                        row.update(phase="correctness", deployment_variant=variant,
                                   seed=seed, invocation_index=len(rows))
                        rows.append(row)
                        append(output / "calls.jsonl", row)
                    _, row = invoke_once(executor, image, functions, False)
                    row.update(phase="explicit_warmup", deployment_variant=variant,
                               seed="person_image", invocation_index=len(rows))
                    rows.append(row)
                    append(output / "calls.jsonl", row)
                    for repeat in range(TIMING_REPEATS):
                        actual, row = invoke_once(executor, image, functions, True)
                        if not math.isfinite(row["latency_ms"]) or row["latency_ms"] <= 0:
                            raise ValueError("invalid graph latency")
                        timed_outputs[variant, repeat] = actual
                        row.update(phase="timing", deployment_variant=variant,
                                   repeat=repeat, invocation_index=len(rows))
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
                    try:
                        board.stop()
                    finally:
                        board.pid = None
                log = board.ssh("cat " + board.runtime + "/rpc.log 2>/dev/null || true")
                (output / ("rpc_" + variant + ".log")).write_text(log, encoding="utf-8")
                print("variant {} complete".format(variant), flush=True)

        for seed in SEEDS:
            reference = correctness_outputs[VARIANTS[0], seed]
            for variant in VARIANTS[1:]:
                if any(mismatch_counts(reference, correctness_outputs[variant, seed])):
                    raise RuntimeError("cross-deployment random-seed output mismatch")
        for repeat in range(TIMING_REPEATS):
            reference = timed_outputs[VARIANTS[0], repeat]
            for variant in VARIANTS[1:]:
                if any(mismatch_counts(reference, timed_outputs[variant, repeat])):
                    raise RuntimeError("cross-deployment timed output mismatch")
        if not all(item["nonzero"] > 0 for row in rows for item in row["outputs"]):
            raise RuntimeError("a graph output is entirely zero")

        statistics_by_variant = {}
        for variant in VARIANTS:
            values = [float(row["latency_ms"]) for row in rows
                      if row["phase"] == "timing" and row["deployment_variant"] == variant]
            if len(values) != TIMING_REPEATS:
                raise ValueError("incomplete timing coverage")
            statistics_by_variant[variant] = {
                "values_ms": values,
                "median_ms": float(statistics.median(values)),
                "minimum_ms": min(values),
                "maximum_ms": max(values),
            }
        medians = {name: row["median_ms"] for name, row in statistics_by_variant.items()}
        after = board.state()
        reuse_board.validate_state(after, exclusive=True)
        if after["boot_id"] != initial["boot_id"]:
            raise ValueError("boot changed during graph experiment")
        final_environment = board.environment()
        if final_environment["governor"] != governor:
            raise ValueError("governor changed during graph experiment")
        write(output / "postflight.json", {"state": after, "environment": final_environment})
        summary = {
            "schema": "c3_yolo608_graph_boot_v1",
            "status": "YOLO608_GRAPH_BOOT{}_COMPLETE".format(boot_index),
            "boot_id": initial["boot_id"],
            "all_outputs_equal": True,
            "complete_graph_run_calls": len(rows),
            "correctness_calls": len(VARIANTS) * len(SEEDS),
            "explicit_warmup_calls": len(VARIANTS),
            "timing_calls": len(VARIANTS) * TIMING_REPEATS,
            "implicit_extra_calls": 0,
            "statistics": statistics_by_variant,
            "improvement_pct": {
                "known_incumbent_vs_stock": 100.0 * (1.0 - medians[VARIANTS[1]] / medians[VARIANTS[0]]),
                "selector_v2_vs_stock": 100.0 * (1.0 - medians[VARIANTS[2]] / medians[VARIANTS[0]]),
                "selector_v2_vs_known_incumbent": 100.0 * (1.0 - medians[VARIANTS[2]] / medians[VARIANTS[1]]),
            },
            "timing_boundary": contract["timing_boundary"],
            "wall_seconds": time.monotonic() - started,
        }
        write(output / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        write(output / "invalid_session.json", {
            "message": str(error), "type": type(error).__name__,
            "traceback": traceback.format_exc()[-10000:],
            "complete_graph_run_calls": len(rows),
            "all_timing_invalid": True,
            "wall_seconds": time.monotonic() - started,
        })
        raise
    finally:
        try:
            board.stop()
        except Exception as error:
            write(output / "cleanup_warning.json", {"message": str(error)})
        seal(output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD))
    parser.add_argument("--runtime-contract", default=str(DEFAULT_RUNTIME_CONTRACT))
    parser.add_argument("--image", default="/home/orange/.tvm_test_data/data/person.jpg")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--previous-boot-dir")
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=1800)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
