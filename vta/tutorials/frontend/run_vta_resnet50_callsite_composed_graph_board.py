#!/usr/bin/env python3
"""Validate one graph-node VTA residency replacement on a clean-start board."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import time
import traceback

import numpy as np
from tvm import rpc
from tvm.contrib import graph_executor

from qualify_vta_residency_fsim import array_sha256
from run_vta_p7r132_y00_search_confirmation import CleanStartBoard
from run_vta_resnet50_relay_residency_dispatch_pair_board import input_for_seed


MODES = ("incumbent_all4", "callsite0_barrier")
SEEDS = (0, 20250901, 20260910)
ROUNDS = 7
PROFILE_KEYS = (
    "driver_run_calls",
    "load_buffer_2d_bytes",
    "load_buffer_2d_calls",
    "load_buffer_2d_wgt_bytes",
    "load_buffer_2d_wgt_calls",
    "synchronize_calls",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_build_artifacts(build_dir):
    recorded = read_json(Path(build_dir) / "artifact_hashes.json")["artifacts"]
    mismatches = {}
    for relative, expected in recorded.items():
        path = Path(build_dir) / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError("frozen call-site build hash mismatch: " + repr(mismatches))
    return recorded


def upload_modules(remote, build_dir, temporary):
    names = (
        "incumbent_graphlib.so",
        "c3_r50_barrier_graphlib.so",
        "c3_r50_callsite_wrapper.so",
    )
    for name in names:
        local = Path(temporary) / name
        shutil.copyfile(Path(build_dir) / name, local)
        remote.upload(str(local))
    # Loading the barrier module through TVM initializes its module context
    # before the wrapper resolves the exact DSO symbol with dlopen/dlsym.
    barrier = remote.load_module("c3_r50_barrier_graphlib.so")
    wrapper = remote.load_module("c3_r50_callsite_wrapper.so")
    incumbent = remote.load_module("incumbent_graphlib.so")
    incumbent.import_module(wrapper)
    return incumbent, barrier, wrapper


def make_executor(remote, module, graph_path, params_path, contexts):
    executor = graph_executor.create(
        Path(graph_path).read_text(encoding="utf-8"), module, contexts
    )
    executor.load_params(Path(params_path).read_bytes())
    return executor


def invoke(executor, data, functions, device, timed):
    executor.set_input("data", data)
    functions["clear"]()
    started = time.perf_counter()
    if timed:
        measured = executor.module.time_evaluator("run", device, number=1, repeat=1)()
        latency_ms = float(measured.mean * 1000.0)
    else:
        executor.run()
        latency_ms = None
    host_wall_ms = (time.perf_counter() - started) * 1000.0
    output = executor.get_output(0).numpy()
    return output, {
        "latency_ms": latency_ms,
        "host_wall_ms": host_wall_ms,
        "output_sha256": array_sha256(output),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_nonzero": int(np.count_nonzero(output)),
        "output_min": float(output.min()),
        "output_max": float(output.max()),
        "runtime_profile_complete": json.loads(functions["status"]()),
    }


def profile_delta(rows, left, right, divisor=1.0):
    result = {}
    for key in PROFILE_KEYS:
        lhs = statistics.median(
            row["runtime_profile_complete"][key] for row in rows if row["mode"] == left
        )
        rhs = statistics.median(
            row["runtime_profile_complete"][key] for row in rows if row["mode"] == right
        )
        result[key] = (rhs - lhs) / divisor
    return result


def run(args):
    build_dir = Path(args.build_dir)
    build_artifacts = verify_build_artifacts(build_dir)
    manifest = read_json(build_dir / "manifest.json")
    if manifest["status"] != "cross_built_unmeasured":
        raise ValueError("call-site graph build is not in the expected state")
    if manifest["selected_graph_node"] != 57 or manifest["matching_graph_nodes"] != [57, 72, 85, 98]:
        raise RuntimeError("exact ResNet50 graph-node identity changed")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    contract = {
        "schema": "c3_vta_resnet50_callsite_composed_board_contract_v1",
        "status": "frozen_before_board_execution",
        "build_manifest_sha256": sha256(build_dir / "manifest.json"),
        "artifact_hash_manifest_sha256": sha256(build_dir / "artifact_hashes.json"),
        "verified_build_artifact_count": len(build_artifacts),
        "executor_source_sha256": sha256(Path(__file__).resolve()),
        "modes": MODES,
        "seeds": SEEDS,
        "rounds": ROUNDS,
        "expected_per_inference_delta": {
            "load_buffer_2d_wgt_bytes": -393216,
            "load_buffer_2d_wgt_calls": -432,
            "synchronize_calls": 16,
        },
        "failure_policy": "preserve raw failure and restore default RPC if needed",
    }
    write_json(output / "contract.json", contract)

    board = CleanStartBoard(args.host, args.ssh_known_hosts)
    before = board.state()
    if (
        before.get("FPGA") != "operating"
        or before.get("UDMABUF") != "201326592"
        or before.get("STORAGE_ERRORS")
    ):
        raise RuntimeError("board preflight failed")
    prior_rpc = board.rpc_state()
    if prior_rpc.get("CWD") != args.default_runtime:
        raise RuntimeError("unexpected RPC directory: " + repr(prior_rpc))

    fresh_rpc = None
    remote = None
    correctness = []
    timings = []
    caught = None
    try:
        board.stop_rpc(prior_rpc)
        bitstream = board.reload_frozen_bitstream()
        fresh_rpc = board.start_rpc(args.default_runtime, "p7r265_r50_callsite0.log")
        write_json(output / "clean_board_start.json", {
            "before": before,
            "stopped_rpc": prior_rpc,
            "bitstream_reload": bitstream,
            "fresh_rpc": fresh_rpc,
        })
        remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
        device = remote.ext_dev(0)
        contexts = [device, remote.cpu(0)]
        functions = {
            "clear": remote.get_function("vta.runtime.profiler_clear"),
            "status": remote.get_function("vta.runtime.profiler_status"),
        }
        with tempfile.TemporaryDirectory(prefix="c3_r50_callsite_") as temporary:
            incumbent, barrier, wrapper = upload_modules(remote, build_dir, temporary)
            # Keep barrier/wrapper remote handles alive for the composed executor.
            modules = (incumbent, barrier, wrapper)
            executors = {
                "incumbent_all4": make_executor(
                    remote, incumbent, build_dir / "incumbent_graph.json",
                    build_dir / "params.bin", contexts,
                ),
                "callsite0_barrier": make_executor(
                    remote, incumbent, build_dir / "callsite0_graph.json",
                    build_dir / "params.bin", contexts,
                ),
            }
            for seed in SEEDS:
                outputs = {}
                for mode in MODES:
                    actual, row = invoke(
                        executors[mode], input_for_seed(seed), functions, device, False
                    )
                    outputs[mode] = actual
                    row.update(seed=int(seed), mode=mode)
                    correctness.append(row)
                mismatch = int(np.count_nonzero(outputs[MODES[0]] != outputs[MODES[1]]))
                for row in correctness[-2:]:
                    row["paired_mismatch_count"] = mismatch
                    row["paired_equal"] = mismatch == 0
                if mismatch:
                    raise RuntimeError("call-site composed graph output mismatch")
            write_json(output / "correctness.json", correctness)
            observed = profile_delta(correctness, MODES[0], MODES[1])
            for key, expected in contract["expected_per_inference_delta"].items():
                if observed[key] != expected:
                    raise RuntimeError(
                        "call-site DMA identity mismatch for {}: {} != {}".format(
                            key, observed[key], expected
                        )
                    )
            data = input_for_seed(0)
            for round_index in range(ROUNDS):
                order = MODES if round_index % 2 == 0 else tuple(reversed(MODES))
                round_outputs = {}
                for position, mode in enumerate(order):
                    actual, row = invoke(executors[mode], data, functions, device, True)
                    round_outputs[mode] = actual
                    row.update(round=round_index, position=position, mode=mode)
                    timings.append(row)
                mismatch = int(
                    np.count_nonzero(round_outputs[MODES[0]] != round_outputs[MODES[1]])
                )
                for row in timings[-2:]:
                    row["paired_mismatch_count"] = mismatch
                    row["paired_equal"] = mismatch == 0
                if mismatch:
                    raise RuntimeError("timed call-site graph output mismatch")
                print("timing round {}/{}".format(round_index + 1, ROUNDS), flush=True)
            del modules
    except Exception as error:
        caught = error
        write_json(output / "failure.json", {
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()[-12000:],
        })
    finally:
        if remote is not None:
            del remote
            gc.collect()
            time.sleep(0.25)
        state = board.rpc_state()
        if state.get("CWD") != args.default_runtime:
            if state.get("PID") and state.get("CWD"):
                board.stop_rpc(state)
            board.reload_frozen_bitstream()
            board.start_rpc(args.default_runtime, "p7r265_default_rpc_restored.log")
    if caught is not None:
        raise caught

    write_json(output / "correctness.json", correctness)
    (output / "timing.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in timings),
        encoding="utf-8",
    )
    medians = {
        mode: statistics.median(row["latency_ms"] for row in timings if row["mode"] == mode)
        for mode in MODES
    }
    wins = sum(
        next(row["latency_ms"] for row in timings if row["round"] == index and row["mode"] == MODES[1])
        < next(row["latency_ms"] for row in timings if row["round"] == index and row["mode"] == MODES[0])
        for index in range(ROUNDS)
    )
    after = board.state()
    if after.get("STORAGE_ERRORS"):
        raise RuntimeError("storage errors appeared during call-site run")
    summary = {
        "schema": "c3_vta_resnet50_callsite_composed_board_v1",
        "status": "callsite0_graph_composition_correctness_dma_and_timing_complete",
        "boot_id": after.get("BOOT"),
        "selected_graph_node": manifest["selected_graph_node"],
        "matching_graph_nodes": manifest["matching_graph_nodes"],
        "correctness_calls": len(correctness),
        "timing_calls": len(timings),
        "all_outputs_equal": all(row["paired_equal"] for row in correctness + timings),
        "all_outputs_nonzero": all(row["output_nonzero"] > 0 for row in correctness + timings),
        "per_inference_profile_delta": profile_delta(correctness, MODES[0], MODES[1]),
        "timed_profile_delta": profile_delta(timings, MODES[0], MODES[1], divisor=2.0),
        "median_latency_ms": medians,
        "callsite0_speedup_percent": (medians[MODES[0]] / medians[MODES[1]] - 1.0) * 100.0,
        "callsite0_paired_wins": wins,
        "paired_rounds": ROUNDS,
        "board_after": after,
        "claim_boundary": (
            "One exact graph node composed from two pre-qualified modules on one boot; "
            "not native Relay call-site tuning, ImageNet accuracy, or physical AXI traffic"
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(
        " ".join(__import__("sys").argv) + "\n", encoding="utf-8"
    )
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
