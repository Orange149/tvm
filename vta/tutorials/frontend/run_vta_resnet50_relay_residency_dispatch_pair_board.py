#!/usr/bin/env python3
"""Clean-start board comparison of two exact-dispatch ResNet50 builds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import time

import numpy as np
from tvm import relay, rpc
from tvm.contrib import graph_executor

from qualify_vta_residency_fsim import array_sha256
from run_vta_p7r132_y00_search_confirmation import CleanStartBoard


SEEDS = (0, 20250901, 20260910)
ROUNDS = 7
DEFAULT_RUNTIME = "/var/volatile/vta_c3_ram/runtime"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic_params_hash(path):
    params = relay.load_param_dict(Path(path).read_bytes())
    digest = hashlib.sha256()
    inventory = []
    for name in sorted(params):
        array = np.ascontiguousarray(params[name].numpy())
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
        inventory.append((name, list(array.shape), str(array.dtype)))
    return digest.hexdigest(), inventory


def load_executor(remote, build_dir, mode, contexts, temporary):
    local = Path(build_dir) / mode
    remote_name = mode + "_resnet50.so"
    upload = Path(temporary) / remote_name
    shutil.copyfile(local / "graphlib.so", upload)
    remote.upload(str(upload))
    module = remote.load_module(remote_name)
    executor = graph_executor.create(
        (local / "graph.json").read_text(encoding="utf-8"), module, contexts
    )
    executor.load_params((local / "params.bin").read_bytes())
    return executor


def input_for_seed(seed):
    rng = np.random.default_rng(int(seed))
    return rng.uniform(-1.0, 1.0, size=(1, 3, 224, 224)).astype("float32")


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


def run(args):
    build_dir = Path(args.build_dir)
    build = read_json(build_dir / "summary.json")
    if build["status"] != "resnet50_pair_cross_build_dispatch_verified":
        raise ValueError("input build is not verified")
    modes = [row["public_mode"] for row in build["builds"]]
    if modes != ["original", "weight_resident_barrier"]:
        raise ValueError("unexpected build modes")
    semantic = [semantic_params_hash(build_dir / mode / "params.bin") for mode in modes]
    if semantic[0] != semantic[1]:
        raise RuntimeError("A/B parameter content differs")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    write_json(output / "parameter_equivalence.json", {
        "status": "identical", "semantic_sha256": semantic[0][0],
        "parameter_count": len(semantic[0][1]), "inventory": semantic[0][1],
    })

    board = CleanStartBoard(args.host, args.ssh_known_hosts)
    before = board.state()
    if before.get("FPGA") != "operating" or before.get("UDMABUF") != "201326592":
        raise RuntimeError("FPGA/u-dma-buf preflight failed")
    if before.get("STORAGE_ERRORS"):
        raise RuntimeError("storage errors present before run")
    prior_rpc = board.rpc_state()
    if prior_rpc.get("CWD") != args.default_runtime:
        raise RuntimeError("unexpected RPC directory: " + repr(prior_rpc))
    board.stop_rpc(prior_rpc)
    bitstream = board.reload_frozen_bitstream()
    fresh_rpc = board.start_rpc(args.default_runtime, "p7r230_resnet50_dispatch.log")
    write_json(output / "clean_board_start.json", {
        "before": before, "stopped_rpc": prior_rpc,
        "bitstream_reload": bitstream, "fresh_rpc": fresh_rpc,
    })

    remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
    device = remote.ext_dev(0)
    contexts = [device, remote.cpu(0)]
    functions = {
        "clear": remote.get_function("vta.runtime.profiler_clear"),
        "status": remote.get_function("vta.runtime.profiler_status"),
    }
    correctness = []
    timings = []
    with tempfile.TemporaryDirectory(prefix="c3_r50_dispatch_") as temporary:
        executors = {
            mode: load_executor(remote, build_dir, mode, contexts, temporary) for mode in modes
        }
        for seed in SEEDS:
            outputs = {}
            for mode in modes:
                actual, row = invoke(executors[mode], input_for_seed(seed), functions, device, False)
                outputs[mode] = actual
                row.update(seed=int(seed), public_mode=mode)
                correctness.append(row)
            mismatch = int(np.count_nonzero(outputs[modes[0]] != outputs[modes[1]]))
            for row in correctness[-2:]:
                row["paired_mismatch_count"] = mismatch
                row["paired_equal"] = mismatch == 0
            if mismatch:
                write_json(output / "correctness.json", correctness)
                raise RuntimeError("whole-model A/B output mismatch")
        timing_data = input_for_seed(0)
        for round_index in range(ROUNDS):
            order = modes if round_index % 2 == 0 else list(reversed(modes))
            round_outputs = {}
            for position, mode in enumerate(order):
                actual, row = invoke(executors[mode], timing_data, functions, device, True)
                round_outputs[mode] = actual
                row.update(round=round_index, position=position, public_mode=mode)
                timings.append(row)
            mismatch = int(np.count_nonzero(round_outputs[modes[0]] != round_outputs[modes[1]]))
            for row in timings[-2:]:
                row["paired_mismatch_count"] = mismatch
                row["paired_equal"] = mismatch == 0
            if mismatch:
                raise RuntimeError("timed whole-model A/B output mismatch")
            print("timing round {}/{}".format(round_index + 1, ROUNDS), flush=True)

    write_json(output / "correctness.json", correctness)
    (output / "timing.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in timings), encoding="utf-8"
    )
    medians = {
        mode: statistics.median(row["latency_ms"] for row in timings if row["public_mode"] == mode)
        for mode in modes
    }
    wins = sum(
        next(row["latency_ms"] for row in timings if row["round"] == index and row["public_mode"] == modes[1])
        < next(row["latency_ms"] for row in timings if row["round"] == index and row["public_mode"] == modes[0])
        for index in range(ROUNDS)
    )
    after = board.state()
    if after.get("STORAGE_ERRORS"):
        raise RuntimeError("storage errors appeared during run")
    summary = {
        "schema": "c3_vta_resnet50_exact_residency_dispatch_board_v1",
        "status": "resnet50_pair_board_correctness_and_timing_complete",
        "boot_id": after.get("BOOT"),
        "model": "resnet50_v2",
        "pretrained": bool(build.get("pretrained")),
        "parameter_semantic_sha256": semantic[0][0],
        "exact_route_candidate_ids": [row["candidate_id"] for row in build["builds"]],
        "correctness_calls": len(correctness),
        "all_paired_outputs_equal": all(row["paired_equal"] for row in correctness + timings),
        "timing_calls": len(timings),
        "median_latency_ms": medians,
        "residency_speedup_percent": (medians[modes[0]] / medians[modes[1]] - 1.0) * 100.0,
        "residency_paired_wins": wins,
        "paired_rounds": ROUNDS,
        "board_after": after,
        "claim_boundary": (
            "Full ResNet50 graph with paired schedule-regression correctness; this is not an "
            "ImageNet accuracy evaluation"
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
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
    parser.add_argument("--default-runtime", default=DEFAULT_RUNTIME)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
