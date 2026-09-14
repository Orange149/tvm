#!/usr/bin/env python3
"""Validate a generic exact-residency ResNet50 pair on a clean-start FPGA."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import tempfile
import time
import traceback

import numpy as np
from tvm import rpc

from run_vta_p7r132_y00_search_confirmation import CleanStartBoard
from run_vta_resnet50_callsite_composed_graph_board import verify_build_artifacts
from run_vta_resnet50_relay_residency_dispatch_pair_board import (
    ROUNDS,
    SEEDS,
    input_for_seed,
    invoke,
    load_executor,
    semantic_params_hash,
    sha256,
    write_json,
)


PROFILE_KEYS = (
    "driver_run_calls",
    "load_buffer_2d_bytes",
    "load_buffer_2d_calls",
    "load_buffer_2d_inp_bytes",
    "load_buffer_2d_inp_calls",
    "load_buffer_2d_wgt_bytes",
    "load_buffer_2d_wgt_calls",
    "synchronize_calls",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_candidate(path, candidate_id):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["candidate_id"] == candidate_id:
            return row
    raise KeyError(candidate_id)


def load_static(path, candidate_id):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["candidate_id"] == candidate_id:
            if row.get("status") != "ok":
                raise RuntimeError("static source is not successful")
            return row
    raise KeyError(candidate_id)


def graph_workload_occurrences(graph, workload):
    input_shape = workload[1][1]
    weight_shape = workload[2][1]
    shapes = graph["attrs"]["shape"][1]
    row_ptr = graph["node_row_ptr"]
    matches = []
    for node_index, node in enumerate(graph["nodes"]):
        if node.get("op") != "tvm_op":
            continue
        input_shapes = [shapes[row_ptr[src] + output] for src, output, _ in node.get("inputs", [])]
        if input_shape in input_shapes and weight_shape in input_shapes:
            matches.append({
                "graph_node": node_index,
                "func_name": node["attrs"]["func_name"],
                "input_shapes": input_shapes,
            })
    if not matches:
        raise RuntimeError("exact workload has no graph execution occurrence")
    return matches


def static_delta(original, residency):
    left = original["static_feature_vector"]
    right = residency["static_feature_vector"]
    mapping = {
        "load_buffer_2d_bytes": "load_dma_bytes",
        "load_buffer_2d_calls": "load_dma_calls",
        "load_buffer_2d_inp_bytes": "input_dma_bytes",
        "load_buffer_2d_inp_calls": "input_dma_calls",
        "load_buffer_2d_wgt_bytes": "weight_dma_bytes",
        "load_buffer_2d_wgt_calls": "weight_dma_calls",
    }
    result = {runtime: right[static] - left[static] for runtime, static in mapping.items()}
    result["synchronize_calls"] = (
        residency["sync"]["function_final_drain"] + residency["sync"]["residency_drains"]
        - original["sync"]["function_final_drain"] - original["sync"]["residency_drains"]
    )
    result["driver_run_calls"] = result["synchronize_calls"]
    return result


def profile_delta(rows, modes, divisor=1.0):
    return {
        key: (
            statistics.median(row["runtime_profile_complete"][key]
                              for row in rows if row["public_mode"] == modes[1])
            - statistics.median(row["runtime_profile_complete"][key]
                                for row in rows if row["public_mode"] == modes[0])
        ) / divisor
        for key in PROFILE_KEYS
    }


def run(args):
    build_dir = Path(args.build_dir)
    artifacts = verify_build_artifacts(build_dir)
    build = read_json(build_dir / "summary.json")
    if build.get("status") != "resnet50_generic_pair_cross_build_dispatch_verified":
        raise RuntimeError("generic ResNet50 pair is not verified")
    modes = build["public_modes"]
    if len(modes) != 2 or modes[0] != "original":
        raise RuntimeError("unexpected pair modes")
    semantic = [semantic_params_hash(build_dir / mode / "params.bin") for mode in modes]
    if semantic[0] != semantic[1]:
        raise RuntimeError("A/B parameter semantics differ")
    candidates = [load_candidate(args.candidates, row["candidate_id"]) for row in build["builds"]]
    workload = candidates[0]["identity"]["workload"]
    if candidates[1]["identity"]["workload"] != workload:
        raise RuntimeError("candidate workload changed")
    graph = read_json(build_dir / modes[0] / "graph.json")
    occurrences = graph_workload_occurrences(graph, workload)
    static_rows = [load_static(args.static_results, row["candidate_id"]) for row in build["builds"]]
    per_occurrence = static_delta(*static_rows)
    expected = {key: value * len(occurrences) for key, value in per_occurrence.items()}

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    contract = {
        "schema": "c3_vta_resnet50_generic_residency_pair_board_contract_v1",
        "status": "frozen_before_board_execution",
        "build_manifest_sha256": sha256(build_dir / "summary.json"),
        "build_artifact_hash_manifest_sha256": sha256(build_dir / "artifact_hashes.json"),
        "verified_build_artifact_count": len(artifacts),
        "executor_source_sha256": sha256(Path(__file__).resolve()),
        "candidate_source_sha256": sha256(args.candidates),
        "static_source_sha256": sha256(args.static_results),
        "modes": modes,
        "seeds": list(SEEDS),
        "rounds": ROUNDS,
        "graph_occurrences": occurrences,
        "per_occurrence_static_delta": per_occurrence,
        "expected_full_graph_delta": expected,
        "failure_policy": "fail closed before timing on output or DMA identity mismatch",
    }
    write_json(output / "contract.json", contract)

    board = CleanStartBoard(args.host, args.ssh_known_hosts)
    before = board.state()
    if (before.get("FPGA") != "operating" or before.get("UDMABUF") != "201326592"
            or before.get("STORAGE_ERRORS")):
        raise RuntimeError("board preflight failed")
    prior = board.rpc_state()
    if prior.get("CWD") != args.default_runtime:
        raise RuntimeError("unexpected RPC directory")

    remote = None
    correctness = []
    timings = []
    caught = None
    try:
        board.stop_rpc(prior)
        bitstream = board.reload_frozen_bitstream()
        fresh = board.start_rpc(args.default_runtime, "p7r273_r50a_full_graph_pair.log")
        write_json(output / "clean_board_start.json", {
            "before": before, "stopped_rpc": prior,
            "bitstream_reload": bitstream, "fresh_rpc": fresh,
        })
        remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
        device = remote.ext_dev(0)
        contexts = [device, remote.cpu(0)]
        functions = {
            "clear": remote.get_function("vta.runtime.profiler_clear"),
            "status": remote.get_function("vta.runtime.profiler_status"),
        }
        with tempfile.TemporaryDirectory(prefix="c3_r50_generic_pair_") as temporary:
            executors = {
                mode: load_executor(remote, build_dir, mode, contexts, temporary) for mode in modes
            }
            for seed in SEEDS:
                values = {}
                for mode in modes:
                    actual, row = invoke(
                        executors[mode], input_for_seed(seed), functions, device, False
                    )
                    values[mode] = actual
                    row.update(seed=int(seed), public_mode=mode)
                    correctness.append(row)
                mismatch = int(np.count_nonzero(values[modes[0]] != values[modes[1]]))
                for row in correctness[-2:]:
                    row["paired_mismatch_count"] = mismatch
                    row["paired_equal"] = mismatch == 0
                if mismatch:
                    raise RuntimeError("whole-model generic A/B output mismatch")
            write_json(output / "correctness.json", correctness)
            observed = profile_delta(correctness, modes)
            if observed != expected:
                raise RuntimeError("whole-graph DMA delta differs from occurrence aggregation")

            data = input_for_seed(0)
            for round_index in range(ROUNDS):
                order = modes if round_index % 2 == 0 else list(reversed(modes))
                values = {}
                for position, mode in enumerate(order):
                    actual, row = invoke(executors[mode], data, functions, device, True)
                    values[mode] = actual
                    row.update(round=round_index, position=position, public_mode=mode)
                    timings.append(row)
                mismatch = int(np.count_nonzero(values[modes[0]] != values[modes[1]]))
                for row in timings[-2:]:
                    row["paired_mismatch_count"] = mismatch
                    row["paired_equal"] = mismatch == 0
                if mismatch:
                    raise RuntimeError("timed generic A/B output mismatch")
                print("timing round {}/{}".format(round_index + 1, ROUNDS), flush=True)
    except Exception as error:
        caught = error
        write_json(output / "failure.json", {
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()[-16000:],
            "correctness_rows": len(correctness),
            "timing_rows": len(timings),
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
            board.start_rpc(args.default_runtime, "p7r273_default_rpc_restored.log")
    if caught is not None:
        raise caught

    write_json(output / "correctness.json", correctness)
    (output / "timing.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in timings), encoding="utf-8"
    )
    observed_timed = profile_delta(timings, modes, divisor=2.0)
    if observed_timed != expected:
        raise RuntimeError("timed whole-graph DMA delta differs from occurrence aggregation")
    medians = {
        mode: statistics.median(row["latency_ms"] for row in timings
                                if row["public_mode"] == mode)
        for mode in modes
    }
    wins = sum(
        next(row["latency_ms"] for row in timings
             if row["round"] == index and row["public_mode"] == modes[1])
        < next(row["latency_ms"] for row in timings
               if row["round"] == index and row["public_mode"] == modes[0])
        for index in range(ROUNDS)
    )
    after = board.state()
    if after.get("STORAGE_ERRORS"):
        raise RuntimeError("storage errors appeared during generic pair run")
    summary = {
        "schema": "c3_vta_resnet50_generic_residency_pair_board_v1",
        "status": "generic_resnet50_pair_correctness_dma_and_timing_complete",
        "boot_id": after.get("BOOT"),
        "modes": modes,
        "graph_occurrence_count": len(occurrences),
        "graph_occurrences": occurrences,
        "correctness_calls": len(correctness),
        "timing_calls": len(timings),
        "all_outputs_equal": all(row["paired_equal"] for row in correctness + timings),
        "all_outputs_nonzero": all(row["output_nonzero"] > 0 for row in correctness + timings),
        "observed_full_graph_delta": profile_delta(correctness, modes),
        "timed_full_graph_delta": observed_timed,
        "occurrence_aggregation_exact": True,
        "median_latency_ms": medians,
        "residency_speedup_percent": (medians[modes[0]] / medians[modes[1]] - 1.0) * 100.0,
        "residency_paired_wins": wins,
        "paired_rounds": ROUNDS,
        "board_after": after,
        "claim_boundary": (
            "One exact R50A workload broadcast to three graph occurrences on one boot; "
            "not call-site isolation, ImageNet accuracy, or physical AXI traffic"
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
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--static-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
