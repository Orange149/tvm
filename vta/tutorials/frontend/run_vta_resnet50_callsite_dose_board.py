#!/usr/bin/env python3
"""Measure k=0..4 exact-node ResNet50 residency dose on a clean-start FPGA."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import tempfile
import time
import traceback

from tvm import rpc

from run_vta_p7r132_y00_search_confirmation import CleanStartBoard
from run_vta_resnet50_callsite_composed_graph_board import (
    PROFILE_KEYS,
    SEEDS,
    invoke,
    make_executor,
    read_json,
    sha256,
    upload_modules,
    verify_build_artifacts,
    write_json,
)
from run_vta_resnet50_relay_residency_dispatch_pair_board import input_for_seed


ROUNDS = 7
MODES = ("baseline_k0", "resident_prefix_k")


def median_profile_delta(rows, left, right, divisor=1.0):
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


def close_remote(remote):
    if remote is not None:
        del remote
        gc.collect()
        time.sleep(0.25)


def run_pair(args, board, build_dir, output, count):
    prior = board.rpc_state()
    if prior.get("CWD") != args.default_runtime:
        raise RuntimeError("unexpected RPC before dose pair: " + repr(prior))
    board.stop_rpc(prior)
    bitstream = board.reload_frozen_bitstream()
    fresh = board.start_rpc(
        args.default_runtime,
        "p7r267_r50_callsite_dose_k{}.log".format(count),
    )
    clean = {"resident_node_count": count, "stopped_rpc": prior,
             "bitstream_reload": bitstream, "fresh_rpc": fresh}
    write_json(output / "clean_start_k{}.json".format(count), clean)

    remote = None
    correctness = []
    timings = []
    try:
        remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
        device = remote.ext_dev(0)
        contexts = [device, remote.cpu(0)]
        functions = {
            "clear": remote.get_function("vta.runtime.profiler_clear"),
            "status": remote.get_function("vta.runtime.profiler_status"),
        }
        with tempfile.TemporaryDirectory(prefix="c3_r50_dose_k{}_".format(count)) as temporary:
            incumbent, barrier, wrapper = upload_modules(remote, build_dir, temporary)
            keep_alive = (incumbent, barrier, wrapper)
            executors = {
                MODES[0]: make_executor(
                    remote, incumbent, build_dir / "prefix0_graph.json",
                    build_dir / "params.bin", contexts,
                ),
                MODES[1]: make_executor(
                    remote, incumbent, build_dir / "prefix{}_graph.json".format(count),
                    build_dir / "params.bin", contexts,
                ),
            }
            for seed in SEEDS:
                values = {}
                for mode in MODES:
                    actual, row = invoke(
                        executors[mode], input_for_seed(seed), functions, device, False
                    )
                    values[mode] = actual
                    row.update(resident_node_count=count, seed=int(seed), mode=mode)
                    correctness.append(row)
                mismatch = int((values[MODES[0]] != values[MODES[1]]).sum())
                for row in correctness[-2:]:
                    row["paired_mismatch_count"] = mismatch
                    row["paired_equal"] = mismatch == 0
                if mismatch:
                    raise RuntimeError("dose correctness mismatch at k={}".format(count))

            data = input_for_seed(0)
            for round_index in range(ROUNDS):
                order = MODES if round_index % 2 == 0 else tuple(reversed(MODES))
                values = {}
                for position, mode in enumerate(order):
                    actual, row = invoke(executors[mode], data, functions, device, True)
                    values[mode] = actual
                    row.update(resident_node_count=count, round=round_index,
                               position=position, mode=mode)
                    timings.append(row)
                mismatch = int((values[MODES[0]] != values[MODES[1]]).sum())
                for row in timings[-2:]:
                    row["paired_mismatch_count"] = mismatch
                    row["paired_equal"] = mismatch == 0
                if mismatch:
                    raise RuntimeError("dose timing mismatch at k={}".format(count))
                print("k={} timing round {}/{}".format(count, round_index + 1, ROUNDS), flush=True)
            del keep_alive
    finally:
        close_remote(remote)
    return clean, correctness, timings


def run(args):
    build_dir = Path(args.build_dir)
    artifacts = verify_build_artifacts(build_dir)
    manifest = read_json(build_dir / "manifest.json")
    if manifest.get("status") != "five_graph_variants_frozen_before_board":
        raise RuntimeError("dose build is not frozen")
    if manifest.get("matching_graph_nodes") != [57, 72, 85, 98]:
        raise RuntimeError("dose graph identity changed")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    contract = {
        "schema": "c3_vta_resnet50_callsite_dose_board_contract_v1",
        "status": "frozen_before_board_execution",
        "build_manifest_sha256": sha256(build_dir / "manifest.json"),
        "build_artifact_hash_manifest_sha256": sha256(build_dir / "artifact_hashes.json"),
        "verified_build_artifact_count": len(artifacts),
        "executor_source_sha256": sha256(Path(__file__).resolve()),
        "resident_node_counts": [1, 2, 3, 4],
        "seeds": list(SEEDS),
        "rounds_per_pair": ROUNDS,
        "expected_per_resident_node_delta": manifest["expected_per_resident_node_delta"],
        "execution_isolation": "one clean bitstream/RPC and two executors per k pair",
        "failure_policy": "fail closed, preserve partial rows, restore default RPC",
    }
    write_json(output / "contract.json", contract)

    board = CleanStartBoard(args.host, args.ssh_known_hosts)
    before = board.state()
    if (before.get("FPGA") != "operating" or before.get("UDMABUF") != "201326592"
            or before.get("STORAGE_ERRORS")):
        raise RuntimeError("board preflight failed")
    if board.rpc_state().get("CWD") != args.default_runtime:
        raise RuntimeError("default RPC is not active before dose experiment")

    all_correctness = []
    all_timings = []
    clean_starts = []
    caught = None
    try:
        for count in contract["resident_node_counts"]:
            clean, correctness, timings = run_pair(
                args, board, build_dir, output, count
            )
            clean_starts.append(clean)
            all_correctness.extend(correctness)
            all_timings.extend(timings)
            write_json(output / "correctness.json", all_correctness)
            (output / "timing.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_timings),
                encoding="utf-8",
            )
    except Exception as error:
        caught = error
        write_json(output / "failure.json", {
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()[-16000:],
            "completed_clean_starts": clean_starts,
            "correctness_rows": len(all_correctness),
            "timing_rows": len(all_timings),
        })
    finally:
        state = board.rpc_state()
        if state.get("CWD") != args.default_runtime:
            if state.get("PID") and state.get("CWD"):
                board.stop_rpc(state)
            board.reload_frozen_bitstream()
            board.start_rpc(args.default_runtime, "p7r267_default_rpc_restored.log")
    if caught is not None:
        raise caught

    expected_unit = contract["expected_per_resident_node_delta"]
    per_count = []
    for count in contract["resident_node_counts"]:
        correctness = [row for row in all_correctness if row["resident_node_count"] == count]
        timings = [row for row in all_timings if row["resident_node_count"] == count]
        observed = median_profile_delta(correctness, MODES[0], MODES[1])
        observed_timed = median_profile_delta(timings, MODES[0], MODES[1], divisor=2.0)
        for key, unit in expected_unit.items():
            expected = count * unit
            if observed[key] != expected or observed_timed[key] != expected:
                raise RuntimeError(
                    "nonlinear logical profile for k={} {}: {}/{} != {}".format(
                        count, key, observed[key], observed_timed[key], expected
                    )
                )
        medians = {
            mode: statistics.median(row["latency_ms"] for row in timings if row["mode"] == mode)
            for mode in MODES
        }
        wins = sum(
            next(row["latency_ms"] for row in timings
                 if row["round"] == round_index and row["mode"] == MODES[1])
            < next(row["latency_ms"] for row in timings
                   if row["round"] == round_index and row["mode"] == MODES[0])
            for round_index in range(ROUNDS)
        )
        per_count.append({
            "resident_node_count": count,
            "resident_graph_nodes": manifest["matching_graph_nodes"][:count],
            "correctness_profile_delta": observed,
            "timed_profile_delta": observed_timed,
            "median_latency_ms": medians,
            "paired_delta_ms": medians[MODES[1]] - medians[MODES[0]],
            "speedup_percent": (medians[MODES[0]] / medians[MODES[1]] - 1.0) * 100.0,
            "paired_wins": wins,
            "paired_rounds": ROUNDS,
        })
    unit_latency_delta = per_count[0]["paired_delta_ms"]
    for row in per_count:
        row["latency_linear_residual_ms_vs_k_times_k1"] = (
            row["paired_delta_ms"] - row["resident_node_count"] * unit_latency_delta
        )

    after = board.state()
    if after.get("STORAGE_ERRORS"):
        raise RuntimeError("storage errors appeared during dose experiment")
    summary = {
        "schema": "c3_vta_resnet50_callsite_dose_board_v1",
        "status": "four_pair_callsite_dose_correctness_dma_and_timing_complete",
        "boot_id": after.get("BOOT"),
        "matching_graph_nodes": manifest["matching_graph_nodes"],
        "correctness_calls": len(all_correctness),
        "timing_calls": len(all_timings),
        "clean_start_pairs": len(clean_starts),
        "all_outputs_equal": all(row["paired_equal"] for row in all_correctness + all_timings),
        "all_outputs_nonzero": all(row["output_nonzero"] > 0 for row in all_correctness + all_timings),
        "logical_dma_and_submission_delta_exactly_linear": True,
        "per_resident_node_count": per_count,
        "board_after": after,
        "claim_boundary": (
            "Four prefix subsets on one boot with a clean RPC/bitstream per pair; latency "
            "residuals are descriptive, not independent samples or arbitrary-subset optimality"
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
