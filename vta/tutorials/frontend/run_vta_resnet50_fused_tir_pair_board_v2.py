#!/usr/bin/env python3
"""Validate and time an R50 residency pair against a frozen fused-TIR contract."""

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
    "load_buffer_2d_bytes",
    "load_buffer_2d_calls",
    "load_buffer_2d_acc_bytes",
    "load_buffer_2d_acc_calls",
    "load_buffer_2d_inp_bytes",
    "load_buffer_2d_inp_calls",
    "load_buffer_2d_wgt_bytes",
    "load_buffer_2d_wgt_calls",
    "store_buffer_2d_bytes",
    "store_buffer_2d_calls",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def profile_delta(rows, modes, divisor=1.0):
    return {
        key: (
            statistics.median(
                row["runtime_profile_complete"][key]
                for row in rows
                if row["public_mode"] == modes[1]
            )
            - statistics.median(
                row["runtime_profile_complete"][key]
                for row in rows
                if row["public_mode"] == modes[0]
            )
        )
        / divisor
        for key in PROFILE_KEYS
    }


def load_fused_contract(path):
    path = Path(path)
    analysis = read_json(path / "fused_tir_occurrence_delta.json")
    status = analysis.get("status")
    accepted = {
        "compiler_fused_tir_prediction_frozen_before_fpga",
        "compiler_fused_tir_prediction_matches_fpga_profile_exactly",
    }
    if status not in accepted:
        raise RuntimeError("fused-TIR analysis is neither prospective nor reconciled")
    if status == "compiler_fused_tir_prediction_frozen_before_fpga" and (
        analysis.get("full_graph_latency_observed") is not False
        or analysis.get("partial_mask_latency_observed") is not False
    ):
        raise RuntimeError("prospective fused-TIR contract has observed latency")
    hashes = read_json(path / "artifact_hashes.json")["artifacts"]
    for relative, expected in hashes.items():
        actual = sha256(path / relative)
        if actual != expected:
            raise RuntimeError("fused-TIR analysis artifact hash mismatch: " + relative)
    expected = analysis["full_graph_fused_tir_delta"]
    if set(expected) != set(PROFILE_KEYS):
        raise RuntimeError("fused-TIR contract has an unexpected profile field set")
    occurrence_count = analysis.get("graph_occurrence_count")
    if not isinstance(occurrence_count, int) or occurrence_count <= 0:
        raise RuntimeError("fused-TIR contract has no graph occurrence")
    if len(analysis.get("graph_occurrences", [])) != occurrence_count:
        raise RuntimeError("fused-TIR graph occurrence list is incomplete")
    return analysis, expected, len(hashes)


def run(args):
    build_dir = Path(args.build_dir)
    build_artifacts = verify_build_artifacts(build_dir)
    build = read_json(build_dir / "summary.json")
    if build.get("status") != "resnet50_generic_pair_cross_build_dispatch_verified":
        raise RuntimeError("generic ResNet50 pair is not verified")
    modes = build["public_modes"]
    if modes != ["original", "input_stationary"]:
        raise RuntimeError("this v2 contract expects an original/input-stationary pair")
    semantic = [semantic_params_hash(build_dir / mode / "params.bin") for mode in modes]
    if semantic[0] != semantic[1]:
        raise RuntimeError("A/B parameter semantics differ")

    analysis_dir = Path(args.fused_analysis)
    fused, expected, analysis_artifact_count = load_fused_contract(analysis_dir)
    if fused["modes"] != modes:
        raise RuntimeError("fused-TIR analysis modes differ from build modes")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    contract = {
        "schema": "c3_vta_resnet50_fused_tir_pair_board_contract_v2",
        "status": "frozen_before_board_execution",
        "build_manifest_sha256": sha256(build_dir / "summary.json"),
        "build_artifact_hash_manifest_sha256": sha256(build_dir / "artifact_hashes.json"),
        "verified_build_artifact_count": len(build_artifacts),
        "fused_analysis_sha256": sha256(analysis_dir / "fused_tir_occurrence_delta.json"),
        "fused_analysis_artifact_hash_manifest_sha256": sha256(
            analysis_dir / "artifact_hashes.json"
        ),
        "verified_fused_analysis_artifact_count": analysis_artifact_count,
        "executor_source_sha256": sha256(Path(__file__).resolve()),
        "modes": modes,
        "seeds": list(SEEDS),
        "rounds": ROUNDS,
        "graph_occurrences": fused["graph_occurrences"],
        "expected_full_graph_fused_tir_delta": expected,
        "failure_policy": (
            "fail closed before timing unless outputs and every frozen fused-TIR "
            "LOAD/STORE field match exactly"
        ),
        "claim_boundary": (
            "A prospective fused-TIR prediction is checked before timing; this contract "
            "does not claim call-site isolation, physical AXI traffic, or ImageNet accuracy"
        ),
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
        fresh = board.start_rpc(args.default_runtime, "c3_resnet50_fused_tir_pair.log")
        write_json(
            output / "clean_board_start.json",
            {
                "before": before,
                "stopped_rpc": prior,
                "bitstream_reload": bitstream,
                "fresh_rpc": fresh,
            },
        )
        remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
        device = remote.ext_dev(0)
        contexts = [device, remote.cpu(0)]
        functions = {
            "clear": remote.get_function("vta.runtime.profiler_clear"),
            "status": remote.get_function("vta.runtime.profiler_status"),
        }
        with tempfile.TemporaryDirectory(prefix="c3_r50_fused_pair_") as temporary:
            executors = {
                mode: load_executor(remote, build_dir, mode, contexts, temporary)
                for mode in modes
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
                    raise RuntimeError("whole-model fused-TIR A/B output mismatch")
            write_json(output / "correctness.json", correctness)
            observed = profile_delta(correctness, modes)
            if observed != expected:
                raise RuntimeError("FPGA profile differs from frozen fused-TIR prediction")

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
                    raise RuntimeError("timed fused-TIR A/B output mismatch")
                print("timing round {}/{}".format(round_index + 1, ROUNDS), flush=True)
    except Exception as error:
        caught = error
        write_json(
            output / "failure.json",
            {
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc()[-16000:],
                "correctness_rows": len(correctness),
                "timing_rows": len(timings),
            },
        )
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
            board.start_rpc(args.default_runtime, "c3_resnet50_default_rpc_restored.log")
    if caught is not None:
        raise caught

    write_json(output / "correctness.json", correctness)
    (output / "timing.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in timings),
        encoding="utf-8",
    )
    observed_timed = profile_delta(timings, modes, divisor=2.0)
    if observed_timed != expected:
        raise RuntimeError("timed FPGA profile differs from frozen fused-TIR prediction")
    medians = {
        mode: statistics.median(
            row["latency_ms"] for row in timings if row["public_mode"] == mode
        )
        for mode in modes
    }
    wins = sum(
        next(
            row["latency_ms"]
            for row in timings
            if row["round"] == index and row["public_mode"] == modes[1]
        )
        < next(
            row["latency_ms"]
            for row in timings
            if row["round"] == index and row["public_mode"] == modes[0]
        )
        for index in range(ROUNDS)
    )
    after = board.state()
    if after.get("STORAGE_ERRORS"):
        raise RuntimeError("storage errors appeared during fused-TIR pair run")
    summary = {
        "schema": "c3_vta_resnet50_fused_tir_pair_board_v2",
        "status": "fused_tir_predicted_pair_correctness_dma_and_timing_complete",
        "boot_id": after.get("BOOT"),
        "modes": modes,
        "graph_occurrence_count": fused["graph_occurrence_count"],
        "correctness_calls": len(correctness),
        "timing_calls": len(timings),
        "all_outputs_equal": all(row["paired_equal"] for row in correctness + timings),
        "all_outputs_nonzero": all(
            row["output_nonzero"] > 0 for row in correctness + timings
        ),
        "expected_full_graph_fused_tir_delta": expected,
        "observed_full_graph_delta": profile_delta(correctness, modes),
        "timed_full_graph_delta": observed_timed,
        "fused_tir_prediction_exact": True,
        "median_latency_ms": medians,
        "residency_speedup_percent": (medians[modes[0]] / medians[modes[1]] - 1.0) * 100.0,
        "residency_paired_wins": wins,
        "paired_rounds": ROUNDS,
        "board_after": after,
        "claim_boundary": (
            "One exact ResNet50 workload at {} graph occurrences on one boot; "
            "not ImageNet accuracy, call-site isolation, or physical AXI traffic"
        ).format(fused["graph_occurrence_count"]),
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(
        " ".join(__import__("sys").argv) + "\n", encoding="utf-8"
    )
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            }
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--fused-analysis", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
