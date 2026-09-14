#!/usr/bin/env python3
"""Validate a frozen fused-program reranker choice on the VTA FPGA."""

from __future__ import annotations

import argparse
import gc
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

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from run_vta_p7r132_y00_search_confirmation import CleanStartBoard
from run_vta_resnet50_relay_residency_dispatch_pair_board import (
    ROUNDS,
    SEEDS,
    input_for_seed,
    invoke,
    semantic_params_hash,
    sha256,
    write_json,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_named_executor(remote, build_dir, remote_name, contexts, temporary):
    local = Path(build_dir) / "input_stationary"
    upload = Path(temporary) / remote_name
    shutil.copyfile(local / "graphlib.so", upload)
    remote.upload(str(upload))
    module = remote.load_module(remote_name)
    executor = graph_executor.create(
        (local / "graph.json").read_text(encoding="utf-8"), module, contexts
    )
    executor.load_params((local / "params.bin").read_bytes())
    return executor


def profile_delta(rows, program_ids, divisor=1.0):
    return {
        key: (
            statistics.median(
                row["runtime_profile_complete"][key]
                for row in rows if row["program_id"] == program_ids[1]
            ) - statistics.median(
                row["runtime_profile_complete"][key]
                for row in rows if row["program_id"] == program_ids[0]
            )
        ) / divisor
        for key in rows[0]["runtime_profile_complete"]
        if key in {
            "load_buffer_2d_bytes", "load_buffer_2d_calls",
            "load_buffer_2d_acc_bytes", "load_buffer_2d_acc_calls",
            "load_buffer_2d_inp_bytes", "load_buffer_2d_inp_calls",
            "load_buffer_2d_wgt_bytes", "load_buffer_2d_wgt_calls",
            "store_buffer_2d_bytes", "store_buffer_2d_calls",
        }
    }


def verify_inputs(contract_dir, build_dirs):
    verified = {
        "reranker_contract": len(verify_artifacts_compatible(contract_dir)),
        "pair_builds": [len(verify_artifacts_compatible(path)) for path in build_dirs],
    }
    contract = read_json(contract_dir / "contract.json")
    if contract.get("status") != "fused_service_choice_frozen_before_fpga_latency":
        raise RuntimeError("reranker choice is not prospectively frozen")
    if contract.get("board_contacted"):
        raise RuntimeError("reranker contract already contacted the board")
    exposure = contract["target_label_exposure"]
    if exposure["full_graph_latency_observed"] or exposure["partial_mask_latency_observed"]:
        raise RuntimeError("target latency was exposed before reranking")
    if exposure["operator_latency_used_for_selection"]:
        raise RuntimeError("operator latency leaked into reranking")
    builds = [read_json(path / "summary.json") for path in build_dirs]
    for index, (path, build) in enumerate(zip(build_dirs, builds)):
        if build.get("status") != "resnet50_generic_pair_cross_build_dispatch_verified":
            raise RuntimeError("pair build is incomplete")
        expected = contract["bound_inputs"]["pair_builds"][index]
        actual = {
            "summary_sha256": sha256(path / "summary.json"),
            "artifact_manifest_sha256": sha256(path / "artifact_hashes.json"),
        }
        if actual != expected:
            raise RuntimeError("reranker-bound pair build hash mismatch")
        if build["public_modes"] != ["original", "input_stationary"]:
            raise RuntimeError("unexpected build modes")
        if build["builds"][1]["candidate_id"] != contract["programs"][index]["candidate_id"]:
            raise RuntimeError("program candidate identity differs from frozen choice")
    graphs = [read_json(path / "input_stationary" / "graph.json") for path in build_dirs]
    if graphs[0] != graphs[1]:
        raise RuntimeError("reranker programs use different full graphs")
    semantic = [
        semantic_params_hash(path / "input_stationary" / "params.bin")
        for path in build_dirs
    ]
    if semantic[0] != semantic[1]:
        raise RuntimeError("reranker program parameter semantics differ")
    return contract, builds, semantic[0], verified


def run(args):
    contract_dir = Path(args.contract)
    build_dirs = [Path(args.build_a), Path(args.build_b)]
    contract, builds, semantic, verified = verify_inputs(contract_dir, build_dirs)
    program_ids = [row["program_id"] for row in contract["programs"]]
    expected = contract["expected_profile_delta_b_minus_a"]
    if set(expected) != set(contract["exact_profile_fields"]):
        raise RuntimeError("frozen profile field set differs from expected delta")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    execution_contract = {
        "schema": "c3_vta_resnet50_fused_program_reranker_board_contract_v1",
        "status": "frozen_before_board_execution",
        "reranker_contract_sha256": sha256(contract_dir / "contract.json"),
        "reranker_artifact_manifest_sha256": sha256(contract_dir / "artifact_hashes.json"),
        "verified_artifact_counts": verified,
        "executor_source_sha256": sha256(Path(__file__).resolve()),
        "program_ids": program_ids,
        "selected_program": contract["selected_program"],
        "expected_profile_delta_b_minus_a": expected,
        "seeds": list(SEEDS),
        "rounds": ROUNDS,
        "failure_policy": "fail closed before timing on any output or exact profile mismatch",
        "claim_boundary": (
            "Prospective two-tile reranker validation on one ResNet50 model and board boot; "
            "logical VTA DMA, not physical AXI or ImageNet accuracy"
        ),
    }
    write_json(output / "contract.json", execution_contract)
    write_json(output / "parameter_equivalence.json", {
        "status": "identical",
        "semantic_sha256": semantic[0],
        "parameter_count": len(semantic[1]),
        "inventory": semantic[1],
    })

    board = CleanStartBoard(args.host, args.ssh_known_hosts)
    before = board.state()
    if (before.get("FPGA") != "operating"
            or before.get("UDMABUF") != "201326592"
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
        fresh = board.start_rpc(args.default_runtime, "c3_r50_fused_reranker.log")
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
        with tempfile.TemporaryDirectory(prefix="c3_r50_fused_reranker_") as temporary:
            executors = {
                program_ids[index]: load_named_executor(
                    remote, build_dirs[index],
                    "r50_reranker_{}.so".format(index), contexts, temporary,
                )
                for index in range(2)
            }
            for seed in SEEDS:
                values = {}
                for program_id in program_ids:
                    actual, row = invoke(
                        executors[program_id], input_for_seed(seed), functions, device, False
                    )
                    values[program_id] = actual
                    row.update(seed=int(seed), program_id=program_id)
                    correctness.append(row)
                mismatch = int(np.count_nonzero(values[program_ids[0]] != values[program_ids[1]]))
                for row in correctness[-2:]:
                    row["paired_mismatch_count"] = mismatch
                    row["paired_equal"] = mismatch == 0
                if mismatch:
                    raise RuntimeError("whole-model reranker program output mismatch")
            write_json(output / "correctness.json", correctness)
            observed = profile_delta(correctness, program_ids)
            if observed != expected:
                raise RuntimeError("FPGA profile differs from frozen program delta")

            data = input_for_seed(0)
            for round_index in range(ROUNDS):
                order = program_ids if round_index % 2 == 0 else list(reversed(program_ids))
                values = {}
                for position, program_id in enumerate(order):
                    actual, row = invoke(executors[program_id], data, functions, device, True)
                    values[program_id] = actual
                    row.update(round=round_index, position=position, program_id=program_id)
                    timings.append(row)
                mismatch = int(np.count_nonzero(values[program_ids[0]] != values[program_ids[1]]))
                for row in timings[-2:]:
                    row["paired_mismatch_count"] = mismatch
                    row["paired_equal"] = mismatch == 0
                if mismatch:
                    raise RuntimeError("timed reranker program output mismatch")
                print("timing round {}/{}".format(round_index + 1, ROUNDS), flush=True)
    except Exception as error:
        caught = error
        write_json(output / "failure.json", {
            "exception_type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()[-16000:],
            "correctness_rows": len(correctness), "timing_rows": len(timings),
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
            board.start_rpc(args.default_runtime, "c3_resnet50_default_rpc_restored.log")
    if caught is not None:
        raise caught

    write_json(output / "correctness.json", correctness)
    (output / "timing.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in timings),
        encoding="utf-8",
    )
    # TVM's graph time_evaluator contributes two profiled graph executions in
    # this runtime path (the same empirical contract used by the fused-pair v2
    # runner). Normalize only the timed profile; correctness invokes once.
    observed_timed = profile_delta(timings, program_ids, divisor=2.0)
    if observed_timed != expected:
        raise RuntimeError("timed FPGA profile differs from frozen program delta")
    medians = {
        program_id: statistics.median(
            row["latency_ms"] for row in timings if row["program_id"] == program_id
        ) for program_id in program_ids
    }
    oracle = min(program_ids, key=lambda program_id: (medians[program_id], program_id))
    selected = contract["selected_program"]
    wins = sum(
        next(row["latency_ms"] for row in timings
             if row["round"] == index and row["program_id"] == selected)
        < next(row["latency_ms"] for row in timings
               if row["round"] == index and row["program_id"] != selected)
        for index in range(ROUNDS)
    )
    after = board.state()
    if after.get("STORAGE_ERRORS"):
        raise RuntimeError("storage errors appeared during reranker run")
    summary = {
        "schema": "c3_vta_resnet50_fused_program_reranker_board_v1",
        "status": "prospective_fused_service_reranker_board_evaluation_complete",
        "boot_id": after.get("BOOT"),
        "program_ids": program_ids,
        "selected_program": selected,
        "oracle_program": oracle,
        "selection_correct": selected == oracle,
        "selected_regret_percent": (medians[selected] / medians[oracle] - 1.0) * 100.0,
        "median_latency_ms": medians,
        "selected_paired_wins": wins,
        "paired_rounds": ROUNDS,
        "correctness_calls": len(correctness),
        "timing_calls": len(timings),
        "all_outputs_equal": all(row["paired_equal"] for row in correctness + timings),
        "all_outputs_nonzero": all(row["output_nonzero"] > 0 for row in correctness + timings),
        "expected_profile_delta_b_minus_a": expected,
        "observed_profile_delta_b_minus_a": profile_delta(correctness, program_ids),
        "timed_profile_delta_b_minus_a": observed_timed,
        "fused_tir_prediction_exact": True,
        "board_after": after,
        "claim_boundary": (
            "One prospective two-tile reranker test on one ResNet50 model/boot; no refit, "
            "not physical AXI traffic, ImageNet accuracy, or broad generalization"
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
    parser.add_argument("--contract", required=True)
    parser.add_argument("--build-a", required=True)
    parser.add_argument("--build-b", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
