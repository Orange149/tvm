#!/usr/bin/env python3
"""Compare singleton-greedy and whole-group R50A call-site admission on FPGA."""

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

from build_vta_resnet50_r50a_callsite_subsets import mask_name
from run_vta_p7r132_y00_search_confirmation import CleanStartBoard
from run_vta_resnet50_callsite_composed_graph_board import (
    invoke,
    make_executor,
    read_json,
    sha256,
    verify_build_artifacts,
    write_json,
)
from run_vta_resnet50_relay_residency_dispatch_pair_board import input_for_seed
from run_vta_resnet50_fused_tir_pair_board_v2 import PROFILE_KEYS


SEEDS = (0, 20250901, 20260910)
ROUNDS = 7
MODES = ("incumbent", "proposal")


def graph_file(mask):
    return mask_name(tuple(mask)) + ".json"


def add_delta(rows):
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) for key in keys}


def expected_mask_delta(routes, baseline_mask, proposal_mask):
    changed = [
        row["expected_delta"]
        for left, right, row in zip(baseline_mask, proposal_mask, routes)
        if left == 0 and right == 1
    ]
    if not changed:
        return {key: 0 for key in PROFILE_KEYS}
    return add_delta(changed)


def profile_delta(rows, divisor=1.0):
    return {
        key: (
            statistics.median(
                row["runtime_profile_complete"][key]
                for row in rows
                if row["mode"] == MODES[1]
            )
            - statistics.median(
                row["runtime_profile_complete"][key]
                for row in rows
                if row["mode"] == MODES[0]
            )
        )
        / divisor
        for key in PROFILE_KEYS
    }


def upload_modules(remote, build_dir, temporary, manifest):
    names = (
        "incumbent_graphlib.so",
        manifest["resident_graphlib"],
        manifest["wrapper_file"],
    )
    for name in names:
        local = Path(temporary) / name
        shutil.copyfile(build_dir / name, local)
        remote.upload(str(local))
    residency = remote.load_module(manifest["resident_graphlib"])
    wrapper = remote.load_module(manifest["wrapper_file"])
    incumbent = remote.load_module("incumbent_graphlib.so")
    incumbent.import_module(wrapper)
    return incumbent, residency, wrapper


def start_clean(board, args, log_name):
    prior = board.rpc_state()
    if prior.get("CWD") != args.default_runtime:
        raise RuntimeError("unexpected RPC before R50A admission pair: " + repr(prior))
    board.stop_rpc(prior)
    bitstream = board.reload_frozen_bitstream()
    fresh = board.start_rpc(args.default_runtime, log_name)
    return {"stopped_rpc": prior, "bitstream_reload": bitstream, "fresh_rpc": fresh}


def paired_wins(rows):
    return sum(
        next(
            row["latency_ms"]
            for row in rows
            if row["round"] == round_index and row["mode"] == MODES[1]
        )
        < next(
            row["latency_ms"]
            for row in rows
            if row["round"] == round_index and row["mode"] == MODES[0]
        )
        for round_index in range(ROUNDS)
    )


def run_pair(
    args,
    board,
    build_dir,
    manifest,
    output,
    pair_name,
    baseline_mask,
    proposal_mask,
):
    clean = start_clean(board, args, "p7r278_r50a_{}.log".format(pair_name))
    write_json(
        output / ("clean_start_" + pair_name + ".json"),
        {
            **clean,
            "pair_name": pair_name,
            "baseline_mask": baseline_mask,
            "proposal_mask": proposal_mask,
        },
    )
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
        with tempfile.TemporaryDirectory(prefix="c3_r50a_admission_") as temporary:
            incumbent, residency, wrapper = upload_modules(
                remote, build_dir, temporary, manifest
            )
            keep_alive = (incumbent, residency, wrapper)
            executors = {
                MODES[0]: make_executor(
                    remote,
                    incumbent,
                    build_dir / graph_file(baseline_mask),
                    build_dir / "params.bin",
                    contexts,
                ),
                MODES[1]: make_executor(
                    remote,
                    incumbent,
                    build_dir / graph_file(proposal_mask),
                    build_dir / "params.bin",
                    contexts,
                ),
            }
            for seed in SEEDS:
                values = {}
                for mode in MODES:
                    actual, row = invoke(
                        executors[mode], input_for_seed(seed), functions, device, False
                    )
                    values[mode] = actual
                    row.update(pair_name=pair_name, seed=int(seed), mode=mode)
                    correctness.append(row)
                mismatch = int(np.count_nonzero(values[MODES[0]] != values[MODES[1]]))
                for row in correctness[-2:]:
                    row["paired_mismatch_count"] = mismatch
                    row["paired_equal"] = mismatch == 0
                if mismatch:
                    raise RuntimeError("R50A admission correctness mismatch: " + pair_name)
            expected = expected_mask_delta(
                manifest["routes"], baseline_mask, proposal_mask
            )
            observed = profile_delta(correctness)
            if observed != expected:
                raise RuntimeError("R50A admission DMA mismatch: " + pair_name)

            data = input_for_seed(0)
            for round_index in range(ROUNDS):
                order = MODES if round_index % 2 == 0 else tuple(reversed(MODES))
                values = {}
                for position, mode in enumerate(order):
                    actual, row = invoke(
                        executors[mode], data, functions, device, True
                    )
                    values[mode] = actual
                    row.update(
                        pair_name=pair_name,
                        round=round_index,
                        position=position,
                        mode=mode,
                    )
                    timings.append(row)
                mismatch = int(np.count_nonzero(values[MODES[0]] != values[MODES[1]]))
                for row in timings[-2:]:
                    row["paired_mismatch_count"] = mismatch
                    row["paired_equal"] = mismatch == 0
                if mismatch:
                    raise RuntimeError("R50A admission timing mismatch: " + pair_name)
                print(
                    "{} round {}/{}".format(pair_name, round_index + 1, ROUNDS),
                    flush=True,
                )
            del keep_alive
    finally:
        if remote is not None:
            del remote
            gc.collect()
            time.sleep(0.25)

    observed_timed = profile_delta(timings, divisor=2.0)
    if observed_timed != expected:
        raise RuntimeError("R50A timed admission DMA mismatch: " + pair_name)
    medians = {
        mode: statistics.median(
            row["latency_ms"] for row in timings if row["mode"] == mode
        )
        for mode in MODES
    }
    wins = paired_wins(timings)
    delta = medians[MODES[1]] - medians[MODES[0]]
    checks = {
        "three_seed_outputs_equal": all(row["paired_equal"] for row in correctness),
        "fused_tir_dma_exact": observed == expected and observed_timed == expected,
        "seven_paired_rounds": len(timings) == 2 * ROUNDS,
        "seven_of_seven_candidate_wins": wins == ROUNDS,
        "strictly_negative_median_delta": delta < 0,
    }
    return correctness, timings, {
        "pair_name": pair_name,
        "baseline_mask": baseline_mask,
        "proposal_mask": proposal_mask,
        "expected_delta": expected,
        "correctness_profile_delta": observed,
        "timed_profile_delta": observed_timed,
        "median_latency_ms": medians,
        "paired_delta_ms": delta,
        "speedup_percent": (medians[MODES[0]] / medians[MODES[1]] - 1.0) * 100.0,
        "paired_wins": wins,
        "paired_rounds": ROUNDS,
        "checks": checks,
        "accepted": all(checks.values()),
        "model_execution_calls": len(correctness) + len(timings),
    }


def run(args):
    build_dir = Path(args.build_dir)
    artifacts = verify_build_artifacts(build_dir)
    manifest = read_json(build_dir / "manifest.json")
    if manifest.get("status") != "eight_route_subsets_frozen_before_callsite_board_labels":
        raise RuntimeError("R50A call-site domain is not frozen")
    nodes = manifest["proposal_order"]
    if nodes != [67, 80, 93] or len(manifest["variants"]) != 8:
        raise RuntimeError("R50A call-site domain changed")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    contract = {
        "schema": "c3_vta_resnet50_r50a_grouped_admission_contract_v1",
        "status": "frozen_before_callsite_board_execution",
        "build_manifest_sha256": sha256(build_dir / "manifest.json"),
        "build_artifact_hash_manifest_sha256": sha256(build_dir / "artifact_hashes.json"),
        "verified_build_artifact_count": len(artifacts),
        "executor_source_sha256": sha256(Path(__file__).resolve()),
        "proposal_order": nodes,
        "gate": manifest["gate"],
        "strict_singleton_policy": (
            "test nodes in graph order against current incumbent; accept only if all gate "
            "conditions hold"
        ),
        "grouped_policy": (
            "test all tied positive-DMA-gain nodes as one group; recursively split only if "
            "the group fails; no split is needed if 000->111 passes"
        ),
        "label_exposure": manifest["label_exposure"],
        "failure_policy": "fail closed and restore default RPC on execution error",
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
    if board.rpc_state().get("CWD") != args.default_runtime:
        raise RuntimeError("default RPC is not active before R50A admission experiment")

    all_correctness = []
    all_timings = []
    singleton_results = []
    group_result = None
    caught = None
    try:
        current = [0, 0, 0]
        for step, node in enumerate(nodes):
            proposal = list(current)
            proposal[step] = 1
            correctness, timings, result = run_pair(
                args,
                board,
                build_dir,
                manifest,
                output,
                "singleton_node{}".format(node),
                list(current),
                proposal,
            )
            singleton_results.append(result)
            all_correctness.extend(correctness)
            all_timings.extend(timings)
            write_json(output / "singleton_progress.json", singleton_results)
            if result["accepted"]:
                current = proposal

        correctness, timings, group_result = run_pair(
            args,
            board,
            build_dir,
            manifest,
            output,
            "whole_group",
            [0, 0, 0],
            [1, 1, 1],
        )
        all_correctness.extend(correctness)
        all_timings.extend(timings)
        write_json(output / "group_progress.json", group_result)
    except Exception as error:
        caught = error
        write_json(
            output / "failure.json",
            {
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc()[-16000:],
                "completed_singleton_pairs": len(singleton_results),
                "group_completed": group_result is not None,
            },
        )
    finally:
        state = board.rpc_state()
        if state.get("CWD") != args.default_runtime:
            if state.get("PID") and state.get("CWD"):
                board.stop_rpc(state)
            board.reload_frozen_bitstream()
            board.start_rpc(args.default_runtime, "p7r278_default_rpc_restored.log")
    if caught is not None:
        raise caught

    singleton_selected = [0, 0, 0]
    for index, result in enumerate(singleton_results):
        if result["accepted"]:
            singleton_selected[index] = 1
    if group_result["accepted"]:
        grouped_selected = [1, 1, 1]
        group_pairs_tested = 1
        recursive_split_exercised = False
    else:
        grouped_selected = None
        group_pairs_tested = 1
        recursive_split_exercised = False
    after = board.state()
    if after.get("STORAGE_ERRORS"):
        raise RuntimeError("storage errors appeared during R50A admission experiment")
    summary = {
        "schema": "c3_vta_resnet50_r50a_grouped_admission_board_v1",
        "status": "singleton_and_whole_group_admission_complete",
        "boot_id": after.get("BOOT"),
        "all_outputs_equal": all(
            row["paired_equal"] for row in all_correctness + all_timings
        ),
        "all_outputs_nonzero": all(
            row["output_nonzero"] > 0 for row in all_correctness + all_timings
        ),
        "singleton_results": singleton_results,
        "singleton_selected_mask": singleton_selected,
        "singleton_pairs_tested": len(singleton_results),
        "singleton_model_execution_calls": sum(
            row["model_execution_calls"] for row in singleton_results
        ),
        "group_result": group_result,
        "grouped_selected_mask": grouped_selected,
        "group_pairs_tested": group_pairs_tested,
        "group_model_execution_calls": group_result["model_execution_calls"],
        "recursive_split_exercised": recursive_split_exercised,
        "board_after": after,
        "claim_boundary": (
            "Raw board measurements only; authoritative development/holdout scope must be "
            "established by a separate audit of the pre-board contracts. Recursive failure "
            "splitting was not exercised when the whole group passed"
        ),
    }
    write_json(output / "correctness.json", all_correctness)
    (output / "timing.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_timings),
        encoding="utf-8",
    )
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
