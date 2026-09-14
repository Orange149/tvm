#!/usr/bin/env python3
"""Run incumbent-protected greedy admission over four ResNet50 graph nodes."""

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

from build_vta_resnet50_callsite_marginal_graphs import mask_name
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


PAIR_MODES = ("current_incumbent", "one_node_proposal")
FINAL_MODES = ("all_original", "greedy_selected", "all_resident")
PAIR_ROUNDS = 7
FINAL_ORDERS = (
    FINAL_MODES,
    (FINAL_MODES[1], FINAL_MODES[2], FINAL_MODES[0]),
    (FINAL_MODES[2], FINAL_MODES[0], FINAL_MODES[1]),
) * 3


def graph_file(mask):
    return mask_name(tuple(mask)) + ".json"


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


def paired_wins(rows, left, right, rounds):
    return sum(
        next(row["latency_ms"] for row in rows
             if row["round"] == round_index and row["mode"] == right)
        < next(row["latency_ms"] for row in rows
               if row["round"] == round_index and row["mode"] == left)
        for round_index in range(rounds)
    )


def start_clean(board, args, log_name):
    prior = board.rpc_state()
    if prior.get("CWD") != args.default_runtime:
        raise RuntimeError("unexpected RPC before greedy action: " + repr(prior))
    board.stop_rpc(prior)
    bitstream = board.reload_frozen_bitstream()
    fresh = board.start_rpc(args.default_runtime, log_name)
    return {"stopped_rpc": prior, "bitstream_reload": bitstream, "fresh_rpc": fresh}


def run_pair(args, board, build_dir, output, step, node, baseline_mask, candidate_mask):
    clean = start_clean(board, args, "p7r271_r50_greedy_step{}.log".format(step))
    write_json(output / "clean_start_step{}.json".format(step), {
        **clean,
        "step": step,
        "graph_node": node,
        "baseline_mask": baseline_mask,
        "candidate_mask": candidate_mask,
    })
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
        with tempfile.TemporaryDirectory(prefix="c3_r50_greedy_step{}_".format(step)) as temporary:
            incumbent, barrier, wrapper = upload_modules(remote, build_dir, temporary)
            keep_alive = (incumbent, barrier, wrapper)
            executors = {
                PAIR_MODES[0]: make_executor(
                    remote, incumbent, build_dir / graph_file(baseline_mask),
                    build_dir / "params.bin", contexts,
                ),
                PAIR_MODES[1]: make_executor(
                    remote, incumbent, build_dir / graph_file(candidate_mask),
                    build_dir / "params.bin", contexts,
                ),
            }
            for seed in SEEDS:
                values = {}
                for mode in PAIR_MODES:
                    actual, row = invoke(
                        executors[mode], input_for_seed(seed), functions, device, False
                    )
                    values[mode] = actual
                    row.update(step=step, graph_node=node, seed=int(seed), mode=mode,
                               baseline_mask=baseline_mask, candidate_mask=candidate_mask)
                    correctness.append(row)
                mismatch = int((values[PAIR_MODES[0]] != values[PAIR_MODES[1]]).sum())
                for row in correctness[-2:]:
                    row["paired_mismatch_count"] = mismatch
                    row["paired_equal"] = mismatch == 0
                if mismatch:
                    raise RuntimeError("greedy correctness mismatch at step {}".format(step))

            data = input_for_seed(0)
            for round_index in range(PAIR_ROUNDS):
                order = PAIR_MODES if round_index % 2 == 0 else tuple(reversed(PAIR_MODES))
                values = {}
                for position, mode in enumerate(order):
                    actual, row = invoke(executors[mode], data, functions, device, True)
                    values[mode] = actual
                    row.update(step=step, graph_node=node, round=round_index,
                               position=position, mode=mode,
                               baseline_mask=baseline_mask, candidate_mask=candidate_mask)
                    timings.append(row)
                mismatch = int((values[PAIR_MODES[0]] != values[PAIR_MODES[1]]).sum())
                for row in timings[-2:]:
                    row["paired_mismatch_count"] = mismatch
                    row["paired_equal"] = mismatch == 0
                if mismatch:
                    raise RuntimeError("greedy timing mismatch at step {}".format(step))
                print("greedy step {}/4 node {} round {}/{}".format(
                    step, node, round_index + 1, PAIR_ROUNDS
                ), flush=True)
            del keep_alive
    finally:
        if remote is not None:
            del remote
            gc.collect()
            time.sleep(0.25)
    return correctness, timings


def summarize_proposal(step, node, baseline_mask, candidate_mask, correctness, timings, expected):
    observed = profile_delta(correctness, *PAIR_MODES)
    observed_timed = profile_delta(timings, *PAIR_MODES, divisor=2.0)
    dma_identity = all(
        observed[key] == value and observed_timed[key] == value
        for key, value in expected.items()
    )
    medians = {
        mode: statistics.median(row["latency_ms"] for row in timings if row["mode"] == mode)
        for mode in PAIR_MODES
    }
    delta = medians[PAIR_MODES[1]] - medians[PAIR_MODES[0]]
    wins = paired_wins(timings, *PAIR_MODES, rounds=PAIR_ROUNDS)
    checks = {
        "three_seed_outputs_equal": all(row["paired_equal"] for row in correctness),
        "logical_dma_and_submission_identity": dma_identity,
        "seven_paired_rounds": len(timings) == 2 * PAIR_ROUNDS,
        "seven_of_seven_candidate_wins": wins == PAIR_ROUNDS,
        "strictly_negative_median_delta": delta < 0,
    }
    return {
        "step": step,
        "graph_node": node,
        "baseline_mask": baseline_mask,
        "candidate_mask": candidate_mask,
        "checks": checks,
        "accepted": all(checks.values()),
        "correctness_profile_delta": observed,
        "timed_profile_delta": observed_timed,
        "median_latency_ms": medians,
        "paired_delta_ms": delta,
        "speedup_percent": (medians[PAIR_MODES[0]] / medians[PAIR_MODES[1]] - 1.0) * 100.0,
        "paired_wins": wins,
        "paired_rounds": PAIR_ROUNDS,
    }


def run_final(args, board, build_dir, output, selected_mask):
    clean = start_clean(board, args, "p7r271_r50_greedy_final.log")
    write_json(output / "clean_start_final.json", {**clean, "selected_mask": selected_mask})
    remote = None
    correctness = []
    timings = []
    masks = {
        FINAL_MODES[0]: [0, 0, 0, 0],
        FINAL_MODES[1]: selected_mask,
        FINAL_MODES[2]: [1, 1, 1, 1],
    }
    try:
        remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
        device = remote.ext_dev(0)
        contexts = [device, remote.cpu(0)]
        functions = {
            "clear": remote.get_function("vta.runtime.profiler_clear"),
            "status": remote.get_function("vta.runtime.profiler_status"),
        }
        with tempfile.TemporaryDirectory(prefix="c3_r50_greedy_final_") as temporary:
            incumbent, barrier, wrapper = upload_modules(remote, build_dir, temporary)
            keep_alive = (incumbent, barrier, wrapper)
            executors = {
                mode: make_executor(
                    remote, incumbent, build_dir / graph_file(mask),
                    build_dir / "params.bin", contexts,
                )
                for mode, mask in masks.items()
            }
            for seed in SEEDS:
                values = {}
                for mode in FINAL_MODES:
                    actual, row = invoke(
                        executors[mode], input_for_seed(seed), functions, device, False
                    )
                    values[mode] = actual
                    row.update(seed=int(seed), mode=mode, mask=masks[mode])
                    correctness.append(row)
                mismatch = max(
                    int((values[FINAL_MODES[0]] != values[mode]).sum())
                    for mode in FINAL_MODES[1:]
                )
                for row in correctness[-3:]:
                    row["all_variants_mismatch_count"] = mismatch
                    row["all_variants_equal"] = mismatch == 0
                if mismatch:
                    raise RuntimeError("final greedy graph output mismatch")

            data = input_for_seed(0)
            for round_index, order in enumerate(FINAL_ORDERS):
                values = {}
                for position, mode in enumerate(order):
                    actual, row = invoke(executors[mode], data, functions, device, True)
                    values[mode] = actual
                    row.update(round=round_index, position=position, mode=mode, mask=masks[mode])
                    timings.append(row)
                mismatch = max(
                    int((values[FINAL_MODES[0]] != values[mode]).sum())
                    for mode in FINAL_MODES[1:]
                )
                for row in timings[-3:]:
                    row["all_variants_mismatch_count"] = mismatch
                    row["all_variants_equal"] = mismatch == 0
                if mismatch:
                    raise RuntimeError("final timed greedy graph output mismatch")
                print("final validation round {}/{}".format(
                    round_index + 1, len(FINAL_ORDERS)
                ), flush=True)
            del keep_alive
    finally:
        if remote is not None:
            del remote
            gc.collect()
            time.sleep(0.25)

    medians = {
        mode: statistics.median(row["latency_ms"] for row in timings if row["mode"] == mode)
        for mode in FINAL_MODES
    }
    profiles = {
        mode: {
            key: statistics.median(
                row["runtime_profile_complete"][key]
                for row in correctness if row["mode"] == mode
            )
            for key in PROFILE_KEYS
        }
        for mode in FINAL_MODES
    }
    return correctness, timings, {
        "selected_mask": selected_mask,
        "median_latency_ms": medians,
        "selected_vs_original_delta_ms": medians[FINAL_MODES[1]] - medians[FINAL_MODES[0]],
        "full_vs_original_delta_ms": medians[FINAL_MODES[2]] - medians[FINAL_MODES[0]],
        "selected_vs_full_delta_ms": medians[FINAL_MODES[1]] - medians[FINAL_MODES[2]],
        "selected_vs_original_wins": paired_wins(
            timings, FINAL_MODES[0], FINAL_MODES[1], len(FINAL_ORDERS)
        ),
        "full_vs_original_wins": paired_wins(
            timings, FINAL_MODES[0], FINAL_MODES[2], len(FINAL_ORDERS)
        ),
        "paired_rounds": len(FINAL_ORDERS),
        "correctness_profiles": profiles,
    }


def run(args):
    build_dir = Path(args.build_dir)
    artifacts = verify_build_artifacts(build_dir)
    manifest = read_json(build_dir / "manifest.json")
    if manifest.get("status") != "sixteen_route_subsets_frozen_before_board":
        raise RuntimeError("all-subsets build is not frozen")
    nodes = manifest["proposal_order"]
    if nodes != [57, 72, 85, 98] or len(manifest.get("variants", [])) != 16:
        raise RuntimeError("greedy route domain changed")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    contract = {
        "schema": "c3_vta_resnet50_callsite_greedy_board_contract_v1",
        "status": "frozen_before_board_execution",
        "build_manifest_sha256": sha256(build_dir / "manifest.json"),
        "build_artifact_hash_manifest_sha256": sha256(build_dir / "artifact_hashes.json"),
        "verified_build_artifact_count": len(artifacts),
        "executor_source_sha256": sha256(Path(__file__).resolve()),
        "proposal_order": nodes,
        "acceptance_gate": manifest["acceptance_gate"],
        "expected_per_edge_delta": manifest["expected_per_edge_delta"],
        "final_validation_modes": FINAL_MODES,
        "final_balanced_orders": FINAL_ORDERS,
        "label_exposure": manifest["label_exposure"],
        "failure_policy": "fail closed, preserve decisions, restore default RPC",
    }
    write_json(output / "contract.json", contract)

    board = CleanStartBoard(args.host, args.ssh_known_hosts)
    before = board.state()
    if (before.get("FPGA") != "operating" or before.get("UDMABUF") != "201326592"
            or before.get("STORAGE_ERRORS")):
        raise RuntimeError("board preflight failed")
    if board.rpc_state().get("CWD") != args.default_runtime:
        raise RuntimeError("default RPC is not active before greedy experiment")

    selected_mask = [0, 0, 0, 0]
    proposals = []
    proposal_correctness = []
    proposal_timings = []
    caught = None
    final_correctness = []
    final_timings = []
    final_result = None
    try:
        for step, node in enumerate(nodes, start=1):
            candidate_mask = list(selected_mask)
            candidate_mask[step - 1] = 1
            correctness, timings = run_pair(
                args, board, build_dir, output, step, node,
                list(selected_mask), candidate_mask,
            )
            proposal = summarize_proposal(
                step, node, list(selected_mask), candidate_mask,
                correctness, timings, contract["expected_per_edge_delta"],
            )
            proposals.append(proposal)
            proposal_correctness.extend(correctness)
            proposal_timings.extend(timings)
            if proposal["accepted"]:
                selected_mask = candidate_mask
            proposal["resulting_incumbent_mask"] = list(selected_mask)
            write_json(output / "proposal_decisions.json", proposals)
            write_json(output / "proposal_correctness.json", proposal_correctness)
            (output / "proposal_timing.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in proposal_timings),
                encoding="utf-8",
            )
            print("step {} node {} accepted={} incumbent={}".format(
                step, node, proposal["accepted"], selected_mask
            ), flush=True)

        final_correctness, final_timings, final_result = run_final(
            args, board, build_dir, output, selected_mask
        )
        write_json(output / "final_correctness.json", final_correctness)
        (output / "final_timing.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in final_timings),
            encoding="utf-8",
        )
    except Exception as error:
        caught = error
        write_json(output / "failure.json", {
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()[-16000:],
            "completed_proposals": proposals,
            "selected_mask": selected_mask,
        })
    finally:
        state = board.rpc_state()
        if state.get("CWD") != args.default_runtime:
            if state.get("PID") and state.get("CWD"):
                board.stop_rpc(state)
            board.reload_frozen_bitstream()
            board.start_rpc(args.default_runtime, "p7r271_default_rpc_restored.log")
    if caught is not None:
        raise caught

    after = board.state()
    if after.get("STORAGE_ERRORS"):
        raise RuntimeError("storage errors appeared during greedy experiment")
    summary = {
        "schema": "c3_vta_resnet50_callsite_greedy_board_v1",
        "status": "incumbent_protected_four_node_greedy_and_final_validation_complete",
        "boot_id": after.get("BOOT"),
        "proposal_count": len(proposals),
        "accepted_nodes": [row["graph_node"] for row in proposals if row["accepted"]],
        "rejected_nodes": [row["graph_node"] for row in proposals if not row["accepted"]],
        "selected_mask": selected_mask,
        "all_proposal_outputs_equal": all(
            row["paired_equal"] for row in proposal_correctness + proposal_timings
        ),
        "proposal_correctness_calls": len(proposal_correctness),
        "proposal_timing_calls": len(proposal_timings),
        "proposal_results": proposals,
        "final_correctness_calls": len(final_correctness),
        "final_timing_calls": len(final_timings),
        "all_final_outputs_equal": all(
            row["all_variants_equal"] for row in final_correctness + final_timings
        ),
        "final_result": final_result,
        "board_after": after,
        "claim_boundary": (
            "One exposed ResNet50 route domain and one boot; this validates fail-closed "
            "graph-node deployment admission, not globally optimal subset search"
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
