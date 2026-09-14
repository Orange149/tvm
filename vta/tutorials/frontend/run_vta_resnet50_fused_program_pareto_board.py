#!/usr/bin/env python3
"""Clean-start FPGA evaluation of a frozen final-fused ResNet50 Pareto wave."""

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
from analyze_vta_resnet50_fused_tir_occurrence_delta import RUNTIME_KEYS
from run_vta_p7r132_y00_search_confirmation import (
    CleanStartBoard,
    EXPECTED_DEFAULT_RUNTIME_HASHES,
)
from run_vta_resnet50_relay_residency_dispatch_pair_board import (
    ROUNDS,
    SEEDS,
    input_for_seed,
    invoke,
    semantic_params_hash,
    sha256,
    write_json,
)
from run_vta_resnet50_fused_program_reranker_board import profile_delta


class CandidateRejected(RuntimeError):
    """A deterministic program-level correctness/profile rejection."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expected_delta(program, stock_features):
    return {
        key: int(program["traffic"][key]) - int(stock_features["traffic"][key])
        for key in RUNTIME_KEYS
    }


def choose_ids(pool, scope):
    front = list(pool["pareto_front_candidate_ids"])
    all_ids = [row["candidate_id"] for row in pool["programs"]]
    if scope == "front":
        return front
    return sorted(set(all_ids) - set(front))


def verify_inputs(pool_dir, scope, prior_front_result):
    verified = {"pool": len(verify_artifacts_compatible(pool_dir))}
    pool = read_json(pool_dir / "pool.json")
    accepted_statuses = {
        "final_fused_pareto_front_frozen_before_fpga",
        "operator_proxy_front_fullgraphs_built_before_fpga",
        "operator_proxy_oracle_completion_pool_after_front_labels",
        "invalid_dominator_peeling_oracle_completion_pool_after_online_labels",
    }
    if pool.get("status") not in accepted_statuses:
        raise RuntimeError("frozen Pareto/proxy-front pool is not pristine")
    completion_pool = pool.get("status") in {
        "operator_proxy_oracle_completion_pool_after_front_labels",
        "invalid_dominator_peeling_oracle_completion_pool_after_online_labels",
    }
    if completion_pool:
        expected_visibilities = ({
            "prospective_front_fpga_correctness": True,
            "prospective_front_full_graph_latency": True,
            "remaining_fpga_correctness": False,
            "remaining_full_graph_latency": False,
            "tophub_latency": False,
        }, {
            "online_fpga_correctness": True,
            "online_full_graph_latency": True,
            "remaining_fpga_correctness": False,
            "remaining_full_graph_latency": False,
            "tophub_latency": False,
        }, {
            "online_fpga_correctness": True,
            "online_full_graph_latency": True,
            "remaining_fpga_correctness": False,
            "remaining_full_graph_latency": False,
            "tophub_latency": False,
            "additional_frozen_control_labels_already_observed": True,
        })
        if scope != "remaining" or pool.get("label_visibility") not in expected_visibilities:
            raise RuntimeError("oracle-completion label boundary drift")
        if pool.get("board_contacted_for_remaining"):
            raise RuntimeError("remaining target labels leaked into oracle-completion pool")
    elif pool.get("board_contacted") or any(pool["label_visibility"].values()):
        raise RuntimeError("target board or latency label leaked into the frozen pool")
    if scope == "front" and prior_front_result is not None:
        raise ValueError("--prior-front-result is only valid for --scope remaining")
    prior = None
    if scope == "remaining":
        if prior_front_result is None:
            raise ValueError("--prior-front-result is required for oracle completion")
        verified["prior_front_result"] = len(
            verify_artifacts_compatible(prior_front_result)
        )
        prior = read_json(prior_front_result / "summary.json")
        if prior.get("status") not in {
            "prospective_final_fused_pareto_front_board_complete",
            "online_peeling_complete",
        }:
            raise RuntimeError("prior front result is incomplete")
        current_manifest = sha256(pool_dir / "artifact_hashes.json")
        if completion_pool:
            if prior.get("status") == "online_peeling_complete":
                if pool.get("online_result_artifact_manifest_sha256") != sha256(
                    prior_front_result / "artifact_hashes.json"
                ):
                    raise RuntimeError("completion pool does not bind the online result")
            else:
                if pool.get("prospective_front_source_artifact_manifest_sha256") != prior[
                    "pool_artifact_manifest_sha256"
                ]:
                    raise RuntimeError("completion pool does not bind the prior front pool")
                if pool.get("prospective_front_board_artifact_manifest_sha256") != sha256(
                    prior_front_result / "artifact_hashes.json"
                ):
                    raise RuntimeError("completion pool does not bind the prior front result")
        elif prior["pool_artifact_manifest_sha256"] != current_manifest:
            raise RuntimeError("prior front result binds a different fused pool")
        prior_ids = prior.get("candidate_ids", prior.get("dispatched_candidate_ids"))
        if prior_ids != pool["pareto_front_candidate_ids"]:
            raise RuntimeError("prior result did not execute the frozen Pareto front")

    by_id = {row["candidate_id"]: row for row in pool["programs"]}
    candidate_ids = choose_ids(pool, scope)
    stock_dir = pool_dir / "stock_reference"
    stock_graph = read_json(stock_dir / "graph.json")
    stock_semantic = semantic_params_hash(stock_dir / "params.bin")
    semantic_hashes = {"stock_reference": stock_semantic[0]}
    for candidate_id in candidate_ids:
        row = by_id[candidate_id]
        local = pool_dir / row["relative_dir"]
        if sha256(local / "graph.json") != row["graph_sha256"]:
            raise RuntimeError("candidate graph hash mismatch: " + candidate_id)
        if sha256(local / "graphlib.so") != row["graphlib_sha256"]:
            raise RuntimeError("candidate DSO hash mismatch: " + candidate_id)
        if sha256(local / "params.bin") != row["params_sha256"]:
            raise RuntimeError("candidate parameter hash mismatch: " + candidate_id)
        if read_json(local / "graph.json") != stock_graph:
            raise RuntimeError("candidate Graph JSON differs from stock: " + candidate_id)
        semantic = semantic_params_hash(local / "params.bin")
        if semantic != stock_semantic:
            raise RuntimeError("candidate parameter semantics differ: " + candidate_id)
        semantic_hashes[candidate_id] = semantic[0]
    return pool, by_id, candidate_ids, verified, semantic_hashes, prior


def load_executor(remote, local, remote_name, contexts, temporary):
    upload = Path(temporary) / remote_name
    shutil.copyfile(local / "graphlib.so", upload)
    remote.upload(str(upload))
    module = remote.load_module(remote_name)
    executor = graph_executor.create(
        (local / "graph.json").read_text(encoding="utf-8"), module, contexts
    )
    executor.load_params((local / "params.bin").read_bytes())
    return executor


def assert_board_state(state):
    if state.get("FPGA") != "operating" or state.get("UDMABUF") != "201326592":
        raise RuntimeError("FPGA/u-dma-buf preflight failed")
    if state.get("STORAGE_ERRORS"):
        raise RuntimeError("storage errors present on board")


def model_input_for_seed(seed, input_spec):
    """Create the deterministic input declared by a full-model experiment."""
    if input_spec is None:
        return input_for_seed(seed)
    rng = np.random.default_rng(int(seed))
    low, high = input_spec.get("uniform_range", [0.0, 1.0])
    return rng.uniform(low, high, size=tuple(input_spec["shape"])).astype("float32")


def invoke_model(executor, data, functions, device, timed, all_outputs):
    """Invoke one graph while retaining the legacy single-output record format."""
    if not all_outputs:
        return invoke(executor, data, functions, device, timed)
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
    outputs = [
        executor.get_output(output_index).numpy()
        for output_index in range(executor.get_num_outputs())
    ]
    return outputs, {
        "latency_ms": latency_ms,
        "host_wall_ms": host_wall_ms,
        "output_count": len(outputs),
        "output_shapes": [list(value.shape) for value in outputs],
        "output_dtypes": [str(value.dtype) for value in outputs],
        "output_nonzero": sum(int(np.count_nonzero(value)) for value in outputs),
        "output_min": min(float(value.min()) for value in outputs),
        "output_max": max(float(value.max()) for value in outputs),
        "runtime_profile_complete": json.loads(functions["status"]()),
    }


def model_mismatch_count(left, right, all_outputs):
    if not all_outputs:
        return int(np.count_nonzero(left != right))
    if len(left) != len(right):
        raise CandidateRejected("whole-model output count differs")
    return sum(int(np.count_nonzero(a != b)) for a, b in zip(left, right))


def run_one(board, args, output, pool_dir, program, stock_features, index, input_spec=None):
    candidate_id = program["candidate_id"]
    expected = expected_delta(program, stock_features)
    before = board.state()
    assert_board_state(before)
    if board.runtime_hashes(args.default_runtime) != EXPECTED_DEFAULT_RUNTIME_HASHES:
        raise RuntimeError("default runtime hash mismatch")
    prior = board.rpc_state()
    if prior.get("CWD") != args.default_runtime:
        raise RuntimeError("unexpected RPC directory before candidate " + candidate_id)
    board.stop_rpc(prior)
    bitstream = board.reload_frozen_bitstream()
    fresh = board.start_rpc(
        args.default_runtime, "c3_r50d_pareto_{:02d}.log".format(index)
    )

    remote = None
    correctness = []
    timings = []
    observed = None
    timed_delta = None
    try:
        remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
        device = remote.ext_dev(0)
        contexts = [device, remote.cpu(0)]
        functions = {
            "clear": remote.get_function("vta.runtime.profiler_clear"),
            "status": remote.get_function("vta.runtime.profiler_status"),
        }
        with tempfile.TemporaryDirectory(prefix="c3_r50d_pareto_") as temporary:
            executors = {
                "stock_reference": load_executor(
                    remote, pool_dir / "stock_reference",
                    "r50d_stock_{:02d}.so".format(index), contexts, temporary,
                ),
                candidate_id: load_executor(
                    remote, pool_dir / program["relative_dir"],
                    "r50d_candidate_{:02d}.so".format(index), contexts, temporary,
                ),
            }
            ids = ["stock_reference", candidate_id]
            for seed_index, seed in enumerate(SEEDS):
                order = ids if seed_index % 2 == 0 else list(reversed(ids))
                values = {}
                seed_rows = []
                for position, program_id in enumerate(order):
                    actual, row = invoke_model(
                        executors[program_id], model_input_for_seed(seed, input_spec),
                        functions, device, False,
                        bool(input_spec and input_spec.get("all_outputs")),
                    )
                    values[program_id] = actual
                    row.update(seed=int(seed), position=position, program_id=program_id)
                    correctness.append(row)
                    seed_rows.append(row)
                mismatch = model_mismatch_count(
                    values[ids[0]], values[ids[1]],
                    bool(input_spec and input_spec.get("all_outputs")),
                )
                for row in seed_rows:
                    row["paired_mismatch_count"] = mismatch
                    row["paired_equal"] = mismatch == 0
                if mismatch or any(row["output_nonzero"] == 0 for row in seed_rows):
                    raise CandidateRejected("whole-model output mismatch or all-zero output")
            observed = profile_delta(correctness, ids)
            if observed != expected:
                raise CandidateRejected("correctness profile differs from fused-TIR prediction")

            data = model_input_for_seed(0, input_spec)
            for round_index in range(ROUNDS):
                order = ids if round_index % 2 == 0 else list(reversed(ids))
                values = {}
                round_rows = []
                for position, program_id in enumerate(order):
                    actual, row = invoke_model(
                        executors[program_id], data, functions, device, True,
                        bool(input_spec and input_spec.get("all_outputs")),
                    )
                    values[program_id] = actual
                    row.update(round=round_index, position=position, program_id=program_id)
                    timings.append(row)
                    round_rows.append(row)
                mismatch = model_mismatch_count(
                    values[ids[0]], values[ids[1]],
                    bool(input_spec and input_spec.get("all_outputs")),
                )
                for row in round_rows:
                    row["paired_mismatch_count"] = mismatch
                    row["paired_equal"] = mismatch == 0
                if mismatch:
                    raise CandidateRejected("timed whole-model output mismatch")
            timed_delta = profile_delta(timings, ids, divisor=2.0)
            if timed_delta != expected:
                raise CandidateRejected("timed profile differs from fused-TIR prediction")

        medians = {
            program_id: statistics.median(
                row["latency_ms"] for row in timings if row["program_id"] == program_id
            ) for program_id in ids
        }
        ratios = []
        deltas = []
        candidate_wins = 0
        for round_index in range(ROUNDS):
            values = {
                row["program_id"]: row["latency_ms"] for row in timings
                if row["round"] == round_index
            }
            ratios.append(values[candidate_id] / values["stock_reference"])
            deltas.append(values[candidate_id] - values["stock_reference"])
            candidate_wins += values[candidate_id] < values["stock_reference"]
        return {
            "candidate_id": candidate_id,
            "family_id": program["family_id"],
            "public_mode": program["public_mode"],
            "status": "passed",
            "clean_start": {"before": before, "stopped_rpc": prior,
                            "bitstream_reload": bitstream, "fresh_rpc": fresh},
            "expected_profile_delta_candidate_minus_stock": expected,
            "observed_profile_delta_candidate_minus_stock": observed,
            "timed_profile_delta_candidate_minus_stock": timed_delta,
            "correctness": correctness,
            "timings": timings,
            "median_latency_ms": medians,
            "median_paired_latency_ratio": statistics.median(ratios),
            "median_paired_delta_ms": statistics.median(deltas),
            "candidate_paired_wins": candidate_wins,
            "paired_rounds": ROUNDS,
        }
    except CandidateRejected as error:
        return {
            "candidate_id": candidate_id,
            "family_id": program["family_id"],
            "public_mode": program["public_mode"],
            "status": "rejected_fail_closed",
            "message": str(error),
            "traceback": traceback.format_exc()[-8000:],
            "clean_start": {"before": before, "stopped_rpc": prior,
                            "bitstream_reload": bitstream, "fresh_rpc": fresh},
            "expected_profile_delta_candidate_minus_stock": expected,
            "observed_profile_delta_candidate_minus_stock": observed,
            "timed_profile_delta_candidate_minus_stock": timed_delta,
            "correctness": correctness,
            "timings": timings,
        }
    finally:
        if remote is not None:
            del remote
            gc.collect()
            time.sleep(0.25)


def finalize(output):
    (output / "command.txt").write_text(
        " ".join(__import__("sys").argv) + "\n", encoding="utf-8"
    )
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})


def run(args):
    outer_started = time.perf_counter()
    pool_dir = Path(args.pool_dir)
    prior_dir = Path(args.prior_front_result) if args.prior_front_result else None
    pool, by_id, candidate_ids, verified, semantic, prior = verify_inputs(
        pool_dir, args.scope, prior_dir
    )
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    pool_manifest_sha256 = sha256(pool_dir / "artifact_hashes.json")
    contract = {
        "schema": "c3_vta_resnet50_final_fused_pareto_board_contract_v1",
        "status": "frozen_before_board_contact",
        "scope": args.scope,
        "collection_status": (
            "prospective_non_dominated_wave" if args.scope == "front"
            else "dominated_oracle_completion_after_front_labels_exposed"
        ),
        "pool_artifact_manifest_sha256": pool_manifest_sha256,
        "prior_front_result_artifact_manifest_sha256": (
            sha256(prior_dir / "artifact_hashes.json") if prior_dir else None
        ),
        "candidate_ids": candidate_ids,
        "primary_latency_metric": "median per-round candidate/stock latency ratio",
        "secondary_latency_metric": "candidate absolute median latency",
        "seeds": list(SEEDS),
        "rounds": ROUNDS,
        "clean_start_per_candidate": True,
        "simultaneous_executors": ["stock_reference", "one_candidate"],
        "failure_policy": (
            "candidate output/profile failure rejects that program before use; "
            "board/RPC/runtime-integrity failure aborts the wave"
        ),
        "verified_artifact_counts": verified,
        "parameter_semantic_sha256": semantic,
        "executor_source_sha256": sha256(Path(__file__).resolve()),
        "board_contacted": False,
        "claim_boundary": (
            "Full {} graph schedule-regression correctness and paired latency; logical "
            "VTA DMA, not physical AXI or task-level accuracy"
        ).format(pool.get("source_model", "ResNet50")),
    }
    write_json(output / "contract.json", contract)
    write_json(output / "pre_observation_hashes.json", {"artifacts": {
        "contract.json": sha256(output / "contract.json")
    }})

    board = CleanStartBoard(args.host, args.ssh_known_hosts)
    initial = board.state()
    assert_board_state(initial)
    boot_id = initial.get("BOOT")
    results = []
    try:
        for index, candidate_id in enumerate(candidate_ids):
            try:
                result = run_one(
                    board, args, output, pool_dir, by_id[candidate_id],
                    pool["stock_fused_features"], index,
                    input_spec=pool.get("input_specification"),
                )
            except CandidateRejected as error:
                result = {
                    "candidate_id": candidate_id,
                    "family_id": by_id[candidate_id]["family_id"],
                    "public_mode": by_id[candidate_id]["public_mode"],
                    "status": "rejected_fail_closed",
                    "message": str(error),
                    "traceback": traceback.format_exc()[-8000:],
                }
            results.append(result)
            write_json(output / "candidate_results.json", results)
            print(
                "board {}/{} {} {}".format(
                    index + 1, len(candidate_ids), candidate_id[:12], result["status"]
                ), flush=True,
            )
        passed = [row for row in results if row["status"] == "passed"]
        if not passed:
            raise RuntimeError("no FPGA-correct program in board wave")
        best = min(
            passed,
            key=lambda row: (row["median_paired_latency_ratio"], row["candidate_id"]),
        )
        after = board.state()
        assert_board_state(after)
        if after.get("BOOT") != boot_id:
            raise RuntimeError("board rebooted during wave")
        summary = {
            "schema": "c3_vta_resnet50_final_fused_pareto_board_v1",
            "status": (
                "prospective_final_fused_pareto_front_board_complete"
                if args.scope == "front" else
                "dominated_final_fused_oracle_completion_board_complete"
            ),
            "scope": args.scope,
            "boot_id": boot_id,
            "pool_artifact_manifest_sha256": pool_manifest_sha256,
            "candidate_ids": candidate_ids,
            "candidate_count": len(candidate_ids),
            "passed_count": len(passed),
            "rejected_count": len(results) - len(passed),
            "best_candidate_id_by_paired_ratio": best["candidate_id"],
            "best_median_paired_latency_ratio": best["median_paired_latency_ratio"],
            "correctness_invocations": sum(len(row.get("correctness", [])) for row in results),
            "timing_invocations": sum(len(row.get("timings", [])) for row in results),
            "candidate_results": results,
            "board_after": after,
            "full_pool_oracle_assembly_ready": args.scope == "remaining",
            "outer_process_seconds_before_summary_write": time.perf_counter() - outer_started,
            "claim_boundary": contract["claim_boundary"],
        }
        write_json(output / "summary.json", summary)
        finalize(output)
        print(json.dumps({
            key: summary[key] for key in (
                "status", "candidate_count", "passed_count", "rejected_count",
                "best_candidate_id_by_paired_ratio", "best_median_paired_latency_ratio",
                "correctness_invocations", "timing_invocations",
            )
        }, indent=2, sort_keys=True))
    except Exception as error:
        write_json(output / "failure.json", {
            "status": "failed_closed", "exception_type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc()[-12000:],
            "completed_candidates": len(results),
            "outer_process_seconds_before_failure_write": time.perf_counter() - outer_started,
        })
        finalize(output)
        raise
    finally:
        state = board.rpc_state()
        if state.get("CWD") != args.default_runtime:
            if state.get("PID") and state.get("CWD"):
                board.stop_rpc(state)
            board.reload_frozen_bitstream()
            board.start_rpc(args.default_runtime, "c3_resnet50_default_rpc_restored.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--scope", choices=("front", "remaining"), required=True)
    parser.add_argument("--prior-front-result")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
