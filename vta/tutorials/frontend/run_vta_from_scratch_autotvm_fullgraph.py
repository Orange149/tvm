#!/usr/bin/env python3
"""Run empty-history AutoTVM-XGB through final full-graph FPGA validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import traceback
from types import SimpleNamespace

import numpy as np
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner
import vta

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from build_vta_resnet50_fused_program_pool import build_stock, fused_features
from build_vta_resnet50_generic_residency_tir_audit import build_one
from run_vta_invalid_dominator_peeling_online import make_source_program
from run_vta_p7r132_y00_search_confirmation import (
    CleanStartBoard,
    EXPECTED_DEFAULT_RUNTIME_HASHES,
)
from run_vta_resnet50_fused_program_pareto_board import assert_board_state, run_one
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json
from tune_resnet18_vta import register_vta_conv2d_template, task_from_entry
from vta_autotvm_measure import VTASequentialBuilder, VTADirectRunner


SEEDS = (0, 20250901, 20260910)
PROCESS_STARTED = time.perf_counter()
PROCESS_STARTED_WALL = time.time()


class WallBudgetReached(RuntimeError):
    """Intentional stop after the current atomic measurement finishes."""


def timeline_callback(path, t0, budget_seconds):
    trial = {"count": 0}

    def callback(_, inputs, results):
        now = time.perf_counter()
        with Path(path).open("a", encoding="utf-8") as stream:
            for inp, result in zip(inputs, results):
                trial["count"] += 1
                costs = []
                if result.error_no == 0:
                    costs = [float(value) for value in result.costs]
                stream.write(json.dumps({
                    "trial": trial["count"],
                    "outer_elapsed_seconds": now - t0,
                    "config_index": int(inp.config.index),
                    "config": inp.config.to_json_dict(),
                    "error_no": int(result.error_no),
                    "costs_seconds": costs,
                    "all_cost": float(result.all_cost),
                    "timestamp": float(result.timestamp),
                }, sort_keys=True) + "\n")
        if now - t0 >= budget_seconds:
            raise WallBudgetReached(
                "wall budget reached after atomic trial {}".format(trial["count"])
            )

    return callback


def best_record(path):
    successful = []
    for inp, result in autotvm.record.load_from_file(str(path)):
        if result.error_no != 0 or not result.costs:
            continue
        costs = [float(value) for value in result.costs]
        if not all(np.isfinite(value) and value > 0 for value in costs):
            continue
        successful.append((float(np.mean(costs)), inp, result))
    if not successful:
        raise RuntimeError("empty-history AutoTVM produced no correct configuration")
    return min(successful, key=lambda item: item[0]), len(successful)


def selected_route(target, config):
    complete = config.to_json_dict()
    complete.pop("index", None)
    identity = {
        "schema": "c3_from_scratch_autotvm_selected_route_v1",
        "workload": target["workload"],
        "complete_config_entity": complete,
        "public_mode": "original",
        "implementation_mode": 0,
    }
    candidate_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "candidate_id": candidate_id,
        "family_id": "AUTOTVM-XGB-SELECTED",
        "public_mode": "original",
        "implementation_mode": 0,
        "workload": target["workload"],
        "complete_config_entity": complete,
        "identity": identity,
    }


def candidate_clean_start(board, default_runtime):
    prior = board.rpc_state()
    if prior.get("PID"):
        board.stop_rpc(prior)
    bitstream = board.reload_frozen_bitstream()
    fresh = board.start_rpc(default_runtime, "c3_p7r445_autotvm_candidate.log")
    return {"prior": prior, "bitstream_reload": bitstream, "fresh_rpc": fresh}


def finish(output):
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    target_dir = Path(args.target_contract)
    verify_artifacts_compatible(target_dir)
    target = json.loads((target_dir / "contract.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n")
    contract = {
        "schema": "c3_from_scratch_autotvm_fullgraph_contract_v1",
        "status": "frozen_before_t0",
        "strategy": "stock_autotvm_xgb_empty_history",
        "workload_id": target["workload_id"],
        "target_manifest_sha256": sha256(target_dir / "artifact_hashes.json"),
        "trial_cap": args.trials,
        "wall_budget_seconds": args.budget_seconds,
        "xgb_seed": args.seed,
        "correctness_seeds": list(SEEDS),
        "tophub_or_history_used_by_search": False,
        "final_graph_other_layers_may_use_tophub": True,
        "board_contacted": False,
        "w0_definition": "host process starts before board preflight/bitstream/RPC preparation",
        "w0_wall_unix": PROCESS_STARTED_WALL,
    }
    write_json(output / "contract.json", contract)
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "results.jsonl").write_text("", encoding="utf-8")

    board = CleanStartBoard(args.host, args.ssh_known_hosts)
    before = board.state()
    assert_board_state(before)
    if board.runtime_hashes(args.default_runtime) != EXPECTED_DEFAULT_RUNTIME_HASHES:
        raise RuntimeError("default runtime hash mismatch")
    prior = board.rpc_state()
    if not prior.get("PID") or prior.get("CWD") != args.default_runtime:
        raise RuntimeError("healthy default RPC is required")
    board.stop_rpc(prior)
    bitstream = board.reload_frozen_bitstream()
    fresh = board.start_rpc(args.default_runtime, "c3_p7r445_autotvm_xgb.log")
    t0 = time.perf_counter()
    contract.update(board_contacted=True, t0_definition=(
        "fresh bitstream/default tmpfs RPC ready; task construction begins"
    ), clean_start={"before": before, "stopped_rpc": prior,
                    "bitstream_reload": bitstream, "fresh_rpc": fresh})
    write_json(output / "contract.json", contract)

    phases = []
    started = time.perf_counter()
    register_vta_conv2d_template()
    env = vta.get_env()
    task = task_from_entry(
        {"workload": target["workload"], "residence_mode": "original_template"},
        env.target, env.target_host,
    )
    phases.append({"phase": "task_and_config_space", "seconds": time.perf_counter() - started})
    if len(task.config_space) != target["complete_original_domain_count"]:
        raise RuntimeError("ConfigSpace differs from frozen target contract")

    # ConfigSpace uses Python's random.randrange while the XGB model and SA
    # optimizer use NumPy.  Seed both or the advertised clean-start seed is not
    # reproducible.
    random.seed(args.seed)
    np.random.seed(args.seed)
    raw_log = output / "autotvm_xgb_raw.log"
    artifacts = output / "isolated_measurements"
    builder = VTASequentialBuilder(artifacts, verbose_errors=False)
    runner = VTADirectRunner(
        args.host, args.port, number=args.measure_number, repeat=args.measure_repeat,
        timeout=args.session_timeout, artifact_dir=artifacts, correctness_seeds=SEEDS,
        before_measure=lambda: candidate_clean_start(board, args.default_runtime),
        verbose_errors=False,
    )
    tuner = XGBTuner(task, loss_type="reg")
    tune_started = time.perf_counter()
    stop_reason = "trial_cap"
    try:
        tuner.tune(
            n_trial=min(args.trials, len(task.config_space)),
            measure_option=autotvm.measure_option(builder=builder, runner=runner),
            callbacks=[
                autotvm.callback.log_to_file(str(raw_log)),
                timeline_callback(output / "timeline.jsonl", t0, args.budget_seconds),
            ],
        )
    except WallBudgetReached as error:
        stop_reason = str(error)
    finally:
        # Tuner.tune resets this only on its normal return path.  A deliberate
        # wall-budget callback exception must not leak tuning mode into the
        # subsequent Relay full-graph build.
        autotvm.GLOBAL_SCOPE.in_tuning = False
        tuner.cost_model._close_pool()  # pylint: disable=protected-access
    phases.append({"phase": "empty_history_autotvm_tune", "seconds": time.perf_counter() - tune_started})

    (best_cost, best_input, _), successful_count = best_record(raw_log)
    best_log = output / "autotvm_xgb_best.log"
    autotvm.record.pick_best(str(raw_log), str(best_log))
    route = selected_route(target, best_input.config)
    write_json(output / "selected_route.json", route)

    model_started = time.perf_counter()
    relay_program, params, input_spec, source_description = make_source_program(target, env)
    target_mode = (
        input_spec.get("target_mode", "ext_dev_only")
        if input_spec else "ext_dev_only"
    )
    phases.append({"phase": "final_graph_model_prepare", "seconds": time.perf_counter() - model_started})
    stock_started = time.perf_counter()
    stock = build_stock(relay_program, params, output, env, target_mode=target_mode)
    stock_graph = json.loads((output / "stock_reference" / "graph.json").read_text())
    stock_features = fused_features(stock, stock_graph, target["workload"])
    stock["fused_features"] = stock_features
    write_json(output / "stock_reference" / "build.json", stock)
    phases.append({"phase": "final_graph_stock_build", "seconds": time.perf_counter() - stock_started})

    candidate_root = output / route["candidate_id"]
    candidate_root.mkdir()
    candidate_started = time.perf_counter()
    built = build_one(route, relay_program, params, candidate_root, env, target_mode=target_mode)
    candidate_graph = json.loads(
        (candidate_root / "original" / "graph.json").read_text(encoding="utf-8")
    )
    if candidate_graph != stock_graph:
        raise RuntimeError("AutoTVM-selected graph structure differs from stock")
    features = fused_features(built, candidate_graph, target["workload"])
    program = {
        **features,
        "candidate_id": route["candidate_id"],
        "family_id": route["family_id"],
        "public_mode": "original",
        "relative_dir": route["candidate_id"] + "/original",
    }
    write_json(candidate_root / "program.json", program)
    phases.append({"phase": "final_graph_selected_build", "seconds": time.perf_counter() - candidate_started})

    board_started = time.perf_counter()
    candidate_clean_start(board, args.default_runtime)
    board_args = SimpleNamespace(
        host=args.host, port=args.port, session_timeout=args.session_timeout,
        default_runtime=args.default_runtime,
    )
    graph_result = run_one(
        board, board_args, output, output, program, stock_features, 0, input_spec=input_spec,
    )
    phases.append({"phase": "final_graph_correctness_and_timing", "seconds": time.perf_counter() - board_started})
    write_json(output / "full_graph_result.json", graph_result)

    timeline = [json.loads(line) for line in (output / "timeline.jsonl").read_text().splitlines()]
    summary = {
        "schema": "c3_from_scratch_autotvm_fullgraph_result_v1",
        "status": (
            "completed_t0_to_t1" if graph_result.get("status") == "passed"
            else "selected_graph_rejected_fail_closed"
        ),
        "workload_id": target["workload_id"],
        "source_description": source_description,
        "relay_build_target_mode": target_mode,
        "config_space_size": len(task.config_space),
        "gross_proposals": len(timeline),
        "successful_isolated_measurements": successful_count,
        "stop_reason": stop_reason,
        "best_isolated_mean_latency_ms": best_cost * 1000.0,
        "selected_config_index": int(best_input.config.index),
        "selected_candidate_id": route["candidate_id"],
        "phases": phases,
        "t0_to_t1_seconds": time.perf_counter() - t0,
        "w0_to_t1_seconds": time.perf_counter() - PROCESS_STARTED,
        "full_graph_result": graph_result,
        "search_used_existing_history_or_tophub": False,
        "claim_boundary": (
            "One empty-history XGB clean-start session. Other graph layers use the normal "
            "deployment context; three XGB seeds and a matched proposed-method run remain "
            "necessary for the final comparative claim."
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "results.jsonl").write_text(
        "".join(
            json.dumps({"record_type": "autotvm_trial", **row}, sort_keys=True)
            + "\n"
            for row in timeline
        ),
        encoding="utf-8",
    )
    finish(output)
    print(json.dumps({
        "status": summary["status"],
        "gross_proposals": summary["gross_proposals"],
        "successful": successful_count,
        "selected_config_index": summary["selected_config_index"],
        "t0_to_t1_seconds": summary["t0_to_t1_seconds"],
        "full_graph_medians": graph_result.get("median_latency_ms"),
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--budget-seconds", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--measure-number", type=int, default=1)
    parser.add_argument("--measure-repeat", type=int, default=3)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output_preexisted = output.exists()
    try:
        run(args)
    except Exception as error:
        if output.is_dir() and not output_preexisted:
            elapsed = time.perf_counter() - PROCESS_STARTED
            invalid = {
                "schema": "c3_from_scratch_autotvm_fullgraph_failure_v1",
                "status": "invalid_entire_session_fail_closed",
                "exception_type": type(error).__name__,
                "message": (str(error) or type(error).__name__)[:4000],
                "traceback": traceback.format_exc()[-12000:],
                "w0_elapsed_seconds": elapsed,
                "may_splice_with_other_session": False,
            }
            write_json(
                output / "invalid_session.json",
                invalid,
            )
            write_json(
                output / "summary.json",
                {
                    "schema": "c3_from_scratch_autotvm_fullgraph_result_v1",
                    "status": invalid["status"],
                    "session_spliced": False,
                    "w0_to_failure_seconds": elapsed,
                    "failure": {
                        "exception_type": invalid["exception_type"],
                        "message": invalid["message"],
                    },
                },
            )
            timeline_path = output / "timeline.jsonl"
            if not timeline_path.exists():
                timeline_path.write_text("", encoding="utf-8")
            with timeline_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "phase": "session_invalidated",
                            "w0_elapsed_seconds": elapsed,
                            "exception_type": invalid["exception_type"],
                            "message": invalid["message"],
                            "may_splice_with_other_session": False,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            results_path = output / "results.jsonl"
            if not results_path.exists():
                results_path.write_text("", encoding="utf-8")
            finish(output)
        raise


if __name__ == "__main__":
    main()
