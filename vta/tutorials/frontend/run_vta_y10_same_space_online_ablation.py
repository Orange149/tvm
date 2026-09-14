#!/usr/bin/env python3
"""Run a label-blind same-candidate-space Y10 online selector ablation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import xgboost as xgb
import vta

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from build_vta_resnet50_fused_program_pool import build_stock, fused_features
from build_vta_resnet50_generic_residency_tir_audit import build_one
from run_vta_invalid_dominator_peeling_online import make_source_program, rows
from run_vta_resnet50_fused_program_pareto_board import (
    CleanStartBoard,
    assert_board_state,
    run_one,
)
from run_vta_resnet50_relay_residency_dispatch_pair_board import (
    semantic_params_hash,
    sha256,
    write_json,
)


FEATURE_NAMES = (
    "tile_b",
    "tile_h",
    "tile_w",
    "tile_ci",
    "tile_co",
    "oc_nthread",
    "h_nthread",
    "mode_original",
    "mode_input_stationary",
    "mode_weight_resident_barrier",
)
MODES = ("original", "input_stationary", "weight_resident_barrier")
WARMUP_COUNT = 4


def read(path):
    return json.loads(Path(path).read_text())


def feature_vector(row):
    knobs = row["knobs"]
    values = [float(knobs[name]) for name in FEATURE_NAMES[:7]]
    values.extend(float(row["public_mode"] == mode) for mode in MODES)
    return values


def make_model(seed):
    return xgb.XGBRegressor(
        n_estimators=64,
        max_depth=3,
        learning_rate=0.1,
        min_child_weight=1,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def hashed_warmup(candidate_ids, seed, workload_id):
    return sorted(
        candidate_ids,
        key=lambda candidate_id: hashlib.sha256(
            f"{seed}:{workload_id}:{candidate_id}".encode()
        ).hexdigest(),
    )[:WARMUP_COUNT]


def finish(output):
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})


def select_next(strategy, remaining, candidates, dma_order, warmup, observations, seed):
    if strategy == "dma_prior":
        candidate_id = next(candidate_id for candidate_id in dma_order if candidate_id in remaining)
        return candidate_id, {
            "selection": "frozen_static_total_dma_order",
            "predicted_paired_ratio": None,
        }
    dispatch_index = len(observations)
    if dispatch_index < WARMUP_COUNT:
        candidate_id = warmup[dispatch_index]
        return candidate_id, {
            "selection": "deterministic_hash_warmup",
            "predicted_paired_ratio": None,
        }
    train = [row for row in observations if row["status"] == "passed"]
    if len(train) < WARMUP_COUNT:
        candidate_id = sorted(remaining)[0]
        return candidate_id, {
            "selection": "fail_closed_lexical_fallback_insufficient_labels",
            "predicted_paired_ratio": None,
        }
    estimator = make_model(seed)
    estimator.fit(
        np.asarray([feature_vector(candidates[row["candidate_id"]]) for row in train]),
        np.asarray([row["median_paired_latency_ratio"] for row in train]),
    )
    choices = sorted(remaining)
    predictions = estimator.predict(
        np.asarray([feature_vector(candidates[candidate_id]) for candidate_id in choices])
    )
    predicted, candidate_id = min(zip(predictions.tolist(), choices), key=lambda item: (item[0], item[1]))
    return candidate_id, {
        "selection": "refit_mode_aware_xgb_min_prediction",
        "predicted_paired_ratio": float(predicted),
    }


def run(args):
    started = time.perf_counter()
    target_dir = Path(args.target_contract).resolve()
    local_dir = Path(args.local_qualification).resolve()
    front_dir = Path(args.front_contract).resolve()
    protocol_path = Path(args.protocol).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.candidate_budget != 6:
        raise ValueError("the frozen attribution protocol requires candidate_budget=6")
    if args.seed not in (44501, 44502, 44503):
        raise ValueError("seed is outside the frozen attribution protocol")

    verified = {
        "target": len(verify_artifacts_compatible(target_dir)),
        "local": len(verify_artifacts_compatible(local_dir)),
        "front": len(verify_artifacts_compatible(front_dir)),
    }
    target = read(target_dir / "contract.json")
    local = read(local_dir / "summary.json")
    front = read(front_dir / "front.json")
    if target["workload_id"] != "Y10" or front["workload_id"] != "Y10":
        raise ValueError("this protocol is bound to Y10")
    if local.get("status") != "completed_local_no_board" or local.get("board_contacted"):
        raise ValueError("local qualification contract is not board-isolated")
    candidates = {row["candidate_id"]: row for row in rows(local_dir / "candidates_v2.jsonl")}
    eligible = sorted(
        row["candidate_id"] for row in rows(local_dir / "fsim_results.jsonl")
        if row.get("status") == "passed"
    )
    if len(eligible) != 12:
        raise ValueError(f"expected exact 12-candidate eligible pool, got {len(eligible)}")
    dma_order = front["control_orders"]["bytes_lazy"]
    if sorted(dma_order) != eligible:
        raise ValueError("frozen DMA order does not cover the exact eligible pool")
    warmup = hashed_warmup(eligible, args.seed, "Y10")

    output.mkdir(parents=True)
    contract = {
        "schema": "c3_y10_same_space_online_ablation_contract_v1",
        "status": "frozen_before_this_independent_reexecution",
        "workload_id": "Y10",
        "strategy": args.strategy,
        "seed": args.seed,
        "candidate_budget": args.candidate_budget,
        "candidate_ids": eligible,
        "dma_order": dma_order,
        "xgb_warmup_candidate_ids": warmup,
        "xgb_feature_names": FEATURE_NAMES,
        "xgb_version": xgb.__version__,
        "xgb_label": "median per-round candidate/stock full-graph latency ratio",
        "bindings": {
            "target_manifest": sha256(target_dir / "artifact_hashes.json"),
            "local_manifest": sha256(local_dir / "artifact_hashes.json"),
            "front_manifest": sha256(front_dir / "artifact_hashes.json"),
            "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
        },
        "verified_artifact_counts": verified,
        "executor_source_sha256": sha256(Path(__file__).resolve()),
        "target_labels_previously_exist": True,
        "target_label_files_read_by_selector": False,
        "claim_boundary": (
            "Independent full-graph cost reexecution after Y10 labels already existed. The selector "
            "does not read those files, but this is an attribution ablation, not a prospective holdout. "
            "Logical VTA counters are not physical AXI traffic, and deterministic random inputs do not "
            "establish COCO mAP."
        ),
        "board_contacted": False,
    }
    write_json(output / "contract.json", contract)
    write_json(output / "pre_observation_hashes.json", {
        "artifacts": {"contract.json": sha256(output / "contract.json")}
    })

    phases = []
    model_started = time.perf_counter()
    env = vta.get_env()
    relay_program, params, input_spec, source_description = make_source_program(target, env)
    phases.append({"phase": "model_prepare", "seconds": time.perf_counter() - model_started})
    stock_started = time.perf_counter()
    target_mode = "ext_dev_only" if input_spec and input_spec.get("all_outputs") else "heterogeneous"
    stock = build_stock(relay_program, params, output, env, target_mode=target_mode)
    stock_graph = read(output / "stock_reference" / "graph.json")
    stock_features = fused_features(stock, stock_graph, target["workload"])
    stock.update(fused_features=stock_features)
    write_json(output / "stock_reference" / "build.json", stock)
    phases.append({"phase": "stock_build", "seconds": time.perf_counter() - stock_started})

    board = CleanStartBoard(args.host, args.ssh_known_hosts)
    initial = board.state()
    assert_board_state(initial)
    boot_id = initial["BOOT"]
    remaining = set(eligible)
    observations = []
    built_programs = []
    decisions = []
    try:
        for dispatch_index in range(args.candidate_budget):
            candidate_id, decision = select_next(
                args.strategy, remaining, candidates, dma_order, warmup, observations, args.seed
            )
            decision.update({
                "dispatch_index": dispatch_index,
                "candidate_id": candidate_id,
                "observed_label_count_before_selection": sum(
                    row["status"] == "passed" for row in observations
                ),
                "selection_elapsed_seconds": time.perf_counter() - started,
            })
            decisions.append(decision)
            write_json(output / "selection_decisions.json", decisions)
            remaining.remove(candidate_id)

            row = candidates[candidate_id]
            root = output / candidate_id
            root.mkdir()
            build_started = time.perf_counter()
            built = build_one(row, relay_program, params, root, env, target_mode=target_mode)
            local_build = root / row["public_mode"]
            graph = read(local_build / "graph.json")
            if graph != stock_graph:
                raise RuntimeError("candidate Graph JSON differs from stock")
            if semantic_params_hash(local_build / "params.bin") != semantic_params_hash(
                output / "stock_reference" / "params.bin"
            ):
                raise RuntimeError("candidate parameter semantics differ from stock")
            features = fused_features(built, graph, target["workload"])
            program = {
                "program_id": candidate_id,
                "candidate_id": candidate_id,
                "family_id": row["family_id"],
                "public_mode": row["public_mode"],
                "knobs": row["knobs"],
                "relative_dir": candidate_id + "/" + row["public_mode"],
                "build_seconds": built["build_seconds"],
                "complete_candidate_build_action_seconds": time.perf_counter() - build_started,
                "dispatch_audit": built["dispatch_audit"],
                "graph_sha256": built["graph_sha256"],
                "params_sha256": built["params_sha256"],
                "graphlib_sha256": built["graphlib_sha256"],
                **features,
            }
            write_json(local_build / "program.json", program)
            built_programs.append(program)
            result = run_one(
                board, args, output, output, program, stock_features, dispatch_index,
                input_spec=input_spec,
            )
            result["outer_elapsed_seconds"] = time.perf_counter() - started
            observations.append(result)
            write_json(output / "online_results.json", observations)
            print(
                "same-space {} seed={} dispatch={} {} {} elapsed={:.3f}s".format(
                    args.strategy, args.seed, dispatch_index + 1, candidate_id[:12],
                    result["status"], result["outer_elapsed_seconds"],
                ),
                flush=True,
            )
            gc.collect()

        after = board.state()
        assert_board_state(after)
        if after["BOOT"] != boot_id:
            raise RuntimeError("board rebooted during same-space online ablation")
        passed = [row for row in observations if row["status"] == "passed"]
        best = min(
            passed,
            key=lambda row: (row["median_paired_latency_ratio"], row["candidate_id"]),
        ) if passed else None
        summary = {
            "schema": "c3_y10_same_space_online_ablation_v1",
            "status": "complete",
            "workload_id": "Y10",
            "strategy": args.strategy,
            "seed": args.seed,
            "candidate_budget": args.candidate_budget,
            "boot_id": boot_id,
            "selection_decisions": decisions,
            "candidate_results": observations,
            "built_programs": built_programs,
            "dispatched_candidate_ids": [row["candidate_id"] for row in observations],
            "passed_candidate_ids": [row["candidate_id"] for row in passed],
            "rejected_candidate_ids": [
                row["candidate_id"] for row in observations if row["status"] != "passed"
            ],
            "best_candidate_id": best["candidate_id"] if best else None,
            "best_median_paired_latency_ratio": (
                best["median_paired_latency_ratio"] if best else None
            ),
            "dispatch_count": len(observations),
            "correctness_invocations": sum(len(row.get("correctness", [])) for row in observations),
            "timing_invocations": sum(len(row.get("timings", [])) for row in observations),
            "phases": phases,
            "outer_process_seconds_before_summary_write": time.perf_counter() - started,
            "stock_reference": stock,
            "stock_fused_features": stock_features,
            "board_after": after,
            "claim_boundary": contract["claim_boundary"],
        }
        write_json(output / "summary.json", summary)
        (output / "command.txt").write_text(" ".join(sys.argv) + "\n")
        finish(output)
        print(json.dumps({key: summary[key] for key in (
            "status", "strategy", "seed", "dispatch_count", "best_candidate_id",
            "best_median_paired_latency_ratio", "outer_process_seconds_before_summary_write",
        )}, indent=2, sort_keys=True))
    except Exception as error:
        write_json(output / "failure.json", {
            "status": "failed_closed",
            "type": type(error).__name__,
            "message": str(error),
            "completed_candidate_count": len(observations),
            "outer_process_seconds_before_failure_write": time.perf_counter() - started,
        })
        (output / "command.txt").write_text(" ".join(sys.argv) + "\n")
        finish(output)
        raise
    finally:
        state = board.rpc_state()
        if state.get("CWD") != args.default_runtime:
            if state.get("PID") and state.get("CWD"):
                board.stop_rpc(state)
            board.reload_frozen_bitstream()
            board.start_rpc(args.default_runtime, "c3_y10_same_space_default_restored.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-contract", required=True)
    parser.add_argument("--local-qualification", required=True)
    parser.add_argument("--front-contract", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strategy", choices=("dma_prior", "mode_aware_xgb"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate-budget", type=int, default=6)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
