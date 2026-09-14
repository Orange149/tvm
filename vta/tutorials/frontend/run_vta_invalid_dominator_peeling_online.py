#!/usr/bin/env python3
"""Execute frozen invalid-dominator peeling one full graph at a time."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

PROCESS_STARTED = time.perf_counter()

import vta

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from analyze_vta_operator_proxy_pareto_development import operator_proxy_program
from build_vta_resnet50_fused_program_pool import build_stock, dominates, fused_features, pareto_front
from build_vta_resnet50_generic_residency_tir_audit import build_one
from build_vta_resnet50_relay_residency_dispatch_pair import (
    make_relay_testing_resnet_program,
    make_relay_testing_resnet50_program,
)
from build_vta_yolov3_tiny_relay_residency_dispatch_pair import (
    make_relay_program as make_yolov3_tiny_program,
)
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


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def next_wave(programs, invalid_ids, dispatched_ids):
    active = [row for row in programs if row["candidate_id"] not in invalid_ids]
    return sorted(
        (row["candidate_id"] for row in pareto_front(active)
         if row["candidate_id"] not in dispatched_ids),
    )


def initial_wave(front, strategy, candidate_budget=None):
    if strategy == "peeling":
        order = list(front["wave0_candidate_ids"])
    else:
        order = list(front["control_orders"][strategy])
    if candidate_budget is not None:
        if candidate_budget <= 0:
            raise ValueError("candidate budget must be positive")
        order = order[:candidate_budget]
    return order


def finish(output):
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})


def make_source_program(target, env):
    """Rebuild the exact full model bound by a pristine target contract."""
    source = target.get("source_model_implementation", {})
    source_path = source.get("path")
    repository = Path(__file__).resolve().parents[3]
    if source_path == "python/tvm/relay/testing/resnet.py":
        path = repository / source_path
        if source.get("sha256") and sha256(path) != source["sha256"]:
            raise RuntimeError("bound ResNet source hash mismatch")
        num_layers = int(source.get("num_layers", target.get("source_num_layers", 50)))
        if num_layers == 50:
            # Keep the historical code path explicit for frozen ResNet50 contracts.
            relay_program, params = make_relay_testing_resnet50_program(env)
            input_spec = None
        else:
            relay_program, params = make_relay_testing_resnet_program(env, num_layers)
            input_spec = {
                "shape": [env.BATCH, 3, 224, 224],
                "uniform_range": [0.0, 1.0],
                "all_outputs": True,
                "target_mode": "heterogeneous",
            }
        return (
            relay_program,
            params,
            input_spec,
            "relay.testing.resnet.get_workload(num_layers={})".format(num_layers),
        )
    if target.get("source_model", "").startswith("yolov3_tiny"):
        cfg = repository / source_path
        assets = target.get("source_model_assets", {})
        weights = Path(assets["darknet_weights"]["path"])
        darknet_lib = Path(assets["darknet_lib"]["path"])
        for path, expected in (
            (cfg, source.get("sha256")),
            (weights, assets["darknet_weights"].get("sha256")),
            (darknet_lib, assets["darknet_lib"].get("sha256")),
        ):
            if not path.is_file() or (expected and sha256(path) != expected):
                raise RuntimeError("bound YOLO model asset mismatch: " + str(path))
        relay_program, params, shape = make_yolov3_tiny_program(
            env,
            SimpleNamespace(cfg=str(cfg), weights=str(weights), darknet_lib=str(darknet_lib)),
        )
        input_spec = {
            "shape": list(shape),
            "uniform_range": [0.0, 1.0],
            "all_outputs": True,
        }
        return relay_program, params, input_spec, "contract-bound Darknet YOLOv3-tiny import"
    raise ValueError("unsupported source model implementation: " + repr(source_path))


def run(args):
    target_dir = Path(args.target_contract)
    policy_dir = Path(args.policy_contract)
    local_dir = Path(args.local_qualification)
    front_dir = Path(args.front_contract)
    suite_dir = Path(args.suite_contract) if args.suite_contract else None
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    verified = {
        "target": len(verify_artifacts_compatible(target_dir)),
        "policy": len(verify_artifacts_compatible(policy_dir)),
        "local": len(verify_artifacts_compatible(local_dir)),
        "front": len(verify_artifacts_compatible(front_dir)),
    }
    if suite_dir is not None:
        verified["suite"] = len(verify_artifacts_compatible(suite_dir))
    target = read(target_dir / "contract.json")
    policy = read(policy_dir / "policy.json")
    local = read(local_dir / "summary.json")
    front = read(front_dir / "front.json")
    suite = read(suite_dir / "suite.json") if suite_dir is not None else None
    if policy.get("status") != "frozen_before_target_lowering_fsim_fullgraph_fpga_or_latency":
        raise ValueError("policy is not pristine")
    if local.get("status") != "completed_local_no_board" or local.get("board_contacted"):
        raise ValueError("local qualification is not pristine")
    if front.get("status") != "wave0_and_controls_frozen_before_fullgraph_fpga_or_latency":
        raise ValueError("front is not pristine")
    if len({target["workload_id"], policy["workload_id"], front["workload_id"]}) != 1:
        raise ValueError("workload binding mismatch")
    if suite is not None:
        if suite.get("status") != "frozen_before_fullgraph_build_fpga_correctness_or_latency":
            raise ValueError("execution suite is not pristine")
        if suite.get("workload_id") != target["workload_id"]:
            raise ValueError("execution suite workload mismatch")
        expected_run = next(
            (row for row in suite["executions"] if row["strategy"] == args.strategy), None
        )
        if expected_run is None:
            raise ValueError("strategy absent from frozen execution suite")
        if (expected_run["budget_seconds"] != args.budget_seconds
                or expected_run["candidate_budget"] != args.candidate_budget):
            raise ValueError("runtime budget differs from frozen execution suite")
        if suite["executor_source_sha256"] != sha256(Path(__file__).resolve()):
            raise ValueError("online executor differs from frozen execution suite")
        source_dir = Path(__file__).resolve().parent
        bound_sources = {
            "board_adapter_source_sha256": "run_vta_resnet50_fused_program_pareto_board.py",
            "stock_builder_source_sha256": "build_vta_resnet50_fused_program_pool.py",
            "candidate_builder_source_sha256": (
                "build_vta_resnet50_generic_residency_tir_audit.py"
            ),
        }
        for key, name in bound_sources.items():
            if suite[key] != sha256(source_dir / name):
                raise ValueError("frozen execution dependency drift: " + name)
    candidates = {row["candidate_id"]: row for row in rows(local_dir / "candidates_v2.jsonl")}
    static = {row["candidate_id"]: row for row in rows(local_dir / "static_results.jsonl")}
    eligible = [row["candidate_id"] for row in rows(local_dir / "fsim_results.jsonl")
                if row.get("status") == "passed"]
    proxy_programs = [operator_proxy_program(candidates[cid], static[cid]) for cid in eligible]
    if sorted(front["wave0_candidate_ids"]) != next_wave(proxy_programs, set(), set()):
        raise ValueError("frozen wave0 differs from recomputed proxy front")

    output.mkdir(parents=True)
    contract = {
        "schema": "c3_invalid_dominator_peeling_online_contract_v1",
        "status": "frozen_before_fullgraph_build_or_board_contact",
        "workload_id": target["workload_id"],
        "source_layer": target["source_layer"],
        "budget_seconds": args.budget_seconds,
        "execution_strategy": args.strategy,
        "candidate_budget": args.candidate_budget,
        "wave0_candidate_ids": front["wave0_candidate_ids"],
        "peeling_algorithm": policy["peeling_algorithm"],
        "control_orders": front["control_orders"],
        "full_graph_source": target["source_model_implementation"],
        "source_model_assets": target.get("source_model_assets"),
        "bindings": {
            "target_manifest": sha256(target_dir / "artifact_hashes.json"),
            "policy_manifest": sha256(policy_dir / "artifact_hashes.json"),
            "local_manifest": sha256(local_dir / "artifact_hashes.json"),
            "front_manifest": sha256(front_dir / "artifact_hashes.json"),
            "suite_manifest": (
                sha256(suite_dir / "artifact_hashes.json") if suite_dir is not None else None
            ),
        },
        "verified_artifact_counts": verified,
        "executor_source_sha256": sha256(Path(__file__).resolve()),
        "board_contacted": False,
        "claim_boundary": (
            "Prospective online execution only. Pool oracle and undispatched correctness/latency "
            "remain unavailable; deterministic model inputs establish schedule equivalence, not "
            "ImageNet accuracy or COCO mAP."
        ),
    }
    write_json(output / "contract.json", contract)
    write_json(output / "pre_observation_hashes.json", {
        "artifacts": {"contract.json": sha256(output / "contract.json")}
    })

    phases = []
    model_started = time.perf_counter()
    env = vta.get_env()
    relay_program, params, input_spec, source_description = make_source_program(target, env)
    contract["full_graph_source_description"] = source_description
    contract["input_specification"] = input_spec
    write_json(output / "contract.json", contract)
    write_json(output / "pre_observation_hashes.json", {
        "artifacts": {"contract.json": sha256(output / "contract.json")}
    })
    phases.append({"phase": "model_prepare", "seconds": time.perf_counter() - model_started})
    stock_started = time.perf_counter()
    target_mode = "ext_dev_only" if input_spec and input_spec.get("all_outputs") else "heterogeneous"
    contract["relay_build_target_mode"] = target_mode
    write_json(output / "contract.json", contract)
    write_json(output / "pre_observation_hashes.json", {
        "artifacts": {"contract.json": sha256(output / "contract.json")}
    })
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
    invalid_ids, dispatched_ids = set(), set()
    wave_ids = initial_wave(front, args.strategy, args.candidate_budget)
    wave_rows, all_results, built_programs = [], [], []
    stop_reason = None
    try:
        wave_index = 0
        while wave_ids:
            current_wave = {
                "wave": wave_index,
                "candidate_ids": list(wave_ids),
                "started_seconds": time.perf_counter() - PROCESS_STARTED,
                "results": [],
            }
            wave_complete = True
            for candidate_id in wave_ids:
                if time.perf_counter() - PROCESS_STARTED >= args.budget_seconds:
                    wave_complete = False
                    stop_reason = "budget_exhausted_before_next_candidate"
                    break
                row = candidates[candidate_id]
                root = output / candidate_id
                root.mkdir()
                build_started = time.perf_counter()
                built = build_one(
                    row, relay_program, params, root, env, target_mode=target_mode
                )
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
                board_started = time.perf_counter()
                result = run_one(
                    board, args, output, output, program, stock_features, len(all_results),
                    input_spec=input_spec,
                )
                result["complete_board_action_seconds"] = time.perf_counter() - board_started
                result["outer_elapsed_seconds"] = time.perf_counter() - PROCESS_STARTED
                all_results.append(result)
                current_wave["results"].append({
                    "candidate_id": candidate_id,
                    "status": result["status"],
                    "outer_elapsed_seconds": result["outer_elapsed_seconds"],
                })
                dispatched_ids.add(candidate_id)
                if result["status"] != "passed":
                    invalid_ids.add(candidate_id)
                write_json(output / "online_results.json", all_results)
                print("online wave={} {} {} elapsed={:.3f}s".format(
                    wave_index, candidate_id[:12], result["status"],
                    result["outer_elapsed_seconds"],
                ), flush=True)
                gc.collect()
            current_wave["complete"] = wave_complete
            current_wave["finished_seconds"] = time.perf_counter() - PROCESS_STARTED
            wave_rows.append(current_wave)
            write_json(output / "waves.json", wave_rows)
            if not wave_complete:
                break
            if args.strategy != "peeling":
                stop_reason = "completed_frozen_control_prefix"
                break
            invalid_in_wave = [cid for cid in wave_ids if cid in invalid_ids]
            if not invalid_in_wave:
                stop_reason = "completed_wave_had_no_fpga_invalid_candidate"
                break
            new_wave = next_wave(proxy_programs, invalid_ids, dispatched_ids)
            if not new_wave:
                stop_reason = "no_new_identity_exposed_after_invalid_removal"
                break
            wave_ids = new_wave
            wave_index += 1

        after = board.state()
        assert_board_state(after)
        if after["BOOT"] != boot_id:
            raise RuntimeError("board rebooted during online execution")
        passed = [row for row in all_results if row["status"] == "passed"]
        summary = {
            "schema": "c3_invalid_dominator_peeling_online_v1",
            "status": "online_peeling_complete" if args.strategy == "peeling"
            else "online_frozen_control_prefix_complete",
            "workload_id": target["workload_id"],
            "execution_strategy": args.strategy,
            "candidate_budget": args.candidate_budget,
            "boot_id": boot_id,
            "stop_reason": stop_reason,
            "waves": wave_rows,
            "candidate_results": all_results,
            "built_programs": built_programs,
            "dispatched_candidate_ids": [row["candidate_id"] for row in all_results],
            "invalid_candidate_ids": sorted(invalid_ids),
            "passed_candidate_ids": [row["candidate_id"] for row in passed],
            "dispatch_count": len(all_results),
            "correctness_invocations": sum(len(row.get("correctness", [])) for row in all_results),
            "timing_invocations": sum(len(row.get("timings", [])) for row in all_results),
            "phases": phases,
            "outer_process_seconds_before_summary_write": time.perf_counter() - PROCESS_STARTED,
            "budget_seconds": args.budget_seconds,
            "budget_overrun_charged": max(
                time.perf_counter() - PROCESS_STARTED - args.budget_seconds, 0.0
            ),
            "stock_reference": stock,
            "stock_fused_features": stock_features,
            "board_after": after,
            "claim_boundary": contract["claim_boundary"],
        }
        write_json(output / "summary.json", summary)
        (output / "command.txt").write_text(" ".join(sys.argv) + "\n")
        finish(output)
        print(json.dumps({key: summary[key] for key in (
            "status", "stop_reason", "dispatch_count", "invalid_candidate_ids",
            "passed_candidate_ids", "outer_process_seconds_before_summary_write",
            "budget_overrun_charged",
        )}, indent=2, sort_keys=True))
    except Exception as error:
        write_json(output / "failure.json", {
            "status": "failed_closed", "type": type(error).__name__, "message": str(error),
            "outer_process_seconds_before_failure_write": time.perf_counter() - PROCESS_STARTED,
            "completed_candidate_count": len(all_results),
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
            board.start_rpc(args.default_runtime, "c3_invalid_peeling_default_restored.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-contract", required=True)
    parser.add_argument("--policy-contract", required=True)
    parser.add_argument("--local-qualification", required=True)
    parser.add_argument("--front-contract", required=True)
    parser.add_argument(
        "--suite-contract",
        help="Optional immutable multi-strategy execution contract frozen before board labels.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-seconds", type=float, default=600.0)
    parser.add_argument(
        "--strategy",
        choices=("peeling", "bytes_lazy", "calls_lazy", "random_lazy", "fixed_front"),
        default="peeling",
        help="Execute peeling or an immutable control order stored in the front contract.",
    )
    parser.add_argument(
        "--candidate-budget",
        type=int,
        help="Optional frozen dispatch-count cap; wall-clock admission remains controlled separately.",
    )
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
