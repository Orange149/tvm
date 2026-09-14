#!/usr/bin/env python3
"""Build the candidates not dispatched by a completed online peeling run.

This is deliberately a post-selection oracle-completion step.  The immutable
online result is copied, never re-ranked, and the remaining candidates are
used only to audit the quality and cost of the already-finished decision.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import vta

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from build_vta_resnet50_fused_program_pool import fused_features
from build_vta_resnet50_generic_residency_tir_audit import build_one
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json
from run_vta_invalid_dominator_peeling_online import make_source_program


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run(args):
    target_dir = Path(args.target_contract)
    policy_dir = Path(args.policy_contract)
    local_dir = Path(args.local_qualification)
    front_dir = Path(args.front_contract)
    online_dir = Path(args.online_result)
    control_dirs = [Path(path) for path in args.additional_observed_result]
    verified = {
        "target_contract": len(verify_artifacts_compatible(target_dir)),
        "policy_contract": len(verify_artifacts_compatible(policy_dir)),
        "local_qualification": len(verify_artifacts_compatible(local_dir)),
        "front_contract": len(verify_artifacts_compatible(front_dir)),
        "online_result": len(verify_artifacts_compatible(online_dir)),
    }
    verified["additional_observed_results"] = [
        len(verify_artifacts_compatible(path)) for path in control_dirs
    ]
    target = read_json(target_dir / "contract.json")
    local = read_json(local_dir / "summary.json")
    front_path = front_dir / (
        "front.json" if (front_dir / "front.json").is_file() else "contract.json"
    )
    front = read_json(front_path)
    online = read_json(online_dir / "summary.json")
    controls = [read_json(path / "summary.json") for path in control_dirs]
    if local.get("status") != "completed_local_no_board":
        raise RuntimeError("local qualification is incomplete")
    if front.get("status") != "wave0_and_controls_frozen_before_fullgraph_fpga_or_latency":
        raise RuntimeError("peeling front contract mismatch")
    if online.get("status") != "online_peeling_complete":
        raise RuntimeError("online peeling result is incomplete")
    if online.get("workload_id") != target.get("workload_id"):
        raise RuntimeError("online result binds a different workload")
    if any(row.get("workload_id") != target.get("workload_id") for row in controls):
        raise RuntimeError("additional observed result binds a different workload")

    candidates = {
        row["candidate_id"]: row
        for row in read_jsonl(local_dir / "candidates_v2.jsonl")
    }
    eligible_ids = [
        row["candidate_id"]
        for row in read_jsonl(local_dir / "fsim_results.jsonl")
        if row.get("status") == "passed"
    ]
    dispatched_ids = list(online["dispatched_candidate_ids"])
    if len(dispatched_ids) != len(set(dispatched_ids)):
        raise RuntimeError("online dispatch contains duplicate candidates")
    if not set(dispatched_ids).issubset(eligible_ids):
        raise RuntimeError("online dispatch escaped the frozen eligible set")
    online_programs = {row["candidate_id"]: row for row in online["built_programs"]}
    if set(online_programs) != set(dispatched_ids):
        raise RuntimeError("online builds differ from dispatched candidates")
    remaining_ids = [cid for cid in eligible_ids if cid not in set(dispatched_ids)]

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    copy_started = time.perf_counter()
    shutil.copytree(online_dir / "stock_reference", output / "stock_reference")
    for cid in dispatched_ids:
        shutil.copytree(online_dir / cid, output / cid)
    copy_seconds = time.perf_counter() - copy_started

    env = vta.get_env()
    relay_program, params, input_spec, source_description = make_source_program(target, env)
    target_mode = "ext_dev_only" if input_spec and input_spec.get("all_outputs") else "heterogeneous"
    stock = read_json(output / "stock_reference" / "build.json")
    stock_graph = read_json(output / "stock_reference" / "graph.json")
    stock_features = online["stock_fused_features"]
    newly_built = {}
    completion_started = time.perf_counter()
    for position, candidate_id in enumerate(remaining_ids):
        row = candidates[candidate_id]
        root = output / candidate_id
        root.mkdir()
        action_started = time.perf_counter()
        build = build_one(row, relay_program, params, root, env, target_mode=target_mode)
        local_build = root / row["public_mode"]
        graph = read_json(local_build / "graph.json")
        if graph != stock_graph:
            raise RuntimeError("candidate Graph JSON differs from online stock")
        features = fused_features(build, graph, target["workload"])
        program = {
            "program_id": candidate_id,
            "candidate_id": candidate_id,
            "family_id": row["family_id"],
            "public_mode": row["public_mode"],
            "knobs": row["knobs"],
            "relative_dir": candidate_id + "/" + row["public_mode"],
            "build_seconds": build["build_seconds"],
            "complete_candidate_build_action_seconds": time.perf_counter() - action_started,
            "dispatch_audit": build["dispatch_audit"],
            "graph_sha256": build["graph_sha256"],
            "params_sha256": build["params_sha256"],
            "graphlib_sha256": build["graphlib_sha256"],
            **features,
        }
        newly_built[candidate_id] = program
        write_json(local_build / "program.json", program)
        print(
            "oracle-build {}/{} {} {} B={} N={} S={}".format(
                position + 1,
                len(remaining_ids),
                row["family_id"],
                row["public_mode"],
                program["dma_bytes"],
                program["dma_calls"],
                program["extra_submissions"],
            ),
            flush=True,
        )
        gc.collect()
    completion_wall_seconds = time.perf_counter() - completion_started

    programs = [
        online_programs[cid] if cid in online_programs else newly_built[cid]
        for cid in eligible_ids
    ]
    new_model_build_seconds = sum(row["build_seconds"] for row in newly_built.values())
    new_complete_build_seconds = sum(
        row["complete_candidate_build_action_seconds"] for row in newly_built.values()
    )
    result = {
        "schema": "c3_vta_invalid_dominator_peeling_oracle_pool_v1",
        "status": "invalid_dominator_peeling_oracle_completion_pool_after_online_labels",
        "workload_id": target["workload_id"],
        "source_model": target["source_model"],
        "source_layer": target["source_layer"],
        "full_graph_source": source_description,
        "input_specification": input_spec,
        "relay_build_target_mode": target_mode,
        "eligible_program_count": len(programs),
        "programs": programs,
        "stock_reference": stock,
        "stock_fused_features": stock_features,
        "selection_kind": "immutable_online_invalid_dominator_peeling",
        "pareto_front_candidate_ids": dispatched_ids,
        "online_dispatched_candidate_ids": dispatched_ids,
        "online_dispatched_count": len(dispatched_ids),
        "online_invalid_candidate_ids": online["invalid_candidate_ids"],
        "remaining_candidate_ids": remaining_ids,
        "remaining_candidate_count": len(remaining_ids),
        "build_cost": {
            "online_outer_process_seconds_before_summary_write": online[
                "outer_process_seconds_before_summary_write"
            ],
            "oracle_completion_new_candidate_model_build_seconds": new_model_build_seconds,
            "oracle_completion_complete_candidate_build_action_seconds": (
                new_complete_build_seconds
            ),
            "oracle_completion_process_wall_seconds": completion_wall_seconds,
            "copy_online_artifacts_seconds": copy_seconds,
        },
        "label_visibility": {
            "online_fpga_correctness": True,
            "online_full_graph_latency": True,
            "remaining_fpga_correctness": False,
            "remaining_full_graph_latency": False,
            "tophub_latency": False,
            "additional_frozen_control_labels_already_observed": bool(controls),
        },
        "additional_observed_control_candidate_ids": sorted({
            candidate_id
            for control in controls
            for candidate_id in control["dispatched_candidate_ids"]
        }),
        "additional_observed_result_artifact_manifest_sha256": [
            sha256(path / "artifact_hashes.json") for path in control_dirs
        ],
        "online_result_artifact_manifest_sha256": sha256(
            online_dir / "artifact_hashes.json"
        ),
        "bound_inputs": {
            "target_contract_sha256": sha256(target_dir / "contract.json"),
            "policy_artifact_manifest_sha256": sha256(policy_dir / "artifact_hashes.json"),
            "local_artifact_manifest_sha256": sha256(local_dir / "artifact_hashes.json"),
            "front_artifact_manifest_sha256": sha256(front_dir / "artifact_hashes.json"),
        },
        "verified_artifact_counts": verified,
        "builder_source_sha256": sha256(Path(__file__).resolve()),
        "board_contacted_for_remaining": False,
        "claim_boundary": (
            "Post-selection oracle completion. Online correctness and latency labels were "
            "already exposed, but cannot alter the immutable dispatch trace; remaining "
            "candidates are built and measured only to audit the final oracle and controls."
        ),
    }
    write_json(output / "pool.json", result)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
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
    print(
        json.dumps(
            {
                "status": result["status"],
                "eligible_program_count": len(programs),
                "online_dispatched_count": len(dispatched_ids),
                "remaining_count": len(remaining_ids),
                "build_cost": result["build_cost"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-contract", required=True)
    parser.add_argument("--policy-contract", required=True)
    parser.add_argument("--local-qualification", required=True)
    parser.add_argument("--front-contract", required=True)
    parser.add_argument("--online-result", required=True)
    parser.add_argument(
        "--additional-observed-result", action="append", default=[],
        help="Bind already-run frozen controls for label-disclosure only; remaining is still "
        "defined relative to the immutable peeling trace and is remeasured independently.",
    )
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
