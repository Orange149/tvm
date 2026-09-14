#!/usr/bin/env python3
"""Build only new route signatures after the R18 graph-correctness fallback."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import vta

import build_vta_c3_resnet18_fullgraph_programs as original_builder
from run_vta_resnet50_relay_residency_dispatch_pair_board import semantic_params_hash


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def finish(output):
    repo = Path(__file__).resolve().parents[3]
    sources = [
        Path(__file__).resolve(),
        repo / "vta/tutorials/frontend/build_vta_c3_resnet18_fullgraph_programs.py",
        repo / "vta/python/vta/top/residency_dispatch.py",
        repo / "vta/python/vta/top/vta_conv2d.py",
    ]
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                str(path.relative_to(output)): original_builder.sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_hashes": {
                str(path): original_builder.sha256(path) for path in sources
            },
        },
    )


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    fallback_dir = Path(args.fallback_contract_dir).resolve()
    source_dir = Path(args.source_contract_dir).resolve()
    prior_dir = Path(args.prior_build_dir).resolve()
    input_hashes = {
        "fallback": original_builder.verify(fallback_dir),
        "source_contract": original_builder.verify(source_dir),
        "prior_build": original_builder.verify(prior_dir),
    }
    fallback = read_json(fallback_dir / "contract.json")
    source = read_json(source_dir / "contract.json")
    prior = read_json(prior_dir / "summary.json")
    if fallback.get("status") != "frozen_after_graph_correctness_failure_before_any_selected_fullgraph_timing":
        raise RuntimeError("fallback contract is not pristine")
    if fallback.get("fullgraph_performance_labels_read") is not False:
        raise RuntimeError("fallback contract contains fullgraph performance labels")
    if source.get("status") != "frozen_before_resnet18_fullgraph_build_fpga_or_latency":
        raise RuntimeError("fullgraph source contract is not pristine")
    if prior.get("status") != "selected_fullgraphs_built_once_before_board":
        raise RuntimeError("prior selected build is incomplete")

    output.mkdir(parents=True)
    write_json(
        output / "contract.json",
        {
            "schema": "c3_resnet18_fullgraph_fallback_build_contract_v1",
            "status": "frozen_before_fallback_fullgraph_build",
            "input_hashes": input_hashes,
            "selection_seed": fallback["selection_seed"],
            "selection_budget": fallback["selection_budget"],
            "reuse_exact_prior_route_signatures": True,
            "board_contacted": False,
            "fullgraph_performance_labels_read": False,
        },
    )

    shutil.copytree(prior_dir / "stock_reference", output / "stock_reference")
    stock = prior["stock_reference"]
    stock_graph = read_json(output / "stock_reference" / "graph.json")
    stock_semantic = semantic_params_hash(output / "stock_reference" / "params.bin")
    prior_by_signature = {row["route_signature"]: row for row in prior["programs"]}
    by_signature = {}
    policy_to_program = {}
    new_build_seconds = 0.0
    reused_program_count = 0
    relay_program = params = env = None

    for policy, policy_row in fallback["policy_routes"].items():
        signature = policy_row["route_signature"]
        if signature in by_signature:
            policy_to_program[policy] = by_signature[signature]["program_id"]
            continue
        if signature in prior_by_signature:
            prior_program = prior_by_signature[signature]
            program_id = prior_program["program_id"]
            shutil.copytree(prior_dir / prior_program["relative_dir"], output / program_id)
            program = json.loads(json.dumps(prior_program))
            program["relative_dir"] = program_id
            program["build_provenance"] = {
                "kind": "exact_route_signature_reused",
                "prior_build_manifest_sha256": input_hashes["prior_build"],
                "repaid_build_seconds": 0.0,
            }
            write_json(output / program_id / "program.json", program)
            reused_program_count += 1
        else:
            if relay_program is None:
                env = vta.get_env()
                relay_program, params = original_builder.make_relay_testing_resnet_program(
                    env, 18, device_annot=True
                )
            program_id = "fallback_{}".format(signature[:16])
            local = output / program_id
            built = original_builder.build_routes(
                policy_row["routes"], relay_program, params, local, env
            )
            graph = read_json(local / "graph.json")
            if graph != stock_graph:
                raise RuntimeError("fallback Graph JSON differs from stock")
            if semantic_params_hash(local / "params.bin") != stock_semantic:
                raise RuntimeError("fallback parameter semantics differ from stock")
            workload_features = {
                route["workload_id"]: original_builder.fused_features(
                    built,
                    graph,
                    source["verified_target_workloads"][route["workload_id"]]["workload"],
                )
                for route in policy_row["routes"]
            }
            program = {
                "program_id": program_id,
                "route_signature": signature,
                "routes": policy_row["routes"],
                "relative_dir": program_id,
                "build_seconds": built["build_seconds"],
                "graph_sha256": built["graph_sha256"],
                "params_sha256": built["params_sha256"],
                "graphlib_sha256": built["graphlib_sha256"],
                "dispatch_audit": built["dispatch_audit"],
                "target_workload_features": workload_features,
                "target_logical_dma_bytes": sum(
                    item["dma_bytes"] for item in workload_features.values()
                ),
                "target_logical_dma_calls": sum(
                    item["dma_calls"] for item in workload_features.values()
                ),
                "build_provenance": {
                    "kind": "new_after_correctness_fallback",
                    "repaid_build_seconds": built["build_seconds"],
                },
            }
            write_json(local / "program.json", program)
            new_build_seconds += float(built["build_seconds"])
            gc.collect()
        by_signature[signature] = program
        policy_to_program[policy] = program["program_id"]
        print(
            "prepared {} for {} ({})".format(
                program["program_id"], policy, program["build_provenance"]["kind"]
            ),
            flush=True,
        )

    programs = sorted(by_signature.values(), key=lambda row: row["program_id"])
    summary = {
        "schema": "c3_resnet18_selected_fullgraph_build_v1",
        "status": "selected_fullgraphs_built_once_before_board",
        "selection_kind": "post_graph_correctness_original_fallback",
        "source": "relay.testing.resnet.get_workload(num_layers=18)",
        "stock_reference": stock,
        "stock_target_workload_features": prior["stock_target_workload_features"],
        "programs": programs,
        "policy_to_program": policy_to_program,
        "policy_count": len(policy_to_program),
        "unique_program_count": len(programs),
        "reused_program_count": reused_program_count,
        "new_program_count": len(programs) - reused_program_count,
        "new_build_seconds": new_build_seconds,
        "board_contacted": False,
        "fullgraph_performance_labels_read": False,
        "claim_boundary": (
            "Fallback build/reuse after graph correctness only; no selected fullgraph "
            "latency, FPS, physical AXI traffic or ImageNet accuracy."
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in programs),
        encoding="utf-8",
    )
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    finish(output)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "policy_count": summary["policy_count"],
                "unique_program_count": summary["unique_program_count"],
                "reused_program_count": summary["reused_program_count"],
                "new_program_count": summary["new_program_count"],
                "new_build_seconds": summary["new_build_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fallback-contract-dir", required=True)
    parser.add_argument("--source-contract-dir", required=True)
    parser.add_argument("--prior-build-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
