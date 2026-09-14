#!/usr/bin/env python3
"""Build each unique policy-selected three-route ResNet18 program exactly once."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import tvm
from tvm import autotvm, relay
import vta
from vta.top.residency_dispatch import ExplicitResidencyDispatch

import run_vta_c3_literature_baselines as baseline
from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from build_vta_resnet50_fused_program_pool import (
    build_stock,
    fused_features,
    save_snapshots,
)
from build_vta_resnet50_generic_residency_tir_audit import (
    FullTIRArchive,
    cross_compiler,
)
from build_vta_resnet50_relay_residency_dispatch_pair import (
    make_relay_testing_resnet_program,
    sha256,
    write_json,
)
from run_vta_resnet50_relay_residency_dispatch_pair_board import semantic_params_hash


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(directory):
    directory = Path(directory).resolve()
    verify_artifacts_compatible(directory)
    return sha256(directory / "artifact_hashes.json")


def build_routes(routes, relay_program, params, local, env):
    local.mkdir()
    archive = FullTIRArchive(env)
    target = tvm.target.Target(env.target, host=env.target_host)
    build_target = {
        "cpu": tvm.target.Target(env.target_vta_cpu, host=env.target_host),
        "ext_dev": target,
    }
    relay.backend.te_compiler.get().clear()
    started = time.perf_counter()
    with autotvm.tophub.context(target):
        with ExplicitResidencyDispatch(routes) as dispatch:
            with vta.build_config(
                opt_level=3,
                disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
                instruments=[archive],
            ):
                graph, lib, lowered_params = relay.build(
                    relay_program, target=build_target, params=params
                )
    audit = dispatch.summary()
    if not audit["all_routes_scheduled"] or not archive.snapshots:
        raise RuntimeError("not every exact H1/H2/H3 route reached final scheduling")
    (local / "graph.json").write_text(graph, encoding="utf-8")
    (local / "params.bin").write_bytes(relay.save_param_dict(lowered_params))
    lib.export_library(str(local / "graphlib.so"), cross_compiler())
    result = {
        "build_seconds": time.perf_counter() - started,
        "dispatch_audit": audit,
        "snapshots": save_snapshots(local, archive),
        "graph_sha256": sha256(local / "graph.json"),
        "params_sha256": sha256(local / "params.bin"),
        "graphlib_sha256": sha256(local / "graphlib.so"),
    }
    write_json(local / "build.json", result)
    return result


def finish(output):
    repo = Path(__file__).resolve().parents[3]
    residency_source = repo / "vta/python/vta/top/residency_dispatch.py"
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_hashes": {
                str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
                str(residency_source): sha256(residency_source),
            },
        },
    )


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    selection_dir = Path(args.selection_contract_dir).resolve()
    source_dir = Path(args.source_contract_dir).resolve()
    input_hashes = {
        "selection": verify(selection_dir),
        "source_contract": verify(source_dir),
    }
    selection = read_json(selection_dir / "contract.json")
    source = read_json(source_dir / "contract.json")
    if selection.get("status") != "frozen_after_complete_operator_pools_before_selected_fullgraph_build_or_label":
        raise RuntimeError("selection contract is not pristine")
    if source.get("status") != "frozen_before_resnet18_fullgraph_build_fpga_or_latency":
        raise RuntimeError("source contract is not pristine")
    output.mkdir(parents=True)
    write_json(
        output / "contract.json",
        {
            "schema": "c3_resnet18_selected_fullgraph_build_contract_v1",
            "status": "frozen_before_selected_fullgraph_build",
            "input_hashes": input_hashes,
            "selection_seed": selection["selection_seed"],
            "selection_budget": selection["selection_budget"],
            "same_route_signature_built_once": True,
            "board_contacted": False,
            "fullgraph_performance_labels_read": False,
        },
    )
    env = vta.get_env()
    relay_program, params = make_relay_testing_resnet_program(env, 18, device_annot=True)
    stock = build_stock(relay_program, params, output, env, target_mode="heterogeneous")
    stock_graph = read_json(output / "stock_reference" / "graph.json")
    stock_semantic = semantic_params_hash(output / "stock_reference" / "params.bin")
    stock_by_workload = {
        workload_id: fused_features(
            stock, stock_graph, details["workload"]
        )
        for workload_id, details in source["verified_target_workloads"].items()
    }
    stock["target_workload_features"] = stock_by_workload
    write_json(output / "stock_reference" / "build.json", stock)

    by_signature = {}
    policy_to_program = {}
    for policy, row in selection["policy_routes"].items():
        signature = row["route_signature"]
        if signature in by_signature:
            policy_to_program[policy] = by_signature[signature]["program_id"]
            continue
        program_id = "selected_{}".format(signature[:16])
        local = output / program_id
        built = build_routes(row["routes"], relay_program, params, local, env)
        graph = read_json(local / "graph.json")
        if graph != stock_graph:
            raise RuntimeError("selected Graph JSON differs from stock")
        if semantic_params_hash(local / "params.bin") != stock_semantic:
            raise RuntimeError("selected parameter semantics differ from stock")
        workload_features = {
            route["workload_id"]: fused_features(
                built,
                graph,
                source["verified_target_workloads"][route["workload_id"]]["workload"],
            )
            for route in row["routes"]
        }
        program = {
            "program_id": program_id,
            "route_signature": signature,
            "routes": row["routes"],
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
        }
        write_json(local / "program.json", program)
        by_signature[signature] = program
        policy_to_program[policy] = program_id
        gc.collect()
        print("built {} for {}".format(program_id, policy), flush=True)

    programs = sorted(by_signature.values(), key=lambda row: row["program_id"])
    summary = {
        "schema": "c3_resnet18_selected_fullgraph_build_v1",
        "status": "selected_fullgraphs_built_once_before_board",
        "source": "relay.testing.resnet.get_workload(num_layers=18)",
        "stock_reference": stock,
        "stock_target_workload_features": stock_by_workload,
        "programs": programs,
        "policy_to_program": policy_to_program,
        "policy_count": len(policy_to_program),
        "unique_program_count": len(programs),
        "total_build_seconds": stock["build_seconds"]
        + sum(row["build_seconds"] for row in programs),
        "board_contacted": False,
        "fullgraph_performance_labels_read": False,
        "claim_boundary": (
            "Exact compiler builds and logical VTA DMA only; no physical AXI, board "
            "correctness, full-graph latency or ImageNet accuracy."
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
                "total_build_seconds": summary["total_build_seconds"],
                "occurrences": {
                    workload_id: len(value["graph_occurrences"])
                    for workload_id, value in stock_by_workload.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-contract-dir", required=True)
    parser.add_argument("--source-contract-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
