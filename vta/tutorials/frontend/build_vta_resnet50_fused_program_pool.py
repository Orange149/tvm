#!/usr/bin/env python3
"""Build every FSim-qualified ResNet50 program once and freeze its fused Pareto front."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import tvm
from tvm import autotvm, relay
import vta

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from analyze_vta_resnet50_fused_tir_occurrence_delta import RUNTIME_KEYS, function_inventory
from build_vta_resnet50_generic_residency_tir_audit import (
    FullTIRArchive,
    build_one,
    cross_compiler,
)
from build_vta_resnet50_relay_residency_dispatch_pair import (
    make_relay_program,
    sha256,
    write_json,
)
from run_vta_resnet50_generic_residency_pair_board import graph_workload_occurrences


AXES = ("dma_bytes", "dma_calls", "extra_submissions")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dominates(left, right):
    return all(left[key] <= right[key] for key in AXES) and any(
        left[key] < right[key] for key in AXES
    )


def pareto_front(programs):
    return [
        row for row in programs
        if not any(dominates(other, row) for other in programs if other is not row)
    ]


def snapshot_sync_inventory(build):
    result = {}
    for snapshot in build["snapshots"]:
        if len(snapshot["functions"]) != 1:
            raise RuntimeError("CPUAccessRewrite snapshot does not contain exactly one PrimFunc")
        symbol = snapshot["functions"][0]["global_symbol"]
        if symbol in result:
            raise RuntimeError("duplicate fused TIR sync symbol: " + repr(symbol))
        result[symbol] = int(snapshot["coproc_sync_occurrences"])
    return result


def fused_features(build, graph, workload):
    occurrences = graph_workload_occurrences(graph, workload)
    dma_inventory = function_inventory(build)
    sync_inventory = snapshot_sync_inventory(build)
    traffic = {key: 0 for key in RUNTIME_KEYS}
    syncs = 0
    occurrence_rows = []
    for occurrence in occurrences:
        symbol = occurrence["func_name"]
        if symbol not in dma_inventory or symbol not in sync_inventory:
            raise RuntimeError("target graph symbol missing from fused TIR: " + symbol)
        dma = {key: int(dma_inventory[symbol].get(key, 0)) for key in RUNTIME_KEYS}
        for key, value in dma.items():
            traffic[key] += value
        syncs += sync_inventory[symbol]
        occurrence_rows.append({
            "graph_node": occurrence["graph_node"],
            "func_name": symbol,
            "dma": dma,
            "coproc_sync_occurrences": sync_inventory[symbol],
        })
    return {
        "graph_occurrences": occurrence_rows,
        "traffic": traffic,
        "dma_bytes": traffic["load_buffer_2d_bytes"] + traffic["store_buffer_2d_bytes"],
        "dma_calls": traffic["load_buffer_2d_calls"] + traffic["store_buffer_2d_calls"],
        "coproc_sync_occurrences": syncs,
        "extra_submissions": max(syncs - len(occurrences), 0),
    }


def save_snapshots(local, archive):
    snapshots = []
    for index, item in enumerate(archive.snapshots):
        item = dict(item)
        stem = "cpu_access_rewrite_{:03d}".format(index)
        (local / (stem + ".json")).write_text(item.pop("json"), encoding="utf-8")
        (local / (stem + ".tir")).write_text(item.pop("tir"), encoding="utf-8")
        snapshots.append({
            **item,
            "json_file": stem + ".json",
            "json_sha256": sha256(local / (stem + ".json")),
            "tir_file": stem + ".tir",
            "tir_sha256": sha256(local / (stem + ".tir")),
        })
    return snapshots


def build_stock(relay_program, params, output, env, target_mode="heterogeneous"):
    local = output / "stock_reference"
    local.mkdir()
    archive = FullTIRArchive(env)
    target = tvm.target.Target(env.target, host=env.target_host)
    if target_mode == "heterogeneous":
        build_target = {
            "cpu": tvm.target.Target(env.target_vta_cpu, host=env.target_host),
            "ext_dev": target,
        }
    elif target_mode == "ext_dev_only":
        build_target = target
    else:
        raise ValueError("unsupported build target mode: " + repr(target_mode))
    relay.backend.te_compiler.get().clear()
    started = time.perf_counter()
    with autotvm.tophub.context(target):
        with vta.build_config(
            opt_level=3,
            disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
            instruments=[archive],
        ):
            graph, lib, lowered_params = relay.build(
                relay_program, target=build_target, params=params
            )
    if not archive.snapshots:
        raise RuntimeError("stock fused TIR archive is empty")
    (local / "graph.json").write_text(graph, encoding="utf-8")
    (local / "params.bin").write_bytes(relay.save_param_dict(lowered_params))
    lib.export_library(str(local / "graphlib.so"), cross_compiler())
    result = {
        "program_id": "stock_reference",
        "candidate_id": None,
        "public_mode": "stock_tophub_reference",
        "build_seconds": time.perf_counter() - started,
        "snapshots": save_snapshots(local, archive),
        "graph_sha256": sha256(local / "graph.json"),
        "params_sha256": sha256(local / "params.bin"),
        "graphlib_sha256": sha256(local / "graphlib.so"),
        "selection_eligible": False,
        "role": "correctness_and_profile_reference_only",
    }
    write_json(local / "build.json", result)
    return result


def run(args):
    target_dir = Path(args.target_contract)
    policy_dir = Path(args.policy_contract)
    local_dir = Path(args.local_qualification)
    verified = {
        "target_contract": len(verify_artifacts_compatible(target_dir)),
        "policy_contract": len(verify_artifacts_compatible(policy_dir)),
        "local_qualification": len(verify_artifacts_compatible(local_dir)),
    }
    target = read_json(target_dir / "contract.json")
    policy = read_json(policy_dir / "contract.json")
    local_summary = read_json(local_dir / "summary.json")
    if policy.get("status") != "frozen_before_target_lower_fsim_fpga_or_latency":
        raise RuntimeError("Pareto policy is not pristine")
    if policy["workload_id"] != target["workload_id"]:
        raise RuntimeError("policy target mismatch")
    if policy["bound_inputs"]["target_contract_sha256"] != sha256(target_dir / "contract.json"):
        raise RuntimeError("policy does not bind target contract")
    if local_summary.get("status") != "completed_local_no_board":
        raise RuntimeError("local qualification is incomplete")
    if local_summary.get("performance_labels_used") or local_summary.get("board_contacted"):
        raise RuntimeError("local qualification leaked target labels")

    # The local qualifier canonicalizes the legacy four-mode freeze into the
    # three public v2 modes and therefore owns the executable candidate IDs.
    candidates = read_jsonl(local_dir / "candidates_v2.jsonl")
    by_id = {row["candidate_id"]: row for row in candidates}
    fsim = read_jsonl(local_dir / "fsim_results.jsonl")
    eligible_ids = [row["candidate_id"] for row in fsim if row.get("status") == "passed"]
    if len(eligible_ids) != local_summary["fsim_passed"]:
        raise RuntimeError("FSim-pass count drift")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    env = vta.get_env()
    relay_program, params = make_relay_program(env, pretrained=True)
    stock = build_stock(relay_program, params, output, env)
    stock_graph = read_json(output / "stock_reference" / "graph.json")
    stock_features = fused_features(stock, stock_graph, target["workload"])
    stock.update(fused_features=stock_features)
    write_json(output / "stock_reference" / "build.json", stock)
    print("fused-pool stock complete {:.3f}s".format(stock["build_seconds"]), flush=True)

    programs = []
    for position, candidate_id in enumerate(eligible_ids):
        row = by_id[candidate_id]
        root = output / candidate_id
        root.mkdir()
        build = build_one(row, relay_program, params, root, env)
        local = root / row["public_mode"]
        graph = read_json(local / "graph.json")
        if graph != stock_graph:
            raise RuntimeError("candidate graph JSON differs from stock reference")
        features = fused_features(build, graph, target["workload"])
        program = {
            "program_id": candidate_id,
            "candidate_id": candidate_id,
            "family_id": row["family_id"],
            "public_mode": row["public_mode"],
            "knobs": row["knobs"],
            "relative_dir": candidate_id + "/" + row["public_mode"],
            "build_seconds": build["build_seconds"],
            "dispatch_audit": build["dispatch_audit"],
            "graph_sha256": build["graph_sha256"],
            "params_sha256": build["params_sha256"],
            "graphlib_sha256": build["graphlib_sha256"],
            **features,
        }
        programs.append(program)
        write_json(local / "program.json", program)
        print(
            "fused-pool {}/{} {} {} B={} N={} S={}".format(
                position + 1, len(eligible_ids), row["family_id"], row["public_mode"],
                program["dma_bytes"], program["dma_calls"], program["extra_submissions"],
            ), flush=True,
        )
        gc.collect()

    front = sorted(pareto_front(programs), key=lambda row: row["candidate_id"])
    front_ids = [row["candidate_id"] for row in front]
    dominated = {
        row["candidate_id"]: sorted(
            other["candidate_id"] for other in programs if dominates(other, row)
        ) for row in programs if row["candidate_id"] not in front_ids
    }
    result = {
        "schema": "c3_vta_resnet50_fused_program_pool_v1",
        "status": "final_fused_pareto_front_frozen_before_fpga",
        "workload_id": target["workload_id"],
        "source_model": target["source_model"],
        "source_layer": target["source_layer"],
        "eligible_program_count": len(programs),
        "programs": programs,
        "stock_reference": stock,
        "stock_fused_features": stock_features,
        "pareto_axes": list(AXES),
        "pareto_front_candidate_ids": front_ids,
        "pareto_front_count": len(front_ids),
        "dominated_by": dominated,
        "total_build_seconds": stock["build_seconds"] + sum(
            row["build_seconds"] for row in programs
        ),
        "label_visibility": {
            "fpga_correctness": False,
            "operator_latency": False,
            "full_graph_latency": False,
            "tophub_latency": False,
        },
        "bound_inputs": {
            "target_contract_sha256": sha256(target_dir / "contract.json"),
            "policy_contract_sha256": sha256(policy_dir / "contract.json"),
            "local_summary_sha256": sha256(local_dir / "summary.json"),
            "local_artifact_manifest_sha256": sha256(local_dir / "artifact_hashes.json"),
        },
        "verified_artifact_counts": verified,
        "builder_source_sha256": sha256(Path(__file__).resolve()),
        "board_contacted": False,
        "claim_boundary": (
            "Prospective compiler-only final-fused Pareto front. Logical VTA DMA and TIR "
            "sync sites are not physical AXI or measured latency; dominance remains under test."
        ),
    }
    write_json(output / "pool.json", result)
    (output / "command.txt").write_text(
        " ".join(__import__("sys").argv) + "\n", encoding="utf-8"
    )
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "status": result["status"],
        "eligible_program_count": len(programs),
        "pareto_front_count": len(front_ids),
        "pareto_front_candidate_ids": front_ids,
        "total_build_seconds": result["total_build_seconds"],
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-contract", required=True)
    parser.add_argument("--policy-contract", required=True)
    parser.add_argument("--local-qualification", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
