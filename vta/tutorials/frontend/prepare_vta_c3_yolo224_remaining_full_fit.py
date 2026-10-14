#!/usr/bin/env python3
"""Freeze latency-blind C3 pools for the remaining full-weight-fit YOLO-224 tasks.

This is a follow-up opportunity screen, not a replacement for any frozen C3
experiment.  It selects every packed convolution that (a) is present in the
frozen YOLOv3-tiny-224 stock graph, (b) has a complete weight tensor no larger
than the physical WGT SRAM capacity, and (c) was not one of the three tasks in
the completed YOLO-224 dual-boot campaign.  Selection never reads latency.

For each selected execution task, three modes (Original, Input and transition-
safe Input+Weight residency/mode11) must lower for the same ConfigEntity.  Three
diverse families are frozen, giving nine candidates per task.  FPGA contact,
FSim and performance measurement are deliberately deferred to later stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from tvm import autotvm
import vta

from c3_candidate_identity import normalize_config_entity
from prepare_vta_c3_resnet18_literature_candidates import hardware_fingerprint
from prepare_vta_c3_yolo_lowering_aware_route_pool import (
    SCHEDULE_SOURCES,
    TEMPLATE,
    digest,
    diverse_families,
    geometry,
    make_candidate,
    probe_lowering,
    rank,
    read_json,
    sha256_file,
    verify_prepare,
    write_json,
    append_jsonl,
)
from prepare_vta_p7r115_yolo_confirmation import hardware_predicate


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_PREPARE = GROUPED / "20260926_c3_yolo224_static_prepare_run01"
DEFAULT_GRAPH = GROUPED / "20260926_c3_yolo224_graph_build_run02/stock/graph.json"
DEFAULT_OUTPUT = GROUPED / "20260927_c3_yolo224_remaining_full_fit_pools_run01"
SCHEMA = "c3_yolo224_remaining_full_fit_pool_v1"
MODE_NUMBERS = {
    "original": 0,
    "input_stationary": 1,
    "input_weight_resident_transition_safe": 11,
}
MODE_ORDER = tuple(MODE_NUMBERS)

# These three task identities already have two-boot operator and graph evidence.
# They are excluded by identity, not by their observed latency.
COMPLETED_DUAL_BOOT_TASKS = frozenset({
    "094b2b164538c2afc67719e78701a221bcfcb3c783e9ff90b89c3f5e7cebd54a",
    "0f5aa8b96e7667eb4573cfb282dc1e217124854ab35047088afed38d7efb5d10",
    "c7fb14254c3edc503c51a8715f58184b6a937f3872738e5e2708a647710ba2d3",
})


def packed_signature_from_workload(workload):
    """Return the input/weight tensor signature used to bind graph call sites."""

    return (
        tuple(int(value) for value in workload[1][1]),
        tuple(int(value) for value in workload[2][1]),
    )


def graph_call_sites(graph):
    """Index packed VTA convolution calls by their input and weight shapes."""

    shapes = graph["attrs"]["shape"][1]
    row_ptr = graph["node_row_ptr"]
    calls = defaultdict(list)
    for node_id, node in enumerate(graph["nodes"]):
        attrs = node.get("attrs") or {}
        if node.get("op") != "tvm_op" or "conv2d" not in node.get("name", ""):
            continue
        if attrs.get("data_layout") != "NCHW1n16c":
            continue
        if len(node.get("inputs", [])) < 2:
            raise ValueError("packed convolution node {} lacks data/weight inputs".format(node_id))
        input_shapes = []
        for source_node, source_output, _ in node["inputs"][:2]:
            entry_id = int(row_ptr[source_node]) + int(source_output)
            input_shapes.append(tuple(int(value) for value in shapes[entry_id]))
        output_entry_id = int(row_ptr[node_id])
        calls[tuple(input_shapes)].append({
            "node_id": int(node_id),
            "node_name": node["name"],
            "func_name": attrs.get("func_name"),
            "function_hash": attrs.get("hash"),
            "input_entry_id": int(row_ptr[node["inputs"][0][0]]) + int(node["inputs"][0][1]),
            "weight_entry_id": int(row_ptr[node["inputs"][1][0]]) + int(node["inputs"][1][1]),
            "output_entry_id": output_entry_id,
            "output_shape": [int(value) for value in shapes[output_entry_id]],
        })
    return calls


def classify_tasks(tasks, graph, env):
    """Build the complete graph inventory and select the latency-blind follow-up set."""

    calls = graph_call_sites(graph)
    inventory = []
    selected = []
    seen_signatures = set()
    for task in sorted(tasks, key=lambda row: row["task_id"]):
        spec = geometry(task["workload"])
        signature = packed_signature_from_workload(task["workload"])
        if signature in seen_signatures:
            raise ValueError("duplicate execution-task signature in prepare manifest")
        seen_signatures.add(signature)
        sites = sorted(calls.get(signature, []), key=lambda row: row["node_id"])
        if not sites:
            raise ValueError("task {} is not bound to the stock graph".format(task["task_id"]))
        weight_shape = signature[1]
        weight_bytes = 1
        for extent in weight_shape:
            weight_bytes *= extent
        fits = weight_bytes <= int(env.WGT_BUFF_SIZE)
        completed = task["task_id"] in COMPLETED_DUAL_BOOT_TASKS
        disposition = (
            "completed_dual_boot"
            if completed
            else "new_full_weight_fit"
            if fits
            else "requires_partial_or_grouped_residency"
        )
        row = {
            "execution_task_id": task["task_id"],
            "geometry": spec,
            "workload": task["workload"],
            "weight_shape": list(weight_shape),
            "full_weight_bytes": int(weight_bytes),
            "wgt_sram_capacity_bytes": int(env.WGT_BUFF_SIZE),
            "full_weight_fits_sram": bool(fits),
            "completed_dual_boot": bool(completed),
            "disposition": disposition,
            "prepare_occurrences_advisory": task.get("occurrences"),
            "graph_call_count": len(sites),
            "graph_call_sites": sites,
        }
        inventory.append(row)
        if disposition == "new_full_weight_fit":
            selected.append((task, spec, row))

    graph_signatures = set(calls)
    unmatched = graph_signatures - seen_signatures
    if unmatched:
        raise ValueError("stock graph contains packed convolution signatures absent from task manifest")
    return inventory, selected


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))

    prepare_contract, prepared, result_path = verify_prepare(args.prepare_dir)
    if int(prepare_contract["selected_resolution"]) != 224:
        raise ValueError("this experiment is frozen to YOLOv3-tiny-224")
    graph_path = Path(args.graph_json).resolve()
    graph = read_json(graph_path)
    env = vta.get_env()
    inventory, selected = classify_tasks(prepared["tasks"], graph, env)
    if len(selected) != 4:
        raise RuntimeError("expected exactly four remaining full-weight-fit tasks, got {}".format(len(selected)))

    source_paths = tuple(SCHEDULE_SOURCES) + (
        Path(__file__).resolve(),
        Path(__file__).with_name("prepare_vta_c3_yolo_lowering_aware_route_pool.py"),
    )
    source_hashes = {str(path.relative_to(REPO)): sha256_file(path) for path in source_paths}
    schedule_version = digest({
        "mechanism": "yolo224-remaining-full-fit-original-input-mode11-v1",
        "source_hashes": source_hashes,
    })
    model_hash = digest({
        "model": "yolov3-tiny",
        "resolution": 224,
        "weights_sha256": prepare_contract["weights"]["sha256"],
        "stock_graph_sha256": sha256_file(graph_path),
    })

    output.mkdir(parents=True)
    scan_path = output / "lowering_scan.jsonl"
    scan_path.write_text("", encoding="utf-8")
    write_json(output / "task_inventory.json", {"schema": SCHEMA, "tasks": inventory})
    pool_files = []
    total_candidates = 0
    task_summaries = {}

    for task_row, spec, inventory_row in sorted(selected, key=lambda item: item[0]["task_id"]):
        workload = task_row["workload"]
        workload_id = "Y224-{}".format(task_row["task_id"][:16])
        mode_tasks = {
            mode: autotvm.task.create(
                TEMPLATE,
                args=tuple(workload[1:]) + (number,),
                target=env.target,
                target_host=env.target_host,
            )
            for mode, number in MODE_NUMBERS.items()
        }
        domains = {
            mode: [
                normalize_config_entity(task.config_space.get(index).to_json_dict())
                for index in range(len(task.config_space))
            ]
            for mode, task in mode_tasks.items()
        }
        reference = domains["original"]
        if any(domain != reference for domain in domains.values()):
            raise RuntimeError("{} mode ConfigSpaces differ".format(workload_id))

        eligible = []
        for index, entity in enumerate(reference):
            certificate = hardware_predicate(workload, entity, env)
            if certificate["passed"]:
                eligible.append((rank(workload_id, entity), digest(entity), index, entity, certificate))
        eligible.sort()

        admitted = []
        attempted = 0
        failure_counts = Counter()
        for selection_rank, entity_hash, index, entity, certificate in eligible[: args.scan_limit]:
            attempted += 1
            family_candidates = []
            for mode in MODE_ORDER:
                candidate = make_candidate(
                    workload_id, workload, spec, entity, index, mode, schedule_version, env
                )
                candidate["model_hash"] = model_hash
                result = probe_lowering(candidate, workload)
                append_jsonl(scan_path, {
                    "schema": SCHEMA,
                    "workload_id": workload_id,
                    "gross_family_position": attempted,
                    "selection_rank_sha256": selection_rank,
                    "semantic_config_sha256": entity_hash,
                    "config_index_debug_only": index,
                    "mode": mode,
                    "candidate_id": candidate["candidate_id"],
                    "visible_features": candidate["visible_features"],
                    **result,
                })
                if result["status"] == "passed":
                    candidate["generation_lowered_tir_sha256"] = result["lowered_tir_sha256"]
                    family_candidates.append(candidate)
                else:
                    key = "dma_2d_pattern" if result["failure"]["dma_2d_pattern_failure"] else "other"
                    failure_counts[key] += 1
            if len(family_candidates) == len(MODE_ORDER):
                admitted.append({
                    "entity": entity,
                    "selection_rank_sha256": selection_rank,
                    "semantic_config_sha256": entity_hash,
                    "config_index_debug_only": index,
                    "hardware_predicate": certificate,
                    "candidates": family_candidates,
                })
            if len(admitted) >= args.admitted_family_target:
                break

        chosen = diverse_families(workload_id, admitted, args.selected_families)
        candidates = []
        for ordinal, family in enumerate(chosen):
            family_id = "{}-F{}".format(workload_id, ordinal)
            for candidate in family["candidates"]:
                candidate["family_id"] = family_id
                candidate["generation_family_ordinal"] = ordinal
                candidate["graph_binding"] = inventory_row["graph_call_sites"]
                candidates.append(candidate)
        incumbent_id = next(
            (row["candidate_id"] for row in candidates if row["residence_mode"] == "original"),
            None,
        )
        complete = len(chosen) == args.selected_families
        pool = {
            "schema": SCHEMA,
            "status": (
                "frozen_after_lowering_gate_before_fsim_fpga_or_latency"
                if complete
                else "insufficient_fully_lowerable_families"
            ),
            "workload_id": workload_id,
            "execution_task_id": task_row["task_id"],
            "model": "yolov3_tiny_darknet",
            "input_resolution": 224,
            "call_count": inventory_row["graph_call_count"],
            "graph_call_sites": inventory_row["graph_call_sites"],
            "workload": workload,
            "geometry": spec,
            "full_weight_bytes": inventory_row["full_weight_bytes"],
            "hardware_fingerprint": hardware_fingerprint(env),
            "candidate_modes": list(MODE_ORDER),
            "candidates": candidates,
            "incumbent": {
                "kind": "first_frozen_hash_rank_original_that_passes_three-mode_lowering_gate",
                "candidate_id": incumbent_id,
                "current_board_latency": None,
                "global_optimum_claimed": False,
            },
            "selection": {
                "task_rule": (
                    "all stock-graph packed conv tasks whose complete int8 weights fit WGT SRAM, "
                    "excluding the three completed dual-boot task identities"
                ),
                "family_rule": (
                    "hardware predicate, then same-entity Original/Input/mode11 real VTA lowering; "
                    "deterministic max-min selection from admitted prefix"
                ),
                "eligible_config_count": len(eligible),
                "gross_families_attempted": attempted,
                "fully_lowerable_families": len(admitted),
                "selected_family_count": len(chosen),
                "scan_limit": args.scan_limit,
                "admitted_family_target": args.admitted_family_target,
                "performance_labels_used": False,
                "fsim_used": False,
            },
            "source_hashes": source_hashes,
            "board_contacted": False,
            "performance_labels_collected": False,
        }
        filename = workload_id + ".json"
        write_json(output / filename, pool)
        pool_files.append(filename)
        total_candidates += len(candidates)
        task_summaries[workload_id] = {
            "execution_task_id": task_row["task_id"],
            "geometry": spec,
            "graph_call_count": inventory_row["graph_call_count"],
            "graph_node_ids": [row["node_id"] for row in inventory_row["graph_call_sites"]],
            "full_weight_bytes": inventory_row["full_weight_bytes"],
            "eligible_config_count": len(eligible),
            "gross_families_attempted": attempted,
            "fully_lowerable_families": len(admitted),
            "selected_family_count": len(chosen),
            "candidate_count": len(candidates),
            "failure_counts": dict(failure_counts),
            "status": pool["status"],
        }

    complete_tasks = sum(
        row["selected_family_count"] == args.selected_families
        for row in task_summaries.values()
    )
    contract = {
        "schema": SCHEMA + "_contract",
        "status": (
            "LOWERING_AWARE_POOLS_FROZEN"
            if complete_tasks == len(task_summaries)
            else "LOWERING_AWARE_POOL_INCOMPLETE"
        ),
        "input_resolution": 224,
        "pool_files": pool_files,
        "execution_task_count": len(pool_files),
        "complete_execution_task_count": complete_tasks,
        "candidate_count": total_candidates,
        "candidate_modes": list(MODE_ORDER),
        "task_selection_rule": (
            "all stock-graph packed conv tasks whose complete int8 weights fit WGT SRAM, "
            "excluding the three completed dual-boot task identities"
        ),
        "completed_dual_boot_task_ids": sorted(COMPLETED_DUAL_BOOT_TASKS),
        "selection_used_performance_values": False,
        "stock_graph": {"path": str(graph_path), "sha256": sha256_file(graph_path)},
        "task_inventory_file": "task_inventory.json",
        "prepare_contract_sha256": sha256_file(Path(args.prepare_dir) / "contract.json"),
        "prepare_result_sha256": sha256_file(result_path),
        "schedule_version": schedule_version,
        "source_hashes": source_hashes,
        "task_summaries": task_summaries,
        "board_contacted": False,
        "performance_labels_read": False,
        "fsim_used_for_selection": False,
        "claim_boundary": (
            "static graph membership, full-weight capacity, real-lowering legality and frozen "
            "identities only; FSim, FPGA correctness, latency and speed claims remain pending"
        ),
    }
    write_json(output / "contract.json", contract)
    write_json(output / "summary.json", contract)
    write_json(output / "artifact_hashes.json", {
        "artifacts": {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "artifact_hashes.json"
        },
        "source_sha256": sha256_file(Path(__file__).resolve()),
    })
    print(json.dumps(contract, indent=2, sort_keys=True), flush=True)
    if contract["status"] != "LOWERING_AWARE_POOLS_FROZEN":
        raise SystemExit(2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-dir", type=Path, default=DEFAULT_PREPARE)
    parser.add_argument("--graph-json", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scan-limit", type=int, default=1024)
    parser.add_argument("--admitted-family-target", type=int, default=12)
    parser.add_argument("--selected-families", type=int, default=3)
    args = parser.parse_args()
    if args.selected_families * len(MODE_ORDER) > 15:
        parser.error("selected pool exceeds the frozen 15-candidate bound")
    if args.admitted_family_target < args.selected_families:
        parser.error("admitted-family-target must cover selected-families")
    return args


if __name__ == "__main__":
    run(parse_args())
