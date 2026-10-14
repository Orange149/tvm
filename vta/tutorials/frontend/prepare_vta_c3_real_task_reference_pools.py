#!/usr/bin/env python3
"""Freeze compact reference pools for current-board-latency-unseen real tasks."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from c3_candidate_identity import normalize_config_entity
from build_vta_c3_real_task_inventory import digest, sha256_file, write_json
from prepare_vta_c3_resnet18_literature_candidates import (
    MODES,
    candidate_row,
    create_task,
    hardware_fingerprint,
    max_min_select,
)
import prepare_vta_p7r115_yolo_confirmation as shared


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
C3 = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
GROUPED = C3 / "07_grouped_holdout"
DEFAULT_INVENTORY = GROUPED / "20260924_c3_real_task_inventory_run02"
DEFAULT_EXPOSURE = GROUPED / "20260924_c3_real_task_exposure_audit_run02"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_real_task_reference_prepare_run01"
TOPHUB_CACHE = Path("/home/orange/.tvm/tophub/vta_v0.10.log")
SCHEMA = "c3_real_task_reference_pool_v1"


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def knobs(entity):
    return {
        name: int(value[-1] if kind == "sp" else value)
        for name, kind, value in entity["entity"]
    }


def geometry(workload):
    data, weight = workload[1][1], workload[2][1]
    return {
        "ci": int(data[1] * data[5]),
        "co": int(weight[0] * weight[4]),
        "height": int(data[2]),
        "width": int(data[3]),
        "kernel": int(weight[2]),
        "stride": int(workload[3][0]),
    }


def workload_id(workload):
    spec = geometry(workload)
    return "R18-U{}C{}H{}K{}S{}".format(
        spec["ci"], spec["co"], spec["height"], spec["kernel"], spec["stride"]
    )


def exact_tophub_records(workload, cache=TOPHUB_CACHE):
    matches = []
    for line_number, line in enumerate(Path(cache).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        record = json.loads(line)
        inp = record["input"]
        if inp[1] == workload[0] and inp[2] == workload[1:] and record["result"][1] == 0:
            matches.append(
                {
                    "line_number": line_number,
                    "mean_source_cost": float(statistics.mean(record["result"][0])),
                    "record": record,
                }
            )
    return sorted(matches, key=lambda row: (row["mean_source_cost"], row["line_number"]))


def select_families(workload, current_task, env, target_id, tophub_record):
    domain = [
        (index, normalize_config_entity(current_task.config_space.get(index).to_json_dict()))
        for index in range(len(current_task.config_space))
    ]
    source_knobs = knobs(tophub_record["record"]["config"])
    incumbent = [(index, entity) for index, entity in domain if knobs(entity) == source_knobs]
    if len(incumbent) != 1:
        raise ValueError("TopHub tile is missing or ambiguous in current ConfigSpace")
    incumbent_index, incumbent_entity = incumbent[0]
    eligible = []
    for index, entity in domain:
        visible = knobs(entity)
        if visible["oc_nthread"] != 1 or visible["h_nthread"] != 1:
            continue
        certificate = shared.hardware_predicate(workload, entity, env)
        if certificate["passed"] and entity != incumbent_entity:
            eligible.append(entity)
    alternatives = max_min_select(eligible, 2, target_id)
    index_by_hash = {digest(entity): index for index, entity in domain}
    families = [
        {
            "family_id": target_id + "F00",
            "role": "exact_tophub_incumbent_tile",
            "config_index": incumbent_index,
            "entity": incumbent_entity,
        }
    ]
    for ordinal, entity in enumerate(alternatives, 1):
        families.append(
            {
                "family_id": "{}F{:02d}".format(target_id, ordinal),
                "role": "label_blind_static_diversity_tile",
                "config_index": index_by_hash[digest(entity)],
                "entity": entity,
            }
        )
    return families, len(domain), len(eligible)


def build_pool(dispatch, env, fingerprint, schedule_version, cache):
    workload = dispatch["vta_workload"]
    target_id = workload_id(workload)
    records = exact_tophub_records(workload, cache)
    if not records:
        raise ValueError("no exact TopHub incumbent for {}".format(target_id))
    task = create_task(workload, 0, env)
    families, domain_count, eligible_count = select_families(
        workload, task, env, target_id, records[0]
    )
    spec = geometry(workload)
    rows = []
    for family in families:
        for mode in MODES:
            row = candidate_row(
                target_id,
                spec,
                workload,
                family["entity"],
                family["config_index"],
                family["family_id"],
                mode,
                schedule_version,
                fingerprint,
                env,
            )
            row["selection_role"] = family["role"]
            row["dispatch_scope_id"] = dispatch["dispatch_scope_id"]
            row["candidate_identity_fields"] = {
                "tile": row["visible_features"],
                "residence_mode": mode,
                "implementation_mode": row["implementation_mode"],
                "weight_load_policy": (
                    "barrier_full_or_group_residency" if mode in ("weight_resident_barrier", "input_weight_resident_barrier")
                    else "template_default"
                ),
                "schedule_version": schedule_version,
            }
            rows.append(row)
    return {
        "schema": SCHEMA,
        "status": "frozen_before_real_lowering_fsim_or_current_board_latency",
        "role": "compact_four_mode_reference_candidate_pool",
        "workload_id": target_id,
        "model": dispatch["model"],
        "source_layers": dispatch["source_layers"],
        "dispatch_scope_id": dispatch["dispatch_scope_id"],
        "workload": workload,
        "geometry": spec,
        "hardware_fingerprint": fingerprint,
        "candidates": rows,
        "selection": {
            "method": "exact TopHub tile plus two normalized-knob max-min single-vthread static-feasible tiles",
            "family_count": 3,
            "gross_candidate_count": len(rows),
            "config_domain_count": domain_count,
            "eligible_alternative_count": eligible_count,
            "current_board_performance_labels_used": False,
            "lowering_or_correctness_labels_used": False,
            "source_tophub_cost_used_only_to_bind_incumbent": True,
            "invalid_modes_remain_explicit_not_applicable": True,
        },
        "incumbent": {
            "kind": "exact_workload_tophub_source_record_pending_current_board_requalification",
            "source_cache": str(Path(cache).resolve()),
            "source_cache_sha256": sha256_file(cache),
            "exact_record_count": len(records),
            "selected_line_number": records[0]["line_number"],
            "source_target": records[0]["record"]["input"][0],
            "source_complete_config_entity": records[0]["record"]["config"],
            "current_config_index": families[0]["config_index"],
            "candidate_id": next(
                row["candidate_id"]
                for row in rows
                if row["family_id"] == families[0]["family_id"] and row["residence_mode"] == "original"
            ),
            "current_board_latency": None,
            "global_optimum_claimed": False,
        },
        "board_contacted": False,
        "performance_labels_collected": False,
        "claim_boundary": "candidate generation and external-cache incumbent binding only",
    }


def run(args):
    import vta

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    dispatch = {
        row["dispatch_scope_id"]: row
        for row in load_jsonl(Path(args.inventory_dir) / "dispatch_tasks.jsonl")
    }
    exposure = load_jsonl(Path(args.exposure_dir) / "exposure_bindings.jsonl")
    selected = [
        dispatch[row["dispatch_scope_id"]]
        for row in exposure
        if row["model"] == "relay_testing_resnet18" and row["strict_latency_unseen_eligible"]
    ]
    if len(selected) != 2:
        raise ValueError("protocol requires exactly two current-board-latency-unseen ResNet18 tasks")
    env = vta.get_env()
    fingerprint = hardware_fingerprint(env)
    sources = [
        Path(__file__).resolve(),
        HERE / "prepare_vta_c3_resnet18_literature_candidates.py",
        REPO / "vta/python/vta/top/vta_conv2d.py",
        REPO / "vta/python/vta/top/vta_conv2d_residency.py",
    ]
    source_hashes = {str(path): sha256_file(path) for path in sources}
    schedule_version = digest(source_hashes)
    pools = [
        build_pool(row, env, fingerprint, schedule_version, args.tophub_cache)
        for row in sorted(selected, key=lambda item: item["dispatch_scope_id"])
    ]
    output.mkdir(parents=True)
    pool_files = []
    for ordinal, pool in enumerate(pools, 4):
        name = "r18-h{}_four_mode_96.json".format(ordinal)
        pool["source_hashes"] = source_hashes
        write_json(output / name, pool)
        pool_files.append(name)
    contract = {
        "schema": "c3_real_task_reference_prepare_contract_v1",
        "status": "two_resnet18_reference_pools_frozen_before_local_or_board",
        "pool_files": pool_files,
        "pool_workload_ids": [pool["workload_id"] for pool in pools],
        "gross_candidate_count": sum(len(pool["candidates"]) for pool in pools),
        "inventory_manifest_sha256": sha256_file(Path(args.inventory_dir) / "artifact_hashes.json"),
        "exposure_manifest_sha256": sha256_file(Path(args.exposure_dir) / "artifact_hashes.json"),
        "tophub_cache_sha256": sha256_file(args.tophub_cache),
        "schedule_version": schedule_version,
        "source_hashes": source_hashes,
        "board_contacted": False,
        "current_board_performance_labels_read": False,
        "external_tophub_source_cost_read_for_incumbent_only": True,
        "yolo_reference_status": "no_strict_current_board_latency_unseen_dispatch_scope_in_yolov3_tiny_416",
        "claim_boundary": "compact proposed pools, not local legality, current-board latency, or oracle",
    }
    write_json(output / "contract.json", contract)
    write_json(
        output / "summary.json",
        {
            "status": contract["status"],
            "pool_workload_ids": contract["pool_workload_ids"],
            "gross_candidate_count": contract["gross_candidate_count"],
            "yolo_reference_status": contract["yolo_reference_status"],
            "board_contacted": False,
        },
    )
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_sha256": sha256_file(Path(__file__).resolve()),
        },
    )
    print(json.dumps(json.loads((output / "summary.json").read_text()), indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-dir", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--exposure-dir", type=Path, default=DEFAULT_EXPOSURE)
    parser.add_argument("--tophub-cache", type=Path, default=TOPHUB_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
