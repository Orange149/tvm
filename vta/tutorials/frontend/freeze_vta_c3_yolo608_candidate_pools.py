#!/usr/bin/env python3
"""Freeze eight label-blind candidate identities for each prepared YOLO task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import vta

from build_vta_c3_real_task_inventory import darknet_call_instances
from c3_candidate_identity import canonical_json_bytes
from prepare_vta_c3_resnet18_literature_candidates import (
    MODES,
    candidate_row,
    hardware_fingerprint,
)


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_PREPARE = GROUPED / "20260924_c3_yolo_unseen_resolution_prepare_run02"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo608_candidate_pools_run01"
SCHEMA = "c3_yolo_unseen_candidate_pool_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def geometry(workload):
    data, weight = workload[1][1], workload[2][1]
    return {
        "ci": int(data[1] * data[5]), "co": int(weight[0] * weight[4]),
        "height": int(data[2]), "width": int(data[3]),
        "kernel": int(weight[2]), "stride": int(workload[3][0]),
    }


def source_layers(cfg_path: Path, spec):
    matches = []
    for call in darknet_call_instances(cfg_path):
        contract = call["operator_contract"]
        key = {
            "ci": contract["ci"], "co": contract["co"],
            "height": contract["input_height"], "width": contract["input_width"],
            "kernel": contract["kernel_height"], "stride": contract["strides"][0],
        }
        if key == spec:
            matches.append(call["source_layer"])
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path, default=DEFAULT_PREPARE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable output: {args.output}")
    contract = json.loads((args.prepare / "contract.json").read_text())
    resolution = contract.get("selected_resolution")
    if not isinstance(resolution, int) or contract.get("latency_values_read") is not False:
        raise RuntimeError("input is not a frozen label-blind YOLO resolution preparation")
    exposure = (contract.get("latency_exposure_registry") or {}).get(str(resolution))
    if exposure is None:
        exposure = (contract.get("latency_exposure_registry") or {}).get(resolution)
    if exposure is not None:
        raise RuntimeError("selected resolution is latency exposed")
    result_path = args.prepare / "resolution_{}/result.json".format(resolution)
    result = json.loads(result_path.read_text())
    if result.get("resolution") != resolution or not str(result.get("status", "")).startswith(
        "static_feasible"
    ):
        raise RuntimeError("selected resolution result is not statically feasible")
    cfg_path = args.prepare / "resolution_{}/yolov3-tiny-{}.cfg".format(
        resolution, resolution)
    env = vta.get_env()
    fingerprint = hardware_fingerprint(env)
    sources = [Path(__file__).resolve(), HERE / "prepare_vta_c3_yolo_unseen_resolution.py",
               HERE / "prepare_vta_c3_resnet18_literature_candidates.py",
               HERE.parents[2] / "vta/python/vta/top/vta_conv2d.py",
               HERE.parents[2] / "vta/python/vta/top/vta_conv2d_residency.py"]
    source_hashes = {str(path): sha256(path) for path in sources}
    schedule_version = digest(source_hashes)
    args.output.mkdir(parents=True)
    pool_files = []
    total = 0
    for task in sorted(result["tasks"], key=lambda row: row["task_id"]):
        workload = task["workload"]
        spec = geometry(workload)
        workload_id = "Y{}-".format(resolution) + task["task_id"][:16]
        candidates = []
        for family_ordinal, family in enumerate(task["selected_families"]):
            entity = family["complete_config_entity"]
            index = int(family["debug"]["config_index"])
            family_id = f"{workload_id}-F{family_ordinal}"
            for mode in MODES:
                candidate = candidate_row(workload_id, spec, workload, entity, index,
                                          family_id, mode, schedule_version,
                                          fingerprint, env)
                candidate["model_hash"] = digest({"model": "yolov3-tiny", "resolution": resolution,
                                                   "weights_sha256": contract["weights"]["sha256"]})
                candidate["candidate_identity_fields"] = {
                    "tile": candidate["visible_features"],
                    "residence_mode": mode,
                    "implementation_mode": candidate["implementation_mode"],
                    "weight_load_policy": ("barrier_full_or_group_residency"
                                           if "resident" in mode else "template_default"),
                    "schedule_version": schedule_version,
                }
                candidates.append(candidate)
        if len(candidates) != 8 or len({row["candidate_id"] for row in candidates}) != 8:
            raise RuntimeError(f"{workload_id}: expected eight unique frozen identities")
        pool = {
            "schema": SCHEMA,
            "status": "frozen_before_lowering_fsim_or_board_latency",
            "workload_id": workload_id,
            "execution_task_id": task["task_id"],
            "model": "yolov3_tiny_darknet",
            "input_resolution": resolution,
            "source_layers": source_layers(cfg_path, spec),
            "call_count": task["occurrences"],
            "workload": workload,
            "geometry": spec,
            "hardware_fingerprint": fingerprint,
            "candidates": candidates,
            "incumbent": {
                "kind": "frozen_stock_mode0_fallback_pending_board_requalification",
                "candidate_id": next(row["candidate_id"] for row in candidates
                                     if row["family_id"].endswith("F0")
                                     and row["residence_mode"] == "original"),
                "current_board_latency": None,
                "global_optimum_claimed": False,
            },
            "selection": {
                "method": "two deterministic label-blind static-feasible families times four modes",
                "gross_candidate_count": 8,
                "performance_labels_used": False,
                "local_correctness_used": False,
            },
            "source_hashes": source_hashes,
            "board_contacted": False,
            "performance_labels_collected": False,
        }
        name = f"{workload_id}.json"
        write_json(args.output / name, pool)
        pool_files.append(name)
        total += 8
    freeze = {
        "schema": SCHEMA + "_contract",
        "status": "yolo{}_pools_frozen_before_local_qualification".format(resolution),
        "input_resolution": resolution,
        "pool_files": pool_files,
        "candidate_count": total,
        "execution_task_count": len(pool_files),
        "prepare_contract_sha256": sha256(args.prepare / "contract.json"),
        "prepare_result_sha256": sha256(result_path),
        "derived_cfg_sha256": sha256(cfg_path),
        "weights_sha256": contract["weights"]["sha256"],
        "source_hashes": source_hashes,
        "board_contacted": False,
        "performance_labels_read": False,
        "claim_boundary": "candidate identities only; lowering, FSim and program equivalence pending",
    }
    write_json(args.output / "contract.json", freeze)
    write_json(args.output / "summary.json", freeze)
    write_json(args.output / "artifact_hashes.json", {
        "artifacts": {path.name: sha256(path) for path in sorted(args.output.iterdir())
                      if path.is_file() and path.name != "artifact_hashes.json"},
        "source": sha256(Path(__file__).resolve()),
    })
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
