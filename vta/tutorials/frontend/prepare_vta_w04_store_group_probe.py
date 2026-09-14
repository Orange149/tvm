#!/usr/bin/env python3
"""Freeze a W04 input-residency output-store grouping correctness probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import tvm
from tvm import autotvm
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import candidate_identity_record
from extract_vta_p7_dma_shapes import call_rows, uop_scope_counts
from run_vta_p7_holdout_correctness import (
    file_sha256,
    load_contract,
    semantic_config_key,
    validate_workload,
    write_json,
)


CONFIG_INDICES = (11, 75, 139, 203, 267)
INHERITED_CONFIG_INDEX = 139


def split_knobs(config):
    result = {}
    for name, kind, value in config["entity"]:
        result[name] = int(value[-1] if kind == "sp" else value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p7-contract", required=True)
    parser.add_argument("--p4-pool", required=True)
    parser.add_argument("--inherited-diagnostic", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("refusing to overwrite probe contract")

    p7 = load_contract(args.p7_contract)
    p4 = json.loads(Path(args.p4_pool).read_text())
    entries, _ = validate_workload(p7, "W04")
    failed = entries[
        "906c1bdc905dc2ebbbd18569941dcfd4f11fe9e32ed55b9bfa03eb027b3b9c40"
    ]
    sentinel = entries[
        "056aba501b8dc2b43c91d52c4449f201db368fe8458fae6b42d4717fecefcbae"
    ]
    workload = failed["workload"]
    env = vta.get_env()
    task = autotvm.task.create(
        "conv2d_packed_residency.vta",
        args=tuple(workload[1:]) + (1,),
        target=env.target,
        target_host=env.target_host,
    )

    candidates = []
    for index in CONFIG_INDICES:
        config = task.config_space.get(index)
        config_json = config.to_json_dict()
        knobs = split_knobs(config_json)
        expected = {
            "tile_h": 14,
            "tile_w": 7,
            "tile_ci": 1,
            "oc_nthread": 1,
            "h_nthread": 1,
        }
        for name, value in expected.items():
            if knobs[name] != value:
                raise RuntimeError("unexpected probe ConfigSpace mapping")
        with task.target:
            schedule, tensors = task.instantiate(config)
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            lowered = tvm.lower(schedule, tensors, name="main")
        tir_json = tvm.ir.save_json(lowered)
        tir_hash = hashlib.sha256(tir_json.encode()).hexdigest()
        identity = candidate_identity_record(
            p4["hardware_fingerprint"],
            p4["template_name"],
            p4["schedule_version"],
            workload,
            "input_stationary",
            config_json,
            config_index=index,
        )
        stores = [row for row in call_rows(lowered) if row["kind"] == "store"]
        candidates.append(
            {
                "candidate_id": identity["candidate_id"],
                "workload_id": "W04",
                "workload": workload,
                "residence_mode": "input_stationary",
                "config_index": index,
                "complete_config_entity": config_json,
                "split_knobs": knobs,
                "tir_sha256": tir_hash,
                "source_guards_sha256": failed["source_guards_sha256"],
                "store_sites": stores,
                "uop_kernel_explicit_push_counts": uop_scope_counts(str(lowered)),
                "accumulator_vectors_per_residency_region": 16 * 14 * 7,
                "accumulator_vector_capacity": 2048,
                "label_source": "inherited_p7c1" if index == INHERITED_CONFIG_INDEX else "new_board_probe",
            }
        )

    inherited_path = Path(args.inherited_diagnostic)
    inherited = json.loads((inherited_path / "summary.json").read_text())
    if inherited.get("candidate_id") != failed["candidate_id"] or inherited.get(
        "target_all_correct"
    ) is not False:
        raise RuntimeError("inherited diagnostic does not bind the known W04 failure")
    new_ids = [
        row["candidate_id"] for row in candidates if row["config_index"] != INHERITED_CONFIG_INDEX
    ]
    board_order = sorted(new_ids, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    contract = {
        "schema": "c3_w04_store_group_probe_contract_v1",
        "status": "frozen_before_new_probe_labels",
        "hypothesis": "with fixed tile_h=14, tile_w=7, tile_ci=1 and equal 1568-vector accumulator footprint, changing tile_co changes the number/grouping of consecutive output-store commands; correctness may expose a hardware-specific grouping boundary that FSim misses",
        "p7_contract": {
            "path": str(Path(args.p7_contract).resolve()),
            "sha256": file_sha256(args.p7_contract),
        },
        "p4_identity_source": {
            "path": str(Path(args.p4_pool).resolve()),
            "sha256": file_sha256(args.p4_pool),
        },
        "hardware_fingerprint": p4["hardware_fingerprint"],
        "board_contract": p7["board_contract"],
        "correctness_seeds": p7["measurement"]["correctness_seeds"],
        "candidate_count": len(candidates),
        "new_board_candidate_count": len(board_order),
        "board_order": board_order,
        "candidates": candidates,
        "inherited_failure": {
            "candidate_id": failed["candidate_id"],
            "config_index": INHERITED_CONFIG_INDEX,
            "diagnostic_path": str(inherited_path.resolve()),
            "diagnostic_sha256": file_sha256(inherited_path / "diagnostic.jsonl"),
            "target_repeat_count": inherited["target_repeat_count"],
            "target_all_correct": inherited["target_all_correct"],
        },
        "sentinel": sentinel,
        "measurement_policy": {
            "new_candidates": "three exact seeds once each; retain all pass/fail outcomes",
            "sentinel": "known-correct original config139 seed0 after every new candidate; stop if sentinel fails",
            "timing": "forbidden",
            "replacement": "forbidden",
        },
    }
    contract_path = output / "contract.json"
    write_json(contract_path, contract)
    write_json(
        output / "artifact_hashes.json",
        {
            "schema": "artifact_hashes_v1",
            "output_sha256": {"contract.json": file_sha256(contract_path)},
            "source_sha256": {
                "vta/tutorials/frontend/prepare_vta_w04_store_group_probe.py": file_sha256(
                    __file__
                )
            },
        },
    )
    print("frozen W04 store-group probe: 4 new + 1 inherited candidate")


if __name__ == "__main__":
    main()
