#!/usr/bin/env python3
"""Freeze a W04 input-residency tile-height correctness boundary probe."""

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
from prepare_vta_w04_store_group_probe import split_knobs
from run_vta_p7_holdout_correctness import (
    file_sha256,
    load_contract,
    validate_workload,
    write_json,
)


CONFIG_INDICES = (136, 137, 138, 139)
INHERITED_CONFIG_INDEX = 139


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
        raise FileExistsError("refusing to overwrite height-probe contract")

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
        if not (
            knobs["tile_w"] == 7
            and knobs["tile_ci"] == 1
            and knobs["tile_co"] == 4
            and knobs["oc_nthread"] == 1
            and knobs["h_nthread"] == 1
            and knobs["tile_h"] in (1, 2, 7, 14)
        ):
            raise RuntimeError("unexpected height-probe ConfigSpace mapping")
        identity = candidate_identity_record(
            p4["hardware_fingerprint"],
            p4["template_name"],
            p4["schedule_version"],
            workload,
            "input_stationary",
            config_json,
            config_index=index,
        )
        row = {
            "candidate_id": identity["candidate_id"],
            "workload_id": "W04",
            "workload": workload,
            "residence_mode": "input_stationary",
            "config_index": index,
            "complete_config_entity": config_json,
            "split_knobs": knobs,
            "source_guards_sha256": failed["source_guards_sha256"],
            "accumulator_vectors_per_residency_region": 16 * knobs["tile_h"] * 7,
            "accumulator_vector_capacity": 2048,
        }
        try:
            with task.target:
                schedule, tensors = task.instantiate(config)
            with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
                lowered = tvm.lower(schedule, tensors, name="main")
            row.update(
                {
                    "tir_sha256": hashlib.sha256(
                        tvm.ir.save_json(lowered).encode()
                    ).hexdigest(),
                    "store_sites": [
                        item for item in call_rows(lowered) if item["kind"] == "store"
                    ],
                    "uop_kernel_explicit_push_counts": uop_scope_counts(str(lowered)),
                    "lower_status": "passed",
                    "label_source": "inherited_p7c1"
                    if index == 139
                    else "new_board_probe",
                }
            )
        except Exception as error:  # Static rejection is itself part of the probe.
            row.update(
                {
                    "tir_sha256": None,
                    "store_sites": [],
                    "uop_kernel_explicit_push_counts": [],
                    "lower_status": "rejected",
                    "lower_failure_type": type(error).__name__,
                    "lower_failure_message": str(error)[:2000],
                    "label_source": "static_lower_rejected",
                }
            )
        candidates.append(row)

    inherited_path = Path(args.inherited_diagnostic)
    inherited = json.loads((inherited_path / "summary.json").read_text())
    if inherited.get("candidate_id") != failed["candidate_id"] or inherited.get(
        "target_all_correct"
    ) is not False:
        raise RuntimeError("inherited diagnostic does not bind the known W04 failure")
    board_order = sorted(
        [
            row["candidate_id"]
            for row in candidates
            if row["label_source"] == "new_board_probe"
        ],
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    )
    contract = {
        "schema": "c3_w04_height_probe_contract_v1",
        "status": "frozen_before_new_probe_labels",
        "hypothesis": "with W04 input_stationary tile_w=7, tile_ci=1 and tile_co=4 fixed, real-board correctness changes at a tile-height/residency-footprint boundary not visible in FSim",
        "p7_contract": {
            "path": str(Path(args.p7_contract).resolve()),
            "sha256": file_sha256(args.p7_contract),
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
            "config_index": 139,
            "diagnostic_path": str(inherited_path.resolve()),
            "diagnostic_sha256": file_sha256(inherited_path / "diagnostic.jsonl"),
            "target_all_correct": False,
        },
        "sentinel": sentinel,
        "measurement_policy": {
            "new_candidates": "three exact seeds once each; retain all outcomes",
            "sentinel": "known-correct original config139 seed0 after every candidate",
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
                "vta/tutorials/frontend/prepare_vta_w04_height_probe.py": file_sha256(__file__)
            },
        },
    )
    print("frozen W04 height probe: 3 new + 1 inherited candidate")


if __name__ == "__main__":
    main()
