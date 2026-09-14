#!/usr/bin/env python3
"""Dump hash-checked lowered TIR for frozen P7 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import tvm
from tvm import autotvm
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from run_vta_p7_holdout_correctness import (
    MODE_NUMBERS,
    load_contract,
    semantic_config_key,
    validate_source_guards,
    validate_workload,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--candidate-id", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("refusing to overwrite TIR dump")

    contract = load_contract(args.contract)
    entries, _ = validate_workload(contract, args.workload)
    env = vta.get_env()
    rows = []
    for candidate_id in args.candidate_id:
        entry = entries[candidate_id]
        validate_source_guards(entry)
        task = autotvm.task.create(
            "conv2d_packed_residency.vta",
            args=tuple(entry["workload"][1:]) + (MODE_NUMBERS[entry["residence_mode"]],),
            target=env.target,
            target_host=env.target_host,
        )
        config = task.config_space.get(int(entry["config_index"]))
        if semantic_config_key(config.to_json_dict()) != semantic_config_key(
            entry["complete_config_entity"]
        ):
            raise RuntimeError("ConfigEntity mismatch")
        with task.target:
            schedule, tensors = task.instantiate(config)
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            lowered = tvm.lower(schedule, tensors, name="main")
        serialized = tvm.ir.save_json(lowered)
        tir_hash = hashlib.sha256(serialized.encode()).hexdigest()
        if tir_hash != entry["tir_sha256"]:
            raise RuntimeError("TIR certificate mismatch")
        stem = candidate_id[:12]
        (output / (stem + ".tir.txt")).write_text(str(lowered) + "\n", encoding="utf-8")
        (output / (stem + ".tir.json")).write_text(serialized + "\n", encoding="utf-8")
        rows.append(
            {
                "candidate_id": candidate_id,
                "residence_mode": entry["residence_mode"],
                "config_index": entry["config_index"],
                "tir_sha256": tir_hash,
                "text_file": stem + ".tir.txt",
                "json_file": stem + ".tir.json",
            }
        )
    write_json(output / "manifest.json", {"schema": "c3_p7_tir_dump_v1", "rows": rows})


if __name__ == "__main__":
    main()
