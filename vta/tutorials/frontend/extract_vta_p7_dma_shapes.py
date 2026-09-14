#!/usr/bin/env python3
"""Extract per-call lowered DMA geometries for every frozen P7 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
)


def constant_or_text(expr):
    if isinstance(expr, tvm.tir.IntImm):
        return int(expr)
    return str(expr)


def call_rows(lowered):
    result = []

    def visit(node):
        if not isinstance(node, tvm.tir.Call) or node.op.name != "tir.call_extern":
            return
        if not node.args or not isinstance(node.args[0], tvm.tir.StringImm):
            return
        name = node.args[0].value
        if name == "VTALoadBuffer2D":
            result.append(
                {
                    "kind": "load",
                    "x_size": constant_or_text(node.args[4]),
                    "y_size": constant_or_text(node.args[5]),
                    "x_stride": constant_or_text(node.args[6]),
                    "memory_type": constant_or_text(node.args[12]),
                }
            )
        elif name == "VTAStoreBuffer2D":
            result.append(
                {
                    "kind": "store",
                    "x_size": constant_or_text(node.args[6]),
                    "y_size": constant_or_text(node.args[7]),
                    "x_stride": constant_or_text(node.args[8]),
                    "memory_type": constant_or_text(node.args[3]),
                }
            )

    for function in lowered.functions.values():
        tvm.tir.stmt_functor.post_order_visit(function.body, visit)
    return result


def uop_scope_counts(tir_text):
    """Count explicit UOP pushes executed while recording each UopKernel."""
    lines = tir_text.splitlines()
    result = []
    for start, line in enumerate(lines):
        if '"coproc_uop_scope"' not in line:
            continue
        base_indent = len(line) - len(line.lstrip())
        loops = []
        count = 0
        for body_line in lines[start + 1 :]:
            stripped = body_line.lstrip()
            if not stripped:
                continue
            indent = len(body_line) - len(stripped)
            if indent <= base_indent:
                break
            while loops and indent <= loops[-1][0]:
                loops.pop()
            if stripped.startswith("for "):
                match = re.search(r"range\((\d+)\)", stripped)
                if match:
                    loops.append((indent, int(match.group(1))))
                    continue
                match = re.search(r"T\.grid\(([^)]*)\)", stripped)
                if match:
                    extent = 1
                    for value in match.group(1).split(","):
                        extent *= int(value.strip())
                    loops.append((indent, extent))
                    continue
            if "T.tir.vta.uop_push(" in stripped:
                multiplicity = 1
                for _, extent in loops:
                    multiplicity *= extent
                count += multiplicity
        result.append(count)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite DMA geometry output")
    output.parent.mkdir(parents=True, exist_ok=True)
    contract = load_contract(args.contract)
    env = vta.get_env()
    rows = []
    for workload_id in contract["pool"]["holdouts"]:
        entries, _ = validate_workload(contract, workload_id)
        for candidate_id, entry in entries.items():
            validate_source_guards(entry)
            task = autotvm.task.create(
                "conv2d_packed_residency.vta",
                args=tuple(entry["workload"][1:])
                + (MODE_NUMBERS[entry["residence_mode"]],),
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
            tir_text = str(lowered)
            tir_hash = hashlib.sha256(tvm.ir.save_json(lowered).encode()).hexdigest()
            if tir_hash != entry["tir_sha256"]:
                raise RuntimeError("TIR certificate mismatch")
            calls = call_rows(lowered)
            stores = [call for call in calls if call["kind"] == "store"]
            rows.append(
                {
                    "workload_id": workload_id,
                    "candidate_id": candidate_id,
                    "residence_mode": entry["residence_mode"],
                    "config_index": entry["config_index"],
                    "tir_sha256": tir_hash,
                    "dma_calls_in_lowered_sites": calls,
                    "store_site_count": len(stores),
                    "constant_store_x_sizes": sorted(
                        {call["x_size"] for call in stores if isinstance(call["x_size"], int)}
                    ),
                    "constant_store_y_sizes": sorted(
                        {call["y_size"] for call in stores if isinstance(call["y_size"], int)}
                    ),
                    "constant_store_strides": sorted(
                        {call["x_stride"] for call in stores if isinstance(call["x_stride"], int)}
                    ),
                    "uop_kernel_explicit_push_counts": uop_scope_counts(tir_text),
                    "max_uop_kernel_explicit_pushes": max(uop_scope_counts(tir_text), default=0),
                }
            )
    payload = {
        "schema": "c3_p7_dma_geometry_v1",
        "contract_sha256": hashlib.sha256(Path(args.contract).read_bytes()).hexdigest(),
        "candidate_count": len(rows),
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
