#!/usr/bin/env python3
"""Compare board-pass and board-fail P7R configurations without measuring latency.

The report is intentionally diagnostic: it lowers the frozen original schedules,
extracts allocation/DMA/UOP/dependency features, and keeps the TIR text so a
real-FPGA legality rule can be derived before dispatching more candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import tvm
from tvm import autotvm
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from analyze_vta_vthread_residency_space import EXTRA_GEOMETRIES, knob_values
from extract_static_vta_dma import extract_module_dma_compact
from extract_vta_p7_dma_shapes import call_rows, uop_scope_counts


CASES = (
    ("W04", 330, "fpga_fail"),
    ("W04", 455, "fpga_pass"),
    ("W04", 461, "fpga_pass"),
    ("E00", 1061, "fpga_fail"),
    ("E00", 1081, "runtime_uop_reject"),
    ("E01", 951, "fpga_fail"),
    ("E01", 755, "fpga_fail"),
    ("E01", 957, "fpga_fail"),
    ("E01", 877, "fpga_fail"),
    ("E01", 301, "fpga_fail"),
    ("E02", 1163, "fpga_pass"),
    ("E02", 1167, "fpga_pass"),
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _constant_product(extents):
    product = 1
    for extent in extents:
        if not isinstance(extent, tvm.tir.IntImm):
            return None
        product *= int(extent)
    return product


def _allocation_rows(lowered):
    rows = []

    def visit(node):
        if not isinstance(node, tvm.tir.Allocate):
            return
        annotation = node.buffer_var.type_annotation
        scope = getattr(annotation, "storage_scope", "")
        rows.append(
            {
                "name": node.buffer_var.name,
                "dtype": str(node.dtype),
                "scope": str(scope),
                "extents": [int(value) if isinstance(value, tvm.tir.IntImm) else str(value)
                            for value in node.extents],
                "scalar_elements": _constant_product(node.extents),
            }
        )

    for function in lowered.functions.values():
        tvm.tir.stmt_functor.post_order_visit(function.body, visit)
    return rows


def _extern_histogram(lowered):
    histogram = Counter()

    def visit(node):
        if not isinstance(node, tvm.tir.Call) or node.op.name != "tir.call_extern":
            return
        if node.args and isinstance(node.args[0], tvm.tir.StringImm):
            histogram[node.args[0].value] += 1

    for function in lowered.functions.values():
        tvm.tir.stmt_functor.post_order_visit(function.body, visit)
    return dict(sorted(histogram.items()))


def _tir_text_features(tir_text):
    keys = (
        "coproc_dep_push",
        "coproc_dep_pop",
        "coproc_sync",
        "coproc_uop_scope",
        "virtual_thread",
        "VTALoadBuffer2D",
        "VTAStoreBuffer2D",
        "VTAUopPush",
    )
    return {key: tir_text.count(key) for key in keys}


def _selected_workloads(selected_tasks):
    selected = json.loads(Path(selected_tasks).read_text(encoding="utf-8"))
    result = {"W{:02d}".format(index): row for index, row in enumerate(selected)}
    result.update(EXTRA_GEOMETRIES)
    return result


def _task(workload, env):
    return autotvm.task.create(
        "conv2d_packed_residency.vta",
        args=tuple(workload[1:]) + (0,),
        target=env.target,
        target_host=env.target_host,
    )


def _one_case(workload_id, config_index, board_status, selected, env, output):
    workload = selected[workload_id]["workload"]
    task = _task(workload, env)
    config = task.config_space.get(config_index)
    knobs = knob_values(config)
    with task.target:
        schedule, tensors = task.instantiate(config)
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        lowered = tvm.lower(schedule, tensors, name="main")
    tir_text = str(lowered)
    name = "{}_config{}_{}".format(workload_id.lower(), config_index, board_status)
    (output / (name + ".tir.txt")).write_text(tir_text + "\n", encoding="utf-8")
    calls = call_rows(lowered)
    stores = [row for row in calls if row["kind"] == "store"]
    uops = uop_scope_counts(tir_text)
    weight_shape = workload[2][1]
    co_blocks = int(weight_shape[0])
    output_tile_vectors = knobs["tile_h"] * knobs["tile_w"] * knobs["tile_co"]
    return {
        "workload_id": workload_id,
        "config_index": config_index,
        "board_status": board_status,
        "knobs": knobs,
        "derived": {
            "output_tile_vectors": output_tile_vectors,
            "cthread_output_tile_vectors": knobs["oc_nthread"] * output_tile_vectors,
            "co_outer_groups_after_cthread": co_blocks
            // (knobs["oc_nthread"] * knobs["tile_co"]),
        },
        "tir_sha256": hashlib.sha256(tvm.ir.save_json(lowered).encode()).hexdigest(),
        "allocations": _allocation_rows(lowered),
        "logical_dma": extract_module_dma_compact(lowered, env)["totals"],
        "lowered_dma_sites": calls,
        "store_x_sizes": sorted({str(row["x_size"]) for row in stores}),
        "store_y_sizes": sorted({str(row["y_size"]) for row in stores}),
        "store_strides": sorted({str(row["x_stride"]) for row in stores}),
        "uop_kernel_explicit_push_counts": uops,
        "max_uop_kernel_explicit_pushes": max(uops, default=0),
        "extern_site_histogram": _extern_histogram(lowered),
        "tir_text_counts": _tir_text_features(tir_text),
        "tir_text_file": name + ".tir.txt",
    }


def _markdown(rows):
    lines = [
        "# P7R real-FPGA legality differential",
        "",
        "This is a label-preserving diagnostic; it collects no latency.",
        "",
        "| workload/config | FPGA | tile h×w×ci×co | output tile vectors | cthread tile vectors | co groups | max UOP pushes | store x/y/stride |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        knob = row["knobs"]
        derived = row["derived"]
        store = "{}/{}/{}".format(
            ",".join(row["store_x_sizes"]),
            ",".join(row["store_y_sizes"]),
            ",".join(row["store_strides"]),
        )
        lines.append(
            "| {}/{} | {} | {}×{}×{}×{} | {} | {} | {} | {} | {} |".format(
                row["workload_id"], row["config_index"], row["board_status"],
                knob["tile_h"], knob["tile_w"], knob["tile_ci"], knob["tile_co"],
                derived["output_tile_vectors"], derived["cthread_output_tile_vectors"],
                derived["co_outer_groups_after_cthread"],
                row["max_uop_kernel_explicit_pushes"], store,
            )
        )
    lines += [
        "",
        "Interpretation is deliberately deferred until the full TIR/DMA feature table is compared.",
        "A feature that only separates these observations is a hypothesis, not yet a hardware rule.",
    ]
    return "\n".join(lines) + "\n"


def main():
    base = Path(__file__).resolve().parent / "report_out" / "stage_tile_cotuning"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-tasks",
        default=str(base / "iteration3_tophub_incumbent" / "selected_tasks.json"),
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    env = vta.get_env()
    selected = _selected_workloads(args.selected_tasks)
    rows = [
        _one_case(workload_id, index, status, selected, env, output)
        for workload_id, index, status in CASES
    ]
    payload = {
        "schema": "c3_p7r_hardware_legality_differential_v1",
        "scope": "local lowering diagnostics with frozen real-FPGA pass/fail labels; no latency",
        "cases": rows,
        "inputs": {
            "selected_tasks": args.selected_tasks,
            "selected_tasks_sha256": _sha256(args.selected_tasks),
            "schedule_sha256": _sha256(
                Path(__file__).resolve().parents[2] / "python" / "vta" / "top" / "vta_conv2d.py"
            ),
        },
    }
    (output / "diagnostic.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "RESULTS.md").write_text(_markdown(rows), encoding="utf-8")
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": artifacts}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(_markdown(rows), end="")


if __name__ == "__main__":
    main()
