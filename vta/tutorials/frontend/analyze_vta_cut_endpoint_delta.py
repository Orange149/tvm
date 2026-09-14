#!/usr/bin/env python3
"""Build a compile-time DMA/boundary delta table for adjacent VTA cut endpoints."""

import argparse
import hashlib
import json
from pathlib import Path

from analyze_stage_memory_additivity import canonical, occurrence_count, workload_label


ENDPOINTS = {
    15: "cpu-00-02__vta-03-15__cpu-16-20",
    16: "cpu-00-02__vta-03-16__cpu-17-20",
    17: "cpu-00-02__vta-03-17__cpu-18-20",
}
METRICS = (
    "load_buffer_2d_calls",
    "load_buffer_2d_bytes",
    "load_buffer_2d_small_calls",
    "load_buffer_2d_inp_bytes",
    "load_buffer_2d_wgt_bytes",
    "store_buffer_2d_calls",
    "store_buffer_2d_bytes",
)


def aggregate_static_dma(units, selected, static_by_workload):
    totals = {key: 0 for key in METRICS}
    occurrences = []
    for item in selected:
        count = occurrence_count(item["workload"], units)
        if not count:
            continue
        label = workload_label(item["workload"])
        static = static_by_workload[canonical(item["workload"])]["static_dma"]["totals"]
        occurrences.append({"workload_id": label, "occurrences": count})
        for key in METRICS:
            totals[key] += count * int(static[key])
    totals["load_average_request_bytes"] = (
        float(totals["load_buffer_2d_bytes"]) / totals["load_buffer_2d_calls"]
        if totals["load_buffer_2d_calls"] else 0.0
    )
    return totals, occurrences


def observed_profile(measurements, units):
    key = "__".join(units)
    source = measurements[key]["incumbent_probe"]["runtime_profile"]
    result = {name: int(source[name]) for name in METRICS if name in source}
    result["load_average_request_bytes"] = (
        float(result["load_buffer_2d_bytes"]) / result["load_buffer_2d_calls"]
    )
    return result


def representative_by_topology(ranking):
    result = {}
    for row in ranking:
        result.setdefault(row["topology_id"], row)
    return result


def endpoint_boundary(row):
    output = next(item for item in row["boundaries"] if item["direction"] == "vta_to_cpu")
    return {
        "vta_to_cpu_bytes": int(output["logical_bytes"]),
        "vta_to_cpu_tensor_count": len(output["host_accounting_ids"]),
        "all_cpu_vta_boundary_bytes": sum(int(item["logical_bytes"]) for item in row["boundaries"]),
    }


def subtract(right, left, keys):
    return {key: right[key] - left[key] for key in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("vta/tutorials/frontend/report_out/stage_tile_cotuning")
    parser.add_argument(
        "--ranking",
        default="vta/tutorials/frontend/report_out/resource_aware_maxplus/v1_p5b_iteration2_ranked_candidates.json",
    )
    parser.add_argument("--selected-tasks", default=str(base / "iteration6_safe_overlay_final/selected_tasks.json"))
    parser.add_argument("--static-dma", default=str(base / "stage_memory_experiments/static_workload_dma.json"))
    parser.add_argument("--stage-measurements", default=str(base / "iteration6_safe_overlay_final/stage_measurements.json"))
    parser.add_argument("--output-dir", default=str(base / "stage_memory_experiments"))
    args = parser.parse_args()

    ranking = json.loads(Path(args.ranking).read_text())["rows"]
    selected = json.loads(Path(args.selected_tasks).read_text())
    static_rows = json.loads(Path(args.static_dma).read_text())["rows"]
    measurements = json.loads(Path(args.stage_measurements).read_text())
    static_by_workload = {canonical(row["workload"]): row for row in static_rows}
    representatives = representative_by_topology(ranking)

    endpoints = []
    for end, topology_id in ENDPOINTS.items():
        row = representatives[topology_id]
        vta_stage = next(stage for stage in row["scheme_cfg"] if stage["device"] == "vta")
        units = vta_stage["unit_names"]
        if len(units) != end - 3 + 1:
            raise RuntimeError("unexpected units for endpoint {}".format(end))
        static_totals, occurrences = aggregate_static_dma(units, selected, static_by_workload)
        endpoints.append({
            "start_unit_index": 3,
            "end_unit_index": end,
            "topology_id": topology_id,
            "tail_unit": units[-1],
            "unit_names": units,
            "workload_occurrences": occurrences,
            "static_conv_dma": static_totals,
            "observed_stage_dma": observed_profile(measurements, units),
            "boundary": endpoint_boundary(row),
        })

    deltas = []
    for left, right in zip(endpoints, endpoints[1:]):
        static_delta = subtract(right["static_conv_dma"], left["static_conv_dma"], METRICS)
        observed_delta = subtract(right["observed_stage_dma"], left["observed_stage_dma"], METRICS)
        boundary_delta = subtract(right["boundary"], left["boundary"], (
            "vta_to_cpu_bytes", "vta_to_cpu_tensor_count", "all_cpu_vta_boundary_bytes"
        ))
        before_occ = {item["workload_id"]: item["occurrences"] for item in left["workload_occurrences"]}
        after_occ = {item["workload_id"]: item["occurrences"] for item in right["workload_occurrences"]}
        added_workloads = {
            key: after_occ.get(key, 0) - before_occ.get(key, 0)
            for key in sorted(set(before_occ) | set(after_occ))
            if after_occ.get(key, 0) != before_occ.get(key, 0)
        }
        if static_delta["load_buffer_2d_bytes"] == 0 and boundary_delta["vta_to_cpu_bytes"] < 0:
            classification = "boundary_only_reduction_no_new_conv_dma"
        elif static_delta["load_buffer_2d_bytes"] > 0 and boundary_delta["vta_to_cpu_bytes"] < 0:
            classification = "vta_dma_vs_boundary_tradeoff"
        else:
            classification = "requires_full_resource_comparison"
        deltas.append({
            "transition": "03..{} -> 03..{}".format(left["end_unit_index"], right["end_unit_index"]),
            "added_unit": right["tail_unit"],
            "added_workloads": added_workloads,
            "static_conv_dma_delta": static_delta,
            "observed_stage_dma_delta": observed_delta,
            "boundary_delta": boundary_delta,
            "classification": classification,
        })

    payload = {
        "schema_version": 1,
        "kind": "vta_cut_endpoint_memory_delta",
        "numbering": "zero-based UNIT_ORDER indices",
        "scope": "fixed VTA start at unit 03; adjacent endpoints 15, 16, and 17",
        "endpoints": endpoints,
        "deltas": deltas,
        "search_rule": {
            "safe_use": (
                "Use deltas as a vector. If an endpoint extension reduces boundary bytes without "
                "increasing static/observed VTA DMA, retain it and prune the shorter endpoint only "
                "when compute, legality, and resource-balance terms are also no worse."
            ),
            "tradeoff_use": (
                "When both VTA DMA and boundary bytes change, keep both candidates until the CPU, "
                "single-VTA, longest-stage, and shared-memory resource bounds are compared."
            ),
            "double_count_warning": (
                "Do not add a DMA-derived time to measured VTA service time; its LOAD/STORE is already included."
            ),
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "cut_endpoint_delta.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Static VTA cut-endpoint delta: units 03..15/16/17",
        "",
        "Unit indices are zero-based. The VTA start is fixed at unit 03; only its right endpoint",
        "moves. Static convolution DMA comes from the selected TopHub tile and lowered TIR. The",
        "boundary column is the graph-level VTA-to-CPU tensor contract and is a separate cost.",
        "",
        "| VTA interval | tail unit | VTA->CPU boundary | tensors | static LOAD calls | static LOAD bytes | static WGT bytes | static STORE bytes | observed LOAD bytes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in endpoints:
        static = row["static_conv_dma"]
        observed = row["observed_stage_dma"]
        boundary = row["boundary"]
        lines.append(
            "| 03..{} | `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                row["end_unit_index"], row["tail_unit"], boundary["vta_to_cpu_bytes"],
                boundary["vta_to_cpu_tensor_count"], static["load_buffer_2d_calls"],
                static["load_buffer_2d_bytes"], static["load_buffer_2d_wgt_bytes"],
                static["store_buffer_2d_bytes"], observed["load_buffer_2d_bytes"],
            )
        )
    lines += [
        "",
        "| extension | added unit/workload | static LOAD delta | observed LOAD delta | STORE delta | VTA->CPU boundary delta | classification |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in deltas:
        workload = ", ".join("{} x{}".format(k, v) for k, v in row["added_workloads"].items()) or "none"
        lines.append(
            "| {} | `{}`; {} | {:+} calls / {:+} B | {:+} calls / {:+} B | {:+} B | {:+} B | `{}` |".format(
                row["transition"], row["added_unit"], workload,
                row["static_conv_dma_delta"]["load_buffer_2d_calls"],
                row["static_conv_dma_delta"]["load_buffer_2d_bytes"],
                row["observed_stage_dma_delta"]["load_buffer_2d_calls"],
                row["observed_stage_dma_delta"]["load_buffer_2d_bytes"],
                row["observed_stage_dma_delta"]["store_buffer_2d_bytes"],
                row["boundary_delta"]["vta_to_cpu_bytes"], row["classification"],
            )
        )
    lines += [
        "",
        "## Search interpretation",
        "",
        "- `03..15 -> 03..16` moves the layer4 projection to VTA. It adds tile-dependent",
        "  convolution DMA while reducing the boundary by one 100352-byte live tensor. This is",
        "  a genuine DMA-versus-shared-boundary tradeoff, not a scalar-byte dominance result.",
        "- `03..16 -> 03..17` moves the add-ReLU tail to VTA. It adds no convolution workload",
        "  and the observed VTA LOAD/STORE counts and bytes also remain unchanged, while the",
        "  boundary falls by another 100352 bytes. Under memory features the longer endpoint is",
        "  no worse; final pruning must still check CPU/VTA service and pipeline balance.",
        "- These deltas are available before deploying each complete topology: workload DMA is",
        "  extracted once from the incumbent tile, and boundary bytes come from the graph tensor",
        "  contract. Runtime profile is validation evidence, not a per-candidate requirement.",
    ]
    report_path = output / "CUT_ENDPOINT_DELTA.md"
    report_path.write_text("\n".join(lines) + "\n")

    hashes_path = output / "artifact_hashes.json"
    hashes = json.loads(hashes_path.read_text()) if hashes_path.exists() else {}
    hashes[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    hashes[json_path.name] = hashlib.sha256(json_path.read_bytes()).hexdigest()
    hashes[report_path.name] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    hashes_path.write_text(json.dumps(hashes, indent=2) + "\n")
    print("wrote", json_path)
    print("wrote", report_path)


if __name__ == "__main__":
    main()
