#!/usr/bin/env python3
"""Build the initial Experiment-D table from frozen topology replays."""

import argparse
import json
from pathlib import Path
import statistics


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def median(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(statistics.median(values)) if values else 0.0


def short_topology(text):
    return text.split(":", 1)[0]


def analyze_row(root, summary_row, warmup):
    rank = int(summary_row["representative_rank"])
    folder = root / "rank{:02d}".format(rank)
    manifest = read_json(folder / "manifest.json")
    rows = read_jsonl(folder / "native_result.jsonl")[warmup:]
    stages = sorted(manifest["stages"], key=lambda value: int(value["index"]))
    stage_rows = []
    for stage in stages:
        index = int(stage["index"])
        prefix = "stage{}".format(index)
        stage_rows.append({
            "index": index,
            "device": stage["device"],
            "units": stage["unit_names"],
            "median_total_ms": median(rows, prefix + "_ms"),
            "median_scheduled_ms": median(rows, prefix + "_scheduled_service_ms"),
            "median_set_ms": median(rows, prefix + "_set_ms"),
            "median_run_ms": median(rows, prefix + "_run_ms"),
            "median_get_ms": median(rows, prefix + "_get_ms"),
            "median_vta_mutex_wait_ms": median(rows, prefix + "_vta_mutex_wait_ms"),
        })

    interstage_api_ms = 0.0
    heterogeneous_api_ms = 0.0
    for left, right in zip(stage_rows, stage_rows[1:]):
        cost = left["median_get_ms"] + right["median_set_ms"]
        interstage_api_ms += cost
        if left["device"] != right["device"]:
            heterogeneous_api_ms += cost

    max_cpu = max(
        (stage["median_scheduled_ms"] for stage in stage_rows if stage["device"] == "cpu"),
        default=0.0,
    )
    vta_run_sum = sum(
        stage["median_run_ms"] for stage in stage_rows if stage["device"] == "vta"
    )
    longest = max(stage["median_scheduled_ms"] for stage in stage_rows)
    measured_cycle = 1000.0 / float(summary_row["pipeline_fps"])
    framework_bytes = (
        float(summary_row["mem_copy_from_host_bytes_per_frame"])
        + float(summary_row["mem_copy_to_host_bytes_per_frame"])
    )
    vta_dma_bytes = (
        float(summary_row["load_bytes_per_frame"])
        + float(summary_row["store_bytes_per_frame"])
    )
    resource_max_proxy = max(max_cpu, vta_run_sum, longest)
    return {
        "id": short_topology(summary_row["topology"]),
        "representative_rank": rank,
        "candidate_id": manifest["candidate_id"],
        "topology": summary_row["topology"],
        "vta_islands": int(summary_row["vta_islands"]),
        "pipeline_fps": float(summary_row["pipeline_fps"]),
        "measured_cycle_ms": measured_cycle,
        "stage_medians": stage_rows,
        "max_cpu_stage_scheduled_ms": max_cpu,
        "vta_run_sum_ms": vta_run_sum,
        "longest_stage_scheduled_ms": longest,
        "resource_max_proxy_ms": resource_max_proxy,
        "cycle_minus_resource_proxy_ms": measured_cycle - resource_max_proxy,
        "interstage_boundary_api_ms": interstage_api_ms,
        "heterogeneous_boundary_api_ms": heterogeneous_api_ms,
        "vta_mutex_wait_sum_ms": float(summary_row["vta_mutex_wait_median_sum_ms"]),
        "framework_copy_bytes_per_frame": framework_bytes,
        "vta_dma_bytes_per_frame": vta_dma_bytes,
        "combined_logical_memory_bytes_per_frame": framework_bytes + vta_dma_bytes,
        "load_calls_per_frame": float(summary_row["load_calls_per_frame"]),
        "load_bytes_per_frame": float(summary_row["load_bytes_per_frame"]),
        "weight_load_bytes_per_frame": float(summary_row["weight_load_bytes_per_frame"]),
        "store_bytes_per_frame": float(summary_row["store_bytes_per_frame"]),
        "mem_copy_from_host_bytes_per_frame":
            float(summary_row["mem_copy_from_host_bytes_per_frame"]),
        "mem_copy_to_host_bytes_per_frame":
            float(summary_row["mem_copy_to_host_bytes_per_frame"]),
    }


def percent_change(new, old):
    return (float(new) / float(old) - 1.0) * 100.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = "vta/tutorials/frontend/report_out/stage_tile_cotuning"
    parser.add_argument("--input-dir", default=base + "/legacy_topology_profiles")
    parser.add_argument("--output-dir", default=base + "/stage_memory_experiments")
    args = parser.parse_args()
    root = Path(args.input_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = read_json(root / "summary.json")
    warmup = int(source["protocol"]["warmup_frames"])
    rows = [analyze_row(root, row, warmup) for row in source["rows"]]
    by_id = {row["id"]: row for row in rows}

    contrasts = {
        "a_vs_c_same_dma": {
            "load_calls_equal": by_id["A"]["load_calls_per_frame"]
            == by_id["C"]["load_calls_per_frame"],
            "load_bytes_equal": by_id["A"]["load_bytes_per_frame"]
            == by_id["C"]["load_bytes_per_frame"],
            "store_bytes_equal": by_id["A"]["store_bytes_per_frame"]
            == by_id["C"]["store_bytes_per_frame"],
            "framework_copy_delta_bytes": by_id["C"]["framework_copy_bytes_per_frame"]
            - by_id["A"]["framework_copy_bytes_per_frame"],
            "cycle_delta_ms": by_id["C"]["measured_cycle_ms"]
            - by_id["A"]["measured_cycle_ms"],
        },
        "b_vs_d_extra_island": {
            "island_delta": by_id["D"]["vta_islands"] - by_id["B"]["vta_islands"],
            "framework_copy_bytes_percent": percent_change(
                by_id["D"]["framework_copy_bytes_per_frame"],
                by_id["B"]["framework_copy_bytes_per_frame"],
            ),
            "load_calls_percent": percent_change(
                by_id["D"]["load_calls_per_frame"], by_id["B"]["load_calls_per_frame"]
            ),
            "load_bytes_percent": percent_change(
                by_id["D"]["load_bytes_per_frame"], by_id["B"]["load_bytes_per_frame"]
            ),
            "weight_load_bytes_percent": percent_change(
                by_id["D"]["weight_load_bytes_per_frame"],
                by_id["B"]["weight_load_bytes_per_frame"],
            ),
            "mutex_wait_delta_ms": by_id["D"]["vta_mutex_wait_sum_ms"]
            - by_id["B"]["vta_mutex_wait_sum_ms"],
            "cycle_delta_ms": by_id["D"]["measured_cycle_ms"]
            - by_id["B"]["measured_cycle_ms"],
        },
    }
    payload = {
        "schema_version": 1,
        "kind": "experiment_d_initial_shared_ddr_topology_table",
        "source": str(root / "summary.json"),
        "sample_scope": "four topology representatives, one boot; mechanism evidence only",
        "rows": rows,
        "controlled_contrasts": contrasts,
        "model_decision": {
            "per_workload_dma": "aggregate statically after TopHub config selection",
            "block_aligned_stage_dma_correction": 0,
            "separate_outer_features": [
                "framework boundary bytes and copy mode",
                "VTA island count and re-entry bytes",
                "single physical VTA service",
                "mutex wait diagnostic",
                "maximum CPU stage service",
                "VTA LOAD calls, total bytes, and weight bytes",
            ],
            "fit_status": "not_fit",
            "fit_reason": "n=4 topology representatives is insufficient and confounded",
        },
    }
    (output / "experiment_d_initial.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Experiment D: initial shared-DDR topology table",
        "",
        "This table re-analyzes the four frozen Top-20 topology representatives. It is a one-boot",
        "mechanism study, not a fitted contention model or a cross-boot significance claim.",
        "Per-stage values are medians over frames, while cycle uses the first/last completion span;",
        "their maximum is a diagnostic proxy and is not treated as a strict finite-sample lower bound.",
        "",
        "| topology | islands | FPS | cycle | max CPU stage | VTA run sum | framework copy | VTA DMA | mutex wait |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {id} | {vta_islands} | {pipeline_fps:.3f} | {measured_cycle_ms:.3f} ms | "
            "{max_cpu_stage_scheduled_ms:.3f} ms | {vta_run_sum_ms:.3f} ms | "
            "{copy:.3f} MB | {dma:.3f} MB | {vta_mutex_wait_sum_ms:.3f} ms |".format(
                copy=row["framework_copy_bytes_per_frame"] / 1e6,
                dma=row["vta_dma_bytes_per_frame"] / 1e6,
                **row
            )
        )
    ac = contrasts["a_vs_c_same_dma"]
    bd = contrasts["b_vs_d_extra_island"]
    lines += [
        "",
        "## Controlled contrasts",
        "",
        "- A and C have exactly equal VTA LOAD calls, LOAD bytes, and STORE bytes. C nevertheless",
        "  adds {:+.0f} framework-copy bytes and changes cycle by {:+.3f} ms. Their different CPU tail".format(
            ac["framework_copy_delta_bytes"], ac["cycle_delta_ms"]
        ),
        "  placement means this pair proves DMA invariance, not a per-byte latency coefficient.",
        "- D versus B adds one VTA island: framework-copy bytes change {:+.2f}%, LOAD calls {:+.2f}%,".format(
            bd["framework_copy_bytes_percent"], bd["load_calls_percent"]
        ),
        "  LOAD bytes {:+.2f}%, and weight LOAD bytes {:+.2f}%. Mutex wait rises {:+.3f} ms, while".format(
            bd["load_bytes_percent"], bd["weight_load_bytes_percent"], bd["mutex_wait_delta_ms"]
        ),
        "  measured cycle changes {:+.3f} ms. The improved CPU partition can hide or outweigh higher".format(
            bd["cycle_delta_ms"]
        ),
        "  memory demand, so a scalar 'more bytes = lower FPS' rule is invalid.",
        "",
        "## Model decision",
        "",
        "The outer search must use a resource maximum, not add every cost to the critical path.",
        "Per-workload DMA is statically aggregated after tile selection. Boundary bytes/copy mode,",
        "VTA re-entry, single-VTA service, maximum CPU-stage service, and DMA calls/weight payload",
        "remain separate features. No coefficients are fit to these four confounded samples.",
        "Cross-boot repetitions and a larger calibration/holdout set are still required before M2",
        "may claim better ranking accuracy.",
    ]
    (output / "EXPERIMENT_D_INITIAL.md").write_text("\n".join(lines) + "\n")
    print("wrote {} topology rows".format(len(rows)))


if __name__ == "__main__":
    main()
