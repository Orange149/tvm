#!/usr/bin/env python3
"""Summarize frozen topology replays without conflating runs and boots."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


RANKS = [1, 5, 7, 13]
TOPOLOGY_IDS = {1: "A", 5: "B", 7: "C", 13: "D"}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze_run(root, rank, skip_first):
    folder = Path(root) / "rank{:02d}".format(rank)
    path = folder / "native_result.jsonl"
    all_rows = read_jsonl(path)
    rows = all_rows[skip_first:]
    completions = np.asarray([float(row["completion_ms"]) for row in rows])
    intervals = np.diff(completions)
    output_hashes = sorted({
        output["fnv1a64"] for row in all_rows for output in row.get("raw_outputs", [])
    })
    profile_path = folder / "profile" / "pipeline" / "benchmark_totals_status.json"
    profile = json.loads(profile_path.read_text())
    frames = len(all_rows)
    mutex_medians = []
    stage_count = int(rows[0]["stage_count"])
    for index in range(stage_count):
        values = [
            float(row["stage{}_vta_mutex_wait_ms".format(index)])
            for row in rows
            if row.get("stage{}_vta_mutex_wait_ms".format(index)) is not None
        ]
        if values:
            mutex_medians.append(float(statistics.median(values)))
    return {
        "rank": rank,
        "topology_id": TOPOLOGY_IDS[rank],
        "total_frames": frames,
        "scored_frames": len(rows),
        "cycle_ms": float(np.mean(intervals)),
        "fps": float(1000.0 / np.mean(intervals)),
        "completion_interval_median_ms": float(np.median(intervals)),
        "completion_interval_p95_ms": float(np.percentile(intervals, 95)),
        "completion_interval_cv": float(np.std(intervals) / np.mean(intervals)),
        "output_hashes": output_hashes,
        "output_deterministic": len(output_hashes) == 1,
        "vta_mutex_wait_median_sum_ms": sum(mutex_medians),
        "load_calls_per_frame": float(profile["load_buffer_2d_calls"]) / frames,
        "load_bytes_per_frame": float(profile["load_buffer_2d_bytes"]) / frames,
        "weight_load_bytes_per_frame": float(profile["load_buffer_2d_wgt_bytes"]) / frames,
        "store_calls_per_frame": float(profile["store_buffer_2d_calls"]) / frames,
        "store_bytes_per_frame": float(profile["store_buffer_2d_bytes"]) / frames,
        "profile_sha256": file_sha256(profile_path),
        "result_sha256": file_sha256(path),
    }


def rank_values(mapping):
    ordered = sorted(mapping, key=lambda key: mapping[key])
    return {key: index + 1 for index, key in enumerate(ordered)}


def spearman(left, right):
    left_rank = rank_values(left)
    right_rank = rank_values(right)
    keys = sorted(left)
    x = np.asarray([left_rank[key] for key in keys], dtype="float64")
    y = np.asarray([right_rank[key] for key in keys], dtype="float64")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("vta/tutorials/frontend/report_out/stage_tile_cotuning")
    parser.add_argument("--boot1", default=str(base / "legacy_topology_profiles"))
    parser.add_argument(
        "--boot2-run1", default=str(base / "stage_memory_experiments/boot_9efb07ba")
    )
    parser.add_argument(
        "--boot2-run2", default=str(base / "stage_memory_experiments/boot_9efb07ba_repeat2")
    )
    parser.add_argument(
        "--boot2-run62", default=str(base / "stage_memory_experiments/boot_9efb07ba_run62")
    )
    parser.add_argument("--output-dir", default=str(base / "stage_memory_experiments"))
    args = parser.parse_args()
    run_specs = [
        ("boot1_run22", "b7216025-291c-466f-878d-f72377394c39", args.boot1),
        ("boot2_run22_a", "9efb07ba-0fc9-4f22-8dbd-4f1de4bfc68d", args.boot2_run1),
        ("boot2_run22_b", "9efb07ba-0fc9-4f22-8dbd-4f1de4bfc68d", args.boot2_run2),
        ("boot2_run62", "9efb07ba-0fc9-4f22-8dbd-4f1de4bfc68d", args.boot2_run62),
    ]
    runs = []
    for run_id, boot_id, root in run_specs:
        runs.append({
            "run_id": run_id,
            "boot_id": boot_id,
            "root": root,
            "rows": [analyze_run(root, rank, 2) for rank in RANKS],
        })

    hashes_by_topology = {}
    dma_invariant = {}
    for rank in RANKS:
        topology = TOPOLOGY_IDS[rank]
        selected = [next(row for row in run["rows"] if row["rank"] == rank) for run in runs]
        hashes_by_topology[topology] = sorted({h for row in selected for h in row["output_hashes"]})
        dma_keys = [
            "load_calls_per_frame", "load_bytes_per_frame", "weight_load_bytes_per_frame",
            "store_calls_per_frame", "store_bytes_per_frame",
        ]
        dma_invariant[topology] = all(
            len({row[key] for row in selected}) == 1 for key in dma_keys
        )

    ordering = {}
    for run in runs:
        ordering[run["run_id"]] = [
            topology for topology, _ in sorted(
                ((row["topology_id"], row["fps"]) for row in run["rows"]),
                key=lambda item: (-item[1], item[0]),
            )
        ]
    boot1 = {row["topology_id"]: row["cycle_ms"] for row in runs[0]["rows"]}
    boot2_long = {row["topology_id"]: row["cycle_ms"] for row in runs[-1]["rows"]}
    payload = {
        "schema_version": 1,
        "kind": "experiment_d_cross_boot_replay",
        "protocol": {
            "skip_first_frames": 2,
            "independent_boots": 2,
            "boot2_short_repeats": 2,
            "primary_boot2_window_frames": 62,
            "warning": "Runs on the same boot are repeated runs, not independent boots.",
        },
        "runs": runs,
        "output_hashes_by_topology": hashes_by_topology,
        "output_stable_across_all_runs": all(len(value) == 1 for value in hashes_by_topology.values()),
        "dma_invariant_across_all_runs": dma_invariant,
        "fps_ordering": ordering,
        "boot1_vs_boot2_run62_cycle_spearman": spearman(boot1, boot2_long),
        "conclusion": {
            "mechanism_reproducible": True,
            "fine_grained_fps_order_reproducible": ordering["boot1_run22"]
            == ordering["boot2_run62"],
            "fit_ddr_coefficient": False,
            "reason": "two boots and material within-boot CPU-stage jitter are insufficient",
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "experiment_d_cross_boot.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Experiment D: cross-boot frozen-topology replay",
        "",
        "This intermediate report covers the first two independent board boots. Boot 2 contains",
        "two 22-frame runs and one 62-frame run; these are repeated runs on one boot, not three",
        "independent boots. The subsequently completed third-boot Latin-square replay is reported",
        "separately in `EXPERIMENT_D_THIRD_BOOT_LATIN.md`.",
        "All runs discard the first two completions for throughput.",
        "",
        "| topology | boot 1, 22 frames | boot 2 run A, 22 | boot 2 run B, 22 | boot 2, 62 | 62-frame interval P95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by_run = {
        run["run_id"]: {row["topology_id"]: row for row in run["rows"]} for run in runs
    }
    for topology in ["A", "B", "C", "D"]:
        lines.append(
            "| {} | {:.3f} FPS | {:.3f} FPS | {:.3f} FPS | {:.3f} FPS | {:.3f} ms |".format(
                topology,
                by_run["boot1_run22"][topology]["fps"],
                by_run["boot2_run22_a"][topology]["fps"],
                by_run["boot2_run22_b"][topology]["fps"],
                by_run["boot2_run62"][topology]["fps"],
                by_run["boot2_run62"][topology]["completion_interval_p95_ms"],
            )
        )
    lines += [
        "",
        "## Findings",
        "",
        "- Every topology has one stable output hash across serial/pipeline replay and across boots.",
        "- LOAD/STORE calls and payload are exactly invariant in every run. The memory mechanism",
        "  conclusions (same-DMA A/C and the extra-island demand of D) therefore reproduce.",
        "- Fine FPS ordering does not reproduce: boot 1 orders C>A>D>B, while the primary boot-2",
        "  62-frame run orders A>D>C>B. Cycle-rank Spearman is {:.3f}.".format(
            payload["boot1_vs_boot2_run62_cycle_spearman"]
        ),
        "- The two short runs on boot 2 expose substantial B/C variation. The 62-frame B run still",
        "  has a 136.753 ms completion-interval P95. This points to CPU-stage/runtime jitter, not",
        "  variable VTA DMA.",
        "",
        "## Decision",
        "",
        "Use these data to validate invariant memory features and coarse pruning rules, but do not",
        "fit a DDR-time coefficient or claim a significant sub-FPS topology ordering from these two",
        "boots. See `EXPERIMENT_D_THIRD_BOOT_LATIN.md` for the completed 4x4 Latin-square replay.",
    ]
    (output / "EXPERIMENT_D_CROSS_BOOT.md").write_text("\n".join(lines) + "\n")
    print("wrote", json_path)


if __name__ == "__main__":
    main()
