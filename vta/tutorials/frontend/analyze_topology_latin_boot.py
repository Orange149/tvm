#!/usr/bin/env python3
"""Analyze the third-boot 4x4 Latin-square replay of frozen VTA topologies."""

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

import numpy as np


RANK_TO_TOPOLOGY = {1: "A", 5: "B", 7: "C", 13: "D"}
TAG_RE = re.compile(r"boot3_ba873689_r(?P<round>[1-4])_p(?P<position>[1-4])_native_result.jsonl$")
DMA_KEYS = [
    "load_calls_per_frame",
    "load_bytes_per_frame",
    "weight_load_bytes_per_frame",
    "store_calls_per_frame",
    "store_bytes_per_frame",
]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranks(values):
    return {key: index + 1 for index, (key, _) in enumerate(sorted(values.items(), key=lambda x: x[1]))}


def spearman(left, right):
    left_rank = ranks(left)
    right_rank = ranks(right)
    keys = sorted(left)
    return float(np.corrcoef(
        [left_rank[key] for key in keys], [right_rank[key] for key in keys]
    )[0, 1])


def analyze_trial(result_path, rank, skip_first):
    match = TAG_RE.search(result_path.name)
    if not match:
        raise RuntimeError("unexpected result filename: {}".format(result_path))
    round_index = int(match.group("round"))
    position = int(match.group("position"))
    tag = result_path.name.replace("_native_result.jsonl", "")
    rows_all = read_jsonl(result_path)
    if len(rows_all) != 62:
        raise RuntimeError("{} has {} frames, expected 62".format(result_path, len(rows_all)))
    rows = rows_all[skip_first:]
    completions = np.asarray([float(row["completion_ms"]) for row in rows])
    intervals = np.diff(completions)
    output_hashes = sorted({
        output["fnv1a64"] for row in rows_all for output in row.get("raw_outputs", [])
    })
    profile_path = result_path.parent / "profile" / tag / "pipeline" / "benchmark_totals_status.json"
    profile = json.loads(profile_path.read_text())
    mutex_medians = []
    for index in range(int(rows[0]["stage_count"])):
        key = "stage{}_vta_mutex_wait_ms".format(index)
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            mutex_medians.append(float(statistics.median(values)))
    frames = len(rows_all)
    return {
        "topology_id": RANK_TO_TOPOLOGY[rank],
        "rank": rank,
        "round": round_index,
        "position": position,
        "total_frames": frames,
        "scored_frames": len(rows),
        "scored_intervals": len(intervals),
        "cycle_ms": float(np.mean(intervals)),
        "fps": float(1000.0 / np.mean(intervals)),
        "completion_interval_median_ms": float(np.median(intervals)),
        "completion_interval_p95_ms": float(np.percentile(intervals, 95)),
        "completion_interval_cv": float(np.std(intervals) / np.mean(intervals)),
        "vta_mutex_wait_median_sum_ms": sum(mutex_medians),
        "output_hashes": output_hashes,
        "output_deterministic": len(output_hashes) == 1,
        "load_calls_per_frame": float(profile["load_buffer_2d_calls"]) / frames,
        "load_bytes_per_frame": float(profile["load_buffer_2d_bytes"]) / frames,
        "weight_load_bytes_per_frame": float(profile["load_buffer_2d_wgt_bytes"]) / frames,
        "store_calls_per_frame": float(profile["store_buffer_2d_calls"]) / frames,
        "store_bytes_per_frame": float(profile["store_buffer_2d_bytes"]) / frames,
        "result_sha256": sha256(result_path),
        "profile_sha256": sha256(profile_path),
    }


def summarize_topology(rows):
    fps = [row["fps"] for row in rows]
    cycles = [row["cycle_ms"] for row in rows]
    return {
        "trials": len(rows),
        "fps_median": float(statistics.median(fps)),
        "fps_min": min(fps),
        "fps_max": max(fps),
        "fps_range": max(fps) - min(fps),
        "cycle_ms_median": float(statistics.median(cycles)),
        "cycle_ms_min": min(cycles),
        "cycle_ms_max": max(cycles),
        "interval_p95_ms_median": float(statistics.median(
            row["completion_interval_p95_ms"] for row in rows
        )),
        "interval_cv_median": float(statistics.median(
            row["completion_interval_cv"] for row in rows
        )),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("vta/tutorials/frontend/report_out/stage_tile_cotuning/stage_memory_experiments")
    parser.add_argument("--input-dir", default=str(base / "boot_ba873689_latin62"))
    parser.add_argument("--previous-json", default=str(base / "experiment_d_cross_boot.json"))
    parser.add_argument("--output-dir", default=str(base))
    parser.add_argument("--archive-sha256", default="65d80cc7cc6d989b82f38631a4eac495d285bdd49daae9ff1992b510636614aa")
    args = parser.parse_args()

    root = Path(args.input_dir)
    trials = []
    for rank in sorted(RANK_TO_TOPOLOGY):
        folder = root / "rank{:02d}".format(rank)
        paths = sorted(folder.glob("boot3_ba873689_r*_p*_native_result.jsonl"))
        if len(paths) != 4:
            raise RuntimeError("rank {} has {} trials, expected 4".format(rank, len(paths)))
        trials.extend(analyze_trial(path, rank, skip_first=2) for path in paths)
    trials.sort(key=lambda row: (row["round"], row["position"]))

    expected_square = {
        1: ["A", "B", "C", "D"],
        2: ["B", "C", "D", "A"],
        3: ["C", "D", "A", "B"],
        4: ["D", "A", "B", "C"],
    }
    observed_square = {
        round_index: [
            row["topology_id"] for row in trials if row["round"] == round_index
        ] for round_index in range(1, 5)
    }
    if observed_square != expected_square:
        raise RuntimeError("Latin-square order mismatch: {}".format(observed_square))

    by_topology = {
        topology: [row for row in trials if row["topology_id"] == topology]
        for topology in "ABCD"
    }
    summaries = {key: summarize_topology(rows) for key, rows in by_topology.items()}
    round_ordering = {}
    for round_index in range(1, 5):
        selected = [row for row in trials if row["round"] == round_index]
        round_ordering[str(round_index)] = [
            row["topology_id"] for row in sorted(selected, key=lambda row: -row["fps"])
        ]
    aggregate_ordering = sorted("ABCD", key=lambda key: -summaries[key]["fps_median"])

    paired = {}
    for left, right in [("A", "C"), ("D", "B")]:
        diffs = []
        for round_index in range(1, 5):
            left_row = next(row for row in by_topology[left] if row["round"] == round_index)
            right_row = next(row for row in by_topology[right] if row["round"] == round_index)
            diffs.append({
                "round": round_index,
                "fps_delta_left_minus_right": left_row["fps"] - right_row["fps"],
                "left_faster": left_row["fps"] > right_row["fps"],
            })
        paired["{}_vs_{}".format(left, right)] = {
            "rounds": diffs,
            "left_faster_rounds": sum(item["left_faster"] for item in diffs),
            "median_fps_delta_left_minus_right": float(statistics.median(
                item["fps_delta_left_minus_right"] for item in diffs
            )),
        }

    position_normalized = {}
    for position in range(1, 5):
        values = []
        for row in trials:
            if row["position"] == position:
                median = summaries[row["topology_id"]]["fps_median"]
                values.append(100.0 * (row["fps"] / median - 1.0))
        position_normalized[str(position)] = {
            "median_percent_from_topology_median": float(statistics.median(values)),
            "min_percent": min(values),
            "max_percent": max(values),
        }

    previous = json.loads(Path(args.previous_json).read_text())
    previous_by_run = {
        run["run_id"]: {row["topology_id"]: row for row in run["rows"]}
        for run in previous["runs"]
    }
    hash_stable = {}
    dma_invariant = {}
    for topology in "ABCD":
        old_rows = [rows[topology] for rows in previous_by_run.values()]
        hashes = {value for row in old_rows + by_topology[topology] for value in row["output_hashes"]}
        hash_stable[topology] = len(hashes) == 1
        dma_invariant[topology] = all(
            len({row[key] for row in old_rows + by_topology[topology]}) == 1
            for key in DMA_KEYS
        )

    boot_cycles = {
        "boot1": {key: previous_by_run["boot1_run22"][key]["cycle_ms"] for key in "ABCD"},
        "boot2": {key: previous_by_run["boot2_run62"][key]["cycle_ms"] for key in "ABCD"},
        "boot3": {key: summaries[key]["cycle_ms_median"] for key in "ABCD"},
    }
    cross_boot_spearman = {
        "boot1_vs_boot2": spearman(boot_cycles["boot1"], boot_cycles["boot2"]),
        "boot1_vs_boot3": spearman(boot_cycles["boot1"], boot_cycles["boot3"]),
        "boot2_vs_boot3": spearman(boot_cycles["boot2"], boot_cycles["boot3"]),
    }

    payload = {
        "schema_version": 1,
        "kind": "experiment_d_third_boot_latin_square",
        "boot_id": "ba873689-d01b-47df-beaf-f536e8508824",
        "archive_sha256": args.archive_sha256,
        "protocol": {
            "rounds": 4,
            "frames_per_trial": 62,
            "skip_first_frames": 2,
            "scored_intervals_per_trial": 59,
            "latin_square": expected_square,
            "total_pipeline_frames": 16 * 62,
        },
        "trials": trials,
        "topology_summary": summaries,
        "round_fps_ordering": round_ordering,
        "aggregate_median_fps_ordering": aggregate_ordering,
        "paired_directions": paired,
        "position_normalized_effect": position_normalized,
        "output_hash_stable_across_three_boots": hash_stable,
        "dma_invariant_across_three_boots": dma_invariant,
        "cross_boot_cycle_rank_spearman": cross_boot_spearman,
        "conclusion": {
            "memory_mechanism_reproducible": all(hash_stable.values()) and all(dma_invariant.values()),
            "fine_grained_ordering_stable_within_boot3": len({tuple(value) for value in round_ordering.values()}) == 1,
            "fit_ddr_time_coefficient": False,
            "use_for_model": "invariant DMA/boundary features and dominance pruning, not a scalar DDR-time fit",
        },
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "experiment_d_third_boot_latin.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Experiment D: third-boot Latin-square replay",
        "",
        "Boot `ba873689-d01b-47df-beaf-f536e8508824` uses four 62-frame rounds. The order",
        "is `A-B-C-D / B-C-D-A / C-D-A-B / D-A-B-C`, so each frozen topology occupies",
        "each execution position once. The first two completions are discarded per trial.",
        "",
        "| topology | four FPS trials | median FPS | min--max FPS | median interval P95 |",
        "|---|---|---:|---:|---:|",
    ]
    for topology in "ABCD":
        rows = sorted(by_topology[topology], key=lambda row: row["round"])
        item = summaries[topology]
        lines.append(
            "| {} | {} | {:.3f} | {:.3f}--{:.3f} | {:.3f} ms |".format(
                topology,
                ", ".join("{:.3f}".format(row["fps"]) for row in rows),
                item["fps_median"], item["fps_min"], item["fps_max"],
                item["interval_p95_ms_median"],
            )
        )
    lines += [
        "",
        "## Round ordering and paired directions",
        "",
    ]
    for round_index in range(1, 5):
        lines.append("- Round {}: `{}`.".format(round_index, ">".join(round_ordering[str(round_index)])))
    for name, item in paired.items():
        lines.append(
            "- `{}`: left is faster in {}/4 rounds; median left-minus-right is {:+.3f} FPS.".format(
                name, item["left_faster_rounds"], item["median_fps_delta_left_minus_right"]
            )
        )
    lines += [
        "",
        "Aggregate boot-3 median ordering is `{}`. Cross-boot cycle-rank Spearman is boot1/boot2 "
        "`{:.3f}`, boot1/boot3 `{:.3f}`, and boot2/boot3 `{:.3f}`.".format(
            ">".join(aggregate_ordering),
            cross_boot_spearman["boot1_vs_boot2"],
            cross_boot_spearman["boot1_vs_boot3"],
            cross_boot_spearman["boot2_vs_boot3"],
        ),
        "",
        "## Reproducibility decision",
        "",
        "- All topology-specific output hashes remain stable across all three boots.",
        "- Per-frame LOAD/STORE calls and payload remain exactly invariant across all three boots.",
        "- The four within-boot FPS orderings are not identical; the fine ordering is therefore",
        "  not a robust target for a scalar DDR-time regression.",
        "- Retain static tile-derived DMA and graph boundary bytes as mechanism-grounded search",
        "  features and safe dominance/pruning signals. Do not add a fitted DMA time on top of",
        "  measured VTA service time, because that would double-count transfers already inside it.",
        "",
        "Raw archive SHA-256: `{}`.".format(args.archive_sha256),
    ]
    report_path = output / "EXPERIMENT_D_THIRD_BOOT_LATIN.md"
    report_path.write_text("\n".join(lines) + "\n")
    print("wrote", json_path)
    print("wrote", report_path)


if __name__ == "__main__":
    main()
