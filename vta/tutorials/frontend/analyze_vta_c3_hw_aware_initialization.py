#!/usr/bin/env python3
"""Evaluate the four frozen HW-Aware initialization levels on full R18 spaces."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import run_vta_c3_literature_baselines as baseline


SCHEMA = "c3_hw_aware_initialization_analysis_v1"
SEEDS = tuple(range(57001, 57021))
LEVELS = baseline.RIEBER_LEVELS


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def verify(directory):
    directory = Path(directory).resolve()
    ledger = read_json(directory / "artifact_hashes.json")["artifacts"]
    for name, expected in ledger.items():
        if baseline.sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: {}".format(directory / name))
    return ledger


def first_valid_position(order, labels, target=25):
    count = 0
    for position, candidate_id in enumerate(order, 1):
        count += int(labels[candidate_id]["is_valid"])
        if count == target:
            return position
    return None


def cost_for(order, labels, count=None):
    chosen = order if count is None else order[:count]
    return sum(float(labels[candidate_id]["wall_seconds"]) for candidate_id in chosen)


def metrics_for_e0(level, seed, e0, discovery_order, labels, presample_count, presample_wall, post=None):
    valid = sum(labels[row["candidate_id"]]["is_valid"] for row in e0)
    result = {
        "schema": SCHEMA,
        "level": level,
        "seed": seed,
        "e0_size": len(e0),
        "e0_valid": valid,
        "e0_invalid": len(e0) - valid,
        "e0_valid_yield": valid / len(e0),
        "e0_invalid_ratio": (len(e0) - valid) / len(e0),
        "compiler_calls_to_first_25_valid": first_valid_position(discovery_order, labels),
        "compiler_wall_to_first_25_valid_seconds": None,
        "presampling_compiler_calls": presample_count,
        "presampling_wall_seconds": presample_wall,
    }
    position = result["compiler_calls_to_first_25_valid"]
    if position is not None:
        result["compiler_wall_to_first_25_valid_seconds"] = cost_for(discovery_order, labels, position)
    if post is not None:
        post_valid = sum(labels[row["candidate_id"]]["is_valid"] for row in post)
        result.update(
            validity_biased_next_count=len(post),
            validity_biased_next_valid=post_valid,
            validity_biased_next_valid_yield=post_valid / len(post) if post else None,
            validity_biased_next_wall_seconds=cost_for([row["candidate_id"] for row in post], labels),
        )
    return result


def one_seed(candidates, labels, workload_id, seed):
    random_order = sorted(
        (row["candidate_id"] for row in candidates),
        key=lambda candidate_id: baseline.stable_rank(seed, workload_id, candidate_id, "rieber-random-e0"),
    )
    by_id = {row["candidate_id"]: row for row in candidates}
    random_e0 = [by_id[candidate_id] for candidate_id in random_order[:50]]
    rows = [
        metrics_for_e0(
            "random_initialization", seed, random_e0, random_order, labels, 50,
            cost_for(random_order, labels, 50),
        )
    ]

    revealed = []

    def callback(candidate_id):
        revealed.append(candidate_id)
        return labels[candidate_id]["is_valid"]

    presample_order = baseline.rieber_presampling_order(
        candidates, seed, callback, limit=min(1000, len(candidates)), parallel=8
    )
    if presample_order != revealed or len(presample_order) != min(1000, len(candidates)):
        raise AssertionError("presampling feedback/order contract failed")
    presample = [
        {**by_id[candidate_id], "is_valid": bool(labels[candidate_id]["is_valid"])}
        for candidate_id in presample_order
    ]
    presample_wall = cost_for(presample_order, labels)
    neighbour_e0 = [by_id[candidate_id] for candidate_id in presample_order[:50]]
    rows.append(
        metrics_for_e0(
            "neighbor_presampling", seed, neighbour_e0, presample_order, labels,
            len(presample_order), presample_wall,
        )
    )
    balanced = baseline.balanced_e0(presample, per_class=25)
    if len(balanced) != 50 or sum(row["is_valid"] for row in balanced) != 25:
        raise AssertionError("balanced E0 25+25 contract failed")
    balanced_candidates = [by_id[row["candidate_id"]] for row in balanced]
    rows.append(
        metrics_for_e0(
            "balanced_e0", seed, balanced_candidates, presample_order, labels,
            len(presample_order), presample_wall,
        )
    )
    presampled_ids = set(presample_order)
    outside = [row for row in candidates if row["candidate_id"] not in presampled_ids]
    ranked = baseline.select_model_rank(
        outside, [], presample, seed, baseline.visible_vector, validity=True
    )
    post = [row for row, _ in ranked[: min(50, len(ranked))]]
    rows.append(
        metrics_for_e0(
            "balanced_e0_validity_bias", seed, balanced_candidates, presample_order, labels,
            len(presample_order), presample_wall, post=post,
        )
    )
    return rows


def distribution(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return {
        "n": len(clean),
        "median": statistics.median(clean),
        "q1": float(np.percentile(clean, 25)),
        "q3": float(np.percentile(clean, 75)),
        "iqr": float(np.percentile(clean, 75) - np.percentile(clean, 25)),
        "minimum": min(clean),
        "maximum": max(clean),
    }


def sign_test(left, right):
    differences = [float(a) - float(b) for a, b in zip(left, right) if a is not None and b is not None and a != b]
    if not differences:
        return {"non_ties": 0, "left_greater": 0, "two_sided_p": 1.0}
    wins = sum(value > 0 for value in differences)
    n = len(differences)
    tail = sum(math.comb(n, index) for index in range(0, min(wins, n - wins) + 1)) / (2 ** n)
    return {"non_ties": n, "left_greater": wins, "two_sided_p": min(1.0, 2 * tail)}


def aggregate(rows):
    result = {}
    for workload_id in sorted({row["workload_id"] for row in rows}):
        result[workload_id] = {}
        workload_rows = [row for row in rows if row["workload_id"] == workload_id]
        random_by_seed = {row["seed"]: row for row in workload_rows if row["level"] == "random_initialization"}
        for level in LEVELS:
            selected = sorted((row for row in workload_rows if row["level"] == level), key=lambda row: row["seed"])
            random_rows = [random_by_seed[row["seed"]] for row in selected]
            item = {
                "e0_valid_yield": distribution([row["e0_valid_yield"] for row in selected]),
                "e0_invalid_ratio": distribution([row["e0_invalid_ratio"] for row in selected]),
                "compiler_calls_to_first_25_valid": distribution([row["compiler_calls_to_first_25_valid"] for row in selected]),
                "compiler_wall_to_first_25_valid_seconds": distribution([row["compiler_wall_to_first_25_valid_seconds"] for row in selected]),
                "presampling_compiler_calls": distribution([row["presampling_compiler_calls"] for row in selected]),
                "presampling_wall_seconds": distribution([row["presampling_wall_seconds"] for row in selected]),
                "e0_yield_vs_random_sign_test": sign_test(
                    [row["e0_valid_yield"] for row in selected],
                    [row["e0_valid_yield"] for row in random_rows],
                ),
            }
            if level == "balanced_e0_validity_bias":
                item["validity_biased_next_valid_yield"] = distribution([row["validity_biased_next_valid_yield"] for row in selected])
                item["validity_biased_next_wall_seconds"] = distribution([row["validity_biased_next_wall_seconds"] for row in selected])
            result[workload_id][level] = item
    return result


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    candidate_dir = Path(args.candidate_dir).resolve()
    verify(candidate_dir)
    scan_dirs = [Path(value).resolve() for value in args.scan_dir]
    scan_by_workload = {}
    for directory in scan_dirs:
        verify(directory)
        summary = read_json(directory / "summary.json")
        scan_by_workload[summary["workload_id"]] = directory
    if set(scan_by_workload) != {"R18-H1", "R18-H2", "R18-H3"}:
        raise ValueError("all three complete validity scans are required")
    output.mkdir(parents=True)
    results_path = output / "results.jsonl"
    results_path.write_text("", encoding="utf-8")
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_seed_level_analysis",
        "levels": list(LEVELS),
        "seeds": list(SEEDS),
        "e0_size": 50,
        "balanced_e0": {"valid": 25, "invalid": 25},
        "presampling": "min(1000, complete original ConfigSpace)",
        "neighbour_definition": "one knob moves one position in its observed discrete domain",
        "validity_bias_evaluation": "top 50 unseen candidates after fitting on presampling labels",
        "candidate_dir": str(candidate_dir),
        "scan_dirs": [str(path) for path in scan_dirs],
        "board_contacted": False,
        "performance_labels_used": False,
    }
    write_json(output / "contract.json", contract)
    started = time.monotonic()
    rows = []
    for workload_id in ("R18-H1", "R18-H2", "R18-H3"):
        candidate_payload = read_json(candidate_dir / "{}_complete_original_space.json".format(workload_id.lower()))
        candidates = candidate_payload["candidates"]
        labels = {row["candidate_id"]: row for row in read_jsonl(scan_by_workload[workload_id] / "results.jsonl")}
        if set(labels) != {row["candidate_id"] for row in candidates}:
            raise ValueError("scan/candidate identity mismatch for {}".format(workload_id))
        for seed in SEEDS:
            for row in one_seed(candidates, labels, workload_id, seed):
                row["workload_id"] = workload_id
                rows.append(row)
                append_jsonl(results_path, row)
            print("{} seed {} complete".format(workload_id, seed), flush=True)
    aggregates = aggregate(rows)
    summary = {
        "schema": SCHEMA,
        "status": "complete_20_seed_four_level_hw_aware_analysis",
        "workloads": aggregates,
        "full_scan_common_cost": {
            workload_id: read_json(path / "summary.json")
            for workload_id, path in sorted(scan_by_workload.items())
        },
        "analysis_wall_seconds": time.monotonic() - started,
        "interpretation_gate": "positive only when 20-seed valid yield rises or IQR falls; all presampling compiler calls and measured lowering wall remain charged",
        "board_contacted": False,
        "performance_labels_used": False,
    }
    write_json(output / "summary.json", summary)
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "command.txt").write_text(" ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n", encoding="utf-8")
    artifacts = {path.name: baseline.sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "artifact_hashes.json"}
    write_json(output / "artifact_hashes.json", {"artifacts": artifacts, "source_hashes": {str(Path(__file__).resolve()): baseline.sha256_file(Path(__file__).resolve()), str(Path(baseline.__file__).resolve()): baseline.sha256_file(Path(baseline.__file__).resolve())}})
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--scan-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
