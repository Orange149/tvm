#!/usr/bin/env python3
"""Evaluate ML2Tuner-style Model V by leave-one-ResNet-geometry-out transfer."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

import run_vta_c3_literature_baselines as baseline


SCHEMA = "c3_ml2tuner_validity_transfer_v1"
SEEDS = tuple(range(57001, 57021))
BUDGETS = (10, 20, 50, 100)


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


def ndcg(labels):
    if not any(labels):
        return 0.0
    dcg = sum(float(value) / math.log2(index + 2) for index, value in enumerate(labels))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(sum(labels)))
    return dcg / ideal


def classification(labels, probabilities):
    predictions = [value >= 0.5 for value in probabilities]
    tp = sum(prediction and label for prediction, label in zip(predictions, labels))
    fp = sum(prediction and not label for prediction, label in zip(predictions, labels))
    fn = sum(not prediction and label for prediction, label in zip(predictions, labels))
    tn = sum(not prediction and not label for prediction, label in zip(predictions, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ranked_labels = [label for _, label in sorted(zip(probabilities, labels), reverse=True)]
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": (tp + tn) / len(labels),
        "brier": statistics.mean((float(probability) - float(label)) ** 2 for probability, label in zip(probabilities, labels)),
        "validity_ndcg": ndcg(ranked_labels),
    }


def distribution(values):
    values = [float(value) for value in values]
    return {
        "median": statistics.median(values),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "minimum": min(values), "maximum": max(values), "n": len(values),
    }


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    candidate_dir = Path(args.candidate_dir).resolve()
    verify(candidate_dir)
    scans = {}
    for value in args.scan_dir:
        directory = Path(value).resolve()
        verify(directory)
        summary = read_json(directory / "summary.json")
        scans[summary["workload_id"]] = directory
    if set(scans) != {"R18-H1", "R18-H2", "R18-H3"}:
        raise ValueError("three complete scans are required")
    datasets = {}
    for workload_id, directory in scans.items():
        contract = read_json(candidate_dir / "{}_complete_original_space.json".format(workload_id.lower()))
        labels = {row["candidate_id"]: row for row in read_jsonl(directory / "results.jsonl")}
        datasets[workload_id] = [
            {**candidate, "is_valid": bool(labels[candidate["candidate_id"]]["is_valid"])}
            for candidate in contract["candidates"]
        ]
    output.mkdir(parents=True)
    results_path = output / "results.jsonl"
    results_path.write_text("", encoding="utf-8")
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_model_fit",
        "method": "leave-one-workload-out Model V; train other two complete original spaces, test untouched target geometry",
        "features": list(baseline.VISIBLE_KNOBS) + ["four mode one-hot"] + list(baseline.WORKLOAD_FEATURES),
        "seeds": list(SEEDS),
        "top_compile_budgets": list(BUDGETS),
        "future_target_performance_used": False,
        "board_contacted": False,
    }
    write_json(output / "contract.json", contract)
    rows = []
    for target in sorted(datasets):
        training = [row for workload_id, items in datasets.items() if workload_id != target for row in items]
        test = datasets[target]
        test_x = [baseline.visible_vector(row) for row in test]
        labels = [int(row["is_valid"]) for row in test]
        global_valid_yield = sum(labels) / len(labels)
        for seed in SEEDS:
            probabilities = baseline._xgb_validity(
                [baseline.visible_vector(row) for row in training],
                [int(row["is_valid"]) for row in training],
                test_x,
                seed,
            )
            metrics = classification(labels, probabilities)
            ranked = sorted(zip(test, probabilities), key=lambda item: (-item[1], item[0]["candidate_id"]))
            top = {}
            for budget in BUDGETS:
                chosen = ranked[: min(budget, len(ranked))]
                valid = sum(row[0]["is_valid"] for row in chosen)
                top[str(budget)] = {
                    "compiler_attempts": len(chosen),
                    "valid": valid,
                    "invalid": len(chosen) - valid,
                    "valid_yield": valid / len(chosen),
                    "invalid_ratio": (len(chosen) - valid) / len(chosen),
                    "relative_valid_yield_vs_unfiltered": (valid / len(chosen)) / global_valid_yield,
                }
            row = {
                "schema": SCHEMA,
                "target_workload": target,
                "training_workloads": sorted(set(datasets) - {target}),
                "seed": seed,
                "training_rows": len(training),
                "test_rows": len(test),
                "test_valid_yield": global_valid_yield,
                "classification": metrics,
                "top_compile_budget": top,
            }
            rows.append(row)
            append_jsonl(results_path, row)
        print("{} 20 seeds complete".format(target), flush=True)
    summary_workloads = {}
    for target in sorted(datasets):
        selected = [row for row in rows if row["target_workload"] == target]
        summary_workloads[target] = {
            "test_rows": selected[0]["test_rows"],
            "unfiltered_valid_yield": selected[0]["test_valid_yield"],
            "f1": distribution([row["classification"]["f1"] for row in selected]),
            "recall": distribution([row["classification"]["recall"] for row in selected]),
            "precision": distribution([row["classification"]["precision"] for row in selected]),
            "validity_ndcg": distribution([row["classification"]["validity_ndcg"] for row in selected]),
            "brier": distribution([row["classification"]["brier"] for row in selected]),
            "top_compile_budget": {
                str(budget): {
                    "valid_yield": distribution([row["top_compile_budget"][str(budget)]["valid_yield"] for row in selected]),
                    "invalid_ratio": distribution([row["top_compile_budget"][str(budget)]["invalid_ratio"] for row in selected]),
                    "relative_valid_yield_vs_unfiltered": distribution([row["top_compile_budget"][str(budget)]["relative_valid_yield_vs_unfiltered"] for row in selected]),
                }
                for budget in BUDGETS
            },
        }
    summary = {
        "schema": SCHEMA,
        "status": "complete_leave_one_workload_out_validity_analysis",
        "workloads": summary_workloads,
        "claim_boundary": "compiler-lowering validity only; FPGA errors and performance Model P/A remain unavailable until complete board pool",
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
