#!/usr/bin/env python3
"""Freeze the selector-v3 same-budget evaluation protocol without labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import c3_selector_v3_evaluation


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
    / "20260924_c3_selector_v3_evaluation_protocol_run02"
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def protocol():
    return {
        "schema": "c3_selector_v3_same_budget_protocol_v1",
        "status": "FROZEN_BEFORE_NEW_GRAPH_POOL_LABELS",
        "selection_unit": "exact_graph_candidate_set",
        "budgets": {"B1": 1, "B2": 2, "B3": 3,
                    "unit": "additional new candidates; incumbent cost reported separately"},
        "deterministic_methods": [
            "incumbent_only", "bytes_only", "calls_only", "command_only", "selector_v3",
        ],
        "random_baseline": {
            "seed": c3_selector_v3_evaluation.DEFAULT_RANDOM_SEED,
            "replicates": c3_selector_v3_evaluation.DEFAULT_RANDOM_REPLICATES,
        },
        "invalid_candidate_policy": "attempt consumes budget and is not replaced or timed",
        "measured_candidate_policy": "select minimum measured latency including incumbent",
        "metrics": [
            "simple_regret_fraction", "improvement_over_incumbent_fraction",
            "oracle_plus_1pct_hit", "oracle_plus_2pct_hit", "fpga_invalid_attempt_count",
        ],
        "oracle_scope": "FPGA-valid candidates in each fully measured declared reference pool",
        "leakage_control": (
            "orders use only frozen model scores or static features; latency labels only evaluate "
            "the frozen order; selection data are not confirmation data"
        ),
        "claim_boundary": (
            "protocol freeze contains no new latency and cannot establish selector superiority"
        ),
    }


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    value = protocol()
    write(output / "protocol.json", value)
    summary = {
        "schema": "c3_selector_v3_same_budget_protocol_freeze_v1",
        "status": value["status"],
        "performance_labels_read": False,
        "board_contacted": False,
        "random_replicates": value["random_baseline"]["replicates"],
    }
    write(output / "summary.json", summary)
    write(output / "artifact_hashes.json", {
        "artifacts": {path.name: sha256(path) for path in sorted(output.iterdir())
                      if path.is_file() and path.name != "artifact_hashes.json"},
        "source_sha256": sha256(Path(__file__).resolve()),
        "evaluation_source_sha256": sha256(
            Path(c3_selector_v3_evaluation.__file__).resolve()),
    })
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
