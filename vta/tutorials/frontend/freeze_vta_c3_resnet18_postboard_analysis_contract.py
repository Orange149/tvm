#!/usr/bin/env python3
"""Freeze the merged R18 board and post-board analysis contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import run_vta_c3_literature_baselines as baseline


SCHEMA = "c3_resnet18_postboard_analysis_contract_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify(directory):
    directory = Path(directory).resolve()
    artifacts = read_json(directory / "artifact_hashes.json")["artifacts"]
    for name, expected in artifacts.items():
        if isinstance(expected, str) and baseline.sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: {}".format(directory / name))
    return baseline.sha256_file(directory / "artifact_hashes.json")


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    merged = Path(args.merged_pool_dir).resolve()
    cross = Path(args.cross_compile_dir).resolve()
    pool_summary = read_json(merged / "summary.json")
    cross_summary = read_json(cross / "summary.json")
    if pool_summary.get("performance_labels_collected") is not False:
        raise ValueError("merged pool is no longer label blind")
    if cross_summary.get("passed") != pool_summary.get("candidate_count"):
        raise ValueError("cross-compile pool is incomplete")
    inputs = {
        "merged_pool": verify(merged),
        "cross_compile": verify(cross),
        "hw_union": verify(args.hw_union_dir),
        "hw_qualification": verify(args.hw_qualification_dir),
        "policy_amendment": verify(args.policy_amendment_dir),
    }
    here = Path(__file__).resolve().parent
    source_paths = [
        here / "run_vta_c3_resnet18_board_pool.py",
        here / "prepare_vta_c3_resnet18_literature_replay.py",
        here / "analyze_vta_c3_resnet18_literature_pool.py",
        here / "run_vta_c3_literature_baselines.py",
    ]
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_any_merged_pool_fpga_correctness_or_latency",
        "candidate_count": 214,
        "workload_counts": {"R18-H1": 104, "R18-H2": 44, "R18-H3": 66},
        "board_execution": {
            "single_non_spliceable_clean_start_session": True,
            "correctness_seeds": [0, 20250901, 20260910],
            "correctness_fail_fast": True,
            "timing_rounds": 5,
            "oracle_after_complete_pool_only": True,
            "boundaries": ["W0", "T0", "T1"],
        },
        "analysis": {
            "main_policies": list(baseline.POLICIES),
            "ml2tuner_A_plus_DMA_ablation": True,
            "search_seeds": list(range(57001, 57021)),
            "budgets": [10, 20, 50, 75, 96],
            "holdout": "target workload labels revealed only through phase ledger; other two complete workloads are development data",
            "metrics": [
                "success@oracle+2%/+5%",
                "trials/time-to-target",
                "best-so-far regret",
                "compiler/FPGA/kernel counts",
                "logical DMA bytes/calls",
                "per-phase and total wall",
                "Model P/A RMSE and nDCG",
            ],
            "cost_views_kept_separate": [
                "actual complete-pool common cost",
                "counterfactual lazy per-policy phase cost",
            ],
        },
        "input_artifact_ledgers": inputs,
        "source_hashes": {
            str(path): baseline.sha256_file(path) for path in source_paths
        },
        "board_contacted": False,
        "target_performance_labels_read": False,
    }
    output.mkdir(parents=True)
    write_json(output / "contract.json", contract)
    write_json(
        output / "summary.json",
        {
            "schema": SCHEMA,
            "status": contract["status"],
            "candidate_count": 214,
            "source_count": len(source_paths),
            "board_contacted": False,
            "target_performance_labels_read": False,
        },
    )
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "results.jsonl").write_text("", encoding="utf-8")
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                path.name: baseline.sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_hashes": {
                str(Path(__file__).resolve()): baseline.sha256_file(Path(__file__).resolve())
            },
        },
    )
    print(json.dumps(read_json(output / "summary.json"), indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-pool-dir", required=True)
    parser.add_argument("--cross-compile-dir", required=True)
    parser.add_argument("--hw-union-dir", required=True)
    parser.add_argument("--hw-qualification-dir", required=True)
    parser.add_argument("--policy-amendment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
