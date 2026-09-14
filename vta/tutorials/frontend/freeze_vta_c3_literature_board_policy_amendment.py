#!/usr/bin/env python3
"""Freeze pre-board corrections to the C3 literature policy implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import run_vta_c3_literature_baselines as baseline


SCHEMA = "c3_literature_board_policy_amendment_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify(directory):
    directory = Path(directory).resolve()
    ledger = read_json(directory / "artifact_hashes.json")
    artifacts = ledger.get("artifacts", ledger.get("files"))
    if not isinstance(artifacts, dict):
        raise ValueError("unknown artifact ledger schema")
    for name, expected in artifacts.items():
        if isinstance(expected, str) and baseline.sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: {}".format(directory / name))
    return baseline.sha256_file(directory / "artifact_hashes.json")


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    inputs = {
        "original_protocol": verify(args.original_protocol_dir),
        "board_pool": verify(args.board_pool_dir),
        "cross_compile": verify(args.cross_compile_dir),
        "hw_aware_analysis": verify(args.hw_aware_dir),
        "model_v_analysis": verify(args.model_v_dir),
    }
    source_paths = [
        Path(baseline.__file__).resolve(),
        Path(__file__).resolve().parent
        / "prepare_vta_c3_resnet18_literature_replay.py",
        Path(__file__).resolve().parent / "run_vta_c3_resnet18_board_pool.py",
        Path(__file__).resolve().parent
        / "analyze_vta_c3_hw_aware_initialization.py",
        Path(__file__).resolve().parent / "analyze_vta_c3_ml2tuner_validity.py",
    ]
    amendment = {
        "schema": SCHEMA,
        "status": "frozen_after_local_validity_before_any_r18_fpga_or_performance_label",
        "supersedes_only_policy_implementation_hashes_in_p7r469": True,
        "does_not_change": {
            "candidate_identities": True,
            "board_pool_candidate_order": True,
            "search_seeds": list(range(57001, 57021)),
            "ml2tuner_N": 10,
            "ml2tuner_alpha": 1,
            "cheng_modes": list(baseline.CHENG_MODES),
            "oracle_hidden_until_complete_pool": True,
        },
        "corrections": [
            {
                "name": "discrete_knob_neighbourhood",
                "reason": "factor domains are nonuniform; adjacency is one observed-domain position, not raw numeric distance one",
            },
            {
                "name": "parallel_frontier_exhaustion",
                "reason": "remove nodes consumed later in the same compiler batch so every requested presampling call remains unique",
            },
            {
                "name": "repeated_ml2tuner_waves",
                "reason": "budgets above ten repeat fixed 20-compile to 10-profile waves while keeping N=10 and alpha=1",
            },
            {
                "name": "cheng_resource_scope",
                "reason": "residency resource bounds constrain the corresponding residency modes; original remains subject to real lowering rather than analytic pre-rejection",
            },
            {
                "name": "post_board_identity_adapter",
                "reason": "separate label-free workload contracts from one-way phase outcome ledgers and preserve measured wall/DMA costs",
            },
        ],
        "r18_board_state": {
            "candidate_count": 50,
            "workload_counts": {"R18-H1": 13, "R18-H2": 12, "R18-H3": 25},
            "cross_compile_passed": 50,
            "fpga_labels_observed": False,
            "performance_labels_observed": False,
            "board_blocker": "current Dropbear host key differs from the previously serial-verified key",
        },
        "cost_reporting": {
            "actual_all_pool_qualification": "reported separately and never hidden",
            "lazy_policy_replay": "counterfactual per-candidate phase cost, reported separately from the actual all-pool run",
            "W0_T0_T1": True,
            "logical_dma_is_physical_axi": False,
        },
        "input_artifact_ledgers": inputs,
        "source_hashes": {
            str(path): baseline.sha256_file(path) for path in source_paths
        },
        "board_contacted_by_amendment": False,
        "target_performance_labels_read": False,
    }
    output.mkdir(parents=True)
    write_json(output / "contract.json", amendment)
    write_json(
        output / "summary.json",
        {
            "schema": SCHEMA,
            "status": amendment["status"],
            "correction_count": len(amendment["corrections"]),
            "candidate_count_unchanged": 50,
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
            }
        },
    )
    print(json.dumps(read_json(output / "summary.json"), indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-protocol-dir", required=True)
    parser.add_argument("--board-pool-dir", required=True)
    parser.add_argument("--cross-compile-dir", required=True)
    parser.add_argument("--hw-aware-dir", required=True)
    parser.add_argument("--model-v-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
