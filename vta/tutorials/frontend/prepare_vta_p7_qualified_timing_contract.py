#!/usr/bin/env python3
"""Freeze the post-correctness P7Q timing contract without replacing failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import xgboost

from run_vta_p7_holdout_correctness import file_sha256, load_contract, write_json


EVIDENCE_DIRS = {
    "W01": ("20260911_p7b_w01_correctness_run01",),
    "W04": ("20260911_p7c_w04_correctness_run01",),
    "W07": ("20260911_p7d_w07_correctness_run01",),
    "W08": (
        "20260911_p7e_w08_correctness_run01",
        "20260911_p7e1_w08_correctness_resume_run01",
    ),
}
INVALID_ID = "906c1bdc905dc2ebbbd18569941dcfd4f11fe9e32ed55b9bfa03eb027b3b9c40"


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p7-contract", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--timing-runner", required=True)
    parser.add_argument("--b3-runner", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("refusing to overwrite qualified timing contract")

    p7 = load_contract(args.p7_contract)
    evidence_root = Path(args.evidence_root).resolve()
    evidence_ledger = {}
    observed = {}
    for workload_id, directories in EVIDENCE_DIRS.items():
        rows = []
        evidence_ledger[workload_id] = []
        for directory in directories:
            path = evidence_root / directory / "correctness.jsonl"
            part = jsonl(path)
            rows.extend(part)
            evidence_ledger[workload_id].append(
                {"path": str(path), "sha256": file_sha256(path), "rows": len(part)}
            )
        observed[workload_id] = rows

    invalid_diagnostic = evidence_root / "20260911_p7c1_w04_mismatch_diagnostic_run01"
    invalid_summary = json.loads((invalid_diagnostic / "summary.json").read_text())
    if (
        invalid_summary["candidate_id"] != INVALID_ID
        or invalid_summary["target_all_correct"] is not False
        or invalid_summary["sentinels_all_correct"] is not True
    ):
        raise RuntimeError("invalid-candidate evidence is incomplete")
    evidence_ledger["W04_invalid"] = {
        "path": str(invalid_diagnostic / "diagnostic.jsonl"),
        "sha256": file_sha256(invalid_diagnostic / "diagnostic.jsonl"),
    }

    workload_contracts = {}
    total_valid = 0
    for workload_id in p7["pool"]["holdouts"]:
        source = p7["workloads"][workload_id]
        entries = {row["candidate_id"]: row for row in source["candidates"]}
        rows_by_position = {int(row["correctness_position"]): row for row in observed[workload_id]}
        invalid_ids = [INVALID_ID] if workload_id == "W04" else []
        expected_valid_positions = [
            position
            for position, candidate_id in enumerate(source["correctness_order"], 1)
            if candidate_id not in invalid_ids
        ]
        if sorted(rows_by_position) != expected_valid_positions:
            raise RuntimeError("correctness position coverage mismatch: " + workload_id)
        for position, row in rows_by_position.items():
            expected_id = source["correctness_order"][position - 1]
            if row["candidate_id"] != expected_id or row["overall_status"] != "passed":
                raise RuntimeError("incorrect candidate binding: " + workload_id)
            if len(row["seeds"]) != 3 or not all(seed["correct"] for seed in row["seeds"]):
                raise RuntimeError("incomplete seed correctness: " + workload_id)

        valid_ids = sorted(set(entries) - set(invalid_ids))
        total_valid += len(valid_ids)
        timing_orders = []
        for block in range(1, 6):
            timing_orders.append(
                sorted(
                    valid_ids,
                    key=lambda cid: hashlib.sha256(
                        "20260911:timing:{}:{}:{}".format(block, workload_id, cid).encode()
                    ).hexdigest(),
                )
            )
        warmup_order = sorted(
            valid_ids,
            key=lambda cid: hashlib.sha256(
                "20260911:warmup:{}:{}".format(workload_id, cid).encode()
            ).hexdigest(),
        )
        incumbent = source["policy_orders"]["B0"][0]
        if incumbent not in valid_ids:
            raise RuntimeError("incumbent was not correctness-qualified")
        workload_contracts[workload_id] = {
            "gross_candidate_count": len(entries),
            "timed_candidate_count": len(valid_ids),
            "invalid_candidate_ids": invalid_ids,
            "timed_candidate_ids": valid_ids,
            "incumbent_candidate_id": incumbent,
            "warmup_order": warmup_order,
            "timing_block_orders": timing_orders,
            "gross_candidates": source["candidates"],
            "policy_orders_gross": source["policy_orders"],
        }

    if total_valid != 79:
        raise RuntimeError("qualified timing pool must contain exactly 79 candidates")
    contract = {
        "schema": "c3_p7_qualified_timing_contract_v1",
        "status": "frozen_after_correctness_before_timing",
        "claim_status": "post-qualification P7Q; the original 80/80 confirmatory timing condition was not met",
        "original_p7_contract": {
            "path": str(Path(args.p7_contract).resolve()),
            "sha256": file_sha256(args.p7_contract),
        },
        "workload_order": p7["pool"]["holdouts"],
        "gross_candidate_count": 80,
        "timed_candidate_count": 79,
        "invalid_candidate_ids": [INVALID_ID],
        "no_replacement": True,
        "failure_accounting": "the invalid candidate remains in every gross policy order, consumes a dispatch if reached, has no latency label, and is never used as an oracle",
        "measurement": {
            "warmup": 3,
            "blocks": 5,
            "number_per_block": 1,
            "candidate_statistic": "median of five randomized complete-block samples",
            "sentinel": "incumbent before and after each block; excluded from candidate samples",
            "timing_order_seed": 20260911,
            "buffers": "one input/weight/output set reused per workload",
        },
        "b3": {
            "description": "sequential XGB over seven numeric ConfigEntity knobs only",
            "seed": 20250901,
            "warmup_gross_dispatches": 4,
            "invalid_policy": "consume gross dispatch and do not train",
            "xgboost_version": xgboost.__version__,
            "runner_path": str(Path(args.b3_runner).resolve()),
            "runner_sha256": file_sha256(args.b3_runner),
        },
        "timing_runner": {
            "path": str(Path(args.timing_runner).resolve()),
            "sha256": file_sha256(args.timing_runner),
        },
        "correctness_evidence": evidence_ledger,
        "workloads": workload_contracts,
    }
    contract_path = output / "contract.json"
    write_json(contract_path, contract)
    write_json(
        output / "artifact_hashes.json",
        {
            "schema": "artifact_hashes_v1",
            "output_sha256": {"contract.json": file_sha256(contract_path)},
            "source_sha256": {
                "vta/tutorials/frontend/prepare_vta_p7_qualified_timing_contract.py": file_sha256(
                    __file__
                )
            },
        },
    )
    print("frozen P7Q timing contract: 79 valid, 1 retained invalid")


if __name__ == "__main__":
    main()
