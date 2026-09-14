#!/usr/bin/env python3
"""Freeze the 20-seed HW-Aware valid-E0 union and its four Cheng modes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import vta

import analyze_vta_c3_hw_aware_initialization as hw_analysis
import prepare_vta_c3_resnet18_literature_candidates as candidates_lib
import run_vta_c3_literature_baselines as baseline


SCHEMA = "c3_hw_aware_performance_union_v1"
SEEDS = tuple(range(57001, 57021))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def balanced_valid_e0(candidates, labels, workload_id, seed):
    by_id = {row["candidate_id"]: row for row in candidates}
    order = baseline.rieber_presampling_order(
        candidates,
        seed,
        lambda candidate_id: labels[candidate_id]["is_valid"],
        limit=min(1000, len(candidates)),
        parallel=8,
    )
    labelled = [
        {**by_id[candidate_id], "is_valid": bool(labels[candidate_id]["is_valid"])}
        for candidate_id in order
    ]
    e0 = baseline.balanced_e0(labelled, 25)
    valid = [row for row in e0 if row["is_valid"]]
    if len(e0) != 50 or len(valid) != 25:
        raise AssertionError("balanced E0 contract failed for {} seed {}".format(workload_id, seed))
    return valid


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    candidate_dir = Path(args.candidate_dir).resolve()
    input_hashes = {
        "candidate_dir": verify(candidate_dir),
        "hw_analysis": verify(args.hw_analysis_dir),
        "policy_amendment": verify(args.policy_amendment_dir),
    }
    scan_dirs = {}
    for value in args.scan_dir:
        directory = Path(value).resolve()
        input_hashes["scan:{}".format(directory.name)] = verify(directory)
        summary = read_json(directory / "summary.json")
        scan_dirs[summary["workload_id"]] = directory
    if set(scan_dirs) != set(candidates_lib.GEOMETRIES):
        raise ValueError("all three complete original scans are required")

    env = vta.get_env()
    fingerprint = candidates_lib.hardware_fingerprint(env)
    all_seed_records = []
    pools = {}
    counts = {}
    for workload_id, spec in candidates_lib.GEOMETRIES.items():
        source = read_json(
            candidate_dir / "{}_complete_original_space.json".format(workload_id.lower())
        )
        rows = source["candidates"]
        labels = {
            row["candidate_id"]: row
            for row in read_jsonl(scan_dirs[workload_id] / "results.jsonl")
        }
        if set(labels) != {row["candidate_id"] for row in rows}:
            raise ValueError("candidate/scan identity mismatch for {}".format(workload_id))
        union = {}
        seeds_by_id = defaultdict(list)
        for seed in SEEDS:
            valid = balanced_valid_e0(rows, labels, workload_id, seed)
            ids = [row["candidate_id"] for row in valid]
            all_seed_records.append(
                {
                    "workload_id": workload_id,
                    "seed": seed,
                    "valid_e0_candidate_ids": ids,
                    "count": len(ids),
                }
            )
            for row in valid:
                union[row["candidate_id"]] = row
                seeds_by_id[row["candidate_id"]].append(seed)
        workload = source["workload"]
        schedule_version = source["source_hashes"]
        expanded = []
        for original in sorted(union.values(), key=lambda row: row["debug"]["config_index"]):
            index = int(original["debug"]["config_index"])
            entity = original["complete_config_entity"]
            family_id = "{}HW{:04d}".format(workload_id, index)
            for mode in candidates_lib.MODES:
                row = candidates_lib.candidate_row(
                    workload_id,
                    spec,
                    workload,
                    entity,
                    index,
                    family_id,
                    mode,
                    schedule_version,
                    fingerprint,
                    env,
                )
                row["hw_aware_selection"] = {
                    "balanced_e0_valid_for_seeds": seeds_by_id[original["candidate_id"]],
                    "selected_using_performance_labels": False,
                    "original_lowering_validity_known": True,
                }
                expanded.append(row)
        pools[workload_id] = {
            "schema": "c3_literature_workload_contract_v1",
            "status": "frozen_after_original_lowering_validity_before_fsim_fpga_or_latency",
            "role": "hw_aware_20_seed_valid_e0_union_four_mode_pool",
            "workload_id": workload_id,
            "geometry": spec,
            "workload": workload,
            "hardware_fingerprint": fingerprint,
            "source_hashes": schedule_version,
            "selection": {
                "method": "union of valid half of 20 balanced 25-valid+25-invalid E0 sets",
                "seeds": list(SEEDS),
                "original_validity_labels_used": True,
                "FSim_labels_used": False,
                "FPGA_labels_used": False,
                "performance_labels_used": False,
            },
            "candidates": expanded,
            "board_contacted": False,
        }
        counts[workload_id] = {
            "unique_valid_originals": len(union),
            "four_mode_identities": len(expanded),
        }

    output.mkdir(parents=True)
    for workload_id, pool in pools.items():
        write_json(
            output / "{}_four_mode_96.json".format(workload_id.lower()), pool
        )
    (output / "seed_e0_membership.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_seed_records),
        encoding="utf-8",
    )
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_supplemental_fsim_cross_compile_fpga_or_latency",
        "workload_counts": counts,
        "total_unique_originals": sum(
            value["unique_valid_originals"] for value in counts.values()
        ),
        "total_four_mode_identities": sum(
            value["four_mode_identities"] for value in counts.values()
        ),
        "search_seeds": list(SEEDS),
        "identity_expansion": "each selected original ConfigEntity expands to the unchanged four Cheng modes",
        "scope": "supplemental performance pool required to evaluate HW-Aware initialization beyond validity yield",
        "input_hashes": input_hashes,
        "board_contacted": False,
        "performance_labels_read": False,
    }
    write_json(output / "contract.json", contract)
    write_json(output / "summary.json", contract)
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
                str(Path(__file__).resolve()): baseline.sha256_file(Path(__file__).resolve()),
                str(Path(baseline.__file__).resolve()): baseline.sha256_file(Path(baseline.__file__).resolve()),
                str(Path(candidates_lib.__file__).resolve()): baseline.sha256_file(Path(candidates_lib.__file__).resolve()),
                str(Path(hw_analysis.__file__).resolve()): baseline.sha256_file(Path(hw_analysis.__file__).resolve()),
            },
        },
    )
    print(json.dumps(contract, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--scan-dir", action="append", required=True)
    parser.add_argument("--hw-analysis-dir", required=True)
    parser.add_argument("--policy-amendment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
