#!/usr/bin/env python3
"""Merge frozen R18 board pools by exact candidate identity before board labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import run_vta_c3_literature_baselines as baseline


SCHEMA = "c3_resnet18_merged_board_pool_v1"
MODES = baseline.CHENG_MODES


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
    ledger = read_json(directory / "artifact_hashes.json")["artifacts"]
    for name, expected in ledger.items():
        if isinstance(expected, str) and baseline.sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: {}".format(directory / name))
    return baseline.sha256_file(directory / "artifact_hashes.json")


def equivalent_identity(left, right):
    keys = (
        "candidate_id",
        "workload_id",
        "model_hash",
        "hardware_fingerprint",
        "complete_config_entity",
        "residence_mode",
        "implementation_mode",
        "tir_sha256",
        "implementation_candidate_id",
        "dma_features",
        "command_features",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    sources = [Path(value).resolve() for value in args.pool_dir]
    if len(sources) < 2:
        raise ValueError("at least two board pools are required")
    input_hashes = {str(path): verify(path) for path in sources}
    merged = {}
    memberships = Counter()
    duplicates = 0
    for source in sources:
        summary = read_json(source / "summary.json")
        if summary.get("performance_labels_collected") is not False:
            raise ValueError("source pool is no longer label-free")
        for row in read_jsonl(source / "candidates.jsonl"):
            candidate_id = row["candidate_id"]
            memberships[candidate_id] += 1
            if candidate_id in merged:
                duplicates += 1
                if not equivalent_identity(merged[candidate_id], row):
                    raise ValueError("duplicate candidate has conflicting exact identity")
            # Later inputs may add canonical hidden-feature aliases while the
            # exact candidate/TIR/command identity remains unchanged.
            merged[candidate_id] = row
    rows = []
    for row in merged.values():
        value = json.loads(json.dumps(row))
        index = int(value.get("debug", {}).get("config_index", -1))
        value["source_family_id"] = value["family_id"]
        value["family_id"] = "{}D{:04d}".format(value["workload_id"], index)
        value["merged_source_count"] = memberships[value["candidate_id"]]
        rows.append(value)
    rows.sort(
        key=lambda row: (
            row["workload_id"],
            int(row.get("debug", {}).get("config_index", -1)),
            MODES.index(row["residence_mode"]),
            row["candidate_id"],
        )
    )
    counts = Counter(row["workload_id"] for row in rows)
    output.mkdir(parents=True)
    (output / "candidates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_merged_cross_compile_fpga_or_latency",
        "candidate_count": len(rows),
        "candidate_order": [row["candidate_id"] for row in rows],
        "workload_counts": dict(sorted(counts.items())),
        "source_candidate_occurrences": sum(memberships.values()),
        "deduplicated_occurrences": duplicates,
        "source_pool_hashes": input_hashes,
        "correctness_policy": "one non-spliceable clean-start session; three seeds fail-fast",
        "timing_policy": "five balanced interleaved rounds for every FPGA-correct candidate",
        "oracle_policy": "connect only after the complete correct timing pool",
        "board_contacted": False,
        "performance_labels_collected": False,
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
                str(Path(__file__).resolve()): baseline.sha256_file(Path(__file__).resolve())
            },
        },
    )
    print(json.dumps(contract, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
