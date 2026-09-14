#!/usr/bin/env python3
"""Freeze the cumulative FSim-pass ResNet18 pool before board contact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prepare_vta_c3_resnet18_literature_candidates import sha256_file, write_json


MODES = ("original", "input_stationary", "weight_resident_barrier", "input_weight_resident_barrier")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def verify(directory):
    directory = Path(directory).resolve()
    ledger = read_json(directory / "artifact_hashes.json")["artifacts"]
    for name, expected in ledger.items():
        if sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: {}".format(directory / name))
    return {name: value for name, value in sorted(ledger.items())}


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    pool_dirs = [Path(value).resolve() for value in args.pool_dir]
    qualification_dirs = [Path(value).resolve() for value in args.qualification_dir]
    pool_hashes = {str(path): verify(path) for path in pool_dirs}
    qualification_hashes = {str(path): verify(path) for path in qualification_dirs}
    candidates = {}
    for directory in pool_dirs:
        for path in directory.glob("r18-h*_four_mode_96.json"):
            pool = read_json(path)
            for row in pool["candidates"]:
                candidate = json.loads(json.dumps(row))
                candidate["identity"] = {
                    "workload": pool["workload"],
                    "complete_config_entity": row["complete_config_entity"],
                }
                if candidate["candidate_id"] in candidates:
                    raise ValueError("duplicate candidate across frozen pools")
                candidates[candidate["candidate_id"]] = candidate
    static, fsim = {}, {}
    for directory in qualification_dirs:
        for row in read_jsonl(directory / "static_results.jsonl"):
            if row["candidate_id"] in static:
                raise ValueError("duplicate static result")
            static[row["candidate_id"]] = row
        for row in read_jsonl(directory / "fsim_results.jsonl"):
            if row["candidate_id"] in fsim:
                raise ValueError("duplicate FSim result")
            fsim[row["candidate_id"]] = row
    selected = []
    for candidate_id, certificate in fsim.items():
        if certificate["status"] != "passed":
            continue
        if candidate_id not in candidates or candidate_id not in static:
            raise ValueError("FSim pass lacks candidate/static source")
        srow = static[candidate_id]
        if srow["status"] != "ok" or certificate["relowered_tir_sha256"] != srow["lowered_tir_sha256"]:
            raise ValueError("FSim/static identity mismatch")
        candidate = candidates[candidate_id]
        if args.workload_id and candidate["workload_id"] != args.workload_id:
            continue
        selected.append({
            **candidate,
            "tir_sha256": srow["lowered_tir_sha256"],
            "implementation_candidate_id": srow["implementation_candidate_id"],
            "hidden_compiler_features": srow["hidden_compiler_features"],
            "dma_features": srow["dma_features"],
            "dependency_audit": srow["dependency_audit"],
            "sync": srow["sync"],
            "command_features": certificate["command_features"],
            "local_qualification": {
                "status": "three_seed_fsim_passed",
                "seeds": certificate["seeds"],
                "source_tir_hash_matches": True,
            },
            "performance_label": None,
            "board_contacted": False,
        })
    selected.sort(
        key=lambda row: (
            row["workload_id"],
            int(row.get("debug", {}).get("config_index", -1)),
            row["family_id"],
            MODES.index(row["residence_mode"]),
        )
    )
    workload_ids = (args.workload_id,) if args.workload_id else ("R18-H1", "R18-H2", "R18-H3")
    counts = {workload_id: sum(row["workload_id"] == workload_id for row in selected) for workload_id in workload_ids}
    if any(count < 12 for count in counts.values()):
        raise ValueError("local legal threshold was not met: {}".format(counts))
    if len({row["candidate_id"] for row in selected}) != len(selected):
        raise ValueError("selected board pool has duplicate identities")
    output.mkdir(parents=True)
    (output / "candidates.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
    contract = {
        "schema": "c3_resnet18_board_pool_v1",
        "status": "frozen_before_cross_compile_or_board_contact",
        "candidate_count": len(selected),
        "candidate_order": [row["candidate_id"] for row in selected],
        "workload_counts": counts,
        "correctness_policy": "three deterministic FPGA seeds, fail-fast; only all-pass identities enter timing",
        "timing_policy": "five balanced interleaved operator rounds after full correctness pool",
        "oracle_policy": "construct only after every FPGA-correct candidate is timed",
        "failure_accounting": "compile, RPC, wrong answer, recovery and wall time are retained",
        "board_contacted": False,
        "performance_labels_collected": False,
    }
    write_json(output / "contract.json", contract)
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "results.jsonl").write_text("", encoding="utf-8")
    (output / "command.txt").write_text(" ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n", encoding="utf-8")
    write_json(output / "summary.json", contract)
    artifacts = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "artifact_hashes.json"}
    write_json(output / "artifact_hashes.json", {
        "artifacts": artifacts,
        "inputs": {"candidate_pools": pool_hashes, "local_qualifications": qualification_hashes},
        "source_hashes": {str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve())},
    })
    print(json.dumps(contract, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", action="append", required=True)
    parser.add_argument("--qualification-dir", action="append", required=True)
    parser.add_argument("--workload-id", choices=("R18-H1", "R18-H2", "R18-H3"))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
