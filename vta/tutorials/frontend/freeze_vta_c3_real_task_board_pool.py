#!/usr/bin/env python3
"""Freeze the 25-program real-task FPGA reference pool before board contact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
C3 = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
GROUPED = C3 / "07_grouped_holdout"
DEFAULT_PREPARE = GROUPED / "20260924_c3_real_task_reference_prepare_run02"
DEFAULT_LOCAL = GROUPED / "20260924_c3_real_task_reference_local_run04"
DEFAULT_SELECTOR = GROUPED / "20260924_c3_real_task_selector_freeze_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_real_task_board_pool_freeze_run01"
SCHEMA = "c3_real_task_board_pool_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify(directory):
    directory = Path(directory).resolve()
    ledger = json.loads((directory / "artifact_hashes.json").read_text(encoding="utf-8"))
    for name, expected in ledger["artifacts"].items():
        if isinstance(expected, str) and sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: {}".format(name))
    return sha256_file(directory / "artifact_hashes.json")


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    manifests = {
        "prepare": verify(args.prepare_dir),
        "local": verify(args.local_dir),
        "selector": verify(args.selector_dir),
    }
    selector = json.loads((Path(args.selector_dir) / "contract.json").read_text(encoding="utf-8"))
    if selector.get("status") != "frozen_before_current_board_latency":
        raise ValueError("selector was not frozen before board latency")
    selected_ids = {
        program["candidate_id"]
        for workload in selector["workloads"].values()
        for program in workload["programs"]
    }
    if len(selected_ids) != 25:
        raise ValueError("expected exactly 25 distinct representative identities")
    pools, candidate_by_id = [], {}
    for path in sorted(Path(args.prepare_dir).glob("r18-h*_four_mode_96.json")):
        pool = json.loads(path.read_text(encoding="utf-8"))
        pools.append(pool)
        for raw in pool["candidates"]:
            row = json.loads(json.dumps(raw))
            row["identity"] = {
                "workload": pool["workload"],
                "complete_config_entity": row["complete_config_entity"],
            }
            candidate_by_id[row["candidate_id"]] = row
    static = {row["candidate_id"]: row for row in read_jsonl(Path(args.local_dir) / "static_results.jsonl")}
    fsim = {row["candidate_id"]: row for row in read_jsonl(Path(args.local_dir) / "fsim_results.jsonl")}
    rows = []
    for candidate_id in selected_ids:
        candidate = candidate_by_id[candidate_id]
        srow, frow = static[candidate_id], fsim[candidate_id]
        if srow["status"] != "ok" or frow["status"] != "passed":
            raise ValueError("selected representative lacks local qualification")
        if frow["relowered_tir_sha256"] != srow["lowered_tir_sha256"]:
            raise ValueError("selected representative TIR changed")
        rows.append(
            {
                **candidate,
                "tir_sha256": srow["lowered_tir_sha256"],
                "implementation_candidate_id": srow["implementation_candidate_id"],
                "hidden_compiler_features": srow["hidden_compiler_features"],
                "dma_features": srow["dma_features"],
                "dependency_audit": srow["dependency_audit"],
                "sync": srow["sync"],
                "command_features": frow["command_features"],
                "local_qualification": {
                    "status": "three_seed_fsim_passed_with_reused_buffers",
                    "seeds": frow["seeds"],
                    "source_tir_hash_matches": True,
                    "buffer_protocol": frow["buffer_protocol"],
                },
                "performance_label": None,
                "board_contacted": False,
            }
        )
    rows.sort(key=lambda row: (row["workload_id"], row["candidate_id"]))
    counts = {
        workload_id: sum(row["workload_id"] == workload_id for row in rows)
        for workload_id in sorted({row["workload_id"] for row in rows})
    }
    if sorted(counts.values()) != [12, 13]:
        raise ValueError("unexpected full-pool counts: {}".format(counts))
    output.mkdir(parents=True)
    (output / "candidates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_cross_compile_or_board_contact",
        "candidate_count": len(rows),
        "candidate_order": [row["candidate_id"] for row in rows],
        "workload_counts": counts,
        "selector_manifest_sha256": manifests["selector"],
        "correctness_policy": "same three seeds on FPGA; fail closed",
        "selection_policy": "frozen selector is immutable; full pool is reference evaluation only",
        "timing_policy": "six rounds per boot and three single-call seed samples per program-round",
        "oracle_policy": "construct only after every FPGA-correct program in the full pool is timed",
        "board_contacted": False,
        "performance_labels_collected": False,
    }
    write_json(output / "contract.json", contract)
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "results.jsonl").write_text("", encoding="utf-8")
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    write_json(output / "summary.json", contract)
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "inputs": manifests,
            "source_sha256": sha256_file(Path(__file__).resolve()),
        },
    )
    print(json.dumps(contract, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-dir", type=Path, default=DEFAULT_PREPARE)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--selector-dir", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
