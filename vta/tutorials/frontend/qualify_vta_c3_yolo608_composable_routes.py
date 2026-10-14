#!/usr/bin/env python3
"""Qualify scoped-completion mode-10 replacements for two frozen YOLO-608 routes.

This is a local-only mechanism gate.  It inherits the exact workloads and
ConfigEntities of the two already confirmed mode-5 routes, assigns fresh
versioned candidate identities, performs final-TIR address/capacity/dependency
checks, and requires exact three-seed FSim equality plus a stable command
signature.  It never contacts the FPGA board or reads latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from build_vta_c3_yolo608_graph_deployments import (
    DEFAULT_ANALYSIS,
    DEFAULT_LOCAL,
    DEFAULT_POOLS,
    DEFAULT_TASKS,
    selected_routes,
)
from c3_candidate_identity import canonical_json_bytes, candidate_identity_record
import run_vta_c3_resnet18_local_qualification as legacy
import run_vta_c3_yolo608_local_qualification as yolo_local


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo608_composable_local_run01"
MODE = "input_weight_resident_composable"
IMPLEMENTATION_MODE = 10
SCHEMA = "c3_yolo608_composable_local_v1"
SCHEDULE_SOURCES = (
    REPO / "vta/python/vta/top/vta_conv2d.py",
    REPO / "vta/python/vta/top/residency_dispatch.py",
    REPO / "vta/python/vta/transform.py",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_parent_candidates(args):
    routes = selected_routes(
        args.pools_dir, args.local_dir, args.analysis_dir, args.tasks_dir
    )["selector_v2"]
    parent_ids = {row["representative_candidate_id"] for row in routes}
    if len(parent_ids) != 2:
        raise ValueError("expected exactly two distinct frozen parent candidates")
    found = []
    for pool_path in sorted(Path(args.pools_dir).glob("Y608-*.json")):
        pool = read_json(pool_path)
        for candidate in pool.get("candidates", []):
            if candidate.get("candidate_id") in parent_ids:
                found.append((pool, candidate))
    if {row[1]["candidate_id"] for row in found} != parent_ids:
        raise ValueError("could not bind every frozen route to its source pool candidate")
    return sorted(found, key=lambda row: row[0]["workload_id"])


def make_candidates(args):
    source_hashes = {str(path.relative_to(REPO)): sha256_file(path) for path in SCHEDULE_SOURCES}
    schedule_version = digest({
        "mechanism": "scoped-force-serial-resident-segment-v1",
        "source_hashes": source_hashes,
    })
    rows = []
    for pool, parent in load_parent_candidates(args):
        identity = candidate_identity_record(
            parent["hardware_fingerprint"],
            "conv2d_packed_residency.vta",
            schedule_version,
            pool["workload"],
            MODE,
            parent["complete_config_entity"],
            config_index=parent["debug"]["config_index"],
        )
        candidate = json.loads(json.dumps(parent))
        candidate.update(
            candidate_id=identity["candidate_id"],
            residence_mode=MODE,
            implementation_mode=IMPLEMENTATION_MODE,
            family_id=pool["workload_id"] + "-composable",
            debug=identity["debug"],
            parent_mode5_candidate_id=parent["candidate_id"],
        )
        candidate["candidate_identity_fields"] = {
            "tile": candidate["visible_features"],
            "residence_mode": MODE,
            "implementation_mode": IMPLEMENTATION_MODE,
            "weight_load_policy": "resident_with_scoped_force_serial_completion",
            "schedule_version": schedule_version,
            "parent_mode5_candidate_id": parent["candidate_id"],
        }
        candidate = legacy.candidate_with_context(candidate, pool)
        rows.append((pool, candidate))
    return source_hashes, schedule_version, rows


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    output.mkdir(parents=True)
    tir_dir = output / "tir"
    tir_dir.mkdir()
    source_hashes, schedule_version, bound = make_candidates(args)
    candidates = [candidate for _, candidate in bound]
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_local_qualification",
        "mechanism": MODE,
        "implementation_mode": IMPLEMENTATION_MODE,
        "candidate_order": [row["candidate_id"] for row in candidates],
        "parent_mode5_candidates": [row["parent_mode5_candidate_id"] for row in candidates],
        "schedule_version": schedule_version,
        "source_hashes": source_hashes,
        "seeds": list(legacy.SEEDS),
        "hard_qualification": [
            "real lowering", "dependency-token closure", "StorageRewrite arena capacity",
            "final DDR/SRAM/UOP/GEMM/ALU address bounds", "three-seed exact FSim equality",
            "three-seed structural command-signature consistency",
        ],
        "board_contacted": False,
        "latency_values_read": False,
    }
    write_json(output / "contract.json", contract)
    write_json(output / "candidates.json", candidates)

    start = time.monotonic()
    static_rows = []
    for pool, candidate in bound:
        row = yolo_local.static_result(candidate, [candidate], tir_dir)
        static_rows.append(row)
        append_jsonl(output / "static_results.jsonl", row)
        print("static {} {}".format(candidate["workload_id"], row["status"]), flush=True)

    fsim_rows = []
    static_by_id = {row["candidate_id"]: row for row in static_rows}
    for candidate in candidates:
        static = static_by_id[candidate["candidate_id"]]
        if static["status"] != "ok":
            continue
        row, stderr = yolo_local.run_fsim_subprocess(candidate, static, args)
        fsim_rows.append(row)
        append_jsonl(output / "fsim_results.jsonl", row)
        if stderr.strip() and row["status"] != "passed":
            append_jsonl(output / "worker_failures.jsonl", {
                "candidate_id": candidate["candidate_id"], "stderr_tail": stderr[-4000:],
            })
        print("fsim {} {}".format(candidate["workload_id"], row["status"]), flush=True)

    per_workload = {}
    for candidate in candidates:
        static = static_by_id[candidate["candidate_id"]]
        fsim = next((row for row in fsim_rows if row["candidate_id"] == candidate["candidate_id"]), None)
        per_workload[candidate["workload_id"]] = {
            "candidate_id": candidate["candidate_id"],
            "parent_mode5_candidate_id": candidate["parent_mode5_candidate_id"],
            "static_status": static["status"],
            "arena_capacity_status": static.get("arena_capacity_certificate", {}).get("status"),
            "final_address_status": static.get("final_address_audit", {}).get("status"),
            "dependency_balanced": static.get("dependency_audit", {}).get("balanced"),
            "fsim_status": None if fsim is None else fsim["status"],
            "command_features_status": None if fsim is None else fsim.get("command_features", {}).get("status"),
        }
    passed = len(per_workload) == 2 and all(
        row["static_status"] == "ok"
        and row["arena_capacity_status"] == "passed"
        and row["final_address_status"] == "passed"
        and row["dependency_balanced"] is True
        and row["fsim_status"] == "passed"
        and row["command_features_status"] == "available_three_seed_consistent"
        for row in per_workload.values()
    )
    summary = {
        "schema": SCHEMA,
        "status": "LOCAL_COMPOSABLE_GATE_PASSED" if passed else "LOCAL_COMPOSABLE_GATE_FAILED",
        "workloads": per_workload,
        "wall_seconds": time.monotonic() - start,
        "board_contacted": False,
        "latency_values_read": False,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools-dir", default=str(DEFAULT_POOLS))
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL))
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS))
    parser.add_argument("--tasks-dir", default=str(DEFAULT_TASKS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--fsim-hw-path", default="/tmp/vta-hw-fsim")
    parser.add_argument("--fsim-timeout", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
