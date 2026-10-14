#!/usr/bin/env python3
"""Locally qualify a composable replacement for the failed ResNet18 H3 route."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from c3_candidate_identity import canonical_json_bytes, candidate_identity_record
import run_vta_c3_resnet18_local_qualification as legacy
import run_vta_c3_yolo608_local_qualification as strict_local


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_POOL = GROUPED / (
    "20260914_p7r472_resnet18_candidate_expansion_run01/r18-h3_four_mode_96.json"
)
DEFAULT_OUTPUT = GROUPED / "20260924_c3_resnet18_h3_composable_local_run01"
PARENT_ID = "05a48341e4db17c245227e5bedbb1b2897b3f0317e8d5c4aad6f1fbf2fe25eb8"
MODE = "input_weight_resident_composable"
IMPLEMENTATION_MODE = 10
SOURCES = (
    REPO / "vta/python/vta/top/vta_conv2d.py",
    REPO / "vta/python/vta/top/residency_dispatch.py",
    REPO / "vta/python/vta/transform.py",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def candidate_from_pool(pool):
    matches = [row for row in pool["candidates"] if row["candidate_id"] == PARENT_ID]
    if len(matches) != 1:
        raise ValueError("exact H3 parent candidate not found")
    parent = matches[0]
    source_hashes = {str(path.relative_to(REPO)): sha256(path) for path in SOURCES}
    version = digest({"mechanism": "scoped-force-serial-resident-segment-v1",
                      "source_hashes": source_hashes})
    identity = candidate_identity_record(
        parent["hardware_fingerprint"], "conv2d_packed_residency.vta", version,
        pool["workload"], MODE, parent["complete_config_entity"],
        config_index=parent["debug"]["config_index"],
    )
    candidate = json.loads(json.dumps(parent))
    candidate.update(
        candidate_id=identity["candidate_id"], residence_mode=MODE,
        implementation_mode=IMPLEMENTATION_MODE, family_id="R18-H3-composable",
        debug=identity["debug"], parent_mode5_candidate_id=PARENT_ID,
    )
    candidate["candidate_identity_fields"] = {
        "tile": candidate["visible_features"], "residence_mode": MODE,
        "implementation_mode": IMPLEMENTATION_MODE,
        "weight_load_policy": "resident_with_scoped_force_serial_completion",
        "schedule_version": version, "parent_mode5_candidate_id": PARENT_ID,
    }
    return legacy.candidate_with_context(candidate, pool), source_hashes, version


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    tir_dir = output / "tir"
    tir_dir.mkdir()
    pool = read(args.pool)
    candidate, source_hashes, version = candidate_from_pool(pool)
    write(output / "contract.json", {
        "schema": "c3_resnet18_h3_composable_local_v1",
        "status": "frozen_before_local_qualification",
        "candidate_id": candidate["candidate_id"],
        "parent_mode5_candidate_id": PARENT_ID,
        "schedule_version": version, "source_hashes": source_hashes,
        "seeds": list(legacy.SEEDS), "board_contacted": False,
        "latency_values_read": False,
    })
    write(output / "candidate.json", candidate)
    started = time.monotonic()
    static = strict_local.static_result(candidate, [candidate], tir_dir)
    write(output / "static_result.json", static)
    fsim, stderr = (None, "")
    if static["status"] == "ok":
        fsim, stderr = strict_local.run_fsim_subprocess(candidate, static, args)
        write(output / "fsim_result.json", fsim)
    if stderr.strip() and (fsim is None or fsim["status"] != "passed"):
        (output / "worker_stderr.txt").write_text(stderr[-8000:], encoding="utf-8")
    passed = bool(
        static["status"] == "ok"
        and static.get("arena_capacity_certificate", {}).get("status") == "passed"
        and static.get("final_address_audit", {}).get("status") == "passed"
        and static.get("dependency_audit", {}).get("balanced") is True
        and fsim is not None and fsim["status"] == "passed"
        and fsim.get("command_features", {}).get("status") == "available_three_seed_consistent"
    )
    summary = {
        "schema": "c3_resnet18_h3_composable_local_v1",
        "status": "LOCAL_COMPOSABLE_H3_GATE_PASSED" if passed else "LOCAL_COMPOSABLE_H3_GATE_FAILED",
        "candidate_id": candidate["candidate_id"],
        "parent_mode5_candidate_id": PARENT_ID,
        "static_status": static["status"],
        "final_address_status": static.get("final_address_audit", {}).get("status"),
        "arena_capacity_status": static.get("arena_capacity_certificate", {}).get("status"),
        "debug_mode_transitions": static.get("final_address_audit", {}).get("debug_mode_transitions"),
        "fsim_status": None if fsim is None else fsim["status"],
        "command_features_status": None if fsim is None else fsim.get("command_features", {}).get("status"),
        "board_contacted": False, "latency_values_read": False,
        "wall_seconds": time.monotonic() - started,
    }
    write(output / "summary.json", summary)
    write(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default=str(DEFAULT_POOL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--fsim-hw-path", default="/tmp/vta-hw-fsim")
    parser.add_argument("--fsim-timeout", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
