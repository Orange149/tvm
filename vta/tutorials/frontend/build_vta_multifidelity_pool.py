#!/usr/bin/env python3
"""Build an immutable candidate-by-fidelity VTA search pool.

The existing equal-budget pools start after common lowering/FSim qualification.
This adapter restores every frozen candidate and keeps information revealed by
each action behind that action.  A search policy may see ``prelower`` at time
zero, exact DMA features only after lowering, command evidence only after FSim,
and latency only after measurement.

The input manifest is intentionally explicit::

  {"workloads": [{"workload_id": "Y00", "local_dir": "...",
                   "completed_pool": ".../completed_pool.json",
                   "candidate_ids": null}]}

``candidate_ids`` may restrict a workload to a pre-registered subset (for
example the six-point Y01 recovery pool).  It must never be chosen from target
latencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "c3_vta_multifidelity_pool_v1"
PHASES = ("lower", "fsim", "compile", "fpga", "measure")
RESOURCE_KEYS = (
    "logical_vta_load_bytes",
    "logical_vta_store_bytes",
    "logical_vta_dma_calls",
    "fpga_kernel_invocations",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def zero_resources():
    return {name: 0.0 for name in RESOURCE_KEYS}


def phase(status, wall_ms=None, resources=None, reveal=None):
    return {
        "status": status,
        "wall_ms": wall_ms,
        "resource_cost": resources,
        "reveal": reveal,
    }


def failure_category(row):
    failure = row.get("failure") or {}
    return str(failure.get("category") or failure.get("phase") or "unknown")


def flatten_numeric(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = (prefix + "_" + str(key)).strip("_")
            result.update(flatten_numeric(item, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = float(value)
    return result


def lower_reveal(row):
    if row["status"] != "ok":
        return {"failure_category": failure_category(row)}
    vector = {str(key): float(value) for key, value in row["static_feature_vector"].items()
              if isinstance(value, (int, float)) and not isinstance(value, bool)}
    return {
        "features": vector,
        "tir_sha256": row.get("tir_sha256"),
        "failure_category": None,
    }


def fsim_reveal(row):
    if row["status"] != "passed":
        return {"failure_category": failure_category(row)}
    command = row.get("command_evidence") or {}
    values = command.get("values") or {}
    numeric = flatten_numeric(values)
    return {
        "features": numeric,
        "command_evidence_status": command.get("status"),
        "tir_sha256": row.get("tir_sha256"),
        "failure_category": None,
    }


def cheap_features(candidate):
    hardware = candidate["hardware_predicate"]
    result = {"knob_" + key: float(value) for key, value in candidate["knobs"].items()}
    for key, value in hardware.get("required", {}).items():
        result["required_" + key] = float(value)
    for key, value in hardware.get("opportunities", {}).items():
        result["opportunity_" + key] = float(value)
    mode = candidate["public_mode"]
    for known in ("original", "input_stationary", "weight_resident_barrier"):
        result["mode_" + known] = float(mode == known)
    return result


def validate(pool):
    if pool.get("schema") != SCHEMA or pool.get("frozen_before_target_labels") is not True:
        raise ValueError("invalid multi-fidelity pool header")
    seen = set()
    for workload_id, workload in pool.get("workloads", {}).items():
        candidates = workload.get("candidates")
        if not candidates:
            raise ValueError(workload_id + ": empty candidates")
        for candidate in candidates:
            cid = candidate.get("candidate_id")
            if not cid or cid in seen or candidate.get("workload_id") != workload_id:
                raise ValueError("missing/duplicate/mismatched candidate identity")
            seen.add(cid)
            pre = candidate.get("prelower") or {}
            if pre.get("feature_provenance") != "frozen_before_target_labels":
                raise ValueError(cid + ": invalid feature provenance")
            if not pre.get("features"):
                raise ValueError(cid + ": empty prelower features")
            for identity_key in ("geometry_signature_sha256", "hardware_fingerprint_sha256",
                                 "schedule_version"):
                if not pre.get(identity_key):
                    raise ValueError(cid + ": missing prelower " + identity_key)
            stopped = False
            phases = candidate.get("oracle", {}).get("phases", {})
            if set(phases) != set(PHASES):
                raise ValueError(cid + ": phase order/schema mismatch")
            for name in PHASES:
                record = phases[name]
                status = record.get("status")
                if status not in ("ok", "invalid", "not_run"):
                    raise ValueError(cid + ": invalid phase status")
                if stopped and status != "not_run":
                    raise ValueError(cid + ": downstream phase ran after stop")
                if status in ("invalid", "not_run"):
                    stopped = True
                wall = record.get("wall_ms")
                if wall is not None and (not isinstance(wall, (int, float)) or
                                         isinstance(wall, bool) or wall < 0 or
                                         not math.isfinite(wall)):
                    raise ValueError(cid + ": invalid wall cost")
            measured = phases["measure"]["status"] == "ok"
            latency = candidate["oracle"].get("latency_ms")
            if measured != (isinstance(latency, (int, float)) and not isinstance(latency, bool)
                            and latency > 0 and math.isfinite(latency)):
                raise ValueError(cid + ": latency/measure mismatch")
    return pool


def build_workload(spec):
    workload_id = spec["workload_id"]
    local = Path(spec["local_dir"]).resolve()
    completed_path = Path(spec["completed_pool"]).resolve()
    candidates = read_jsonl(local / "candidates_v2.jsonl")
    static_rows = {row["candidate_id"]: row for row in read_jsonl(local / "static_results.jsonl")}
    fsim_rows = {row["candidate_id"]: row for row in read_jsonl(local / "fsim_results.jsonl")}
    completed = read_json(completed_path)
    completed_workload = completed["workloads"][workload_id]
    board_rows = {row["candidate_id"]: row for row in completed_workload["candidates"]}
    selected = spec.get("candidate_ids")
    if selected is not None:
        selected = set(selected)
        candidates = [row for row in candidates if row["candidate_id"] in selected]
        if {row["candidate_id"] for row in candidates} != selected:
            raise ValueError(workload_id + ": selected candidate identity missing locally")
    result = []
    for candidate in candidates:
        cid = candidate["candidate_id"]
        static = static_rows[cid]
        fsim = fsim_rows[cid]
        lower_ok = static["status"] == "ok"
        fsim_ok = fsim["status"] == "passed"
        phases = {
            "lower": phase(
                "ok" if lower_ok else "invalid",
                float(static["diagnostic_wall_seconds"]) * 1000.0,
                zero_resources(), lower_reveal(static),
            )
        }
        if not lower_ok:
            phases.update({name: phase("not_run") for name in PHASES[1:]})
        else:
            phases["fsim"] = phase(
                "ok" if fsim_ok else "invalid",
                float(fsim["diagnostic_wall_seconds"]) * 1000.0,
                zero_resources(), fsim_reveal(fsim),
            )
            if not fsim_ok:
                phases.update({name: phase("not_run") for name in PHASES[2:]})
            else:
                board = board_rows.get(cid)
                if board is None:
                    raise ValueError(workload_id + ": FSim-pass identity lacks completed board label: " + cid)
                board_oracle = board["oracle"]
                for name in PHASES[2:]:
                    source = board_oracle["phases"][name]
                    phases[name] = phase(
                        source["status"], source.get("wall_ms"),
                        source.get("resource_cost"), None,
                    )
        result.append({
            "candidate_id": cid,
            "workload_id": workload_id,
            "family_id": candidate["family_id"],
            "residence_mode": candidate["public_mode"],
            "prelower": {
                "feature_provenance": "frozen_before_target_labels",
                "features": cheap_features(candidate),
                "geometry_signature_sha256": canonical_sha256(
                    candidate["identity"]["workload"]
                ),
                "hardware_fingerprint_sha256": canonical_sha256(
                    candidate["identity"]["hardware_fingerprint"]
                ),
                "schedule_version": candidate["identity"]["schedule_version"],
                "selection_basis": candidate.get("selection_basis"),
            },
            "oracle": {
                "phases": phases,
                "latency_ms": (board_rows[cid]["oracle"]["latency_ms"]
                               if cid in board_rows else None),
            },
        })
    measured = [row["oracle"]["latency_ms"] for row in result
                if row["oracle"]["latency_ms"] is not None]
    return {
        "candidate_count": len(result),
        "pool_oracle_latency_ms": min(measured) if measured else None,
        "candidates": result,
        "source_bindings": {
            "local_dir": str(local),
            "local_artifact_hashes_sha256": sha256(local / "artifact_hashes.json"),
            "completed_pool": str(completed_path),
            "completed_pool_sha256": sha256(completed_path),
            "candidate_subset_pre_registered": selected is not None,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = read_json(manifest_path)
    workloads = {}
    for spec in manifest["workloads"]:
        wid = spec["workload_id"]
        if wid in workloads:
            raise ValueError("duplicate workload: " + wid)
        workloads[wid] = build_workload(spec)
    pool = {
        "schema": SCHEMA,
        "pool_id": manifest.get("pool_id", "vta-multifidelity"),
        "frozen_before_target_labels": True,
        "claim_status": manifest.get("claim_status", "development_replay_only"),
        "workloads": workloads,
        "manifest_binding": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "information_boundary": {
            "time_zero": "prelower.features only",
            "after_lower": "exact lowered-TIR DMA/SRAM/request features and lower outcome",
            "after_fsim": "command evidence and FSim outcome",
            "after_fpga": "hardware correctness outcome and incurred resource cost",
            "after_measure": "latency label",
        },
    }
    validate(pool)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pool_path = output / "multifidelity_pool.json"
    write_json(pool_path, pool)
    write_json(output / "artifact_hashes.json", {
        "artifacts": {"multifidelity_pool.json": sha256(pool_path)},
        "source_sha256": {str(Path(__file__).resolve()): sha256(__file__)},
        "manifest_sha256": sha256(manifest_path),
    })
    print(json.dumps({
        "output": str(output.resolve()),
        "workloads": {key: value["candidate_count"] for key, value in workloads.items()},
        "pool_sha256": sha256(pool_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
