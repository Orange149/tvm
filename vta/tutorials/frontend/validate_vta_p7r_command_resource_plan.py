#!/usr/bin/env python3
"""Freeze and validate page-aligned W05 command-buffer capacity certificates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from collect_vta_full_pool_fsim_commands import (
    run_candidate_no_rpc,
    strip_diagnostic_timing,
)


LEGACY_PER_QUEUE_BYTES = 1 << 25
ALLOCATION_ALIGNMENT_BYTES = 4096


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_frozen_file(path):
    path = Path(path).resolve()
    ledger = load_json(path.parent / "artifact_hashes.json")["artifacts"]
    observed = sha256_file(path)
    if ledger.get(path.name) != observed:
        raise ValueError("frozen artifact hash mismatch: {}".format(path))
    return {"path": str(path), "sha256": observed}


def align_up(value, alignment=ALLOCATION_ALIGNMENT_BYTES):
    return ((int(value) + alignment - 1) // alignment) * alignment


def peak_by_role(rows):
    result = {}
    for row in rows:
        if row.get("outcome") != "passed_local_signature":
            raise ValueError("baseline command signature did not pass")
        peaks = row["command_signature"]["structural"]["peaks"]
        result[row["candidate_role"]] = {
            "insn_peak_bytes": int(peaks["insn_bytes"]),
            "uop_peak_bytes": int(peaks["uop_bytes"]),
        }
    return result


def capacity_for_roles(peaks, roles):
    return {
        "insn_capacity_bytes": align_up(max(peaks[role]["insn_peak_bytes"] for role in roles)),
        "uop_capacity_bytes": align_up(max(peaks[role]["uop_peak_bytes"] for role in roles)),
    }


def validate_candidate(candidate, capacity, timeout_seconds):
    names = ("VTA_INSN_BUFFER_BYTES", "VTA_UOP_BUFFER_BYTES", "VTA_INSN_SUBMIT_THRESHOLD_BYTES")
    old = {name: os.environ.get(name) for name in names}
    try:
        os.environ["VTA_INSN_BUFFER_BYTES"] = str(capacity["insn_capacity_bytes"])
        os.environ["VTA_UOP_BUFFER_BYTES"] = str(capacity["uop_capacity_bytes"])
        os.environ.pop("VTA_INSN_SUBMIT_THRESHOLD_BYTES", None)
        result, _ = run_candidate_no_rpc(candidate, timeout_seconds)
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    strip_diagnostic_timing(result)
    result["candidate_role"] = candidate["candidate_role"]
    result["requested_capacity"] = capacity
    records = result.get("queue_records", [])
    result["reported_capacity_matches"] = bool(records) and all(
        int(record["insn_capacity"]) == capacity["insn_capacity_bytes"]
        and int(record["uop_capacity"]) == capacity["uop_capacity_bytes"]
        for record in records
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-results", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    input_guards = {
        "signature_results": verify_frozen_file(args.signature_results),
        "candidates": verify_frozen_file(args.candidates),
    }
    baseline = load_jsonl(args.signature_results)
    candidates = load_jsonl(args.candidates)
    by_role = {row["candidate_role"]: row for row in candidates}
    peaks = peak_by_role(baseline)
    deploy_roles = ("protected_tophub_incumbent", "bounded_hybrid_candidate")
    comparison_roles = ("same_tile_original",)
    deploy_capacity = capacity_for_roles(peaks, deploy_roles)
    comparison_capacity = capacity_for_roles(peaks, comparison_roles)
    legacy_total = 2 * LEGACY_PER_QUEUE_BYTES
    deploy_total = deploy_capacity["insn_capacity_bytes"] + deploy_capacity["uop_capacity_bytes"]
    plan = {
        "schema": "c3_p7r_command_resource_plan_v1",
        "status": "frozen_before_reduced_capacity_validation",
        "alignment_bytes": ALLOCATION_ALIGNMENT_BYTES,
        "alignment_rationale": "conservative page alignment; runtime minimum is 256-byte alignment",
        "observed_peaks_include_finish": True,
        "observed_peaks_by_role": peaks,
        "certified_dispatch_set": {
            "roles": list(deploy_roles),
            **deploy_capacity,
            "total_requested_bytes": deploy_total,
            "legacy_total_requested_bytes": legacy_total,
            "requested_byte_reduction_fraction": 1.0 - deploy_total / legacy_total,
        },
        "same_tile_comparison_only": {
            "roles": list(comparison_roles),
            **comparison_capacity,
            "total_requested_bytes": comparison_capacity["insn_capacity_bytes"]
            + comparison_capacity["uop_capacity_bytes"],
        },
        "safety_rule": "capacity = align_up(max exact peak over allowed identities); runtime checks both queues before submission",
        "claim_boundary": (
            "exact frozen W05 identities under the guarded runtime only; not a whole-model or arbitrary-shape bound"
        ),
        "input_guards": input_guards,
    }
    plan_path = output / "resource_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    (output / "pre_validation_hashes.json").write_text(
        json.dumps({"artifacts": {"resource_plan.json": sha256_file(plan_path)}}, indent=2)
        + "\n"
    )

    results = []
    for role in deploy_roles:
        results.append(validate_candidate(by_role[role], deploy_capacity, args.candidate_timeout_seconds))
    for role in comparison_roles:
        results.append(
            validate_candidate(by_role[role], comparison_capacity, args.candidate_timeout_seconds)
        )
    for row in results:
        expected = peaks[row["candidate_role"]]
        observed = row.get("command_signature", {}).get("structural", {}).get("peaks", {})
        row["peak_matches_baseline"] = (
            int(observed.get("insn_bytes", -1)) == expected["insn_peak_bytes"]
            and int(observed.get("uop_bytes", -1)) == expected["uop_peak_bytes"]
        )
        row["capacity_validation_passed"] = (
            row.get("outcome") == "passed_local_signature"
            and row["reported_capacity_matches"]
            and row["peak_matches_baseline"]
        )
    results_path = output / "validation.jsonl"
    results_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in results))
    passed = sum(row["capacity_validation_passed"] for row in results)
    summary = {
        "schema": "c3_p7r_command_resource_validation_summary_v1",
        "status": "passed" if passed == len(results) else "failed",
        "passed": passed,
        "selected": len(results),
        "certified_dispatch_capacity_bytes": deploy_total,
        "legacy_capacity_bytes": legacy_total,
        "requested_byte_reduction_fraction": 1.0 - deploy_total / legacy_total,
        "performance_measurement": "not_collected",
        "board_contacted": False,
        "claim_boundary": plan["claim_boundary"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "STATUS.md").write_text(
        "# P7R W05 reduced command capacity validation\n\n"
        "- Status: `{}` ({}/{})\n"
        "- Certified dispatch capacity: {} B (instruction {} B + UOP {} B)\n"
        "- Legacy requested capacity: {} B\n"
        "- Requested-byte reduction: {:.4%}\n"
        "- Local FSim structural validation only; no board or performance claim\n".format(
            summary["status"], passed, len(results), deploy_total,
            deploy_capacity["insn_capacity_bytes"], deploy_capacity["uop_capacity_bytes"],
            legacy_total, summary["requested_byte_reduction_fraction"],
        )
    )
    hashes = {
        path.name: sha256_file(path)
        for path in output.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
