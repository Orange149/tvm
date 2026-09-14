#!/usr/bin/env python3
"""Analyze generality and transfer limits of the frozen full-pool capacity result."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from validate_vta_p7r_command_resource_plan import align_up


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
    ledger = load_json(path.parent / "artifact_hashes.json")
    hashes = ledger.get("artifacts", ledger.get("output_sha256", ledger))
    observed = sha256_file(path)
    if hashes.get(path.name) != observed:
        raise ValueError("frozen artifact hash mismatch: {}".format(path))
    return {"path": str(path), "sha256": observed}


def peak(row, queue):
    return int(row["command_signature"]["structural"]["peaks"][queue + "_bytes"])


def capacities(rows):
    return {
        "insn_peak_bytes": max(peak(row, "insn") for row in rows),
        "uop_peak_bytes": max(peak(row, "uop") for row in rows),
        "insn_capacity_bytes": align_up(max(peak(row, "insn") for row in rows)),
        "uop_capacity_bytes": align_up(max(peak(row, "uop") for row in rows)),
    }


def percentile(values, fraction):
    values = sorted(values)
    position = int(round(fraction * (len(values) - 1)))
    return values[position]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--capacity-plan", required=True)
    parser.add_argument("--validation-results", required=True)
    parser.add_argument("--validation-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)

    paths = {
        "baseline_results": Path(args.baseline_results),
        "capacity_plan": Path(args.capacity_plan),
        "validation_results": Path(args.validation_results),
        "validation_summary": Path(args.validation_summary),
    }
    guards = {name: verify_frozen_file(path) for name, path in paths.items()}
    baseline = load_jsonl(paths["baseline_results"])
    validation = load_jsonl(paths["validation_results"])
    plan = load_json(paths["capacity_plan"])
    summary = load_json(paths["validation_summary"])
    if len(baseline) != 197 or len(validation) != 197 or summary["passed"] != 197:
        raise ValueError("full-pool coverage is not 197/197")
    if not all(row.get("capacity_validation_passed") for row in validation):
        raise ValueError("at least one reduced-capacity validation failed")
    if {row["candidate_id"] for row in baseline} != {
        row["candidate_id"] for row in validation
    }:
        raise ValueError("baseline and validation populations differ")

    workloads = sorted({row["workload_id"] for row in baseline})
    modes = sorted({row["public_mode"] for row in baseline})
    if workloads != ["W{:02d}".format(index) for index in range(10)]:
        raise ValueError("expected W00--W09")
    overall = capacities(baseline)
    by_workload = {
        workload: capacities([row for row in baseline if row["workload_id"] == workload])
        for workload in workloads
    }
    leave_one_out = {}
    for workload in workloads:
        training = capacities([row for row in baseline if row["workload_id"] != workload])
        target = by_workload[workload]
        leave_one_out[workload] = {
            "derived_from_other_nine": training,
            "heldout": target,
            "heldout_fits": (
                target["insn_peak_bytes"] <= training["insn_capacity_bytes"]
                and target["uop_peak_bytes"] <= training["uop_capacity_bytes"]
            ),
            "extra_insn_pages_if_not_fit": max(
                0,
                (target["insn_capacity_bytes"] - training["insn_capacity_bytes"]) // 4096,
            ),
            "extra_uop_pages_if_not_fit": max(
                0,
                (target["uop_capacity_bytes"] - training["uop_capacity_bytes"]) // 4096,
            ),
        }

    per_identity = [
        align_up(peak(row, "insn")) + align_up(peak(row, "uop")) for row in baseline
    ]
    legacy = 2 * (1 << 25)
    max_insn_rows = [row for row in baseline if peak(row, "insn") == overall["insn_peak_bytes"]]
    max_uop_rows = [row for row in baseline if peak(row, "uop") == overall["uop_peak_bytes"]]
    result = {
        "schema": "c3_full_pool_command_capacity_generality_analysis_v1",
        "status": "completed",
        "scope_correction": {
            "source_text": plan["claim_boundary"],
            "correct_scope": "ten frozen workload identities W00--W09 and five schedule modes",
            "reason": "the source phrase 'three workload families' was descriptive metadata only; its structured workload list contains W00--W09",
            "validation_repeated": False,
        },
        "set_level_rule": {
            "formula": "C_q(S)=align_A(max_{c in S,b in submits(c)} B_q(c,b))",
            "meaning": "capacity follows the compiled allowlist, not a board-specific magic constant",
            "alignment_bytes": 4096,
            "identity_invalidation": plan["invalidation_key"],
            "concurrent_queue_extension": "M_cmd=sum_j(C_insn(S_j)+C_uop(S_j)); reuse is allowed only for non-overlapping lifetimes",
        },
        "coverage": {
            "candidates": len(baseline),
            "workloads": workloads,
            "modes": modes,
            "reduced_capacity_validations_passed": len(validation),
        },
        "pool_capacity": {
            **overall,
            "total_capacity_bytes": overall["insn_capacity_bytes"]
            + overall["uop_capacity_bytes"],
            "legacy_total_capacity_bytes": legacy,
            "requested_byte_reduction_fraction": 1.0
            - (overall["insn_capacity_bytes"] + overall["uop_capacity_bytes"]) / legacy,
        },
        "per_identity_total_capacity_distribution_bytes": {
            "min": min(per_identity),
            "median": statistics.median(per_identity),
            "p90": percentile(per_identity, 0.90),
            "max": max(per_identity),
        },
        "peak_sources": {
            "instruction": [
                {
                    "candidate_id": row["candidate_id"],
                    "workload_id": row["workload_id"],
                    "mode": row["public_mode"],
                    "config_index": row.get("debug", {}).get("config_index"),
                }
                for row in max_insn_rows
            ],
            "uop": [
                {
                    "candidate_id": row["candidate_id"],
                    "workload_id": row["workload_id"],
                    "mode": row["public_mode"],
                    "config_index": row.get("debug", {}).get("config_index"),
                }
                for row in max_uop_rows
            ],
        },
        "by_workload": by_workload,
        "leave_one_workload_out": {
            "results": leave_one_out,
            "fit_count": sum(row["heldout_fits"] for row in leave_one_out.values()),
            "total": len(leave_one_out),
            "interpretation": (
                "leave-one-out tests fixed-capacity transfer only; exact compilation remains the safety mechanism"
            ),
        },
        "claim_boundary": {
            "supported": [
                "one set-derived 36 KiB capacity passed all 197 frozen identities across W00--W09",
                "the method is parameterized by an allowlist and alignment rather than config-index constants",
                "runtime capacity checks preserve fail-closed behavior before submission",
            ],
            "not_supported": [
                "arbitrary unseen shapes fit 36 KiB",
                "board physical allocator consumption equals requested bytes",
                "dynamic shapes, command replay, whole-stage latency, or FPS are qualified",
            ],
        },
        "inputs": guards,
    }
    analysis_path = output / "analysis.json"
    analysis_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "STATUS.md").write_text(
        "# Full-pool command-capacity generality analysis\n\n"
        "- Coverage: 197 identities, W00--W09, five modes; reduced-capacity pass 197/197\n"
        "- Pool capacity: {} KiB instruction + {} KiB UOP = {} KiB\n"
        "- Per-identity total capacity: min {} KiB, median {:.0f} KiB, p90 {} KiB, max {} KiB\n"
        "- Leave-one-workload-out fixed-capacity fit: {}/10\n"
        "- Rule: derive capacity from the exact compiled allowlist; never assume 36 KiB for unseen identities\n".format(
            overall["insn_capacity_bytes"] // 1024,
            overall["uop_capacity_bytes"] // 1024,
            (overall["insn_capacity_bytes"] + overall["uop_capacity_bytes"]) // 1024,
            min(per_identity) // 1024, statistics.median(per_identity) / 1024,
            percentile(per_identity, 0.90) // 1024, max(per_identity) // 1024,
            result["leave_one_workload_out"]["fit_count"],
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
    print(json.dumps({
        "coverage": result["coverage"],
        "pool_capacity": result["pool_capacity"],
        "per_identity": result["per_identity_total_capacity_distribution_bytes"],
        "leave_one_out_fit": result["leave_one_workload_out"]["fit_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
