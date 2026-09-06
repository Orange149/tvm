#!/usr/bin/env python3
"""Summarize P5A build evidence and preserve first-build cost separately from replay cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freeze_cpu_vta_pipeline_v1 import (
    PROTOCOL_ID,
    load_sealed_artifact,
    repo_root,
    seal_artifact,
)


DEFAULT_OUTPUT = repo_root() / "vta/tutorials/frontend/report_out/resource_aware_maxplus"


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def package_storage(packages_dir):
    logical_bytes = 0
    unique_bytes = 0
    seen = set()
    for path in Path(packages_dir).rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        logical_bytes += int(stat.st_size)
        inode = (int(stat.st_dev), int(stat.st_ino))
        if inode not in seen:
            seen.add(inode)
            unique_bytes += int(stat.st_size)
    return {
        "logical_bytes": logical_bytes,
        "unique_inode_bytes": unique_bytes,
        "hardlink_saved_bytes": logical_bytes - unique_bytes,
    }


def review_p5a(output_dir=DEFAULT_OUTPUT, initial_full_build_wall_clock_s=0.0):
    output_dir = Path(output_dir)
    audit = load_sealed_artifact(
        output_dir / "v1_p5a_compile_audit.json",
        "cpu_vta_pipeline_v1_p5a_compile_audit",
    )
    reference = load_sealed_artifact(
        output_dir / "v1_p5a_reference.json",
        "cpu_vta_pipeline_v1_p5a_independent_tensor_reference",
    )
    state = load_sealed_artifact(
        output_dir / "v1_execution_state.json",
        "cpu_vta_pipeline_v1_execution_state",
    )
    if audit["compile_audit_pass_count"] != audit["candidate_count"]:
        raise RuntimeError("P5A review requires all frozen candidates to pass the compile audit")
    if initial_full_build_wall_clock_s <= 0.0:
        raise RuntimeError("the observed first full-build wall clock must be provided")

    stage_keys = {}
    total_stage_references = 0
    cache_hits = 0
    cache_misses = 0
    for row in audit["rows"]:
        manifest = json.loads(
            (Path(row["package_dir"]) / "manifest.json").read_text(encoding="utf-8")
        )
        total_stage_references += len(manifest["stages"])
        for stage in manifest["stages"]:
            stage_keys[stage["stage_build_cache_key"]] = stage["device"]
            cache_hits += int(bool(stage["stage_build_cache_hit"]))
            cache_misses += int(not bool(stage["stage_build_cache_hit"]))

    review = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p5a_review",
            "protocol_id": PROTOCOL_ID,
            "compile_audit_artifact_sha256": audit["artifact_sha256"],
            "reference_artifact_sha256": reference["artifact_sha256"],
            "execution_state_artifact_sha256": state["artifact_sha256"],
            "result": {
                "candidate_package_count": audit["candidate_count"],
                "candidate_compile_audit_pass_count": audit["compile_audit_pass_count"],
                "candidate_compile_audit_failure_count": audit["compile_audit_failure_count"],
                "board_commands_run": 0,
                "performance_results_present": False,
                "total_stage_references": total_stage_references,
                "unique_compiled_stage_keys": len(stage_keys),
                "unique_cpu_stage_keys": sum(device == "cpu" for device in stage_keys.values()),
                "unique_vta_stage_keys": sum(device == "vta" for device in stage_keys.values()),
                "manifest_stage_cache_hits": cache_hits,
                "manifest_stage_cache_misses": cache_misses,
                "tuple_boundary_count": sum(row["tuple_boundary_count"] for row in audit["rows"]),
            },
            "cost": {
                "initial_full_build_and_first_audit_wall_clock_s": float(
                    initial_full_build_wall_clock_s
                ),
                "initial_cost_source": "captured first P5A process result before ABI audit correction",
                "final_cached_replay_and_audit_wall_clock_s": float(audit["total_wall_clock_s"]),
                "cached_replay_is_not_reported_as_initial_compile_cost": True,
                "package_storage": package_storage(output_dir / "v1_p5a_packages"),
            },
            "resolved_issues": [
                "20 frozen candidates all have native cross-compiled packages",
                "VTA stage binaries reference native VTA GEMM or ALU runtime calls",
                "stage threads and independent prefix affinities match the frozen P4R rows",
                "serial and pipeline scripts dump complete final float32 logits",
                "independent MXNet full-logits reference and comparator are frozen before P5B",
                "package hashes remain valid after immutable-artifact hardlink deduplication",
            ],
            "remaining_limits": [
                "P5A proves buildability and interface consistency, not board numerical correctness",
                "tuple ABI is checked by ordinal, shape, and dtype; semantic order must pass P5B tensor checks",
                "the MXNet float reference gate is a quantized semantic sanity check, not bit-exact equivalence",
                "P5B must require serial-pipeline tensor equivalence and semantic-reference pass before timing",
                "shared-DDR concurrent slowdown remains conditional D1 work and is not resolved by P5A",
            ],
            "gate_passed": True,
            "next_stage": "V1-P5B",
            "next_stage_requires_user_confirmation": True,
        }
    )
    path = output_dir / "v1_p5a_review.json"
    write_json(path, review)
    return review


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--initial-full-build-wall-clock-s", type=float, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    review = review_p5a(args.output_dir, args.initial_full_build_wall_clock_s)
    print(json.dumps(review, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
