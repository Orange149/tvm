#!/usr/bin/env python3
"""Check whether historical ResNet18 measurements can label the V1 P3 search space."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from freeze_cpu_vta_pipeline_v1 import (
    PROTOCOL_ID,
    UNIT_ORDER,
    load_sealed_artifact,
    repo_root,
    seal_artifact,
    validate_artifact_sha256,
)
from resource_aware_dataset import DEFAULT_RESNET_ROOTS, collect_resnet_records
from solve_cpu_vta_pipeline_v1_p3 import (
    build_context,
    candidate_id,
    label_from_path,
    segment_id,
    sha256_file,
    topology_id,
)


DEFAULT_OUTPUT = repo_root() / "vta/tutorials/frontend/report_out/resource_aware_maxplus"
MIN_COMPARABLE_POOL = 40
CURRENT_AFFINITY_POLICY = "independent_overlapping_prefix_masks_v1"
CURRENT_QUEUE_DEPTH = 2
CURRENT_POLL_SLEEP_NS = 1000


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def map_historical_record(record):
    unit_indices = {name: index for index, name in enumerate(UNIT_ORDER)}
    parts = list(record.metadata.get("stage_static_parts") or [])
    if len(parts) != len(record.stages):
        raise ValueError("stage metadata count mismatch for {}".format(record.candidate_id))

    path = []
    expected_start = 0
    for stage, part in zip(record.stages, parts):
        names = list(part.get("unit_names") or [])
        if not names or any(name not in unit_indices for name in names):
            raise ValueError("unmapped units for {}".format(record.candidate_id))
        indices = [unit_indices[name] for name in names]
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("non-contiguous stage for {}".format(record.candidate_id))
        if indices[0] != expected_start:
            raise ValueError("stage coverage gap for {}".format(record.candidate_id))
        expected_start = indices[-1] + 1
        path.append(
            (segment_id(stage.device, indices[0], indices[-1]), int(stage.threads))
        )
    if expected_start != len(UNIT_ORDER):
        raise ValueError("incomplete stage coverage for {}".format(record.candidate_id))

    path = tuple(path)
    cpu_threads = [threads for sid, threads in path if sid.startswith("cpu:")]
    return {
        "historical_candidate_id": record.candidate_id,
        "mapped_execution_candidate_id": candidate_id(path),
        "mapped_topology_id": topology_id(path),
        "mapped_path": [list(item) for item in path],
        "stage_threads": [threads for _, threads in path],
        "cpu_thread_sum": sum(cpu_threads),
        "cpu_stage_threads": cpu_threads,
        "vta_island_count": int(record.vta_island_count),
        "measured_cycle_ms": float(record.measured_cycle_ms),
        "source_path": str(record.source_path),
        "runtime_provenance": dict(record.metadata.get("runtime_provenance") or {}),
    }


def strip_outcome(row):
    return {key: value for key, value in row.items() if key != "measured_cycle_ms"}


def build_audit(output_dir=DEFAULT_OUTPUT, roots=DEFAULT_RESNET_ROOTS):
    output_dir = Path(output_dir)
    unit_schema = load_sealed_artifact(
        output_dir / "v1_unit_and_boundary_schema.json",
        "cpu_vta_pipeline_v1_unit_and_boundary_schema",
    )
    profile = load_sealed_artifact(
        output_dir / "v1_profile_manifest.json",
        "cpu_vta_pipeline_v1_profile_manifest",
    )
    local_cost = load_sealed_artifact(
        output_dir / "v1_local_cost_table.json",
        "cpu_vta_pipeline_v1_local_cost_table",
    )
    p3_report = load_sealed_artifact(
        output_dir / "v1_dp_vs_enumeration_report.json",
        "cpu_vta_pipeline_v1_dp_vs_enumeration_report",
    )
    execution_state = load_sealed_artifact(
        output_dir / "v1_execution_state.json",
        "cpu_vta_pipeline_v1_execution_state",
    )
    replayable_stages = {
        "V1-P3",
        "V1-P4",
        "V1-P4B",
        "V1-P4R",
        "V1-P5A",
        "V1-P5B",
    }
    if execution_state["current_stage"] not in replayable_stages:
        raise RuntimeError("P4 requires a sealed P3-or-later V1 execution state")
    if execution_state["current_stage"] == "V1-P3" and execution_state[
        "current_stage_status"
    ] != "completed_awaiting_user_confirmation":
        raise RuntimeError("P3 is not complete")
    if execution_state["current_stage"] == "V1-P4" and execution_state[
        "current_stage_status"
    ] not in {
        "completed_gate_failed_noncomparable_measured_pool",
        "completed_compatibility_passed_awaiting_baseline_scoring",
        "completed_semantic_compatibility_awaiting_retrospective_scoring",
    }:
        raise RuntimeError("existing P4 state is not rerunnable")
    if execution_state["current_stage"] == "V1-P4B" and execution_state[
        "current_stage_status"
    ] not in {
        "completed_exploratory_gate_inconclusive",
        "completed_exploratory_gate_passed_current_runtime_unverified",
    }:
        raise RuntimeError("existing P4B state is not compatible with a P4 re-audit")

    context = build_context(profile, local_cost)
    topology_count = int(profile["coverage"]["legal_candidate_count"])
    execution_count = int(p3_report["legal_execution_candidate_count"])

    records, source_paths = collect_resnet_records(roots=tuple(roots))
    rows = []
    mapping_errors = []
    for record in records:
        try:
            rows.append(map_historical_record(record))
        except ValueError as err:
            mapping_errors.append(
                {"historical_candidate_id": record.candidate_id, "error": str(err)}
            )

    mapped_execution_ids = {row["mapped_execution_candidate_id"] for row in rows}
    mapped_topology_ids = {row["mapped_topology_id"] for row in rows}
    compatible_rows = []
    incompatible_rows = []
    for row in rows:
        path = tuple((str(sid), int(threads)) for sid, threads in row["mapped_path"])
        try:
            label_from_path(path, context)
        except RuntimeError as err:
            incompatible_rows.append(
                {
                    "historical_candidate_id": row["historical_candidate_id"],
                    "reason": str(err),
                }
            )
        else:
            compatible_rows.append(row)
    semantic_execution_overlap = sorted(
        {row["mapped_execution_candidate_id"] for row in compatible_rows}
    )
    topology_overlap = sorted({row["mapped_topology_id"] for row in compatible_rows})
    current_runtime_rows = [
        row
        for row in compatible_rows
        if row["runtime_provenance"].get("cpu_affinity_policy")
        == CURRENT_AFFINITY_POLICY
        and int(row["runtime_provenance"].get("queue_depth", -1))
        == CURRENT_QUEUE_DEPTH
        and int(row["runtime_provenance"].get("poll_sleep_ns", -1))
        == CURRENT_POLL_SLEEP_NS
        and bool(row["runtime_provenance"].get("current_source_fingerprint_available"))
    ]
    current_runtime_execution_overlap = sorted(
        {row["mapped_execution_candidate_id"] for row in current_runtime_rows}
    )
    cpu_thread_sum_counts = Counter(row["cpu_thread_sum"] for row in rows)
    cpu_thread_vector_counts = Counter(
        tuple(row["cpu_stage_threads"]) for row in compatible_rows
    )
    duplicate_mappings = defaultdict(list)
    for row in rows:
        duplicate_mappings[row["mapped_execution_candidate_id"]].append(
            row["historical_candidate_id"]
        )

    semantic_comparison_eligible = (
        bool(rows)
        and not mapping_errors
        and len(semantic_execution_overlap) >= MIN_COMPARABLE_POOL
    )
    current_runtime_comparison_eligible = (
        len(current_runtime_execution_overlap) >= MIN_COMPARABLE_POOL
    )
    compatibility_rows = [strip_outcome(row) for row in rows]
    compatibility = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p4_measured_pool_compatibility",
            "protocol_id": PROTOCOL_ID,
            "scope": "host_only_historical_runtime_semantic_compatibility",
            "candidate_outcome_fields_present": False,
            "candidate_outcomes_used_by_score": False,
            "p3_report_artifact_sha256": p3_report["artifact_sha256"],
            "current_execution_space": {
                "legal_topology_count": topology_count,
                "legal_execution_candidate_count": execution_count,
                "cpu_thread_parameter_scope": "independent per CPU stage",
                "cpu_thread_sum_is_a_constraint": False,
                "cpu_affinity_policy": "not a search decision; masks may overlap",
            },
            "historical_pool": {
                "record_count": len(records),
                "mapped_record_count": len(rows),
                "unique_mapped_execution_count": len(mapped_execution_ids),
                "unique_mapped_topology_count": len(mapped_topology_ids),
                "source_summaries": [
                    {"path": str(path), "sha256": sha256_file(path)}
                    for path in source_paths
                ],
                "cpu_thread_sum_counts": {
                    str(key): value for key, value in sorted(cpu_thread_sum_counts.items())
                },
                "cpu_thread_sum_counts_are_diagnostic_only": True,
                "semantic_cpu_thread_vector_counts": {
                    ",".join(str(value) for value in key): count
                    for key, count in sorted(cpu_thread_vector_counts.items())
                },
                "per_stage_thread_search_coverage_complete": False,
                "mapping_errors": mapping_errors,
                "duplicate_execution_mappings": {
                    key: value
                    for key, value in sorted(duplicate_mappings.items())
                    if len(value) > 1
                },
            },
            "overlap": {
                "semantic_execution_candidate_count": len(semantic_execution_overlap),
                "semantic_execution_candidate_ids": semantic_execution_overlap,
                "current_runtime_exact_candidate_count": len(
                    current_runtime_execution_overlap
                ),
                "current_runtime_exact_candidate_ids": current_runtime_execution_overlap,
                "topology_count": len(topology_overlap),
                "compatible_record_count": len(compatible_rows),
                "incompatible_records": incompatible_rows,
                "topology_only_is_not_execution_equivalence": True,
                "segment_threads_match_is_not_runtime_policy_equivalence": True,
            },
            "prerequisite_checks": {
                "measured_pool_nonempty": bool(records),
                "all_records_map_to_current_units": len(rows) == len(records),
                "current_p3_space_is_972528": execution_count == 972528,
                "semantic_execution_overlap_at_least_40": len(semantic_execution_overlap)
                >= MIN_COMPARABLE_POOL,
                "current_runtime_exact_overlap_at_least_40": (
                    current_runtime_comparison_eligible
                ),
                "cpu_thread_sum_not_used_as_legality_gate": True,
            },
            "comparison_eligible": semantic_comparison_eligible,
            "comparison_scope": "exploratory_same_segments_and_threads_on_historical_runtime",
            "current_runtime_policy_comparison_eligible": (
                current_runtime_comparison_eligible
            ),
            "decision": (
                "run_exploratory_b0_b4_historical_runtime_retrospective"
                if semantic_comparison_eligible
                else "stop_before_retrospective_noncomparable_semantic_labels"
            ),
            "rows": compatibility_rows,
        }
    )
    historical_labels = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p4_historical_labels",
            "protocol_id": PROTOCOL_ID,
            "scope": "retrospective_evaluation_labels_only",
            "compatibility_artifact_sha256": compatibility["artifact_sha256"],
            "candidate_outcomes_used_by_score": False,
            "record_count": len(rows),
            "rows": [
                {
                    "historical_candidate_id": row["historical_candidate_id"],
                    "mapped_execution_candidate_id": row[
                        "mapped_execution_candidate_id"
                    ],
                    "measured_cycle_ms": row["measured_cycle_ms"],
                    "source_path": row["source_path"],
                }
                for row in rows
            ],
        }
    )

    models = {
        name: {
            "metrics": None,
            "status": "not_run",
            "reason": "compatibility audit only; freeze B0-B4 score definitions before reading "
            "measured throughput labels",
        }
        for name in ("B0", "B1", "B2", "B3", "B4")
    }
    report = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p4_retrospective_report",
            "protocol_id": PROTOCOL_ID,
            "compatibility_artifact_sha256": compatibility["artifact_sha256"],
            "historical_labels_artifact_sha256": historical_labels[
                "artifact_sha256"
            ],
            "status": (
                "semantic_compatibility_passed_current_runtime_unverified"
                if semantic_comparison_eligible
                else "gate_failed"
            ),
            "comparison_scope": "same segments and per-stage threads on historical runtime only",
            "models": models,
            "gate": {
                "historical_semantic_pool_comparable_for_exploratory_b4": (
                    semantic_comparison_eligible
                ),
                "historical_labels_comparable_to_current_runtime_b4": (
                    current_runtime_comparison_eligible
                ),
                "b4_low_k_regret_beats_b0_b3": None,
                "p5_may_start": False,
            },
            "finding": (
                "Historical CPU thread sums are not physical-core allocations. There are enough "
                "matching segment/thread labels for an exploratory retrospective, but the old "
                "packages do not record the current overlapping-affinity source fingerprint and "
                "therefore are not exact current-runtime executions."
            ),
            "required_next_evidence": (
                "Freeze B0-B4 scoring rules and compute exploratory regret on the semantic "
                "subset. Any formal current-runtime claim requires the P5 shortlist to be rebuilt, "
                "reference-checked, and measured under the frozen affinity policy."
            ),
            "ddr_contention_follow_up": {
                "id": "D1",
                "status": "conditional_not_started",
                "trigger": "matched concurrent controls and frozen-candidate residuals both show "
                "that aggregate DDR demand is insufficient",
            },
        }
    )
    state = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_execution_state",
            "protocol_id": PROTOCOL_ID,
            "current_stage": "V1-P4",
            "current_stage_status": (
                "completed_semantic_compatibility_awaiting_retrospective_scoring"
                if semantic_comparison_eligible
                else "completed_gate_failed_noncomparable_measured_pool"
            ),
            "next_stage": "V1-P4B" if semantic_comparison_eligible else "V1-P4R",
            "next_stage_may_start": False,
            "advance_requires_user_confirmation": True,
            "blocking_evidence": (
                ["current_runtime_policy_has_no_exact_historical_label_pool"]
                if semantic_comparison_eligible
                else ["semantic_historical_pool_has_fewer_than_40_unique_executions"]
            ),
            "p4_compatibility_artifact_sha256": compatibility["artifact_sha256"],
            "p4_historical_labels_artifact_sha256": historical_labels[
                "artifact_sha256"
            ],
            "p4_report_artifact_sha256": report["artifact_sha256"],
            "ranked_candidates_artifact_sha256": p3_report[
                "ranked_candidates_artifact_sha256"
            ],
        }
    )
    return {
        "v1_p4_measured_pool_compatibility.json": compatibility,
        "v1_p4_historical_labels.json": historical_labels,
        "v1_p4_retrospective_report.json": report,
        "v1_execution_state.json": state,
    }


def run(output_dir=DEFAULT_OUTPUT, roots=DEFAULT_RESNET_ROOTS):
    outputs = build_audit(output_dir=output_dir, roots=roots)
    for name, payload in outputs.items():
        validate_artifact_sha256(payload)
        write_json(Path(output_dir) / name, payload)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resnet-root", action="append", default=[])
    args = parser.parse_args()
    roots = tuple(args.resnet_root) if args.resnet_root else DEFAULT_RESNET_ROOTS
    outputs = run(Path(args.output_dir), roots)
    report = outputs["v1_p4_retrospective_report.json"]
    compatibility = outputs["v1_p4_measured_pool_compatibility.json"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "historical_record_count": compatibility["historical_pool"][
                    "record_count"
                ],
                "semantic_execution_overlap": compatibility["overlap"][
                    "semantic_execution_candidate_count"
                ],
                "current_runtime_exact_overlap": compatibility["overlap"][
                    "current_runtime_exact_candidate_count"
                ],
                "p5_may_start": report["gate"]["p5_may_start"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
