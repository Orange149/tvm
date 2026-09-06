#!/usr/bin/env python3
"""Freeze the V1-P4R prospective comparison and thread-control shortlist."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from build_cpu_vta_pipeline_v1_p1 import enumerate_reachable
from freeze_cpu_vta_pipeline_v1 import (
    PROTOCOL_ID,
    load_sealed_artifact,
    repo_root,
    seal_artifact,
    validate_artifact_sha256,
)
from solve_cpu_vta_pipeline_v1_p3 import (
    CPU_THREAD_CHOICES,
    boundary_components,
    boundary_id,
    build_context,
    candidate_id,
    candidate_record,
    label_from_path,
    path_from_scheme,
    positive_allocations,
    topology_id,
)


DEFAULT_OUTPUT = repo_root() / "vta/tutorials/frontend/report_out/resource_aware_maxplus"
PRIMARY_K = 3
MAX_UNIQUE_CANDIDATES = 20
RANDOM_SEED = 20260902
TYPE_FIXED_PATH = (
    ("cpu:00:00", 4),
    ("vta:01:19", 1),
    ("cpu:20:20", 4),
)


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _b2_score(path, context):
    service = sum(
        float(context["segment_costs"][sid]["service_ms_by_threads"][str(threads)])
        for sid, threads in path
    )
    boundary = 0.0
    for (left_id, _), (right_id, _) in zip(path, path[1:]):
        left = context["profile_segments"][left_id]
        right = context["profile_segments"][right_id]
        direction = "{}_to_{}".format(left["device"], right["device"])
        bid = boundary_id(right["start_index"], direction)
        boundary += boundary_components(context["boundary_costs"][bid], context)["total_ms"]
    return service + boundary


def _b3_score(label):
    return max(
        label.max_cpu_stage_ms,
        label.vta_service_sum_ms + label.vta_mutex_boundary_sum_ms,
    )


def _retain(items, label, score, limit):
    key = (float(score), candidate_id(label.path))
    if len(items) < limit or key < items[-1][0]:
        items.append((key, label))
        items.sort(key=lambda item: item[0])
        del items[limit:]


def _enumerate_baselines(unit_schema, context):
    schemes, _, _ = enumerate_reachable(unit_schema)
    rng = random.Random(RANDOM_SEED)
    b2 = []
    b3 = []
    b4_candidates = []
    random_reservoir = []
    count = 0
    for scheme in schemes:
        cpu_stages = sum(stage["device"] == "cpu" for stage in scheme)
        for allocation in positive_allocations(cpu_stages):
            path = path_from_scheme(scheme, allocation)
            label = label_from_path(path, context)
            count += 1
            _retain(b2, label, _b2_score(path, context), PRIMARY_K)
            _retain(b3, label, _b3_score(label), PRIMARY_K)
            _retain(b4_candidates, label, label.score, 256)
            if len(random_reservoir) < PRIMARY_K:
                random_reservoir.append(label)
            else:
                position = rng.randrange(count)
                if position < PRIMARY_K:
                    random_reservoir[position] = label
    random_reservoir.sort(key=lambda label: candidate_id(label.path))
    return {
        "B2_single_frame": [label for _, label in b2],
        "B3_pipeline_max_load": [label for _, label in b3],
        "B4_v1": [label for _, label in b4_candidates[:PRIMARY_K]],
        "B0_uniform_random": random_reservoir,
        "B4_fill_pool": [label for _, label in b4_candidates],
    }, count


def _thread_controls(reference, context):
    controls = []
    for index, (sid, threads) in enumerate(reference.path):
        if context["profile_segments"][sid]["device"] != "cpu":
            continue
        for replacement in CPU_THREAD_CHOICES:
            if replacement == threads:
                continue
            path = list(reference.path)
            path[index] = (sid, replacement)
            controls.append(label_from_path(tuple(path), context))
    return controls


def _scheme_config(label, context):
    rows = []
    for ordinal, (sid, _) in enumerate(label.path):
        segment = context["profile_segments"][sid]
        rows.append(
            {
                "name": "stage{}".format(ordinal),
                "device": segment["device"],
                "unit_names": list(segment["unit_names"]),
            }
        )
    return rows


def _add(selected, role, label, source_rank=None):
    cid = candidate_id(label.path)
    row = selected.setdefault(cid, {"label": label, "roles": []})
    role_record = {"role": role}
    if source_rank is not None:
        role_record["source_rank"] = int(source_rank)
    if role_record not in row["roles"]:
        row["roles"].append(role_record)


def build_p4r(output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir)
    state = load_sealed_artifact(
        output_dir / "v1_execution_state.json",
        "cpu_vta_pipeline_v1_execution_state",
    )
    if state["current_stage"] not in {"V1-P4B", "V1-P4R"}:
        raise RuntimeError("P4R requires the completed P4B state")
    p4b = load_sealed_artifact(
        output_dir / "v1_p4b_retrospective_report.json",
        "cpu_vta_pipeline_v1_p4b_retrospective_report",
    )
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
    context = build_context(profile, local_cost)
    arms, execution_count = _enumerate_baselines(unit_schema, context)

    selected = {}
    for role in ("B4_v1", "B2_single_frame", "B3_pipeline_max_load", "B0_uniform_random"):
        for rank, label in enumerate(arms[role], 1):
            _add(selected, role, label, rank)
    type_fixed = label_from_path(TYPE_FIXED_PATH, context)
    _add(selected, "B1_type_fixed", type_fixed, 1)

    b4_reference = arms["B4_v1"][0]
    for label in _thread_controls(b4_reference, context):
        _add(selected, "paired_thread_control", label)

    # Spend any remaining budget on distinct high-ranked B4 topologies. This improves
    # partition coverage without changing any model score or looking at outcomes.
    seen_topologies = {topology_id(row["label"].path) for row in selected.values()}
    for rank, label in enumerate(arms["B4_fill_pool"], 1):
        if len(selected) >= MAX_UNIQUE_CANDIDATES:
            break
        tid = topology_id(label.path)
        if tid in seen_topologies:
            continue
        _add(selected, "B4_topology_diversity_fill", label, rank)
        seen_topologies.add(tid)
    for rank, label in enumerate(arms["B4_fill_pool"], 1):
        if len(selected) >= MAX_UNIQUE_CANDIDATES:
            break
        _add(selected, "B4_score_fill", label, rank)

    if len(selected) != MAX_UNIQUE_CANDIDATES:
        raise RuntimeError(
            "P4R expected exactly {} unique candidates, got {}".format(
                MAX_UNIQUE_CANDIDATES, len(selected)
            )
        )

    order = sorted(selected)
    random.Random(RANDOM_SEED).shuffle(order)
    execution_order = {cid: index for index, cid in enumerate(order, 1)}
    rows = []
    for cid, item in sorted(selected.items()):
        label = item["label"]
        record = candidate_record(label, 0, context)
        rows.append(
            {
                "candidate_id": cid,
                "topology_id": topology_id(label.path),
                "path": [list(stage) for stage in label.path],
                "stage_runtime_threads": [int(stage[1]) for stage in label.path],
                "scheme_cfg": _scheme_config(label, context),
                "roles": sorted(
                    item["roles"],
                    key=lambda value: (value["role"], value.get("source_rank", 0)),
                ),
                "frozen_execution_order": execution_order[cid],
                "predicted_scores_ms": {
                    "B2_single_frame": _b2_score(label.path, context),
                    "B3_pipeline_max_load": _b3_score(label),
                    "B4_v1": float(label.score),
                },
                "requires_native_compile": bool(record["requires_shortlist_native_compile"]),
                "candidate_outcome_fields_present": False,
            }
        )

    plan = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p4r_prospective_plan",
            "protocol_id": PROTOCOL_ID,
            "scope": "current_runtime_prospective_pilot",
            "p4b_report_artifact_sha256": p4b["artifact_sha256"],
            "unit_schema_artifact_sha256": unit_schema["artifact_sha256"],
            "profile_manifest_artifact_sha256": profile["artifact_sha256"],
            "local_cost_table_artifact_sha256": local_cost["artifact_sha256"],
            "legal_execution_candidate_count": execution_count,
            "primary_comparison": {
                "metric": "throughput_regret_at_3",
                "primary_k": PRIMARY_K,
                "arms": [
                    "B0_uniform_random",
                    "B2_single_frame",
                    "B3_pipeline_max_load",
                    "B4_v1",
                ],
                "oracle_scope": "best measured throughput in the frozen 20-candidate P4R pool",
                "claim_scope": "pilot evidence only; not the global 972528-configuration oracle",
            },
            "secondary_metrics": [
                "regret_at_1",
                "best_measured_fps_in_each_arm_at_3",
                "paired_thread_control_speedup_and_rank_consistency",
                "compile_reference_failure_rate",
                "profile_build_search_board_wall_clock",
            ],
            "budget": {
                "unique_candidate_compile_limit": MAX_UNIQUE_CANDIDATES,
                "unique_candidate_board_limit": MAX_UNIQUE_CANDIDATES,
                "all_gate_passing_candidates_are_measured": True,
                "oracle_based_early_stopping_allowed": False,
                "failure_or_timeout_consumes_budget": True,
            },
            "correctness_gate": {
                "compile_before_board": True,
                "native_only_no_fallback": True,
                "p5a_independent_expected_tensor_and_comparator_required": True,
                "p5b_board_native_vs_reference_required_before_timing": True,
                "stable_output_hash_alone_is_sufficient": False,
                "serial_pipeline_equivalence_required": True,
            },
            "execution_policy": {
                "frozen_order_seed": RANDOM_SEED,
                "board_host": "192.168.1.234",
                "rpc_port": 9090,
                "queue_depth": 2,
                "poll_sleep_ns": 1000,
                "cpu_affinity": "independent_overlapping_prefix_masks_v1",
                "performance_runs": 20,
                "warmup_frames": 2,
                "candidate_order_changes_after_outcomes": False,
            },
            "decision_rule": "P5B may start only after every candidate has a local compile/package "
            "audit and an independent expected-tensor comparator is available; each candidate must "
            "then pass board native-vs-reference correctness before timing. P4R itself does not "
            "authorize board execution",
        }
    )
    manifest = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p4r_candidate_manifest",
            "protocol_id": PROTOCOL_ID,
            "prospective_plan_artifact_sha256": plan["artifact_sha256"],
            "candidate_count": len(rows),
            "candidate_outcome_fields_present": False,
            "rows": rows,
        }
    )
    next_state = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_execution_state",
            "protocol_id": PROTOCOL_ID,
            "current_stage": "V1-P4R",
            "current_stage_status": "completed_awaiting_user_confirmation",
            "next_stage": "V1-P5A",
            "next_stage_may_start": False,
            "advance_requires_user_confirmation": True,
            "blocking_evidence": [
                "20_candidate_local_compile_and_package_audit_not_run",
                "independent_expected_tensor_and_raw_output_comparator_not_yet_demonstrated",
                "board_performance_experiment_requires_explicit_user_confirmation",
            ],
            "p4r_plan_artifact_sha256": plan["artifact_sha256"],
            "p4r_manifest_artifact_sha256": manifest["artifact_sha256"],
        }
    )
    return {
        "v1_p4r_prospective_plan.json": plan,
        "v1_p4r_candidate_manifest.json": manifest,
        "v1_execution_state.json": next_state,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    outputs = build_p4r(Path(args.output_dir))
    for name, payload in outputs.items():
        validate_artifact_sha256(payload)
        write_json(Path(args.output_dir) / name, payload)
    print(
        json.dumps(
            {
                "candidate_count": outputs["v1_p4r_candidate_manifest.json"]["candidate_count"],
                "next_stage": outputs["v1_execution_state.json"]["next_stage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
