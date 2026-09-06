#!/usr/bin/env python3
"""Build the P5B Iteration2 CPU segment model and rerun exact Top-K search."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from pathlib import Path

from freeze_cpu_vta_pipeline_v1 import (
    PROTOCOL_ID,
    load_sealed_artifact,
    repo_root,
    seal_artifact,
    validate_artifact_sha256,
)
from solve_cpu_vta_pipeline_v1_p3 import (
    build_context,
    candidate_id,
    candidate_record,
    enumerate_oracle,
    label_from_path,
    solve_k_best_label_setting,
)


DEFAULT_OUTPUT = repo_root() / "vta/tutorials/frontend/report_out/resource_aware_maxplus"
ITERATION_DIR = "v1_p5b_iteration2"
THREADS = (1, 2, 3, 4)
WARMUP = 2
EXPECTED_ROWS = 7


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def nearest_rank_percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summarize_atomic_profiles(output_dir):
    root = Path(output_dir) / ITERATION_DIR / "atomic_cpu_profile"
    wall = {}
    core = {}
    sessions = []
    for threads in THREADS:
        path = root / "t{}".format(threads) / "stage_serial_result.jsonl"
        rows = read_jsonl(path)
        if len(rows) != EXPECTED_ROWS:
            raise RuntimeError("atomic profile {} has {} rows".format(path, len(rows)))
        if any(int(row["stage_count"]) != 21 for row in rows):
            raise RuntimeError("atomic profile must contain 21 stages")
        fingerprints = sorted(
            {
                output["fnv1a64"]
                for row in rows
                for output in row.get("raw_outputs", [])
            }
        )
        if len(fingerprints) != 1:
            raise RuntimeError("atomic output is not deterministic for t={}".format(threads))
        scored = rows[WARMUP:]
        units = []
        for index in range(21):
            wall[index, threads] = statistics.median(
                float(row["stage{}_run_ms".format(index)]) for row in scored
            )
            core[index, threads] = statistics.median(
                float(row["stage{}_run_process_cpu_ms".format(index)])
                for row in scored
            )
            units.append(
                {
                    "unit_index": index,
                    "run_ms_median": wall[index, threads],
                    "run_process_cpu_ms_median": core[index, threads],
                }
            )
        sessions.append(
            {
                "threads": threads,
                "path": str(path),
                "rows": len(rows),
                "warmup": WARMUP,
                "scored_rows": len(scored),
                "output_fingerprint": fingerprints[0],
                "top1_values": sorted({int(row["top1"]) for row in rows}),
                "unit_measurements": units,
            }
        )
    reference = read_json(root / "t1/reference_comparison.json")
    if not reference.get("passed"):
        raise RuntimeError("atomic CPU reference comparison failed")
    return wall, core, sessions, reference


def profile_median(path, stage_index, field="run_ms"):
    rows = read_jsonl(path)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError("holdout profile {} has {} rows".format(path, len(rows)))
    return statistics.median(
        float(row["stage{}_{}".format(stage_index, field)]) for row in rows[WARMUP:]
    )


def build_grouped_holdouts(output_dir, wall):
    output_dir = Path(output_dir)
    iteration = output_dir / ITERATION_DIR
    checks = []
    definitions = (
        ("stem", 0, 0, "family_profile", 0),
        ("transition_block", 10, 12, "family_profile", 2),
        ("terminal_block_plus_head", 18, 20, "family_profile", 4),
        ("terminal_regular_block", 18, 19, "head_profile", 2),
        ("head", 20, 20, "head_profile", 3),
    )
    for threads in THREADS:
        for group, start, end, profile_name, stage_index in definitions:
            path = iteration / profile_name / "t{}".format(threads) / "stage_serial_result.jsonl"
            predicted = sum(wall[index, threads] for index in range(start, end + 1))
            observed = profile_median(path, stage_index)
            checks.append(
                {
                    "group": group,
                    "threads": threads,
                    "unit_range": [start, end],
                    "predicted_ms": predicted,
                    "observed_ms": observed,
                    "ape": abs(predicted - observed) / observed,
                    "source": str(path),
                    "source_scored_rows": EXPECTED_ROWS - WARMUP,
                }
            )
    # These older one-row measurements are a deliberately weaker extrapolation check.
    for threads in (2, 3, 4):
        path = (
            output_dir
            / "v1_p5b_iteration1/thread{}/correctness/stage_serial_result.jsonl".format(
                threads
            )
        )
        rows = read_jsonl(path)
        if len(rows) != 1:
            raise RuntimeError("legacy fused holdout must contain one row")
        predicted = sum(wall[index, threads] for index in range(3))
        observed = float(rows[0]["stage0_run_ms"])
        checks.append(
            {
                "group": "early_fused_legacy_single_sample",
                "threads": threads,
                "unit_range": [0, 2],
                "predicted_ms": predicted,
                "observed_ms": observed,
                "ape": abs(predicted - observed) / observed,
                "source": str(path),
                "source_scored_rows": 1,
            }
        )
    errors = [item["ape"] for item in checks]
    metrics = {
        "check_count": len(checks),
        "median_ape": statistics.median(errors),
        "p95_ape": nearest_rank_percentile(errors, 0.95),
        "max_ape": max(errors),
    }
    gate = {
        "median_ape_le_5pct": metrics["median_ape"] <= 0.05,
        "p95_ape_le_15pct": metrics["p95_ape"] <= 0.15,
        "max_ape_le_25pct": metrics["max_ape"] <= 0.25,
        "at_least_five_group_families": len({item["group"] for item in checks}) >= 5,
    }
    return checks, metrics, gate


def apply_atomic_cpu_costs(context, wall, core):
    revised = copy.deepcopy(context)
    updated = 0
    for segment_id, cost in revised["segment_costs"].items():
        if not segment_id.startswith("cpu:"):
            continue
        start, end = (int(value) for value in segment_id.split(":")[1:])
        cost["service_ms_by_threads"] = {
            str(threads): sum(wall[index, threads] for index in range(start, end + 1))
            for threads in THREADS
        }
        cost["host_core_demand_ms_by_threads"] = {
            str(threads): sum(core[index, threads] for index in range(start, end + 1))
            for threads in THREADS
        }
        cost["compiler_status"] = "atomic_composition_grouped_holdout_passed"
        updated += 1
    return revised, updated


def parse_candidate_id(text):
    path = []
    for item in text.split("__"):
        segment, threads = item.rsplit("_t", 1)
        device, start, end = segment.split("-")
        path.append(("{}:{}:{}".format(device, start, end), int(threads)))
    return tuple(path)


def rank_measured_pool(context, output_dir):
    review = read_json(Path(output_dir) / "v1_p5b_iteration1_review.json")
    rows = []
    for item in review["measured_pool"]:
        label = label_from_path(parse_candidate_id(item["candidate_id"]), context)
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "measured_fps": float(item["fps"]),
                "measured_ii_ms": 1000.0 / float(item["fps"]),
                "predicted_fps": 1000.0 / label.score,
                "predicted_ii_ms": label.score,
            }
        )
    measured_order = [
        item["candidate_id"]
        for item in sorted(rows, key=lambda row: (-row["measured_fps"], row["candidate_id"]))
    ]
    predicted_order = [
        item["candidate_id"]
        for item in sorted(rows, key=lambda row: (row["predicted_ii_ms"], row["candidate_id"]))
    ]
    return rows, {
        "measured_order": measured_order,
        "predicted_order": predicted_order,
        "exact_order_match": measured_order == predicted_order,
        "top1_match": measured_order[0] == predicted_order[0],
        "cycle_mape": statistics.mean(
            abs(item["predicted_ii_ms"] - item["measured_ii_ms"])
            / item["measured_ii_ms"]
            for item in rows
        ),
        "evidence_role": "blind replay only; measured throughput was not used to fit or rank",
    }


def run_iteration(output_dir=DEFAULT_OUTPUT, top_k=20):
    output_dir = Path(output_dir)
    unit_schema = load_sealed_artifact(
        output_dir / "v1_unit_and_boundary_schema.json",
        "cpu_vta_pipeline_v1_unit_and_boundary_schema",
    )
    profile = load_sealed_artifact(
        output_dir / "v1_profile_manifest.json", "cpu_vta_pipeline_v1_profile_manifest"
    )
    local_cost = load_sealed_artifact(
        output_dir / "v1_local_cost_table.json", "cpu_vta_pipeline_v1_local_cost_table"
    )
    wall, core, sessions, reference = summarize_atomic_profiles(output_dir)
    holdouts, holdout_metrics, holdout_gate = build_grouped_holdouts(output_dir, wall)
    if not all(holdout_gate.values()):
        raise RuntimeError("atomic CPU composition holdout failed: {}".format(holdout_gate))

    context, updated_count = apply_atomic_cpu_costs(
        build_context(profile, local_cost), wall, core
    )
    dp_labels, dp_stats = solve_k_best_label_setting(context, top_k=top_k)
    oracle, _, topology_count, execution_count = enumerate_oracle(
        unit_schema, context, top_k=top_k
    )
    dp_ids = [candidate_id(label.path) for label in dp_labels]
    oracle_ids = [candidate_id(label.path) for label in oracle]
    if dp_ids != oracle_ids:
        raise RuntimeError("Iteration2 DP Top-K differs from exact enumeration")

    measured_rows, replay = rank_measured_pool(context, output_dir)
    if not replay["top1_match"] or not replay["exact_order_match"]:
        raise RuntimeError("Iteration2 does not preserve the measured-pool ranking")
    measured_ids = {item["candidate_id"] for item in measured_rows}
    records = []
    for rank, label in enumerate(dp_labels, 1):
        record = candidate_record(label, rank, context)
        record["scheme_cfg"] = [
            {
                "name": "stage{}".format(index),
                "device": context["profile_segments"][segment_id]["device"],
                "unit_names": list(context["profile_segments"][segment_id]["unit_names"]),
            }
            for index, (segment_id, _) in enumerate(label.path)
        ]
        record["stage_runtime_threads"] = [threads for _, threads in label.path]
        record["already_measured_candidate"] = record["candidate_id"] in measured_ids
        record["cost_provenance"]["iteration_policy"] = (
            "sum 21 independently measured CPU unit services after grouped fused-segment holdout"
        )
        records.append(record)
    validation = next(record for record in records if not record["already_measured_candidate"])

    measurements = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p5b_iteration2_cpu_measurements",
            "protocol_id": PROTOCOL_ID,
            "atomic_sessions": sessions,
            "reference_comparison": reference,
            "grouped_holdout_checks": holdouts,
            "grouped_holdout_metrics": holdout_metrics,
            "grouped_holdout_gate": holdout_gate,
            "updated_cpu_segment_count": updated_count,
            "candidate_throughput_consumed": False,
        }
    )
    ranked = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p5b_iteration2_ranked_candidates",
            "protocol_id": PROTOCOL_ID,
            "measurement_artifact_sha256": measurements["artifact_sha256"],
            "candidate_throughput_used_for_ranking": False,
            "cpu_resource_correction": "max(total_host_core_ms/4, "
            "max_k(sum(cpu_stage_core_ms where threads<=k)/k))",
            "validation_candidate": validation["candidate_id"],
            "validation_candidate_rank": validation["rank"],
            "rows": records,
        }
    )
    validation_scheme = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p5b_iteration2_validation_scheme",
            "protocol_id": PROTOCOL_ID,
            "candidate_id": validation["candidate_id"],
            "scheme_name": validation["candidate_id"],
            "scheme_cfg": validation["scheme_cfg"],
            "stage_runtime_threads": validation["stage_runtime_threads"],
            "selection_policy": "highest-ranked candidate absent from measured candidate pool",
        }
    )
    report = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p5b_iteration2_review",
            "protocol_id": PROTOCOL_ID,
            "ranked_artifact_sha256": ranked["artifact_sha256"],
            "dp_top_k_equals_enumeration": True,
            "topology_count": topology_count,
            "execution_configuration_count": execution_count,
            "dp_stats": dp_stats,
            "measured_pool_replay": {"rows": measured_rows, **replay},
            "gate": {
                "atomic_reference_passed": bool(reference.get("passed")),
                "grouped_holdout_passed": all(holdout_gate.values()),
                "dp_top_k_exact": dp_ids == oracle_ids,
                "measured_pool_top1_match": replay["top1_match"],
                "measured_pool_exact_order_match": replay["exact_order_match"],
                "validation_candidate_is_unmeasured": validation["candidate_id"]
                not in measured_ids,
            },
            "remaining_limitation": "absolute cycle prediction still omits concurrent "
            "CPU/VTA shared-DDR and scheduler efficiency loss; validate ranking with one unseen "
            "candidate before adding a contention correction",
        }
    )
    outputs = {
        "v1_p5b_iteration2_cpu_measurements.json": measurements,
        "v1_p5b_iteration2_ranked_candidates.json": ranked,
        "v1_p5b_iteration2_validation_scheme.json": validation_scheme,
        "v1_p5b_iteration2_review.json": report,
    }
    for name, payload in outputs.items():
        validate_artifact_sha256(payload)
        write_json(output_dir / name, payload)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    outputs = run_iteration(args.output_dir, args.top_k)
    ranked = outputs["v1_p5b_iteration2_ranked_candidates.json"]
    review = outputs["v1_p5b_iteration2_review.json"]
    print(
        json.dumps(
            {
                "top1": ranked["rows"][0]["candidate_id"],
                "top1_predicted_fps": ranked["rows"][0]["predicted_fps"],
                "validation_candidate": ranked["validation_candidate"],
                "measured_pool_exact_order_match": review["measured_pool_replay"][
                    "exact_order_match"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
