#!/usr/bin/env python3
"""Replay CPU affinity capacity models on the 200 historical ResNet18 rows.

The score phase reads no candidate throughput.  The evaluation phase joins the
separately sealed historical labels and reports both the full historical stage
layouts and the alternating-device subset accepted by the current V1 DP.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from freeze_cpu_vta_pipeline_v1 import (
    PROTOCOL_ID,
    load_sealed_artifact,
    repo_root,
    seal_artifact,
    validate_artifact_sha256,
)
from iterate_cpu_vta_pipeline_v1_p5b_iteration2 import (
    apply_atomic_cpu_costs,
    summarize_atomic_profiles,
)
from solve_cpu_vta_pipeline_v1_p3 import (
    PHYSICAL_CPU_CORES,
    boundary_components,
    boundary_id,
    build_context,
    stage_ddr_demand,
)


DEFAULT_OUTPUT = repo_root() / "vta/tutorials/frontend/report_out/resource_aware_maxplus"
K_VALUES = (1, 3, 5, 10, 20)


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _is_canonical(path):
    devices = [segment_id.split(":", 1)[0] for segment_id, _ in path]
    return all(left != right for left, right in zip(devices, devices[1:]))


def _prefix_capacity_bounds(cpu_stages):
    bounds = []
    for width in range(1, PHYSICAL_CPU_CORES + 1):
        confined = sum(
            stage["core_work_ms"]
            for stage in cpu_stages
            if stage["threads"] <= width
        )
        bounds.append(confined / width)
    return bounds


def _uniform_per_core_loads(cpu_stages):
    loads = [0.0] * PHYSICAL_CPU_CORES
    for stage in cpu_stages:
        share = stage["core_work_ms"] / stage["threads"]
        for core in range(stage["threads"]):
            loads[core] += share
    return loads


def score_path(path, context):
    stages = []
    total_host_core_work = 0.0
    shared_ddr_ms = 0.0
    for segment_id, threads in path:
        segment = context["profile_segments"][segment_id]
        threads = int(threads)
        cost = context["segment_costs"][segment_id]
        run_ms = float(cost["service_ms_by_threads"][str(threads)])
        core_work_ms = float(cost["host_core_demand_ms_by_threads"][str(threads)])
        ddr = stage_ddr_demand(segment, threads, context)
        stages.append(
            {
                "segment_id": segment_id,
                "device": segment["device"],
                "threads": threads,
                "run_ms": run_ms,
                "core_work_ms": core_work_ms,
                "cpu_owned_boundary_ms": 0.0,
                "ddr_ms": float(ddr["service_ms"]),
            }
        )
        total_host_core_work += core_work_ms
        shared_ddr_ms += float(ddr["service_ms"])

    vta_mutex_boundary_ms = 0.0
    boundary_host_core_ms = 0.0
    direct_boundary_ms = 0.0
    same_device_splits = 0
    for index in range(1, len(stages)):
        previous = stages[index - 1]
        current = stages[index]
        if previous["device"] == current["device"]:
            same_device_splits += 1
            continue
        current_segment = context["profile_segments"][current["segment_id"]]
        cut = int(current_segment["start_index"])
        direction = "{}_to_{}".format(previous["device"], current["device"])
        components = boundary_components(
            context["boundary_costs"][boundary_id(cut, direction)], context
        )
        cpu_stage = previous if previous["device"] == "cpu" else current
        cpu_stage["cpu_owned_boundary_ms"] += components["cpu_owner_ms"]
        vta_mutex_boundary_ms += components["vta_mutex_ms"]
        boundary_host_core_ms += components["host_core_demand_ms"]
        direct_boundary_ms += components["total_ms"]

    total_host_core_work += boundary_host_core_ms
    cpu_stages = [stage for stage in stages if stage["device"] == "cpu"]
    max_cpu_stage_ms = max(
        stage["run_ms"] + stage["cpu_owned_boundary_ms"] for stage in cpu_stages
    )
    single_vta_ms = sum(stage["run_ms"] for stage in stages if stage["device"] == "vta")
    single_vta_with_boundary_ms = single_vta_ms + vta_mutex_boundary_ms
    physical_pool_ms = total_host_core_work / PHYSICAL_CPU_CORES
    max_threads = max(stage["threads"] for stage in cpu_stages)
    legacy_union_average_ms = (
        sum(stage["core_work_ms"] for stage in cpu_stages) / max_threads
    )
    prefix_bounds = _prefix_capacity_bounds(cpu_stages)
    fluid_prefix_ms = max(prefix_bounds)
    uniform_core_loads = _uniform_per_core_loads(cpu_stages)
    uniform_per_core_ms = max(uniform_core_loads)

    common = (max_cpu_stage_ms, single_vta_with_boundary_ms, shared_ddr_ms)
    cpu_bounds = {
        "legacy_union_average": max(physical_pool_ms, legacy_union_average_ms),
        "fluid_nested_prefix": max(physical_pool_ms, fluid_prefix_ms),
        "uniform_per_core_diagnostic": max(physical_pool_ms, uniform_per_core_ms),
    }
    scores = {name: max(*common, value) for name, value in cpu_bounds.items()}
    return {
        "canonical_alternating_devices": _is_canonical(path),
        "same_device_split_count": same_device_splits,
        "scores_ms": scores,
        "resource_load_ms": {
            "max_cpu_stage": max_cpu_stage_ms,
            "single_vta_plus_mutex_boundaries": single_vta_with_boundary_ms,
            "shared_ddr_physical_demand": shared_ddr_ms,
            "total_host_physical_pool": physical_pool_ms,
            "legacy_union_average": legacy_union_average_ms,
            "fluid_prefix_capacity_bounds": prefix_bounds,
            "fluid_prefix_max": fluid_prefix_ms,
            "uniform_per_core_loads": uniform_core_loads,
            "uniform_per_core_max": uniform_per_core_ms,
            "direct_boundary_communication": direct_boundary_ms,
        },
        "cpu_stages": [
            {
                "segment_id": stage["segment_id"],
                "threads": stage["threads"],
                "core_work_ms": stage["core_work_ms"],
            }
            for stage in cpu_stages
        ],
    }


def build_scores(output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir)
    compatibility = load_sealed_artifact(
        output_dir / "v1_p4_measured_pool_compatibility.json",
        "cpu_vta_pipeline_v1_p4_measured_pool_compatibility",
    )
    if compatibility.get("candidate_outcome_fields_present") is not False:
        raise RuntimeError("score phase requires the label-free compatibility artifact")
    profile = load_sealed_artifact(
        output_dir / "v1_profile_manifest.json",
        "cpu_vta_pipeline_v1_profile_manifest",
    )
    local_cost = load_sealed_artifact(
        output_dir / "v1_local_cost_table.json",
        "cpu_vta_pipeline_v1_local_cost_table",
    )
    iteration2 = load_sealed_artifact(
        output_dir / "v1_p5b_iteration2_cpu_measurements.json",
        "cpu_vta_pipeline_v1_p5b_iteration2_cpu_measurements",
    )
    wall, core, _, _ = summarize_atomic_profiles(output_dir)
    context, updated_count = apply_atomic_cpu_costs(
        build_context(profile, local_cost), wall, core
    )

    unique = {}
    for source in compatibility["rows"]:
        candidate_id = source["mapped_execution_candidate_id"]
        if candidate_id not in unique:
            unique[candidate_id] = {
                "execution_candidate_id": candidate_id,
                "path": [list(item) for item in source["mapped_path"]],
                "source_paths": [],
                "historical_candidate_ids": [],
            }
        unique[candidate_id]["source_paths"].append(source["source_path"])
        unique[candidate_id]["historical_candidate_ids"].append(
            source["historical_candidate_id"]
        )

    rows = []
    for candidate_id, source in sorted(unique.items()):
        path = tuple((str(item[0]), int(item[1])) for item in source["path"])
        rows.append({**source, **score_path(path, context)})
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "resnet18_historical_affinity_model_scores",
            "protocol_id": PROTOCOL_ID,
            "scope": "host_only_scores_sealed_before_historical_throughput_join",
            "candidate_outcomes_present": False,
            "candidate_outcomes_used": False,
            "historical_record_count": len(compatibility["rows"]),
            "unique_execution_candidate_count": len(rows),
            "canonical_unique_candidate_count": sum(
                row["canonical_alternating_devices"] for row in rows
            ),
            "profile_manifest_artifact_sha256": profile["artifact_sha256"],
            "local_cost_table_artifact_sha256": local_cost["artifact_sha256"],
            "iteration2_cpu_measurements_artifact_sha256": iteration2[
                "artifact_sha256"
            ],
            "updated_cpu_segment_count": updated_count,
            "models": {
                "legacy_union_average": "max(total host work/4, sum CPU-stage work/max threads)",
                "fluid_nested_prefix": "max(total host work/4, max_k(sum work with "
                "threads<=k)/k); production model for shared [0,threads) masks",
                "uniform_per_core_diagnostic": "max(total host work/4, max core of "
                "sum stage_work/threads); diagnostic only because workers may migrate within mask",
            },
            "limitations": [
                "Historical manifests do not contain the current source fingerprint or explicit "
                "CPU affinity mode.",
                "Adjacent same-device historical VTA stages are kept as separate serialized "
                "services, but their VTA-to-VTA materialization boundary has no current cost model.",
                "The CPU atomic service table was collected after these historical runs and is "
                "used only as outcome-free retrospective service calibration.",
            ],
            "rows": rows,
        }
    )


def _average_ranks(values):
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _spearman(predicted, measured):
    left = _average_ranks(predicted)
    right = _average_ranks(measured)
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def absolute_error_metrics(predicted, measured):
    errors = [left - right for left, right in zip(predicted, measured)]
    absolute = [abs(value) for value in errors]
    ape = [abs(error) / value for error, value in zip(errors, measured)]
    return {
        "mae_ms": statistics.mean(absolute),
        "median_ae_ms": statistics.median(absolute),
        "mape": statistics.mean(ape),
        "median_ape": statistics.median(ape),
        "p95_ape": _percentile(ape, 0.95),
        "mean_signed_error_ms": statistics.mean(errors),
    }


def model_metrics(rows, model):
    predicted = [row["scores_ms"][model] for row in rows]
    measured = [row["measured_cycle_ms"] for row in rows]
    order = sorted(
        rows,
        key=lambda row: (row["scores_ms"][model], row["execution_candidate_id"]),
    )
    measured_order = sorted(
        rows,
        key=lambda row: (row["measured_cycle_ms"], row["execution_candidate_id"]),
    )
    oracle = measured_order[0]["measured_cycle_ms"]
    actual_rank = {
        row["execution_candidate_id"]: rank
        for rank, row in enumerate(measured_order, 1)
    }
    ranking = {
        "measured_pool_oracle_cycle_ms": oracle,
        "measured_pool_oracle_fps": 1000.0 / oracle,
        "measured_pool_oracle_candidate_id": measured_order[0][
            "execution_candidate_id"
        ],
        "predicted_top1_candidate_id": order[0]["execution_candidate_id"],
        "predicted_top1_actual_rank": actual_rank[order[0]["execution_candidate_id"]],
    }
    for k in K_VALUES:
        selected = order[: min(k, len(order))]
        best = min(row["measured_cycle_ms"] for row in selected)
        effective_k = len(selected)
        ranking["regret_at_{}".format(k)] = 1.0 - oracle / best
        ranking["top_{}_recall".format(k)] = len(
            {row["execution_candidate_id"] for row in selected}.intersection(
                row["execution_candidate_id"] for row in measured_order[:effective_k]
            )
        ) / effective_k
    threshold = oracle / 0.95
    ranking["evaluations_to_95_percent_oracle"] = next(
        (
            rank
            for rank, row in enumerate(order, 1)
            if row["measured_cycle_ms"] <= threshold
        ),
        len(order),
    )
    return {
        "count": len(rows),
        "cycle_error": absolute_error_metrics(predicted, measured),
        "spearman": _spearman(predicted, measured),
        "ranking": ranking,
        "top10": [
            {
                "predicted_rank": rank,
                "execution_candidate_id": row["execution_candidate_id"],
                "predicted_cycle_ms": row["scores_ms"][model],
                "measured_cycle_ms": row["measured_cycle_ms"],
                "measured_fps": 1000.0 / row["measured_cycle_ms"],
                "actual_rank": actual_rank[row["execution_candidate_id"]],
            }
            for rank, row in enumerate(order[:10], 1)
        ],
    }


def _topology_id(candidate_id):
    return "__".join(part.rsplit("_t", 1)[0] for part in candidate_id.split("__"))


def current_top20_historical_overlap(ranked_rows, historical_rows):
    """Join after selection; historical labels never influence the current Top-20."""

    measured_order = sorted(
        historical_rows,
        key=lambda row: (row["measured_cycle_ms"], row["execution_candidate_id"]),
    )
    actual_rank = {
        row["execution_candidate_id"]: rank
        for rank, row in enumerate(measured_order, 1)
    }
    by_id = {row["execution_candidate_id"]: row for row in historical_rows}
    by_topology = defaultdict(list)
    for row in historical_rows:
        by_topology[_topology_id(row["execution_candidate_id"])].append(row)

    rows = []
    for candidate in ranked_rows:
        candidate_id = candidate["candidate_id"]
        topology = _topology_id(candidate_id)
        exact = by_id.get(candidate_id)
        topology_matches = sorted(
            by_topology.get(topology, []),
            key=lambda row: (row["measured_cycle_ms"], row["execution_candidate_id"]),
        )
        rows.append(
            {
                "static_rank": candidate["rank"],
                "candidate_id": candidate_id,
                "topology_id": topology,
                "predicted_fps": candidate["predicted_fps"],
                "exact_historical_match": exact is not None,
                "exact_historical_measured_fps": (
                    None if exact is None else 1000.0 / exact["measured_cycle_ms"]
                ),
                "topology_historical_matches": [
                    {
                        "candidate_id": row["execution_candidate_id"],
                        "measured_fps": 1000.0 / row["measured_cycle_ms"],
                        "actual_rank_in_199": actual_rank[row["execution_candidate_id"]],
                    }
                    for row in topology_matches
                ],
            }
        )

    current_topologies = {row["topology_id"] for row in rows}
    covered_topologies = {
        row["topology_id"] for row in rows if row["topology_historical_matches"]
    }
    return {
        "selection_policy": "current iteration2 DP Top-20 over 972528 legal configurations",
        "historical_labels_used_during_selection": False,
        "historical_join_performed_after_selection": True,
        "current_top20_count": len(rows),
        "current_unique_topology_count": len(current_topologies),
        "exact_execution_match_count": sum(
            row["exact_historical_match"] for row in rows
        ),
        "topology_covered_candidate_count": sum(
            bool(row["topology_historical_matches"]) for row in rows
        ),
        "topology_covered_unique_count": len(covered_topologies),
        "topology_uncovered_candidate_count": sum(
            not row["topology_historical_matches"] for row in rows
        ),
        "rows": rows,
    }


def cross_batch_absolute_calibration(warm_rows, theory_rows, model):
    """Diagnose additive, proportional, and affine residuals across batches."""

    def evaluate_direction(train_name, train_rows, test_name, test_rows):
        train_predicted = [row["scores_ms"][model] for row in train_rows]
        train_measured = [row["measured_cycle_ms"] for row in train_rows]
        test_predicted = [row["scores_ms"][model] for row in test_rows]
        test_measured = [row["measured_cycle_ms"] for row in test_rows]
        scale = sum(
            predicted * measured
            for predicted, measured in zip(train_predicted, train_measured)
        ) / sum(predicted * predicted for predicted in train_predicted)
        robust_scale = statistics.median(
            measured / predicted
            for predicted, measured in zip(train_predicted, train_measured)
        )
        additive_offset = statistics.mean(
            measured - predicted
            for predicted, measured in zip(train_predicted, train_measured)
        )
        robust_additive_offset = statistics.median(
            measured - predicted
            for predicted, measured in zip(train_predicted, train_measured)
        )
        predicted_mean = statistics.mean(train_predicted)
        measured_mean = statistics.mean(train_measured)
        centered_denominator = sum(
            (predicted - predicted_mean) ** 2 for predicted in train_predicted
        )
        affine_scale = sum(
            (predicted - predicted_mean) * (measured - measured_mean)
            for predicted, measured in zip(train_predicted, train_measured)
        ) / centered_denominator
        affine_intercept = measured_mean - affine_scale * predicted_mean
        methods = {
            "least_squares_through_origin": {
                "scale": scale,
                "predicted": [scale * value for value in test_predicted],
            },
            "median_measured_to_predicted_ratio": {
                "scale": robust_scale,
                "predicted": [robust_scale * value for value in test_predicted],
            },
            "mean_additive_offset": {
                "offset_ms": additive_offset,
                "predicted": [value + additive_offset for value in test_predicted],
            },
            "median_additive_offset": {
                "offset_ms": robust_additive_offset,
                "predicted": [value + robust_additive_offset for value in test_predicted],
            },
            "least_squares_affine": {
                "scale": affine_scale,
                "intercept_ms": affine_intercept,
                "predicted": [
                    affine_scale * value + affine_intercept for value in test_predicted
                ],
            },
        }
        return {
            "train_batch": train_name,
            "test_batch": test_name,
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "raw_test_error": absolute_error_metrics(test_predicted, test_measured),
            "methods": {
                method: {
                    **{key: value for key, value in result.items() if key != "predicted"},
                    "test_error": absolute_error_metrics(result["predicted"], test_measured),
                }
                for method, result in methods.items()
            },
        }

    directions = [
        evaluate_direction("warm100", warm_rows, "theory100", theory_rows),
        evaluate_direction("theory100", theory_rows, "warm100", warm_rows),
    ]
    return {
        "model": model,
        "purpose": "post-hoc residual-form diagnosis only",
        "candidate_ranking_changed": False,
        "formal_prediction_eligible": False,
        "policy": (
            "fit additive, proportional, and affine forms on one 100-record cohort "
            "and evaluate only on the other cohort"
        ),
        "directions": directions,
        "diagnostic_summary": {
            "mean_additive_offsets_ms": [
                direction["methods"]["mean_additive_offset"]["offset_ms"]
                for direction in directions
            ],
            "median_additive_offsets_ms": [
                direction["methods"]["median_additive_offset"]["offset_ms"]
                for direction in directions
            ],
            "additive_test_mape": [
                direction["methods"]["mean_additive_offset"]["test_error"]["mape"]
                for direction in directions
            ],
            "proportional_test_mape": [
                direction["methods"]["least_squares_through_origin"]["test_error"][
                    "mape"
                ]
                for direction in directions
            ],
            "interpretation": (
                "A roughly 34-37 ms additive residual transfers better than a single "
                "proportional scale. This is a profiling target, not an identified "
                "hardware parameter; controlled runtime, synchronization, cache, and "
                "contention experiments must determine its physical ownership."
            ),
        },
    }


def evaluate_scores(scores, output_dir=DEFAULT_OUTPUT, current_ranked=None):
    output_dir = Path(output_dir)
    labels = load_sealed_artifact(
        output_dir / "v1_p4_historical_labels.json",
        "cpu_vta_pipeline_v1_p4_historical_labels",
    )
    if current_ranked is None:
        current_ranked = load_sealed_artifact(
            output_dir / "v1_p5b_iteration2_ranked_candidates.json",
            "cpu_vta_pipeline_v1_p5b_iteration2_ranked_candidates",
        )
    else:
        validate_artifact_sha256(current_ranked)
    score_rows = {row["execution_candidate_id"]: row for row in scores["rows"]}
    samples = defaultdict(list)
    sample_sources = defaultdict(list)
    for row in labels["rows"]:
        candidate_id = row["mapped_execution_candidate_id"]
        samples[candidate_id].append(float(row["measured_cycle_ms"]))
        sample_sources[candidate_id].append(row["source_path"])
    if set(samples) != set(score_rows):
        raise RuntimeError("historical score and label candidate sets differ")

    unique_rows = []
    all_record_rows = []
    for candidate_id, values in sorted(samples.items()):
        base = score_rows[candidate_id]
        unique_rows.append(
            {**base, "measured_cycle_ms": statistics.median(values)}
        )
        for value, source in zip(values, sample_sources[candidate_id]):
            all_record_rows.append(
                {**base, "measured_cycle_ms": value, "label_source_path": source}
            )

    scopes = {
        "all_200_records_absolute_only": all_record_rows,
        "all_199_unique_layouts": unique_rows,
        "canonical_70_records_absolute_only": [
            row for row in all_record_rows if row["canonical_alternating_devices"]
        ],
        "canonical_69_unique_layouts": [
            row for row in unique_rows if row["canonical_alternating_devices"]
        ],
        "warm100_records_absolute_only": [
            row for row in all_record_rows if "20260507_v23_warm100" in row["label_source_path"]
        ],
        "theory100_records_absolute_only": [
            row for row in all_record_rows if "20260512_theory_build100_board" in row["label_source_path"]
        ],
    }
    models = tuple(scores["models"])
    evaluated = {}
    for scope, rows in scopes.items():
        is_unique = "unique" in scope
        evaluated[scope] = {
            "record_count": len(rows),
            "ranking_metrics_valid": is_unique,
            "models": {model: model_metrics(rows, model) for model in models},
        }
        if not is_unique:
            for result in evaluated[scope]["models"].values():
                result.pop("ranking")
                result.pop("top10")
    production_model = "fluid_nested_prefix"
    calibration = cross_batch_absolute_calibration(
        scopes["warm100_records_absolute_only"],
        scopes["theory100_records_absolute_only"],
        production_model,
    )
    current_overlap = current_top20_historical_overlap(
        current_ranked["rows"], unique_rows
    )
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "resnet18_historical_affinity_model_evaluation",
            "protocol_id": PROTOCOL_ID,
            "scope": "retrospective_historical_runtime_only",
            "scores_artifact_sha256": scores["artifact_sha256"],
            "historical_labels_artifact_sha256": labels["artifact_sha256"],
            "current_ranked_candidates_artifact_sha256": current_ranked[
                "artifact_sha256"
            ],
            "candidate_outcomes_used_to_fit_or_change_scores": False,
            "candidate_outcomes_used_for_cross_batch_absolute_calibration": True,
            "duplicate_execution_candidate_rule": "median measured cycle",
            "scopes": evaluated,
            "cross_batch_absolute_calibration": calibration,
            "current_top20_historical_overlap": current_overlap,
            "interpretation_policy": {
                "primary_absolute_scope": "all_200_records_absolute_only",
                "primary_ranking_scope": "all_199_unique_layouts",
                "current_dp_scope": "canonical_69_unique_layouts",
                "production_model": production_model,
                "uniform_per_core_diagnostic_is_production_eligible": False,
                "cross_batch_scale_is_ranking_evidence": False,
            },
        }
    )


def run(output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir)
    scores = build_scores(output_dir)
    validate_artifact_sha256(scores)
    write_json(output_dir / "v1_historical200_affinity_scores.json", scores)
    evaluation = evaluate_scores(scores, output_dir)
    validate_artifact_sha256(evaluation)
    write_json(output_dir / "v1_historical200_affinity_evaluation.json", evaluation)
    return scores, evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    scores, evaluation = run(args.output_dir)
    primary = evaluation["scopes"]["all_199_unique_layouts"]["models"]
    print(
        json.dumps(
            {
                "historical_records": scores["historical_record_count"],
                "unique_layouts": scores["unique_execution_candidate_count"],
                "canonical_unique_layouts": scores["canonical_unique_candidate_count"],
                "models": {
                    name: {
                        "mape": result["cycle_error"]["mape"],
                        "spearman": result["spearman"],
                        "regret_at_5": result["ranking"]["regret_at_5"],
                    }
                    for name, result in primary.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
