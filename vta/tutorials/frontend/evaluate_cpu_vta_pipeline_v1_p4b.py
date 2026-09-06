#!/usr/bin/env python3
"""Freeze V1 baseline scores, then evaluate them on historical-runtime labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from freeze_cpu_vta_pipeline_v1 import (
    PROTOCOL_ID,
    load_sealed_artifact,
    repo_root,
    seal_artifact,
    validate_artifact_sha256,
)
from solve_cpu_vta_pipeline_v1_p3 import build_context, candidate_record, label_from_path


DEFAULT_OUTPUT = repo_root() / "vta/tutorials/frontend/report_out/resource_aware_maxplus"
K_VALUES = (1, 3, 5, 10, 20)
RANDOM_SEED = 20260902
RANDOM_REPETITIONS = 2000
TYPE_FIXED_PATH = (
    ("cpu:00:00", 4),
    ("vta:01:19", 1),
    ("cpu:20:20", 4),
)


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def canonical_sha256(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_scoring_protocol(compatibility):
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p4b_scoring_protocol",
            "protocol_id": PROTOCOL_ID,
            "scope": "host_only_historical_runtime_exploratory_retrospective",
            "compatibility_artifact_sha256": compatibility["artifact_sha256"],
            "candidate_outcomes_available_to_scoring": False,
            "candidate_outcomes_used_by_scoring": False,
            "k_values": list(K_VALUES),
            "ranking_order": "ascending score, then lexicographic execution candidate id",
            "candidate_deduplication": "mapped execution candidate id",
            "duplicate_label_rule": "median measured cycle after scores are sealed",
            "models": {
                "B0_uniform_random": {
                    "kind": "random_order_distribution",
                    "seed": RANDOM_SEED,
                    "repetitions": RANDOM_REPETITIONS,
                },
                "B0_stratified_random": {
                    "kind": "round_robin_over_shuffled_vta_island_count_strata",
                    "seed": RANDOM_SEED,
                    "repetitions": RANDOM_REPETITIONS,
                },
                "B1_type_fixed": {
                    "kind": "single_fixed_policy_candidate",
                    "path": [list(item) for item in TYPE_FIXED_PATH],
                    "cpu_thread_policy": "threads_4_for_both_cpu_stages",
                    "rationale": "stem/head are CPU-only; place every VTA-supported unit "
                    "01..19 in one VTA island and use the fixed maximum TVM thread setting",
                    "missing_label_policy": "report unavailable; do not substitute nearest candidate",
                },
                "B2_single_frame": {
                    "formula": "sum(stage isolated run service)+sum(direct boundary service)",
                    "shared_resource_overlap": False,
                },
                "B3_pipeline_max_load": {
                    "formula": "max(max CPU stage with owned boundary, single VTA service plus "
                    "VTA-mutex boundary)",
                    "cpu_core_pool_bound": False,
                    "shared_ddr_bound": False,
                },
                "B4_v1": {
                    "formula": "max(max CPU stage with owned boundary, single VTA service plus "
                    "VTA-mutex boundary, total host core-ms/4, shared DDR demand)",
                },
            },
            "metrics": {
                "regret_at_k": "1 - measured_pool_oracle_cycle_ms / best_cycle_in_top_k_ms",
                "evaluations_to_95_percent_oracle": "first rank with cycle <= oracle_cycle/0.95",
                "top_k_recall": "intersection(predicted top K, measured top K)/K",
                "oracle_hit_at_k": "whether an exact measured-pool oracle is in predicted top K",
                "spearman": "diagnostic only",
            },
            "gate": {
                "exploratory_same_pool": "B4 must beat both B2 and B3 and both B0 median "
                "regrets at the same preregistered K",
                "full_baseline_gate": "also requires B4 regret@1 to beat the exact one-candidate "
                "B1 regret at the same one-evaluation budget",
                "p5_policy": "historical runtime labels cannot authorize current-runtime P5",
            },
        }
    )


def _semantic_rows(compatibility):
    allowed = set(compatibility["overlap"]["semantic_execution_candidate_ids"])
    unique = {}
    for source in compatibility["rows"]:
        candidate_id = source["mapped_execution_candidate_id"]
        if candidate_id not in allowed or candidate_id in unique:
            continue
        unique[candidate_id] = {
            "execution_candidate_id": candidate_id,
            "path": [list(item) for item in source["mapped_path"]],
            "vta_island_count": int(source["vta_island_count"]),
        }
    return [unique[key] for key in sorted(unique)]


def _random_orders(rows, stratified):
    ids = [row["execution_candidate_id"] for row in rows]
    strata = defaultdict(list)
    for row in rows:
        strata[int(row["vta_island_count"])].append(row["execution_candidate_id"])
    rng = random.Random(RANDOM_SEED)
    orders = []
    for _ in range(RANDOM_REPETITIONS):
        if not stratified:
            order = list(ids)
            rng.shuffle(order)
        else:
            queues = {key: list(values) for key, values in sorted(strata.items())}
            for values in queues.values():
                rng.shuffle(values)
            order = []
            while any(queues.values()):
                keys = [key for key, values in queues.items() if values]
                rng.shuffle(keys)
                order.extend(queues[key].pop() for key in keys)
        orders.append(order)
    return orders


def build_frozen_scores(output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir)
    compatibility = load_sealed_artifact(
        output_dir / "v1_p4_measured_pool_compatibility.json",
        "cpu_vta_pipeline_v1_p4_measured_pool_compatibility",
    )
    if compatibility.get("candidate_outcome_fields_present") is not False:
        raise RuntimeError("P4B score phase requires a physically label-free candidate pool")
    if not compatibility["comparison_eligible"]:
        raise RuntimeError("P4B requires a comparable semantic historical pool")
    profile = load_sealed_artifact(
        output_dir / "v1_profile_manifest.json",
        "cpu_vta_pipeline_v1_profile_manifest",
    )
    local_cost = load_sealed_artifact(
        output_dir / "v1_local_cost_table.json",
        "cpu_vta_pipeline_v1_local_cost_table",
    )
    scoring_protocol = build_scoring_protocol(compatibility)
    context = build_context(profile, local_cost)
    rows = _semantic_rows(compatibility)
    scored_rows = []
    for source in rows:
        path = tuple((str(sid), int(threads)) for sid, threads in source["path"])
        label = label_from_path(path, context)
        record = candidate_record(label, 0, context)
        loads = record["resource_load_ms"]
        stage_run_sum = sum(float(stage["run_service_ms"]) for stage in record["stages"])
        scores = {
            "B2_single_frame_ms": stage_run_sum
            + float(loads["direct_boundary_communication"]),
            "B3_pipeline_max_load_ms": max(
                float(loads["max_cpu_stage"]),
                float(loads["single_vta_plus_mutex_boundaries"]),
            ),
            "B4_v1_ms": float(record["predicted_ii_ms"]),
        }
        scored_rows.append({**source, "scores": scores})

    uniform_orders = _random_orders(rows, stratified=False)
    stratified_orders = _random_orders(rows, stratified=True)
    type_fixed_id = "cpu-00-00_t4__vta-01-19_t1__cpu-20-20_t4"
    type_fixed_label = label_from_path(TYPE_FIXED_PATH, context)
    type_fixed_record = candidate_record(type_fixed_label, 0, context)
    frozen_scores = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p4b_frozen_scores",
            "protocol_id": PROTOCOL_ID,
            "scope": "scores_sealed_before_historical_throughput_join",
            "scoring_protocol_artifact_sha256": scoring_protocol["artifact_sha256"],
            "compatibility_artifact_sha256": compatibility["artifact_sha256"],
            "profile_manifest_artifact_sha256": profile["artifact_sha256"],
            "local_cost_table_artifact_sha256": local_cost["artifact_sha256"],
            "candidate_outcome_fields_present": False,
            "candidate_outcome_fields_used": False,
            "candidate_count": len(scored_rows),
            "type_fixed": {
                "execution_candidate_id": type_fixed_id,
                "present_in_semantic_pool": any(
                    row["execution_candidate_id"] == type_fixed_id for row in scored_rows
                ),
                "current_search_grammar_valid": True,
                "requires_shortlist_native_compile": type_fixed_record[
                    "requires_shortlist_native_compile"
                ],
            },
            "random_order_generators": {
                "uniform_orders_sha256": canonical_sha256(uniform_orders),
                "stratified_orders_sha256": canonical_sha256(stratified_orders),
                "seed": RANDOM_SEED,
                "repetitions": RANDOM_REPETITIONS,
            },
            "rows": scored_rows,
        }
    )
    return {
        "v1_p4b_scoring_protocol.json": scoring_protocol,
        "v1_p4b_frozen_scores.json": frozen_scores,
    }


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
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def ranking_metrics(order, measured):
    oracle = min(measured.values())
    measured_order = sorted(measured, key=lambda key: (measured[key], key))
    oracle_ids = {
        key for key, value in measured.items() if abs(value - oracle) <= 1.0e-12
    }
    result = {
        "measured_pool_oracle_cycle_ms": oracle,
        "measured_pool_oracle_candidate_ids": sorted(oracle_ids),
    }
    for k in K_VALUES:
        effective_k = min(k, len(order))
        selected = order[:effective_k]
        best = min(measured[key] for key in selected)
        result["regret_at_{}".format(k)] = 1.0 - oracle / best
        result["best_cycle_at_{}_ms".format(k)] = best
        result["top_{}_recall".format(k)] = len(
            set(selected).intersection(measured_order[:effective_k])
        ) / float(effective_k)
        result["oracle_hit_at_{}".format(k)] = bool(
            set(selected).intersection(oracle_ids)
        )
    threshold = oracle / 0.95
    result["evaluations_to_95_percent_oracle"] = next(
        (index for index, key in enumerate(order, 1) if measured[key] <= threshold),
        len(order),
    )
    return result


def _random_metrics(orders, measured):
    samples = defaultdict(list)
    for order in orders:
        metrics = ranking_metrics(order, measured)
        for key, value in metrics.items():
            if (
                key.startswith("regret_at_")
                or key.startswith("top_") and key.endswith("_recall")
                or key == "evaluations_to_95_percent_oracle"
            ):
                samples[key].append(float(value))
    return {
        key: {
            "median": statistics.median(values),
            "p10": sorted(values)[int(0.10 * (len(values) - 1))],
            "p90": sorted(values)[int(0.90 * (len(values) - 1))],
        }
        for key, values in sorted(samples.items())
    }


def _random_beats_reference(orders, measured, reference_metrics):
    samples = [ranking_metrics(order, measured) for order in orders]
    result = {}
    for k in K_VALUES:
        key = "regret_at_{}".format(k)
        result["probability_random_regret_le_b4_at_{}".format(k)] = sum(
            item[key] <= reference_metrics[key] + 1.0e-12 for item in samples
        ) / float(len(samples))
    key = "evaluations_to_95_percent_oracle"
    result["probability_random_reaches_95pct_no_later_than_b4"] = sum(
        item[key] <= reference_metrics[key] for item in samples
    ) / float(len(samples))
    return result


def evaluate_frozen_scores(output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir)
    scoring_protocol = load_sealed_artifact(
        output_dir / "v1_p4b_scoring_protocol.json",
        "cpu_vta_pipeline_v1_p4b_scoring_protocol",
    )
    frozen_scores = load_sealed_artifact(
        output_dir / "v1_p4b_frozen_scores.json",
        "cpu_vta_pipeline_v1_p4b_frozen_scores",
    )
    compatibility = load_sealed_artifact(
        output_dir / "v1_p4_measured_pool_compatibility.json",
        "cpu_vta_pipeline_v1_p4_measured_pool_compatibility",
    )
    historical_labels = load_sealed_artifact(
        output_dir / "v1_p4_historical_labels.json",
        "cpu_vta_pipeline_v1_p4_historical_labels",
    )
    if frozen_scores["scoring_protocol_artifact_sha256"] != scoring_protocol["artifact_sha256"]:
        raise RuntimeError("P4B scoring protocol hash mismatch")
    if frozen_scores["compatibility_artifact_sha256"] != compatibility["artifact_sha256"]:
        raise RuntimeError("P4B compatibility hash mismatch")
    if historical_labels["compatibility_artifact_sha256"] != compatibility[
        "artifact_sha256"
    ]:
        raise RuntimeError("P4B historical label pool hash mismatch")

    allowed = {row["execution_candidate_id"] for row in frozen_scores["rows"]}
    label_samples = defaultdict(list)
    for row in historical_labels["rows"]:
        candidate_id = row["mapped_execution_candidate_id"]
        if candidate_id in allowed:
            label_samples[candidate_id].append(float(row["measured_cycle_ms"]))
    measured = {
        key: statistics.median(values) for key, values in sorted(label_samples.items())
    }
    if set(measured) != allowed:
        raise RuntimeError("P4B score/label candidate sets differ")

    score_rows = {row["execution_candidate_id"]: row for row in frozen_scores["rows"]}
    models = {}
    for model, field in (
        ("B2_single_frame", "B2_single_frame_ms"),
        ("B3_pipeline_max_load", "B3_pipeline_max_load_ms"),
        ("B4_v1", "B4_v1_ms"),
    ):
        order = sorted(allowed, key=lambda key: (score_rows[key]["scores"][field], key))
        predicted = [score_rows[key]["scores"][field] for key in sorted(allowed)]
        observed = [measured[key] for key in sorted(allowed)]
        models[model] = {
            "status": "evaluated_exploratory_historical_runtime",
            "metrics": ranking_metrics(order, measured),
            "spearman": _spearman(predicted, observed),
            "ranked_execution_candidate_ids": order,
        }

    rows = [score_rows[key] for key in sorted(allowed)]
    uniform_orders = _random_orders(rows, stratified=False)
    stratified_orders = _random_orders(rows, stratified=True)
    if canonical_sha256(uniform_orders) != frozen_scores["random_order_generators"][
        "uniform_orders_sha256"
    ]:
        raise RuntimeError("uniform random order hash mismatch")
    if canonical_sha256(stratified_orders) != frozen_scores["random_order_generators"][
        "stratified_orders_sha256"
    ]:
        raise RuntimeError("stratified random order hash mismatch")
    models["B0_uniform_random"] = {
        "status": "evaluated_distribution",
        "metrics": _random_metrics(uniform_orders, measured),
    }
    models["B0_stratified_random"] = {
        "status": "evaluated_distribution",
        "metrics": _random_metrics(stratified_orders, measured),
    }

    type_fixed_id = frozen_scores["type_fixed"]["execution_candidate_id"]
    if type_fixed_id in measured:
        oracle = min(measured.values())
        fixed_cycle = measured[type_fixed_id]
        models["B1_type_fixed"] = {
            "status": "evaluated_exact_policy_label",
            "metrics": {
                "measured_pool_oracle_cycle_ms": oracle,
                "fixed_policy_cycle_ms": fixed_cycle,
                "regret_at_1": 1.0 - oracle / fixed_cycle,
                "within_95_percent_oracle": fixed_cycle <= oracle / 0.95,
            },
        }
    else:
        models["B1_type_fixed"] = {
            "status": "not_evaluable_exact_policy_absent_from_historical_pool",
            "metrics": None,
            "nearest_candidate_substitution_used": False,
        }

    exploratory_ks = []
    for k in K_VALUES:
        key = "regret_at_{}".format(k)
        b4 = models["B4_v1"]["metrics"][key]
        if (
            b4 < models["B2_single_frame"]["metrics"][key]
            and b4 < models["B3_pipeline_max_load"]["metrics"][key]
            and b4 < models["B0_uniform_random"]["metrics"][key]["median"]
            and b4 < models["B0_stratified_random"]["metrics"][key]["median"]
        ):
            exploratory_ks.append(k)
    exploratory_gate = bool(exploratory_ks)
    b1_available = models["B1_type_fixed"]["metrics"] is not None
    b1_equal_budget_pass = bool(
        b1_available
        and 1 in exploratory_ks
        and models["B4_v1"]["metrics"]["regret_at_1"]
        < models["B1_type_fixed"]["metrics"]["regret_at_1"]
    )
    full_passing_ks = [1] if b1_equal_budget_pass else []
    full_gate = bool(full_passing_ks)
    b4_evaluations_to_95 = models["B4_v1"]["metrics"][
        "evaluations_to_95_percent_oracle"
    ]
    uniform_random_median_evaluations_to_95 = models["B0_uniform_random"]["metrics"][
        "evaluations_to_95_percent_oracle"
    ]["median"]
    thread_vectors = compatibility["historical_pool"].get(
        "semantic_cpu_thread_vector_counts", {}
    )
    report = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p4b_retrospective_report",
            "protocol_id": PROTOCOL_ID,
            "scope": "exploratory_same_segments_threads_historical_runtime_only",
            "scoring_protocol_artifact_sha256": scoring_protocol["artifact_sha256"],
            "frozen_scores_artifact_sha256": frozen_scores["artifact_sha256"],
            "compatibility_artifact_sha256": compatibility["artifact_sha256"],
            "historical_labels_artifact_sha256": historical_labels[
                "artifact_sha256"
            ],
            "candidate_outcomes_used_to_fit_or_change_scores": False,
            "unique_candidate_count": len(measured),
            "duplicate_label_rule_applied_count": sum(
                len(values) > 1 for values in label_samples.values()
            ),
            "models": models,
            "gate": {
                "b4_beats_b0_b2_b3_at_same_preregistered_k": exploratory_gate,
                "passing_k_values": exploratory_ks,
                "exact_b1_label_available": b1_available,
                "b4_regret_at_1_beats_b1_equal_budget": (
                    b1_equal_budget_pass if b1_available else None
                ),
                "full_gate_passing_k_values": full_passing_ks,
                "full_b0_b4_gate_passed": full_gate,
                "current_runtime_exact_labels_available": False,
                "p5_may_start": False,
            },
            "board_budget_diagnostic": {
                "b4_evaluations_to_95_percent_oracle": b4_evaluations_to_95,
                "uniform_random_median_evaluations_to_95_percent_oracle": (
                    uniform_random_median_evaluations_to_95
                ),
                "b4_beats_uniform_random_median": (
                    b4_evaluations_to_95
                    < uniform_random_median_evaluations_to_95
                ),
                "interpretation": "low-K regret improves, but the primary fewer-board-"
                "evaluations-to-near-optimal claim is not established",
            },
            "random_baseline_comparison": {
                "uniform": _random_beats_reference(
                    uniform_orders, measured, models["B4_v1"]["metrics"]
                ),
                "stratified": _random_beats_reference(
                    stratified_orders, measured, models["B4_v1"]["metrics"]
                ),
            },
            "thread_search_coverage": {
                "observed_cpu_thread_vector_counts": thread_vectors,
                "outer_cpu_threads_fixed_in_historical_pool": [3, 4],
                "middle_cpu_threads_fixed_in_historical_pool": 1,
                "per_stage_thread_joint_optimization_validated": False,
                "interpretation": "P4B primarily evaluates partition ranking, not the full "
                "independent per-stage TVM thread decision space",
            },
            "decision": (
                "stop_before_p5_full_baseline_or_current_runtime_evidence_missing"
                if not full_gate
                else "stop_before_p5_historical_runtime_is_not_current_runtime"
            ),
            "limitations": [
                "historical affinity/source fingerprint differs from current runtime",
                "type-fixed B1 has no exact label in the semantic historical pool"
                if not b1_available
                else "none",
                "measured-pool oracle is not a global legal-space oracle",
                "historical candidate selection is not a random sample of the 972528 legal "
                "execution configurations",
                "historical thread vectors do not cover independent per-stage thread choices",
                "the any-K exploratory result is not a single-primary-K confirmatory test",
            ],
        }
    )
    state = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_execution_state",
            "protocol_id": PROTOCOL_ID,
            "current_stage": "V1-P4B",
            "current_stage_status": "completed_exploratory_gate_inconclusive"
            if not full_gate
            else "completed_exploratory_gate_passed_current_runtime_unverified",
            "next_stage": "V1-P4R" if not full_gate else "V1-P5",
            "next_stage_may_start": False,
            "advance_requires_user_confirmation": True,
            "blocking_evidence": [
                "exact_type_fixed_baseline_label_missing"
                if not b1_available
                else "current_runtime_exact_label_pool_missing",
                "historical_runtime_not_current_runtime",
                "b4_does_not_beat_uniform_random_median_evaluations_to_95pct",
                "historical_pool_does_not_cover_per_stage_thread_decisions",
                "historical_pool_is_not_a_random_sample_of_current_legal_space",
            ],
            "p4b_scoring_protocol_artifact_sha256": scoring_protocol["artifact_sha256"],
            "p4b_frozen_scores_artifact_sha256": frozen_scores["artifact_sha256"],
            "p4b_report_artifact_sha256": report["artifact_sha256"],
        }
    )
    return {
        "v1_p4b_retrospective_report.json": report,
        "v1_execution_state.json": state,
    }


def write_outputs(output_dir, outputs):
    for name, payload in outputs.items():
        validate_artifact_sha256(payload)
        write_json(Path(output_dir) / name, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--mode", choices=("score", "evaluate", "all"), default="all")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if args.mode in {"score", "all"}:
        write_outputs(output_dir, build_frozen_scores(output_dir))
    if args.mode in {"evaluate", "all"}:
        outputs = evaluate_frozen_scores(output_dir)
        write_outputs(output_dir, outputs)
        report = outputs["v1_p4b_retrospective_report.json"]
        print(json.dumps({"gate": report["gate"], "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
