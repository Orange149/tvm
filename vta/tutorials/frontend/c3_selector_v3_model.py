#!/usr/bin/env python3
"""Completion-aware graph-program cost model for C3 selector v3.

The model deliberately operates on exact graph candidate sets rather than on
isolated operators.  It never consumes latency fields as features, fits all
imputation/scaling state inside each task-held-out fold, and fails closed to a
static ordering until both ordinary dependency execution and scoped completion
are represented by several execution tasks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from c3_completion_policy import (
    DEPENDENCY_TOKENS,
    POLICIES,
    SCOPED_FORCE_SERIAL,
    TRANSITION_SAFE_FORCE_SERIAL,
)


SCHEMA = "c3_selector_v3_completion_model_v1"
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
MIN_EXECUTION_TASKS = 8
MIN_COMPARISONS = 40
MIN_TARGET_COVERAGE = 0.80
MIN_SCOPED_TASKS = 3
MIN_DEPENDENCY_ONLY_TASKS = 3

RATIO_FIELDS = (
    "inp_request_bytes", "wgt_request_bytes", "acc_request_bytes", "out_request_bytes",
    "total_request_bytes", "inp_request_calls", "wgt_request_calls",
    "acc_request_calls", "out_request_calls", "total_request_calls",
    "uop_load_bytes", "uop_load_calls", "instruction_bytes", "submissions", "sync_count",
)
ARENA_DELTA_FIELDS = (
    "inp_arena_fraction", "wgt_arena_fraction", "acc_arena_fraction", "uop_arena_fraction",
)
COMPLETION_FIELDS = (
    "replacement_count", "scoped_completion_count", "blocking_completion_count",
    "debug_transition_count", "scoped_function_count", "distinct_policy_count",
    "entry_drain_count",
)
POLICY_CATEGORIES = tuple(sorted(POLICIES))


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _log_ratio(candidate: Any, incumbent: Any) -> Optional[float]:
    left, right = _finite(candidate), _finite(incumbent)
    if left is None or right is None or left < 0.0 or right < 0.0:
        return None
    return math.log((left + 1.0) / (right + 1.0))


def _delta(candidate: Any, incumbent: Any) -> Optional[float]:
    left, right = _finite(candidate), _finite(incumbent)
    return None if left is None or right is None else left - right


def _metrics(row: Mapping[str, Any], incumbent: bool = False) -> Mapping[str, Any]:
    if incumbent:
        return row.get("incumbent_aggregate_metrics") or row.get("incumbent_metrics") or {}
    return row.get("aggregate_metrics") or row.get("candidate_metrics") or {}


def _policies(row: Mapping[str, Any]) -> List[str]:
    explicit = row.get("completion_policy_names")
    if explicit is not None:
        return sorted(str(value) for value in explicit)
    return sorted(str(member.get("completion_policy", "")) for member in row.get("members") or [])


def completion_features(row: Mapping[str, Any]) -> Dict[str, float]:
    explicit = row.get("completion_features")
    if explicit is not None:
        result = {name: float(explicit.get(name, 0.0)) for name in COMPLETION_FIELDS}
        for policy in POLICY_CATEGORIES:
            result["policy_count." + policy] = float(explicit.get("policy_count." + policy, 0.0))
        return result
    members = row.get("members") or []
    policies = _policies(row)
    unknown = sorted(set(policies) - set(POLICIES))
    if unknown:
        raise ValueError("unknown completion policies: {}".format(unknown))
    counts = {policy: policies.count(policy) for policy in POLICY_CATEGORIES}
    result = {
        "replacement_count": float(len(members) if members else len(policies)),
        "scoped_completion_count": float(
            counts.get(SCOPED_FORCE_SERIAL, 0) + counts.get(TRANSITION_SAFE_FORCE_SERIAL, 0)
        ),
        "blocking_completion_count": float(sum(
            count for policy, count in counts.items()
            if POLICIES[policy]["blocking_completion_before_exit"]
        )),
        "debug_transition_count": float(sum(
            count * len(POLICIES[policy]["ordered_debug_transitions"])
            for policy, count in counts.items()
        )),
        "scoped_function_count": float(sum(len(member.get("scoped_functions") or [])
                                             for member in members)),
        "distinct_policy_count": float(len(set(policies))),
        "entry_drain_count": float(sum(
            count * POLICIES[policy]["entry_drains_before_debug"]
            for policy, count in counts.items()
        )),
    }
    result.update({"policy_count." + policy: float(counts[policy])
                   for policy in POLICY_CATEGORIES})
    return result


def numeric_feature_names() -> List[str]:
    return (
        ["log_ratio." + field for field in RATIO_FIELDS]
        + ["arena_delta." + field for field in ARENA_DELTA_FIELDS]
        + ["completion." + field for field in COMPLETION_FIELDS]
        + ["completion.policy_count." + policy for policy in POLICY_CATEGORIES]
    )


def feature_names() -> List[str]:
    names = numeric_feature_names()
    return names + ["missing." + name for name in names]


def row_numeric_features(row: Mapping[str, Any]) -> List[Optional[float]]:
    candidate, incumbent = _metrics(row), _metrics(row, incumbent=True)
    completion = completion_features(row)
    values: List[Optional[float]] = [
        _log_ratio(candidate.get(field), incumbent.get(field)) for field in RATIO_FIELDS
    ]
    values += [_delta(candidate.get(field), incumbent.get(field)) for field in ARENA_DELTA_FIELDS]
    values += [_finite(completion.get(field)) for field in COMPLETION_FIELDS]
    values += [_finite(completion.get("policy_count." + policy)) for policy in POLICY_CATEGORIES]
    return values


def target_value(row: Mapping[str, Any]) -> Optional[float]:
    explicit = _finite(row.get("target_log_latency_ratio"))
    if explicit is not None:
        return explicit
    candidate = _finite(row.get("candidate_latency_ms"))
    incumbent = _finite(row.get("incumbent_latency_ms"))
    if candidate is None or incumbent is None or candidate <= 0.0 or incumbent <= 0.0:
        return None
    return math.log(candidate / incumbent)


@dataclass(frozen=True)
class Preprocessor:
    medians: Tuple[float, ...]
    iqrs: Tuple[float, ...]

    def to_json(self) -> Dict[str, Any]:
        names = numeric_feature_names()
        return {
            "numeric_feature_names": names,
            "medians": dict(zip(names, self.medians)),
            "iqrs": dict(zip(names, self.iqrs)),
            "missing_indicator_policy": "one indicator per numeric feature",
            "iqr_zero_policy": "scale_one",
        }


def fit_preprocessor(rows: Sequence[Mapping[str, Any]]) -> Preprocessor:
    columns: List[List[float]] = [[] for _ in numeric_feature_names()]
    for row in rows:
        for index, value in enumerate(row_numeric_features(row)):
            if value is not None:
                columns[index].append(value)
    medians, iqrs = [], []
    for column in columns:
        if not column:
            medians.append(0.0)
            iqrs.append(1.0)
            continue
        array = np.asarray(column, dtype=np.float64)
        median = float(np.median(array))
        q25, q75 = np.percentile(array, [25.0, 75.0])
        medians.append(median)
        iqrs.append(float(q75 - q25) if q75 > q25 else 1.0)
    return Preprocessor(tuple(medians), tuple(iqrs))


def transform_rows(rows: Sequence[Mapping[str, Any]], preprocessor: Preprocessor) -> np.ndarray:
    encoded = []
    for row in rows:
        raw = row_numeric_features(row)
        missing = [1.0 if value is None else 0.0 for value in raw]
        numeric = [
            ((preprocessor.medians[index] if value is None else value)
             - preprocessor.medians[index]) / preprocessor.iqrs[index]
            for index, value in enumerate(raw)
        ]
        encoded.append(numeric + missing)
    if not encoded:
        return np.empty((0, len(feature_names())), dtype=np.float64)
    return np.asarray(encoded, dtype=np.float64)


@dataclass(frozen=True)
class RidgeFit:
    alpha: float
    intercept: float
    coefficients: Tuple[float, ...]

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return self.intercept + matrix @ np.asarray(self.coefficients, dtype=np.float64)

    def to_json(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "intercept": self.intercept,
                "coefficients": dict(zip(feature_names(), self.coefficients))}


def fit_ridge(matrix: np.ndarray, target: np.ndarray, alpha: float) -> RidgeFit:
    x_mean, y_mean = np.mean(matrix, axis=0), float(np.mean(target))
    centered_x, centered_y = matrix - x_mean, target - y_mean
    coefficients = np.linalg.solve(
        centered_x.T @ centered_x + float(alpha) * np.eye(matrix.shape[1]),
        centered_x.T @ centered_y,
    )
    return RidgeFit(float(alpha), y_mean - float(x_mean @ coefficients),
                    tuple(float(value) for value in coefficients))


def _task_id(row: Mapping[str, Any]) -> str:
    return str(row.get("execution_task_id") or row.get("graph_execution_task_id") or "")


def model_input_reasons(row: Mapping[str, Any]) -> List[str]:
    reasons = []
    if not str(row.get("candidate_set_id", "")):
        reasons.append("missing_candidate_set_id")
    if not str(row.get("hardware_fingerprint_id", "")):
        reasons.append("missing_hardware_fingerprint_id")
    if not _metrics(row, incumbent=True):
        reasons.append("missing_incumbent_aggregate_metrics")
    try:
        completion_features(row)
    except ValueError as error:
        reasons.append(str(error))
    for policy in sorted(set(_policies(row)) & set(POLICIES)):
        if not POLICIES[policy]["production_eligible"]:
            reasons.append("completion_policy_not_production_eligible:{}".format(policy))
    return reasons


def row_contract_reasons(row: Mapping[str, Any], require_target: bool = False) -> List[str]:
    reasons = model_input_reasons(row)
    if not _task_id(row):
        reasons.append("missing_execution_task_id")
    if row.get("label_scope") != "exact_graph_candidate_set":
        reasons.append("label_scope_not_exact_graph_candidate_set")
    if row.get("graph_context_qualified") is not True:
        reasons.append("graph_context_not_qualified")
    if row.get("fpga_correct") is not True:
        reasons.append("fpga_correct_not_true")
    if not str(row.get("measurement_protocol_id", "")):
        reasons.append("missing_measurement_protocol_id")
    if require_target and target_value(row) is None:
        reasons.append("missing_target")
    return reasons


def eligibility(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tasks = sorted({_task_id(row) for row in rows if _task_id(row)})
    targets = [target_value(row) for row in rows]
    invalid = {str(row.get("candidate_set_id", index)): row_contract_reasons(row)
               for index, row in enumerate(rows) if row_contract_reasons(row)}
    fingerprints = sorted({str(row.get("hardware_fingerprint_id")) for row in rows
                           if row.get("hardware_fingerprint_id")})
    scoped_tasks, dependency_tasks = set(), set()
    for row in rows:
        policies = set(_policies(row))
        if policies & {SCOPED_FORCE_SERIAL, TRANSITION_SAFE_FORCE_SERIAL}:
            scoped_tasks.add(_task_id(row))
        if policies and policies <= {DEPENDENCY_TOKENS}:
            dependency_tasks.add(_task_id(row))
    coverage = sum(value is not None for value in targets) / len(rows) if rows else 0.0
    reasons = []
    if invalid:
        reasons.append("rows_fail_exact_graph_contract={}".format(len(invalid)))
    if len(fingerprints) != 1:
        reasons.append("hardware_fingerprint_count={}!=1".format(len(fingerprints)))
    if len(tasks) < MIN_EXECUTION_TASKS:
        reasons.append("execution_tasks={}<{}".format(len(tasks), MIN_EXECUTION_TASKS))
    if len(rows) < MIN_COMPARISONS:
        reasons.append("comparisons={}<{}".format(len(rows), MIN_COMPARISONS))
    if coverage < MIN_TARGET_COVERAGE:
        reasons.append("target_coverage={:.6f}<{:.6f}".format(coverage, MIN_TARGET_COVERAGE))
    if len(scoped_tasks) < MIN_SCOPED_TASKS:
        reasons.append("scoped_completion_tasks={}<{}".format(
            len(scoped_tasks), MIN_SCOPED_TASKS))
    if len(dependency_tasks) < MIN_DEPENDENCY_ONLY_TASKS:
        reasons.append("dependency_only_tasks={}<{}".format(
            len(dependency_tasks), MIN_DEPENDENCY_ONLY_TASKS))
    return {
        "eligible": not reasons,
        "execution_task_count": len(tasks),
        "comparison_count": len(rows),
        "target_count": sum(value is not None for value in targets),
        "target_coverage": coverage,
        "scoped_completion_task_count": len(scoped_tasks),
        "dependency_only_task_count": len(dependency_tasks),
        "hardware_fingerprint_ids": fingerprints,
        "invalid_rows": invalid,
        "reasons": reasons,
    }


def _fold_metrics(rows: Sequence[Mapping[str, Any]], predicted: np.ndarray) -> Dict[str, float]:
    actual = np.asarray([target_value(row) for row in rows], dtype=np.float64)
    selected = min(range(len(rows)), key=lambda index: (
        float(predicted[index]), str(rows[index].get("candidate_set_id", ""))))
    oracle = float(np.min(actual))
    return {
        "mae": float(np.mean(np.abs(predicted - actual))),
        "simple_regret_log_ratio": max(0.0, float(actual[selected]) - oracle),
        "oracle_plus_2pct_hit": bool(float(actual[selected]) <= oracle + math.log(1.02)),
    }


def select_alpha_loto(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    labelled = [row for row in rows if target_value(row) is not None]
    tasks = sorted({_task_id(row) for row in labelled})
    per_alpha = {}
    for alpha in RIDGE_ALPHAS:
        folds = []
        for task in tasks:
            train = [row for row in labelled if _task_id(row) != task]
            test = [row for row in labelled if _task_id(row) == task]
            preprocessor = fit_preprocessor(train)
            fit = fit_ridge(transform_rows(train, preprocessor),
                            np.asarray([target_value(row) for row in train]), alpha)
            metrics = _fold_metrics(test, fit.predict(transform_rows(test, preprocessor)))
            folds.append({"held_out_execution_task_id": task, "train_rows": len(train),
                          "test_rows": len(test), **metrics})
        per_alpha[str(alpha)] = {
            "macro_simple_regret_log_ratio": float(np.mean([
                fold["simple_regret_log_ratio"] for fold in folds])),
            "macro_mae": float(np.mean([fold["mae"] for fold in folds])),
            "oracle_plus_2pct_hit_rate": float(np.mean([
                fold["oracle_plus_2pct_hit"] for fold in folds])),
            "folds": folds,
        }
    selected = min(RIDGE_ALPHAS, key=lambda alpha: (
        per_alpha[str(alpha)]["macro_simple_regret_log_ratio"],
        per_alpha[str(alpha)]["macro_mae"], -alpha,
    ))
    return {
        "criterion": "minimum macro held-task simple regret, then macro MAE; ties larger alpha",
        "selected_alpha": selected,
        "alphas": per_alpha,
    }


def train_selector(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    gate = eligibility(rows)
    base = {
        "schema": SCHEMA,
        "eligibility": gate,
        "training_unit": "exact_graph_candidate_set_relative_to_incumbent",
        "grouping": "leave_one_execution_task_out",
        "latency_feature_prohibition": True,
    }
    if not gate["eligible"]:
        return {**base, "selector_kind": "static_fallback",
                "fallback_order": ["submissions", "uop_load_bytes",
                                   "total_request_bytes", "total_request_calls",
                                   "instruction_bytes", "candidate_set_id"]}
    labelled = [row for row in rows if target_value(row) is not None]
    cv = select_alpha_loto(labelled)
    preprocessor = fit_preprocessor(labelled)
    fit = fit_ridge(transform_rows(labelled, preprocessor),
                    np.asarray([target_value(row) for row in labelled]),
                    float(cv["selected_alpha"]))
    return {**base, "selector_kind": "completion_aware_ridge",
            "cross_validation": cv, "preprocessor": preprocessor.to_json(),
            "ridge": fit.to_json(), "feature_names": feature_names()}


def _model_from_json(contract: Mapping[str, Any]) -> Tuple[Preprocessor, RidgeFit]:
    if contract.get("schema") != SCHEMA or contract.get("selector_kind") != "completion_aware_ridge":
        raise ValueError("contract is not a completion-aware selector-v3 ridge model")
    names = numeric_feature_names()
    prep = contract["preprocessor"]
    preprocessor = Preprocessor(tuple(float(prep["medians"][name]) for name in names),
                                tuple(float(prep["iqrs"][name]) for name in names))
    ridge = contract["ridge"]
    fit = RidgeFit(float(ridge["alpha"]), float(ridge["intercept"]),
                   tuple(float(ridge["coefficients"][name]) for name in feature_names()))
    return preprocessor, fit


def score_graph_sets(rows: Sequence[Mapping[str, Any]],
                     contract: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Score exact sets, refusing partial pools or caller-injected predictions."""

    preprocessor, fit = _model_from_json(contract)
    failures = {str(row.get("candidate_set_id", index)): model_input_reasons(row)
                for index, row in enumerate(rows) if model_input_reasons(row)}
    if failures:
        raise ValueError("graph-set model input contract failed: {}".format(failures))
    expected_fingerprints = set(
        str(value) for value in (contract.get("eligibility") or {}).get(
            "hardware_fingerprint_ids", [])
    )
    actual_fingerprints = {str(row.get("hardware_fingerprint_id")) for row in rows}
    if len(expected_fingerprints) != 1 or actual_fingerprints != expected_fingerprints:
        raise ValueError(
            "model/input hardware fingerprint mismatch: expected {}, got {}".format(
                sorted(expected_fingerprints), sorted(actual_fingerprints)
            )
        )
    scores = fit.predict(transform_rows(rows, preprocessor))
    return [dict(row, predicted_graph_log_latency_ratio=float(score),
                 prediction_source="frozen_completion_aware_ridge")
            for row, score in zip(rows, scores)]
