#!/usr/bin/env python3
"""Core implementation for the C3 compiler-cost-aware selector v2.

The module is intentionally independent of TVM.  It consumes versioned rows
whose performance labels have already passed the experiment-contract checks.
All preprocessing is fit on the training fold; no held-out latency is used to
impute or scale features.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA = "c3_selector_v2_core_v1"
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
MIN_EXECUTION_TASKS = 8
MIN_COMPARISONS = 40
MIN_TARGET_COVERAGE = 0.80

# These quantities are transformed exactly as specified in the v2 protocol.
LOG_RATIO_FIELDS = (
    "inp_request_bytes",
    "wgt_request_bytes",
    "acc_request_bytes",
    "out_request_bytes",
    "total_request_bytes",
    "inp_request_calls",
    "wgt_request_calls",
    "acc_request_calls",
    "out_request_calls",
    "total_request_calls",
    "uop_load_bytes",
    "uop_load_calls",
    "command_uop_bytes",
    "instruction_bytes",
    "submissions",
)
TILE_FIELDS = ("tile_b", "tile_h", "tile_w", "tile_ci", "tile_co")
DIRECT_FIELDS = (
    "sync_count",
    "inp_arena_fraction",
    "wgt_arena_fraction",
    "acc_arena_fraction",
    "uop_arena_fraction",
)
MODE_CATEGORIES = (
    "original",
    "input_stationary",
    "weight_stationary",
    "weight_stationary_barrier",
    "paper_inspired_hybrid",
    "input_weight_staged",
    "other",
)
THREAD_CATEGORIES = ("1", "2", "4", "other")


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _log_ratio(candidate: Any, incumbent: Any) -> Optional[float]:
    candidate = _finite_or_none(candidate)
    incumbent = _finite_or_none(incumbent)
    if candidate is None or incumbent is None or candidate < 0 or incumbent < 0:
        return None
    return math.log((candidate + 1.0) / (incumbent + 1.0))


def _mode_category(value: Any) -> str:
    text = str(value or "other")
    return text if text in MODE_CATEGORIES[:-1] else "other"


def _thread_category(value: Any) -> str:
    number = _finite_or_none(value)
    if number is not None and number.is_integer() and str(int(number)) in THREAD_CATEGORIES[:-1]:
        return str(int(number))
    return "other"


def numeric_feature_names() -> List[str]:
    return [f"log_ratio.{name}" for name in LOG_RATIO_FIELDS] + [
        f"tile.{name}" for name in TILE_FIELDS
    ] + [f"direct.{name}" for name in DIRECT_FIELDS]


def feature_names() -> List[str]:
    result = numeric_feature_names()
    result += [f"missing.{name}" for name in numeric_feature_names()]
    result += [f"mode.{name}" for name in MODE_CATEGORIES]
    result += [f"oc_nthread.{name}" for name in THREAD_CATEGORIES]
    result += [f"h_nthread.{name}" for name in THREAD_CATEGORIES]
    return result


def row_numeric_features(row: Mapping[str, Any]) -> List[Optional[float]]:
    candidate_metrics = row.get("candidate_metrics") or {}
    incumbent_metrics = row.get("incumbent_metrics") or {}
    candidate_tile = row.get("candidate_tile") or {}
    values: List[Optional[float]] = []
    for name in LOG_RATIO_FIELDS:
        values.append(_log_ratio(candidate_metrics.get(name), incumbent_metrics.get(name)))
    for name in TILE_FIELDS:
        values.append(_finite_or_none(candidate_tile.get(name)))
    for name in DIRECT_FIELDS:
        values.append(_finite_or_none(candidate_metrics.get(name)))
    return values


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
    width = len(numeric_feature_names())
    columns: List[List[float]] = [[] for _ in range(width)]
    for row in rows:
        for index, value in enumerate(row_numeric_features(row)):
            if value is not None:
                columns[index].append(value)
    medians: List[float] = []
    iqrs: List[float] = []
    for column in columns:
        if not column:
            # An entirely absent feature remains distinguishable through its
            # missing indicator.  Zero is only the imputation location, not an
            # assertion that the underlying metric was zero.
            medians.append(0.0)
            iqrs.append(1.0)
            continue
        array = np.asarray(column, dtype=np.float64)
        median = float(np.median(array))
        q25, q75 = np.percentile(array, [25.0, 75.0])
        iqr = float(q75 - q25)
        medians.append(median)
        iqrs.append(iqr if iqr > 0.0 else 1.0)
    return Preprocessor(tuple(medians), tuple(iqrs))


def transform_rows(rows: Sequence[Mapping[str, Any]], preprocessor: Preprocessor) -> np.ndarray:
    encoded: List[List[float]] = []
    for row in rows:
        raw = row_numeric_features(row)
        missing = [1.0 if value is None else 0.0 for value in raw]
        numeric = [
            ((preprocessor.medians[i] if value is None else value) - preprocessor.medians[i])
            / preprocessor.iqrs[i]
            for i, value in enumerate(raw)
        ]
        mode = _mode_category(row.get("candidate_mode"))
        oc = _thread_category((row.get("candidate_tile") or {}).get("oc_nthread"))
        hn = _thread_category((row.get("candidate_tile") or {}).get("h_nthread"))
        vector = numeric + missing
        vector += [1.0 if mode == category else 0.0 for category in MODE_CATEGORIES]
        vector += [1.0 if oc == category else 0.0 for category in THREAD_CATEGORIES]
        vector += [1.0 if hn == category else 0.0 for category in THREAD_CATEGORIES]
        encoded.append(vector)
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
        return {
            "alpha": self.alpha,
            "intercept": self.intercept,
            "coefficients": dict(zip(feature_names(), self.coefficients)),
        }


def fit_ridge(matrix: np.ndarray, target: np.ndarray, alpha: float) -> RidgeFit:
    if len(target) == 0:
        raise ValueError("ridge requires at least one target")
    x_mean = np.mean(matrix, axis=0)
    y_mean = float(np.mean(target))
    centered_x = matrix - x_mean
    centered_y = target - y_mean
    gram = centered_x.T @ centered_x
    rhs = centered_x.T @ centered_y
    coefficients = np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0]), rhs)
    intercept = y_mean - float(x_mean @ coefficients)
    return RidgeFit(float(alpha), intercept, tuple(float(x) for x in coefficients))


def target_value(row: Mapping[str, Any]) -> Optional[float]:
    explicit = _finite_or_none(row.get("target_log_latency_ratio"))
    if explicit is not None:
        return explicit
    candidate = _finite_or_none(row.get("candidate_latency_ms"))
    incumbent = _finite_or_none(row.get("incumbent_latency_ms"))
    if candidate is None or incumbent is None or candidate <= 0.0 or incumbent <= 0.0:
        return None
    return math.log(candidate / incumbent)


def eligibility(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tasks = sorted({str(row.get("execution_task_id")) for row in rows})
    targets = [target_value(row) for row in rows]
    coverage = sum(value is not None for value in targets) / len(rows) if rows else 0.0
    reasons = []
    if len(tasks) < MIN_EXECUTION_TASKS:
        reasons.append(f"execution_tasks={len(tasks)}<{MIN_EXECUTION_TASKS}")
    if len(rows) < MIN_COMPARISONS:
        reasons.append(f"comparisons={len(rows)}<{MIN_COMPARISONS}")
    if coverage < MIN_TARGET_COVERAGE:
        reasons.append(f"target_coverage={coverage:.6f}<{MIN_TARGET_COVERAGE:.6f}")
    return {
        "eligible": not reasons,
        "execution_task_count": len(tasks),
        "comparison_count": len(rows),
        "target_count": sum(value is not None for value in targets),
        "target_coverage": coverage,
        "reasons": reasons,
    }


def select_alpha_loto(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    labelled = [row for row in rows if target_value(row) is not None]
    tasks = sorted({str(row["execution_task_id"]) for row in labelled})
    per_alpha: Dict[str, Any] = {}
    for alpha in RIDGE_ALPHAS:
        errors: List[float] = []
        folds: List[Dict[str, Any]] = []
        for task in tasks:
            train = [row for row in labelled if str(row["execution_task_id"]) != task]
            test = [row for row in labelled if str(row["execution_task_id"]) == task]
            if not train or not test:
                continue
            preprocessor = fit_preprocessor(train)
            train_x = transform_rows(train, preprocessor)
            train_y = np.asarray([target_value(row) for row in train], dtype=np.float64)
            fit = fit_ridge(train_x, train_y, alpha)
            predicted = fit.predict(transform_rows(test, preprocessor))
            actual = np.asarray([target_value(row) for row in test], dtype=np.float64)
            fold_errors = np.abs(predicted - actual)
            errors.extend(float(x) for x in fold_errors)
            folds.append({
                "held_out_execution_task_id": task,
                "train_rows": len(train),
                "test_rows": len(test),
                "mae": float(np.mean(fold_errors)),
            })
        per_alpha[str(alpha)] = {
            "mae": float(np.mean(errors)) if errors else None,
            "prediction_count": len(errors),
            "folds": folds,
        }
    usable = [(float(value["mae"]), -alpha, alpha) for alpha in RIDGE_ALPHAS
              if (value := per_alpha[str(alpha)])["mae"] is not None]
    if not usable:
        raise ValueError("leave-one-execution-task-out produced no predictions")
    selected = min(usable)[2]
    return {
        "criterion": "minimum leave-one-execution-task-out MAE; ties choose larger alpha",
        "selected_alpha": selected,
        "alphas": per_alpha,
    }


def train_selector(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    gate = eligibility(rows)
    if not gate["eligible"]:
        return {
            "schema": SCHEMA,
            "selector_kind": "lexicographic_fallback",
            "eligibility": gate,
            "fallback_order": [
                "submissions",
                "uop_load_bytes",
                "total_request_bytes",
                "total_request_calls",
                "instruction_bytes",
                "candidate_id",
            ],
        }
    labelled = [row for row in rows if target_value(row) is not None]
    cv = select_alpha_loto(labelled)
    preprocessor = fit_preprocessor(labelled)
    matrix = transform_rows(labelled, preprocessor)
    target = np.asarray([target_value(row) for row in labelled], dtype=np.float64)
    fit = fit_ridge(matrix, target, float(cv["selected_alpha"]))
    return {
        "schema": SCHEMA,
        "selector_kind": "ridge",
        "eligibility": gate,
        "cross_validation": cv,
        "preprocessor": preprocessor.to_json(),
        "ridge": fit.to_json(),
        "feature_names": feature_names(),
    }


def _unknown_last(value: Any) -> Tuple[int, float]:
    parsed = _finite_or_none(value)
    return (1, 0.0) if parsed is None else (0, parsed)


def fallback_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    metrics = row.get("candidate_metrics") or {}
    return (
        _unknown_last(metrics.get("submissions")),
        _unknown_last(metrics.get("uop_load_bytes")),
        _unknown_last(metrics.get("total_request_bytes")),
        _unknown_last(metrics.get("total_request_calls")),
        _unknown_last(metrics.get("instruction_bytes")),
        str(row.get("candidate_id", "")),
    )


def model_from_json(contract: Mapping[str, Any]) -> Tuple[Preprocessor, RidgeFit]:
    if contract.get("selector_kind") != "ridge":
        raise ValueError("contract is not a ridge selector")
    names = numeric_feature_names()
    prep = contract["preprocessor"]
    preprocessor = Preprocessor(
        tuple(float(prep["medians"][name]) for name in names),
        tuple(float(prep["iqrs"][name]) for name in names),
    )
    ridge = contract["ridge"]
    fit = RidgeFit(
        float(ridge["alpha"]),
        float(ridge["intercept"]),
        tuple(float(ridge["coefficients"][name]) for name in feature_names()),
    )
    return preprocessor, fit


def rank_candidates(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if contract.get("selector_kind") != "ridge":
        ordered = sorted(rows, key=fallback_key)
        return [dict(row, selector_v2_score=None, selector_v2_rank=i + 1) for i, row in enumerate(ordered)]
    preprocessor, fit = model_from_json(contract)
    scores = fit.predict(transform_rows(rows, preprocessor))
    decorated = [dict(row, selector_v2_score=float(score)) for row, score in zip(rows, scores)]
    decorated.sort(key=lambda row: (row["selector_v2_score"], str(row.get("candidate_id", ""))))
    return [dict(row, selector_v2_rank=i + 1) for i, row in enumerate(decorated)]


def shortlist(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("shortlist requires a non-empty candidate pool")
    incumbent = [row for row in rows if bool(row.get("is_incumbent"))]
    if len(incumbent) != 1:
        raise ValueError("shortlist requires exactly one incumbent")
    ranked = rank_candidates(rows, contract)
    new_rows = [row for row in rows if not bool(row.get("is_incumbent"))]
    predicted = next((row for row in ranked if not bool(row.get("is_incumbent"))), None)
    bytes_min = min(
        new_rows,
        key=lambda row: (_unknown_last((row.get("candidate_metrics") or {}).get("total_request_bytes")),
                         str(row.get("candidate_id", ""))),
    ) if new_rows else None
    command_min = min(
        new_rows,
        key=lambda row: (
            _unknown_last((row.get("candidate_metrics") or {}).get("instruction_bytes")),
            _unknown_last((row.get("candidate_metrics") or {}).get("uop_load_bytes")),
            str(row.get("candidate_id", "")),
        ),
    ) if new_rows else None
    selected: List[Dict[str, Any]] = []
    roles: Dict[str, List[str]] = {}
    for role, row in (("incumbent", incumbent[0]), ("selector_v2", predicted),
                      ("bytes_only", bytes_min), ("command_only", command_min)):
        if row is None:
            continue
        cid = str(row["candidate_id"])
        roles.setdefault(cid, []).append(role)
        if not any(str(item["candidate_id"]) == cid for item in selected):
            selected.append(row)
    return {
        "shortlist_candidate_ids": [str(row["candidate_id"]) for row in selected[:4]],
        "roles_by_candidate_id": roles,
        "selector_kind": contract.get("selector_kind"),
    }


def fixed_random_orders(candidate_ids: Iterable[str], seed: int = 20260924,
                        count: int = 1000) -> List[List[str]]:
    ids = sorted(set(str(value) for value in candidate_ids))
    orders = []
    for replicate in range(count):
        orders.append(sorted(ids, key=lambda cid: hashlib.sha256(
            f"{seed}:{replicate}:{cid}".encode("utf-8")).hexdigest()))
    return orders


def budget_prefix(order: Sequence[str], budget: int, incumbent_id: str,
                  fpga_validity: Mapping[str, bool]) -> Dict[str, Any]:
    """Return attempted and eligible candidates; invalid attempts consume budget."""
    new_order = [cid for cid in order if cid != incumbent_id]
    attempted = list(new_order[: max(0, int(budget))])
    eligible = [incumbent_id] + [cid for cid in attempted if bool(fpga_validity.get(cid, False))]
    return {
        "budget_new_candidates": int(budget),
        "attempted_new_candidate_ids": attempted,
        "fpga_invalid_attempt_count": sum(not bool(fpga_validity.get(cid, False)) for cid in attempted),
        "timing_eligible_candidate_ids": eligible,
        "replacement_policy": "no_replacement_after_fpga_invalid",
    }

