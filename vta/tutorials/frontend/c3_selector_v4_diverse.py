#!/usr/bin/env python3
"""Label-free low-movement and Tile-diverse ordering for C3 selector v4.

This module is deliberately a candidate organizer, not a latency predictor.
It first forms one frozen low-movement window and then applies deterministic
farthest-first ordering in normalized Tile/thread coordinates.  B1/B2/B3 are
prefixes of that one order; changing the budget never reruns the selector.

Static quantities are heuristics, not safe performance pruning.  Candidates
outside the window remain in the returned order and may still be measured by
a complete-pool reference evaluation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple


SCHEMA = "c3_selector_v4_low_movement_tile_diversity_v1"
DEFAULT_MAX_BUDGET = 3
DEFAULT_WINDOW_MULTIPLIER = 2

METRIC_FIELDS = (
    "total_request_bytes",
    "total_request_calls",
    "submissions",
    "uop_load_bytes",
    "instruction_bytes",
)
METRIC_ALIASES = {
    # The transition-safe timing freeze predates the canonical selector name.
    # This is a name-only adapter; both fields denote cumulative UOP command
    # bytes and neither is an on-chip occupancy peak.
    "uop_load_bytes": ("uop_load_bytes", "command_uop_bytes"),
}
TILE_FIELDS = (
    "tile_b",
    "tile_h",
    "tile_w",
    "tile_ci",
    "tile_co",
    "oc_nthread",
    "h_nthread",
)
LATENCY_LABEL_FIELDS = (
    "latency",
    "latency_ms",
    "candidate_latency_ms",
    "incumbent_latency_ms",
    "target_log_latency_ratio",
    "performance_label",
)


def _candidate_id(row: Mapping[str, Any]) -> str:
    # program_id is the executable identity and therefore the measurement
    # unit.  candidate_id may also exist on raw rows but can identify an
    # earlier configuration before lowering/program de-duplication.
    return str(row.get("program_id") or row.get("candidate_id") or "")


def _metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return (
        row.get("timing_freeze_metrics")
        or row.get("candidate_metrics")
        or row.get("aggregate_metrics")
        or {}
    )


def _tile(row: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = row.get("visible_features")
    if direct:
        return direct
    identity = row.get("candidate_identity_fields") or {}
    return identity.get("tile") or row.get("tile") or {}


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError("{} must be a nonnegative finite number".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be a nonnegative finite number".format(name))
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("{} must be a nonnegative finite number".format(name))
    return result


def _positive_number(value: Any, name: str) -> float:
    result = _nonnegative_number(value, name)
    if result <= 0.0:
        raise ValueError("{} must be positive".format(name))
    return result


def project_static_candidate(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Project a raw candidate onto the only fields selector v4 may consume."""

    leaked = [name for name in LATENCY_LABEL_FIELDS if row.get(name) is not None]
    leaked += [
        str(name) for name, value in row.items()
        if "latency" in str(name).lower()
        and name not in LATENCY_LABEL_FIELDS
        and value is not None
    ]
    if leaked:
        raise ValueError("latency/performance labels are forbidden: {}".format(sorted(set(leaked))))
    candidate_id = _candidate_id(row)
    if not candidate_id:
        raise ValueError("candidate_id or program_id is required")
    metrics = _metrics(row)
    tile = _tile(row)
    projected_metrics = {}
    for name in METRIC_FIELDS:
        aliases = METRIC_ALIASES.get(name, (name,))
        present = [alias for alias in aliases if metrics.get(alias) is not None]
        if len(present) > 1:
            values = {_nonnegative_number(metrics[alias], "metrics." + alias) for alias in present}
            if len(values) != 1:
                raise ValueError("conflicting metric aliases for {}".format(name))
        value = metrics.get(present[0]) if present else None
        projected_metrics[name] = _nonnegative_number(value, "metrics." + name)
    projected_tile = {
        name: _positive_number(tile.get(name), "tile." + name)
        for name in TILE_FIELDS
    }
    return {
        "candidate_id": candidate_id,
        "metrics": projected_metrics,
        "tile": projected_tile,
    }


def _static_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    metrics = row["metrics"]
    return tuple(metrics[name] for name in METRIC_FIELDS) + (row["candidate_id"],)


def _movement_bucket(row: Mapping[str, Any], minimum_bytes: float) -> int:
    # A factor-of-two bucket avoids treating small byte-count perturbations as
    # a categorical difference while keeping movement as the primary stratum.
    ratio = (row["metrics"]["total_request_bytes"] + 1.0) / (minimum_bytes + 1.0)
    return int(math.floor(math.log(ratio, 2.0))) if ratio > 1.0 else 0


def _tile_vectors(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Tuple[float, ...]]:
    raw = {
        row["candidate_id"]: tuple(math.log(value, 2.0) for value in (
            row["tile"][name] for name in TILE_FIELDS
        ))
        for row in rows
    }
    lows = [min(values[index] for values in raw.values()) for index in range(len(TILE_FIELDS))]
    highs = [max(values[index] for values in raw.values()) for index in range(len(TILE_FIELDS))]
    return {
        candidate_id: tuple(
            0.0 if highs[index] == lows[index]
            else (value - lows[index]) / (highs[index] - lows[index])
            for index, value in enumerate(values)
        )
        for candidate_id, values in raw.items()
    }


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def diverse_static_order(
        rows: Sequence[Mapping[str, Any]],
        max_budget: int = DEFAULT_MAX_BUDGET,
        window_multiplier: int = DEFAULT_WINDOW_MULTIPLIER) -> Dict[str, Any]:
    """Return one label-free order whose B1..Bmax are stable prefixes."""

    if isinstance(max_budget, bool) or int(max_budget) != max_budget or max_budget <= 0:
        raise ValueError("max_budget must be a positive integer")
    if (isinstance(window_multiplier, bool)
            or int(window_multiplier) != window_multiplier
            or window_multiplier <= 0):
        raise ValueError("window_multiplier must be a positive integer")
    if not rows:
        raise ValueError("at least one new candidate is required")
    projected = [project_static_candidate(row) for row in rows]
    ids = [row["candidate_id"] for row in projected]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique")

    static = sorted(projected, key=_static_key)
    window_size = min(len(static), int(max_budget) * int(window_multiplier))
    window = static[:window_size]
    outside = static[window_size:]
    vectors = _tile_vectors(projected)
    minimum_bytes = static[0]["metrics"]["total_request_bytes"]

    remaining = list(window)
    selected: List[Mapping[str, Any]] = [remaining.pop(0)]
    trace = [{
        "candidate_id": selected[0]["candidate_id"],
        "reason": "minimum_static_movement_anchor",
        "movement_bucket": 0,
        "minimum_tile_distance": None,
    }]
    while remaining:
        best_bucket = min(_movement_bucket(row, minimum_bytes) for row in remaining)
        eligible = [row for row in remaining
                    if _movement_bucket(row, minimum_bytes) == best_bucket]

        def choice_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
            vector = vectors[row["candidate_id"]]
            minimum_distance = min(
                _distance(vector, vectors[chosen["candidate_id"]]) for chosen in selected
            )
            # max() gives greatest diversity; the negated static tuple gives
            # deterministic preference to less movement on a distance tie.
            numeric_static = tuple(-float(value) for value in _static_key(row)[:-1])
            return (minimum_distance,) + numeric_static + (
                tuple(-ord(char) for char in row["candidate_id"]),
            )

        chosen = max(eligible, key=choice_key)
        minimum_distance = min(
            _distance(vectors[chosen["candidate_id"]], vectors[row["candidate_id"]])
            for row in selected
        )
        remaining.remove(chosen)
        selected.append(chosen)
        trace.append({
            "candidate_id": chosen["candidate_id"],
            "reason": "farthest_tile_within_lowest_movement_bucket",
            "movement_bucket": best_bucket,
            "minimum_tile_distance": minimum_distance,
        })

    ordered = selected + outside
    ordered_ids = [row["candidate_id"] for row in ordered]
    budgets = {
        "B{}".format(budget): ordered_ids[:min(budget, len(ordered_ids))]
        for budget in range(1, int(max_budget) + 1)
    }
    return {
        "schema": SCHEMA,
        "selector_kind": "label_free_low_movement_tile_diversity",
        "performance_model_claim": False,
        "safe_pruning_claim": False,
        "max_budget": int(max_budget),
        "window_multiplier": int(window_multiplier),
        "window_size": window_size,
        "metric_priority": list(METRIC_FIELDS),
        "tile_coordinates": list(TILE_FIELDS),
        "static_order_candidate_ids": [row["candidate_id"] for row in static],
        "window_candidate_ids": [row["candidate_id"] for row in window],
        "ordered_candidate_ids": ordered_ids,
        "budget_prefixes": budgets,
        "selection_trace": trace,
        "outside_window_policy": "append_in_static_order_without_pruning",
        "latency_fields_consumed": [],
    }
