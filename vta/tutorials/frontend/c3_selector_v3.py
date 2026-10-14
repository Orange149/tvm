#!/usr/bin/env python3
"""Composability-gated measurement and deployment interface for C3 selector v3.

Selector v3 intentionally reuses a frozen v2 cost model for ordering while it
adds a new hard semantic layer.  It does not claim that old v2 coefficients
learn the cost of scoped completion.  A future v3 performance model requires a
new, independently frozen training/holdout split.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from c3_composability import (
    GRAPH_SCHEMA,
    GRAPH_STATUS,
    candidate_disposition,
)
from c3_selector_v2 import _unknown_last, rank_candidates


SCHEMA = "c3_selector_v3_composability_gate_v1"


def graph_certificates_by_set(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        if row.get("schema") != GRAPH_SCHEMA or row.get("status") != GRAPH_STATUS:
            raise ValueError("invalid graph-context certificate")
        candidate_set_id = str(row.get("candidate_set_id", ""))
        if not candidate_set_id:
            raise ValueError("missing candidate_set_id")
        previous = result.get(candidate_set_id)
        if previous is not None:
            # The same exact graph program may be correctness-qualified on more
            # than one boot.  Keep one certificate for the semantic gate, but
            # fail closed if the claimed program membership changed.
            if sorted(previous.get("candidate_ids") or []) != sorted(row.get("candidate_ids") or []):
                raise ValueError("candidate membership changed across graph certificates")
            continue
        result[candidate_set_id] = row
    return result


def qualify_pool(rows: Sequence[Mapping[str, Any]],
                 candidate_certificates: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    qualified, rejected = [], []
    for raw in rows:
        row = dict(raw)
        disposition = candidate_disposition(row, candidate_certificates)
        row["selector_v3_disposition"] = disposition
        (qualified if disposition["eligible"] else rejected).append(row)
    incumbents = [row for row in qualified if bool(row.get("is_incumbent"))]
    if len(incumbents) != 1:
        raise ValueError("qualified pool requires exactly one incumbent")
    return {
        "schema": SCHEMA,
        "qualified": qualified,
        "rejected": rejected,
        "qualified_candidate_ids": [str(row["candidate_id"]) for row in qualified],
        "rejected_candidates": {
            str(row["candidate_id"]): row["selector_v3_disposition"] for row in rejected
        },
    }


def _bytes_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    metrics = row.get("candidate_metrics") or {}
    return (_unknown_last(metrics.get("total_request_bytes")), str(row.get("candidate_id", "")))


def _command_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    metrics = row.get("candidate_metrics") or {}
    return (
        _unknown_last(metrics.get("submissions")),
        _unknown_last(metrics.get("uop_load_bytes")),
        _unknown_last(metrics.get("instruction_bytes")),
        str(row.get("candidate_id", "")),
    )


def measurement_plan(rows: Sequence[Mapping[str, Any]], v2_contract: Mapping[str, Any],
                     candidate_certificates: Mapping[str, Mapping[str, Any]],
                     graph_certificates: Iterable[Mapping[str, Any]] = ()) -> Dict[str, Any]:
    """Build a maximum-four shortlist without allowing unsafe candidates to time."""

    pool = qualify_pool(rows, candidate_certificates)
    qualified = pool["qualified"]
    incumbent = next(row for row in qualified if bool(row.get("is_incumbent")))
    new_rows = [row for row in qualified if not bool(row.get("is_incumbent"))]
    ranked = rank_candidates(qualified, v2_contract)
    predicted = next((row for row in ranked if not bool(row.get("is_incumbent"))), None)
    candidates = [
        ("incumbent", incumbent),
        ("cost_model", predicted),
        ("bytes_only", min(new_rows, key=_bytes_key) if new_rows else None),
        ("command_only", min(new_rows, key=_command_key) if new_rows else None),
    ]
    selected: List[Mapping[str, Any]] = []
    roles: Dict[str, List[str]] = {}
    for role, row in candidates:
        if row is None:
            continue
        candidate_id = str(row["candidate_id"])
        roles.setdefault(candidate_id, []).append(role)
        if all(str(existing["candidate_id"]) != candidate_id for existing in selected):
            selected.append(row)
    graph_by_set = graph_certificates_by_set(graph_certificates)
    gate = {}
    timing_eligible = []
    for row in selected[:4]:
        candidate_id = str(row["candidate_id"])
        if bool(row.get("is_incumbent")):
            gate[candidate_id] = {"status": "INCUMBENT_TIMING_ELIGIBLE", "candidate_set_id": None}
            timing_eligible.append(candidate_id)
            continue
        candidate_set_id = row.get("target_candidate_set_id")
        certificate = graph_by_set.get(str(candidate_set_id)) if candidate_set_id else None
        if certificate is not None and candidate_id in certificate.get("candidate_ids", []):
            gate[candidate_id] = {"status": "EXACT_GRAPH_CONTEXT_QUALIFIED",
                                  "candidate_set_id": candidate_set_id}
            timing_eligible.append(candidate_id)
        else:
            gate[candidate_id] = {"status": "GRAPH_CORRECTNESS_REQUIRED_BEFORE_TIMING",
                                  "candidate_set_id": candidate_set_id}
    return {
        "schema": SCHEMA,
        "status": "MEASUREMENT_PLAN_FROZEN",
        "shortlist_candidate_ids": [str(row["candidate_id"]) for row in selected[:4]],
        "roles_by_candidate_id": roles,
        "timing_eligible_candidate_ids": timing_eligible,
        "gate_by_candidate_id": gate,
        "rejected_candidates": pool["rejected_candidates"],
        "ranking_model": "frozen_selector_v2_ordering_after_selector_v3_semantic_gate",
        "performance_model_claim": "v2 coefficients are not claimed to model scoped-completion cost",
    }


def deployment_gate(candidate_id: str, candidate_set_id: str,
                    graph_certificates: Iterable[Mapping[str, Any]],
                    boot_results: Sequence[Mapping[str, Any]],
                    minimum_improvement_pct: float = 2.0,
                    minimum_wins: int = 5, rounds: int = 6) -> Dict[str, Any]:
    """Require exact graph correctness and two distinct stable-improvement boots."""

    graph_by_set = graph_certificates_by_set(graph_certificates)
    graph = graph_by_set.get(str(candidate_set_id))
    reasons = []
    if graph is None or candidate_id not in graph.get("candidate_ids", []):
        reasons.append("missing_exact_graph_context_certificate")
    boot_ids = [str(row.get("boot_id", "")) for row in boot_results]
    if len(boot_results) != 2 or len(set(boot_ids)) != 2 or any(not value for value in boot_ids):
        reasons.append("requires_two_distinct_boot_ids")
    for index, row in enumerate(boot_results):
        if float(row.get("median_improvement_pct", float("-inf"))) < minimum_improvement_pct:
            reasons.append("boot_{}_improvement_below_threshold".format(index))
        if int(row.get("wins", -1)) < minimum_wins or int(row.get("rounds", -1)) != rounds:
            reasons.append("boot_{}_win_or_round_requirement_failed".format(index))
    return {
        "schema": SCHEMA,
        "status": "DEPLOY" if not reasons else "FALL_BACK_TO_INCUMBENT",
        "candidate_id": candidate_id,
        "candidate_set_id": candidate_set_id,
        "requirements": {
            "minimum_improvement_pct_each_boot": float(minimum_improvement_pct),
            "minimum_wins_each_boot": int(minimum_wins),
            "rounds_each_boot": int(rounds),
            "distinct_boots": 2,
        },
        "reasons": reasons,
    }


def candidate_set_deployment_gate(candidate_set_id: str,
                                  graph_certificates: Iterable[Mapping[str, Any]],
                                  boot_results: Sequence[Mapping[str, Any]],
                                  minimum_improvement_pct: float = 2.0,
                                  minimum_wins: int = 5,
                                  rounds: int = 6) -> Dict[str, Any]:
    """Apply the deployment policy to one exact graph-level candidate set.

    This is the canonical gate for a graph deployment containing more than one
    replaced operator.  Its measured latency belongs to the whole set and must
    not be attributed to an arbitrary member candidate.
    """

    graph_by_set = graph_certificates_by_set(graph_certificates)
    graph = graph_by_set.get(str(candidate_set_id))
    reasons = []
    if graph is None:
        reasons.append("missing_exact_graph_context_certificate")
    boot_ids = [str(row.get("boot_id", "")) for row in boot_results]
    if len(boot_results) != 2 or len(set(boot_ids)) != 2 or any(not value for value in boot_ids):
        reasons.append("requires_two_distinct_boot_ids")
    for index, row in enumerate(boot_results):
        if str(row.get("candidate_set_id", "")) != str(candidate_set_id):
            reasons.append("boot_{}_candidate_set_identity_mismatch".format(index))
        if float(row.get("median_improvement_pct", float("-inf"))) < minimum_improvement_pct:
            reasons.append("boot_{}_improvement_below_threshold".format(index))
        if int(row.get("wins", -1)) < minimum_wins or int(row.get("rounds", -1)) != rounds:
            reasons.append("boot_{}_win_or_round_requirement_failed".format(index))
    return {
        "schema": SCHEMA,
        "status": "DEPLOY" if not reasons else "FALL_BACK_TO_INCUMBENT",
        "deployment_unit": "exact_graph_candidate_set",
        "candidate_set_id": candidate_set_id,
        "candidate_ids": list(graph.get("candidate_ids") or []) if graph else [],
        "requirements": {
            "minimum_improvement_pct_each_boot": float(minimum_improvement_pct),
            "minimum_wins_each_boot": int(minimum_wins),
            "rounds_each_boot": int(rounds),
            "distinct_boots": 2,
        },
        "reasons": reasons,
    }


def _set_metric_key(row: Mapping[str, Any], fields: Sequence[str]) -> Tuple[Any, ...]:
    metrics = row.get("aggregate_metrics") or {}
    return tuple(_unknown_last(metrics.get(field)) for field in fields) + (
        str(row.get("candidate_set_id", "")),
    )


def graph_set_measurement_plan(rows: Sequence[Mapping[str, Any]],
                               graph_certificates: Iterable[Mapping[str, Any]] = (),
                               model_contract: Mapping[str, Any] = None) -> Dict[str, Any]:
    """Choose a maximum-four shortlist whose unit is an exact graph program."""

    if not rows:
        raise ValueError("graph-set plan requires a non-empty pool")
    ids = [str(row.get("candidate_set_id", "")) for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("graph candidate-set IDs must be non-empty and unique")
    incumbents = [row for row in rows if bool(row.get("is_incumbent"))]
    if len(incumbents) != 1:
        raise ValueError("graph-set plan requires exactly one incumbent set")
    incumbent = incumbents[0]
    new_rows = [row for row in rows if not bool(row.get("is_incumbent"))]
    incompatible = [row for row in new_rows
                    if (row.get("compatibility") or {}).get("compatible") is not True]
    eligible = [row for row in new_rows if row not in incompatible]
    model_status, model_reason = "STATIC_FALLBACK", "model_contract_not_provided"
    scored = []
    if model_contract is not None:
        if model_contract.get("selector_kind") != "completion_aware_ridge":
            model_reason = "model_contract_not_completion_aware_ridge"
        else:
            try:
                from c3_selector_v3_model import score_graph_sets
                scored = score_graph_sets(eligible, model_contract)
                model_status, model_reason = "COMPLETION_AWARE_MODEL", None
            except ValueError as error:
                model_reason = "model_input_rejected: {}".format(error)
    has_model_scores = bool(eligible) and model_status == "COMPLETION_AWARE_MODEL"
    if has_model_scores:
        eligible = scored
    selector_role = (
        "selector_v3_model" if has_model_scores else "selector_v3_static_fallback"
    )

    def selector_key(row):
        score = _unknown_last(row.get("predicted_graph_log_latency_ratio"))
        return (score,) + _set_metric_key(
            row, ("submissions", "uop_load_bytes", "instruction_bytes")
        )

    choices = [
        ("incumbent", incumbent),
        (selector_role, min(eligible, key=selector_key) if eligible else None),
        ("bytes_only", min(eligible, key=lambda row: _set_metric_key(
            row, ("total_request_bytes", "total_request_calls")
        )) if eligible else None),
        ("command_only", min(eligible, key=lambda row: _set_metric_key(
            row, ("submissions", "uop_load_bytes", "instruction_bytes")
        )) if eligible else None),
    ]
    selected, roles = [], {}
    for role, row in choices:
        if row is None:
            continue
        candidate_set_id = str(row["candidate_set_id"])
        roles.setdefault(candidate_set_id, []).append(role)
        if all(str(item["candidate_set_id"]) != candidate_set_id for item in selected):
            selected.append(row)

    graph_by_set = graph_certificates_by_set(graph_certificates)
    timing_eligible, gates = [], {}
    for row in selected[:4]:
        candidate_set_id = str(row["candidate_set_id"])
        if bool(row.get("is_incumbent")):
            gates[candidate_set_id] = "INCUMBENT_TIMING_ELIGIBLE"
            timing_eligible.append(candidate_set_id)
        elif candidate_set_id in graph_by_set:
            gates[candidate_set_id] = "EXACT_GRAPH_CONTEXT_QUALIFIED"
            timing_eligible.append(candidate_set_id)
        else:
            gates[candidate_set_id] = "GRAPH_CORRECTNESS_REQUIRED_BEFORE_TIMING"
    return {
        "schema": "c3_selector_v3_graph_set_plan_v1",
        "status": "GRAPH_SET_MEASUREMENT_PLAN_FROZEN",
        "selection_unit": "exact_graph_candidate_set",
        "shortlist_candidate_set_ids": [str(row["candidate_set_id"]) for row in selected[:4]],
        "roles_by_candidate_set_id": roles,
        "timing_eligible_candidate_set_ids": timing_eligible,
        "gate_by_candidate_set_id": gates,
        "incompatible_candidate_set_ids": [str(row["candidate_set_id"]) for row in incompatible],
        "ranking_model": (
            "graph_latency_model"
            if has_model_scores
            else "static_command_fallback_no_completion_aware_model"
        ),
        "performance_model_claim": bool(has_model_scores),
        "model_status": model_status,
        "model_fallback_reason": model_reason,
        "claim_boundary": (
            "static aggregate metrics prioritize graph programs but do not guarantee latency order"
        ),
    }
