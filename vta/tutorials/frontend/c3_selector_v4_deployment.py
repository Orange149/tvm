#!/usr/bin/env python3
"""Production-eligibility gate in front of the frozen selector-v4 organizer.

Selector v4 orders candidates only after semantic legality has been decided.
This module keeps isolated-operator reference candidates in the audit trail,
but prevents legacy completion policies from entering deployment budgets.
No latency or performance label is consumed by this gate.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from c3_composability import candidate_disposition
from c3_selector_v4_diverse import diverse_static_order


SCHEMA = "c3_selector_v4_production_gate_v1"


def production_static_order(
        rows: Sequence[Mapping[str, Any]],
        certificates_by_candidate: Mapping[str, Mapping[str, Any]],
        max_budget: int = 3,
        window_multiplier: int = 2) -> Dict[str, Any]:
    """Gate candidates by completion semantics, then form one v4 order.

    The incumbent remains a measured fallback and is not part of the extra
    candidate budget.  Every non-incumbent must pass the existing fail-closed
    composability disposition before the label-free organizer can see it.
    """

    if not rows:
        raise ValueError("candidate pool is empty")
    candidate_ids = [str(row.get("candidate_id") or "") for row in rows]
    program_ids = [str(row.get("program_id") or "") for row in rows]
    if any(not value for value in candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be present and unique")
    if any(not value for value in program_ids) or len(program_ids) != len(set(program_ids)):
        raise ValueError("program IDs must be present and unique")

    incumbents = [row for row in rows if row.get("is_incumbent") is True]
    if len(incumbents) != 1:
        raise ValueError("exactly one incumbent is required")

    eligible = []
    dispositions = {}
    for row in rows:
        disposition = candidate_disposition(row, certificates_by_candidate)
        dispositions[str(row["candidate_id"])] = {
            **disposition,
            "program_id": str(row["program_id"]),
            "residence_mode": row.get("residence_mode") or row.get("candidate_mode"),
        }
        if disposition["eligible"] and row.get("is_incumbent") is not True:
            eligible.append(row)

    if not eligible:
        raise ValueError("no production-eligible new candidate remains after the hard gate")
    ordered = diverse_static_order(
        eligible, max_budget=max_budget, window_multiplier=window_multiplier
    )
    rejected = sorted(
        candidate_id for candidate_id, value in dispositions.items()
        if not value["eligible"]
    )
    admitted = sorted(
        candidate_id for candidate_id, value in dispositions.items()
        if value["eligible"]
    )
    return {
        "schema": SCHEMA,
        "selector_kind": "production_gate_then_label_free_v4_organizer",
        "incumbent_candidate_id": str(incumbents[0]["candidate_id"]),
        "incumbent_program_id": str(incumbents[0]["program_id"]),
        "production_eligible_candidate_ids": admitted,
        "production_rejected_candidate_ids": rejected,
        "candidate_dispositions": dispositions,
        "selector_v4": ordered,
        "ordered_program_ids": ordered["ordered_candidate_ids"],
        "budget_prefixes": ordered["budget_prefixes"],
        "performance_labels_consumed": [],
        "performance_model_claim": False,
        "safe_pruning_claim": False,
        "reference_pool_retention": (
            "rejected programs may remain in an isolated-operator complete pool, "
            "but cannot enter deployment budgets"
        ),
    }
