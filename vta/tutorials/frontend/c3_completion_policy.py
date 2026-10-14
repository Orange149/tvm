#!/usr/bin/env python3
"""Versioned completion policies for composable C3 VTA programs.

The policy is part of a program's semantics, not merely a performance knob.
This registry keeps selector code independent of numeric experimental mode IDs
and provides a fail-closed extension point for future completion mechanisms.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


SCHEMA = "c3_completion_policy_registry_v1"
DEPENDENCY_TOKENS = "dependency_tokens_v1"
SCOPED_FORCE_SERIAL = "scoped_force_serial_completion_v1"
TRANSITION_SAFE_FORCE_SERIAL = "entry_drain_scoped_force_serial_completion_v1"
LEGACY_UNSCOPED_BARRIER = "legacy_unscoped_barrier_v1"
GLOBAL_FORCE_SERIAL_DIAGNOSTIC = "global_force_serial_diagnostic_v1"


POLICIES: Dict[str, Dict[str, Any]] = {
    DEPENDENCY_TOKENS: {
        "scope": "normal_lowered_program",
        "candidate_local": True,
        "ordered_debug_transitions": [],
        "blocking_completion_before_exit": False,
        "production_eligible": True,
        "requires_local_composability_certificate": False,
        "entry_drains_before_debug": 0,
    },
    SCOPED_FORCE_SERIAL: {
        "scope": "one_lowered_resident_command_segment",
        "candidate_local": True,
        "ordered_debug_transitions": [32, 0],
        "blocking_completion_before_exit": True,
        "production_eligible": False,
        "requires_local_composability_certificate": True,
        "entry_drains_before_debug": 0,
    },
    TRANSITION_SAFE_FORCE_SERIAL: {
        "scope": "one_lowered_resident_command_segment_with_entry_drain",
        "candidate_local": True,
        "ordered_debug_transitions": [32, 0],
        "blocking_completion_before_exit": True,
        "production_eligible": True,
        "requires_local_composability_certificate": True,
        "entry_drains_before_debug": 1,
    },
    LEGACY_UNSCOPED_BARRIER: {
        "scope": "legacy_resident_segment_without_closed_completion_contract",
        "candidate_local": False,
        "ordered_debug_transitions": [],
        "blocking_completion_before_exit": False,
        "production_eligible": False,
        "requires_local_composability_certificate": True,
        "entry_drains_before_debug": 0,
    },
    GLOBAL_FORCE_SERIAL_DIAGNOSTIC: {
        "scope": "whole_graph_runtime",
        "candidate_local": False,
        "ordered_debug_transitions": [],
        "blocking_completion_before_exit": True,
        "production_eligible": False,
        "requires_local_composability_certificate": False,
        "entry_drains_before_debug": 0,
    },
}

MODE_TO_POLICY = {
    "original": DEPENDENCY_TOKENS,
    "input_stationary": DEPENDENCY_TOKENS,
    "input_weight_resident_composable": SCOPED_FORCE_SERIAL,
    "input_weight_resident_transition_safe": TRANSITION_SAFE_FORCE_SERIAL,
    "input_weight_resident_barrier": LEGACY_UNSCOPED_BARRIER,
    "weight_resident_barrier": LEGACY_UNSCOPED_BARRIER,
    "weight_stationary_barrier": LEGACY_UNSCOPED_BARRIER,
    "global_force_serial": GLOBAL_FORCE_SERIAL_DIAGNOSTIC,
}


def policy_record(name: str) -> Dict[str, Any]:
    if name not in POLICIES:
        raise ValueError("unknown completion policy: {}".format(name))
    return {"schema": SCHEMA, "name": name, **POLICIES[name]}


def completion_policy_for_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve and cross-check one candidate's explicit or legacy policy."""

    mode = str(candidate.get("candidate_mode") or candidate.get("residence_mode") or "")
    inferred = MODE_TO_POLICY.get(mode)
    explicit = candidate.get("completion_policy")
    if explicit is None:
        if inferred is None:
            raise ValueError("candidate mode has no registered completion policy: {}".format(mode))
        name = inferred
    else:
        name = str(explicit)
        if name not in POLICIES:
            raise ValueError("candidate declares unknown completion policy: {}".format(name))
        if inferred is not None and name != inferred:
            raise ValueError(
                "candidate mode/policy mismatch: {} requires {}, got {}".format(
                    mode, inferred, name
                )
            )
    record = policy_record(name)
    record["resolved_from"] = "explicit" if explicit is not None else "registered_mode"
    record["candidate_mode"] = mode
    return record


def policy_set_compatibility(policy_names):
    """Return the policy-only compatibility result for a graph candidate set.

    Spatial/function overlap is checked by the graph-set builder because it
    requires graph identity.  This function only handles semantic policy class.
    """

    names = [str(name) for name in policy_names]
    unknown = sorted(set(names) - set(POLICIES))
    reasons = ["unknown_completion_policy:{}".format(name) for name in unknown]
    for name in sorted(set(names) & set(POLICIES)):
        if not POLICIES[name]["production_eligible"]:
            reasons.append("policy_not_production_eligible:{}".format(name))
    return {
        "compatible": not reasons,
        "policy_names": sorted(names),
        "reasons": reasons,
        "rule": "all policies production-eligible; graph overlap checked separately",
    }
