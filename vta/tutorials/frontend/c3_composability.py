#!/usr/bin/env python3
"""Machine-readable composability certificates for C3 VTA candidates.

The certificates deliberately separate three claims:

* local qualification proves one lowered candidate is internally well formed;
* static set qualification proves scoped completion does not escape its linked
  functions, but does not claim that the full graph is correct;
* graph-context qualification records an exact full-graph correctness result.

No latency is required for any correctness certificate.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Sequence

from c3_completion_policy import (
    DEPENDENCY_TOKENS,
    LEGACY_UNSCOPED_BARRIER,
    SCOPED_FORCE_SERIAL,
    TRANSITION_SAFE_FORCE_SERIAL,
    completion_policy_for_candidate,
    policy_record,
)


CANDIDATE_SCHEMA = "c3_composability_candidate_certificate_v1"
SET_SCHEMA = "c3_composability_set_certificate_v1"
GRAPH_SCHEMA = "c3_composability_graph_certificate_v1"
# Backward-compatible public name used by already frozen v1 certificates.
POLICY = SCOPED_FORCE_SERIAL
LOCAL_STATUS = "LOCAL_SCOPED_COMPLETION_QUALIFIED"
STATIC_SET_STATUS = "STATICALLY_SCOPED_PENDING_GRAPH_GATE"
GRAPH_STATUS = "GRAPH_CONTEXT_QUALIFIED"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _all_zero_mismatches(value: Mapping[str, Sequence[int]]) -> bool:
    return bool(value) and all(
        isinstance(row, (list, tuple)) and row and all(int(item) == 0 for item in row)
        for row in value.values()
    )


def candidate_certificate(candidate: Mapping[str, Any], static: Mapping[str, Any],
                          fsim: Mapping[str, Any]) -> Dict[str, Any]:
    """Create a strict local certificate from existing qualification records."""

    candidate_id = str(candidate.get("candidate_id", ""))
    if not candidate_id or static.get("candidate_id") != candidate_id or fsim.get("candidate_id") != candidate_id:
        raise ValueError("candidate/static/FSim identity mismatch")
    policy = completion_policy_for_candidate(candidate)
    if (not policy["production_eligible"] or not policy["candidate_local"]
            or not policy["requires_local_composability_certificate"]):
        raise ValueError("candidate does not declare a certifiable local completion policy")
    address = static.get("final_address_audit") or {}
    arena = static.get("arena_capacity_certificate") or {}
    dependency = static.get("dependency_audit") or {}
    command = fsim.get("command_features") or {}
    checks = {
        "static_ok": static.get("status") == "ok",
        "arena_capacity_passed": arena.get("status") == "passed",
        "final_address_passed": address.get("status") == "passed",
        "dependency_balanced": dependency.get("balanced") is True,
        "forbidden_direct_1_3_absent": dependency.get("forbidden_direct_1_3_count") == 0,
        "debug_transition_closed": (
            address.get("debug_mode_transitions") == policy["ordered_debug_transitions"]
        ),
        "entry_drain_before_debug": (
            policy["entry_drains_before_debug"] == 0
            or (
                len(address.get("sync_event_positions") or [])
                >= policy["entry_drains_before_debug"]
                and (address.get("sync_event_positions") or [None])[0] == 0
            )
        ),
        "fsim_three_seed_passed": fsim.get("status") == "passed" and len(fsim.get("seeds") or []) == 3
        and all(row.get("correct") is True for row in fsim.get("seeds") or []),
        "command_signature_consistent": command.get("status") == "available_three_seed_consistent",
    }
    if not all(checks.values()):
        raise ValueError("local composability qualification failed: {}".format(checks))
    workload_id = str(candidate.get("workload_id", ""))
    if not workload_id:
        raise ValueError("candidate lacks workload_id")
    identity = {
        "candidate_id": candidate_id,
        "workload_id": workload_id,
        "implementation_mode": candidate.get("implementation_mode"),
        "residence_mode": candidate.get("residence_mode") or candidate.get("candidate_mode"),
        "completion_policy": policy["name"],
        "schedule_version": (candidate.get("candidate_identity_fields") or {}).get("schedule_version"),
        "parent_mode5_candidate_id": candidate.get("parent_mode5_candidate_id"),
    }
    return {
        "schema": CANDIDATE_SCHEMA,
        "status": LOCAL_STATUS,
        "certificate_id": digest(identity),
        "identity": identity,
        "completion_contract": {
            "global_debug_flag_required": 0,
            "enter_debug_mode": (
                policy["ordered_debug_transitions"][0]
                if policy["ordered_debug_transitions"] else None
            ),
            "exit_debug_mode": (
                policy["ordered_debug_transitions"][-1]
                if policy["ordered_debug_transitions"] else None
            ),
            "final_transition": policy["ordered_debug_transitions"],
            "blocking_completion_between_enter_and_exit": policy[
                "blocking_completion_before_exit"
            ],
            "scope": policy["scope"],
            "entry_drains_before_debug": policy["entry_drains_before_debug"],
        },
        "checks": checks,
        "evidence": {
            "lowered_tir_sha256": static.get("lowered_tir_sha256"),
            "storage_tir_sha256": static.get("storage_tir_sha256"),
            "implementation_candidate_id": static.get("implementation_candidate_id"),
            "fsim_seed_sha256": {
                str(row["seed"]): row.get("actual_sha256") for row in fsim.get("seeds") or []
            },
            "performance_label": None,
            "board_contacted": False,
        },
    }


def validate_candidate_certificate(value: Mapping[str, Any]) -> None:
    if value.get("schema") != CANDIDATE_SCHEMA or value.get("status") != LOCAL_STATUS:
        raise ValueError("candidate composability certificate is not locally qualified")
    if not value.get("identity", {}).get("candidate_id"):
        raise ValueError("candidate certificate lacks identity")
    if not all((value.get("checks") or {}).values()):
        raise ValueError("candidate certificate contains a failed check")
    contract = value.get("completion_contract") or {}
    policy = policy_record(value.get("identity", {}).get("completion_policy"))
    if (contract.get("global_debug_flag_required") != 0
            or contract.get("final_transition") != policy["ordered_debug_transitions"]
            or contract.get("blocking_completion_between_enter_and_exit")
            is not policy["blocking_completion_before_exit"]
            or contract.get("entry_drains_before_debug")
            != policy["entry_drains_before_debug"]):
        raise ValueError("candidate completion contract is not closed")


def static_set_certificate(candidate_certificates: Iterable[Mapping[str, Any]],
                           graph_identity: Mapping[str, Any],
                           binary_scope_audit: Mapping[str, Any]) -> Dict[str, Any]:
    """Certify static scoping while leaving full-graph correctness pending."""

    certificates = list(candidate_certificates)
    if not certificates:
        raise ValueError("candidate set is empty")
    for certificate in certificates:
        validate_candidate_certificate(certificate)
    candidate_ids = [row["identity"]["candidate_id"] for row in certificates]
    workload_ids = [row["identity"]["workload_id"] for row in certificates]
    if len(candidate_ids) != len(set(candidate_ids)) or len(workload_ids) != len(set(workload_ids)):
        raise ValueError("candidate set repeats a candidate or workload")
    if binary_scope_audit.get("status") != "passed" or binary_scope_audit.get("global_vta_debug_flag") != 0:
        raise ValueError("binary completion scope audit failed")
    calls = binary_scope_audit.get("calls_by_function") or {}
    if not calls or any(int(count) != 2 for count in calls.values()):
        raise ValueError("each scoped function must contain exact enter/exit calls")
    if not graph_identity.get("graph_sha256") or not graph_identity.get("params_sha256"):
        raise ValueError("graph identity is incomplete")
    policies = sorted({row["identity"]["completion_policy"] for row in certificates})
    identity = {
        "graph": dict(graph_identity),
        "candidate_ids": sorted(candidate_ids),
        "scoped_functions": sorted(calls),
        "completion_policy": policies[0] if len(policies) == 1 else policies,
    }
    return {
        "schema": SET_SCHEMA,
        "status": STATIC_SET_STATUS,
        "candidate_set_id": digest(identity),
        "identity": identity,
        "candidate_certificate_ids": sorted(row["certificate_id"] for row in certificates),
        "binary_scope_audit": dict(binary_scope_audit),
        "graph_correctness_claimed": False,
        "required_next_gate": "exact full-graph multi-seed output equality before timing or deployment",
    }


def graph_context_certificate(static_set: Mapping[str, Any],
                              evidence: Mapping[str, Any]) -> Dict[str, Any]:
    """Upgrade one exact static set only after its full graph passes correctness."""

    if static_set.get("schema") != SET_SCHEMA or static_set.get("status") != STATIC_SET_STATUS:
        raise ValueError("input is not a pending static set certificate")
    if evidence.get("candidate_set_id") != static_set.get("candidate_set_id"):
        raise ValueError("graph evidence targets another candidate set")
    mismatches = evidence.get("mismatch_count_by_seed_and_output") or {}
    checks = {
        "all_equal": evidence.get("all_equal") is True,
        "multi_seed": len(mismatches) >= 3,
        "all_mismatch_counts_zero": _all_zero_mismatches(mismatches),
        "boot_identity_present": bool(evidence.get("boot_id")),
        "no_bitstream_reload_required_by_mechanism": evidence.get("bitstream_reloaded") is False,
    }
    if not all(checks.values()):
        raise ValueError("full-graph correctness gate failed: {}".format(checks))
    return {
        "schema": GRAPH_SCHEMA,
        "status": GRAPH_STATUS,
        "graph_certificate_id": digest({
            "candidate_set_id": static_set["candidate_set_id"],
            "boot_id": evidence["boot_id"],
            "mismatches": mismatches,
        }),
        "candidate_set_id": static_set["candidate_set_id"],
        "candidate_ids": static_set["identity"]["candidate_ids"],
        "checks": checks,
        "evidence": {
            "boot_id": evidence["boot_id"],
            "mismatch_count_by_seed_and_output": mismatches,
            "complete_graph_run_calls": evidence.get("complete_graph_run_calls"),
            "timing_collected": bool(evidence.get("timing_collected")),
            "performance_claim": bool(evidence.get("performance_claim")),
        },
        "timing_eligible": True,
        "deployment_eligible_after_performance_policy": True,
    }


def candidate_disposition(candidate: Mapping[str, Any],
                          certificates_by_candidate: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Return the hard selector-v3 disposition without looking at latency."""

    candidate_id = str(candidate.get("candidate_id", ""))
    if bool(candidate.get("is_incumbent")):
        return {"eligible": True, "reason": "incumbent_fallback", "required_next_gate": None}
    try:
        policy = completion_policy_for_candidate(candidate)
    except ValueError:
        return {"eligible": False, "reason": "unsupported_or_unclassified_completion_policy",
                "required_next_gate": "classify_completion_policy"}
    if policy["name"] == LEGACY_UNSCOPED_BARRIER:
        return {"eligible": False, "reason": "legacy_unscoped_residency_not_composable",
                "required_next_gate": "regenerate_as_mode11"}
    if not policy["production_eligible"]:
        if policy["name"] == SCOPED_FORCE_SERIAL:
            return {"eligible": False, "reason": "predecessor_sensitive_mode10_not_transition_safe",
                    "required_next_gate": "regenerate_as_mode11"}
        return {"eligible": False, "reason": "completion_policy_not_production_eligible",
                "required_next_gate": "select_production_completion_policy"}
    if policy["name"] == DEPENDENCY_TOKENS:
        locally_correct = candidate.get("local_qualification_passed") is True or candidate.get("fpga_correct") is True
        if not locally_correct:
            return {"eligible": False, "reason": "missing_standard_local_correctness",
                    "required_next_gate": "standard_local_qualification"}
        return {"eligible": True, "reason": "standard_nonresident_candidate",
                "required_next_gate": "exact_graph_context_correctness"}
    if not policy["requires_local_composability_certificate"]:
        return {"eligible": False, "reason": "unsupported_or_unclassified_completion_policy",
                "required_next_gate": "classify_completion_policy"}
    certificate = certificates_by_candidate.get(candidate_id)
    if certificate is None:
        return {"eligible": False, "reason": "missing_local_composability_certificate",
                "required_next_gate": "local_composability_qualification"}
    try:
        validate_candidate_certificate(certificate)
    except ValueError as error:
        return {"eligible": False, "reason": "invalid_local_composability_certificate: {}".format(error),
                "required_next_gate": "repair_or_reject"}
    return {"eligible": True, "reason": "locally_qualified_scoped_completion",
            "required_next_gate": "exact_graph_context_correctness"}
