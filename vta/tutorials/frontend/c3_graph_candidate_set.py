#!/usr/bin/env python3
"""Exact graph-level candidate sets for composability-aware C3 selection."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Any, Dict, Mapping, Sequence

from c3_completion_policy import completion_policy_for_candidate, policy_set_compatibility
from c3_composability import candidate_disposition
from c3_composability import GRAPH_SCHEMA, GRAPH_STATUS, SET_SCHEMA, STATIC_SET_STATUS


SCHEMA = "c3_graph_candidate_set_v1"
SUM_FIELDS = (
    "inp_request_bytes", "wgt_request_bytes", "acc_request_bytes", "out_request_bytes",
    "total_request_bytes", "inp_request_calls", "wgt_request_calls",
    "acc_request_calls", "out_request_calls", "total_request_calls",
    "uop_load_bytes", "uop_load_calls", "instruction_bytes", "submissions", "sync_count",
)
MAX_FIELDS = ("inp_arena_fraction", "wgt_arena_fraction", "acc_arena_fraction",
              "uop_arena_fraction")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _number(value):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sorted_nonempty_strings(value, field):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("candidate requires non-empty {}".format(field))
    result = sorted(str(item) for item in value)
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError("candidate has invalid or duplicate {}".format(field))
    return result


def _aggregate_metrics(candidates: Sequence[Mapping[str, Any]], source_field: str) -> Dict[str, Any]:
    """Aggregate one metric source without pretending it predicts latency."""

    result: Dict[str, Any] = {}
    for field in SUM_FIELDS:
        values = [_number((row.get(source_field) or {}).get(field)) for row in candidates]
        result[field] = None if any(value is None for value in values) else sum(values)
    for field in MAX_FIELDS:
        values = [_number((row.get(source_field) or {}).get(field)) for row in candidates]
        result[field] = None if any(value is None for value in values) else max(values, default=0.0)
    result["replacement_count"] = len(candidates)
    result["aggregation_claim"] = (
        "sum for requested work, max for per-candidate arena fractions; no latency additivity claim"
    )
    return result


def aggregate_metrics(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return _aggregate_metrics(candidates, "candidate_metrics")


def aggregate_incumbent_metrics(candidates: Sequence[Mapping[str, Any]]):
    """Return a matched incumbent aggregate, or ``None`` when evidence is absent."""

    if candidates and any(not isinstance(row.get("incumbent_metrics"), Mapping)
                          for row in candidates):
        return None
    return _aggregate_metrics(candidates, "incumbent_metrics")


def build_graph_candidate_set(graph_identity: Mapping[str, Any],
                              candidates: Sequence[Mapping[str, Any]],
                              certificates_by_candidate: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Build one exact, statically compatible graph replacement program."""

    if not graph_identity.get("graph_sha256") or not graph_identity.get("params_sha256"):
        raise ValueError("graph identity requires graph_sha256 and params_sha256")
    members = []
    workload_owners, node_owners, function_owners = {}, {}, {}
    policy_names = []
    for raw in candidates:
        candidate = dict(raw)
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id:
            raise ValueError("candidate lacks candidate_id")
        disposition = candidate_disposition(candidate, certificates_by_candidate)
        if not disposition["eligible"] or bool(candidate.get("is_incumbent")):
            raise ValueError("replacement candidate is ineligible: {}".format(disposition["reason"]))
        workload_id = str(candidate.get("execution_task_id") or candidate.get("workload_id") or "")
        if not workload_id:
            raise ValueError("candidate lacks execution_task_id/workload_id")
        graph_nodes = _sorted_nonempty_strings(candidate.get("graph_node_ids"), "graph_node_ids")
        functions = _sorted_nonempty_strings(candidate.get("scoped_functions"), "scoped_functions")
        if workload_id in workload_owners:
            raise ValueError("multiple candidates target workload {}".format(workload_id))
        workload_owners[workload_id] = candidate_id
        for node in graph_nodes:
            if node in node_owners:
                raise ValueError("graph node {} has overlapping candidates".format(node))
            node_owners[node] = candidate_id
        policy = completion_policy_for_candidate(candidate)
        policy_names.append(policy["name"])
        if policy["ordered_debug_transitions"]:
            for function in functions:
                if function in function_owners:
                    raise ValueError("scoped completion function {} overlaps".format(function))
                function_owners[function] = candidate_id
        certificate = certificates_by_candidate.get(candidate_id)
        members.append({
            "candidate_id": candidate_id,
            "workload_id": workload_id,
            "graph_node_ids": graph_nodes,
            "scoped_functions": functions,
            "completion_policy": policy["name"],
            "candidate_certificate_id": (
                certificate.get("certificate_id") if certificate is not None else None
            ),
        })
    compatibility = policy_set_compatibility(policy_names)
    if not compatibility["compatible"]:
        raise ValueError("completion policies are incompatible: {}".format(compatibility["reasons"]))
    members.sort(key=lambda row: (row["workload_id"], row["candidate_id"]))
    identity = {
        "schema": SCHEMA,
        "graph": dict(graph_identity),
        "members": members,
    }
    return {
        "schema": SCHEMA,
        "candidate_set_id": _digest(identity),
        "identity": identity,
        "members": members,
        "candidate_ids": [row["candidate_id"] for row in members],
        "compatibility": compatibility,
        "aggregate_metrics": aggregate_metrics(candidates),
        "incumbent_aggregate_metrics": aggregate_incumbent_metrics(candidates),
        "graph_correctness_claimed": False,
        "required_next_gate": "exact full-graph multi-seed output equality",
    }


def enumerate_graph_candidate_sets(graph_identity: Mapping[str, Any],
                                   task_pools: Mapping[str, Sequence[Mapping[str, Any]]],
                                   certificates_by_candidate: Mapping[str, Mapping[str, Any]],
                                   max_replacements: int = 3,
                                   maximum_combinations: int = 4096) -> Dict[str, Any]:
    """Exactly enumerate a bounded product of per-task replacement pools.

    The implicit first choice for every task is "keep incumbent".  This helper
    never silently truncates a larger product because doing so would turn a
    heuristic omission into an apparent exhaustive candidate-set result.
    """

    tasks = sorted(str(task) for task in task_pools)
    pools = []
    candidate_ids = set()
    for task in tasks:
        rows = [dict(row) for row in task_pools[task]]
        for row in rows:
            declared = str(row.get("execution_task_id") or row.get("workload_id") or "")
            if declared != task:
                raise ValueError("candidate pool/task identity mismatch for {}".format(task))
            candidate_id = str(row.get("candidate_id", ""))
            if not candidate_id or candidate_id in candidate_ids:
                raise ValueError("candidate IDs must be non-empty and globally unique")
            candidate_ids.add(candidate_id)
        pools.append([None] + rows)
    product_size = math.prod(len(pool) for pool in pools)
    if product_size > int(maximum_combinations):
        raise ValueError(
            "candidate-set product {} exceeds exact enumeration limit {}".format(
                product_size, maximum_combinations
            )
        )
    accepted, rejected = [], []
    incumbent_identity = {"schema": SCHEMA, "graph": dict(graph_identity), "members": []}
    incumbent = {
        "schema": SCHEMA,
        "candidate_set_id": _digest(incumbent_identity),
        "identity": incumbent_identity,
        "members": [],
        "candidate_ids": [],
        "is_incumbent": True,
        "compatibility": {"compatible": True, "policy_names": [], "reasons": []},
        "aggregate_metrics": aggregate_metrics([]),
        "incumbent_aggregate_metrics": aggregate_incumbent_metrics([]),
        "graph_correctness_claimed": True,
        "required_next_gate": None,
    }
    accepted.append(incumbent)
    for choices in itertools.product(*pools):
        rows = [row for row in choices if row is not None]
        if not rows or len(rows) > int(max_replacements):
            continue
        try:
            candidate_set = build_graph_candidate_set(
                graph_identity, rows, certificates_by_candidate
            )
        except ValueError as error:
            rejected.append({
                "candidate_ids": sorted(str(row.get("candidate_id", "")) for row in rows),
                "reason": str(error),
            })
            continue
        candidate_set["is_incumbent"] = False
        accepted.append(candidate_set)
    accepted.sort(key=lambda row: (
        0 if row.get("is_incumbent") else 1,
        len(row.get("candidate_ids") or []),
        row["candidate_set_id"],
    ))
    return {
        "schema": "c3_graph_candidate_set_enumeration_v1",
        "status": "EXACT_WITHIN_DECLARED_PRODUCT_BOUND",
        "task_ids": tasks,
        "cartesian_product_size_including_incumbent_choices": product_size,
        "max_replacements": int(max_replacements),
        "maximum_combinations": int(maximum_combinations),
        "accepted_sets": accepted,
        "rejected_combinations": rejected,
        "exhaustiveness_claim": (
            "complete only for the supplied per-task pools and max_replacements bound"
        ),
    }


def inherit_graph_context_certificate(graph_set: Mapping[str, Any],
                                      legacy_static_set: Mapping[str, Any],
                                      legacy_graph_certificate: Mapping[str, Any]) -> Dict[str, Any]:
    """Migrate exact correctness evidence to a richer identity of the same binary.

    This does not infer correctness for a different program.  It requires exact
    equality of graph/parameter hashes, candidate membership, completion scopes,
    and the legacy set/certificate link.  The new identity only adds the explicit
    workload/node/function mapping that v1 omitted.
    """

    if graph_set.get("schema") != SCHEMA:
        raise ValueError("input is not a graph candidate set")
    if (legacy_static_set.get("schema") != SET_SCHEMA
            or legacy_static_set.get("status") != STATIC_SET_STATUS):
        raise ValueError("legacy static set is not qualified")
    if (legacy_graph_certificate.get("schema") != GRAPH_SCHEMA
            or legacy_graph_certificate.get("status") != GRAPH_STATUS):
        raise ValueError("legacy graph certificate is not qualified")
    legacy_set_id = str(legacy_static_set.get("candidate_set_id", ""))
    if legacy_graph_certificate.get("candidate_set_id") != legacy_set_id:
        raise ValueError("legacy graph/static set identity mismatch")
    graph = graph_set.get("identity", {}).get("graph") or {}
    legacy_identity = legacy_static_set.get("identity") or {}
    legacy_graph = legacy_identity.get("graph") or {}
    if (graph.get("graph_sha256") != legacy_graph.get("graph_sha256")
            or graph.get("params_sha256") != legacy_graph.get("params_sha256")):
        raise ValueError("graph or parameter identity changed")
    candidates = sorted(graph_set.get("candidate_ids") or [])
    if candidates != sorted(legacy_identity.get("candidate_ids") or []):
        raise ValueError("candidate membership changed")
    functions = sorted(
        function for member in graph_set.get("members") or []
        for function in member.get("scoped_functions") or []
    )
    if functions != sorted(legacy_identity.get("scoped_functions") or []):
        raise ValueError("completion function scope changed")
    migration = {
        "new_candidate_set_id": graph_set["candidate_set_id"],
        "legacy_candidate_set_id": legacy_set_id,
        "graph_sha256": graph["graph_sha256"],
        "params_sha256": graph["params_sha256"],
        "candidate_ids": candidates,
        "scoped_functions": functions,
        "rule": "identity enrichment only; compiled graph program is unchanged",
    }
    return {
        "schema": GRAPH_SCHEMA,
        "status": GRAPH_STATUS,
        "graph_certificate_id": _digest(migration),
        "candidate_set_id": graph_set["candidate_set_id"],
        "candidate_ids": candidates,
        "checks": {
            **dict(legacy_graph_certificate.get("checks") or {}),
            "identity_migration_exact": True,
        },
        "evidence": {
            **dict(legacy_graph_certificate.get("evidence") or {}),
            "identity_migration": migration,
            "legacy_graph_certificate_id": legacy_graph_certificate.get(
                "graph_certificate_id"
            ),
        },
        "timing_eligible": legacy_graph_certificate.get("timing_eligible") is True,
        "deployment_eligible_after_performance_policy": (
            legacy_graph_certificate.get("deployment_eligible_after_performance_policy") is True
        ),
    }


def inherit_boot_result_identity(graph_set: Mapping[str, Any],
                                 migrated_graph_certificate: Mapping[str, Any],
                                 legacy_boot_result: Mapping[str, Any]) -> Dict[str, Any]:
    """Relabel one boot result only when an exact identity migration proves equivalence."""

    candidate_set_id = str(graph_set.get("candidate_set_id", ""))
    if (not candidate_set_id
            or migrated_graph_certificate.get("candidate_set_id") != candidate_set_id):
        raise ValueError("migrated graph certificate targets another candidate set")
    migration = (migrated_graph_certificate.get("evidence") or {}).get("identity_migration") or {}
    legacy_id = str(migration.get("legacy_candidate_set_id", ""))
    if not legacy_id or legacy_boot_result.get("candidate_set_id") != legacy_id:
        raise ValueError("legacy boot result targets another candidate set")
    if sorted(legacy_boot_result.get("candidate_ids") or []) != sorted(
            graph_set.get("candidate_ids") or []):
        raise ValueError("boot result candidate membership changed")
    required = ("boot_id", "median_improvement_pct", "wins", "rounds")
    if any(key not in legacy_boot_result for key in required):
        raise ValueError("legacy boot result is incomplete")
    return {
        **dict(legacy_boot_result),
        "candidate_set_id": candidate_set_id,
        "legacy_candidate_set_id": legacy_id,
        "identity_migration_graph_certificate_id": migrated_graph_certificate.get(
            "graph_certificate_id"
        ),
        "identity_migration_only": True,
        "latency_values_modified": False,
    }
