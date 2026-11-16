#!/usr/bin/env python3
"""Shared-memory-realization-aware helpers for CPU--VTA Top-K selection.

The module deliberately keeps logical tensor traffic, framework materialization,
shared-slot capacity, VTA-internal DMA, and synchronization in separate units.
It never converts an unqualified zero-copy edge into a latency improvement.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


EPSILON = 1.0e-12
POLICY_COPY = "ordinary_copy"
POLICY_EXTERNAL_K2 = "external_k2_shared"
POLICY_POOL_ANCHOR_K2 = "pool_anchor_k2_shared"
DETAILED_DMA_FIELDS = (
    "load_buffer_2d_inp_bytes",
    "load_buffer_2d_wgt_bytes",
    "load_buffer_2d_acc_bytes",
    "store_buffer_2d_out_bytes",
    "load_buffer_2d_inp_calls",
    "load_buffer_2d_wgt_calls",
    "load_buffer_2d_acc_calls",
    "store_buffer_2d_out_calls",
    "load_buffer_2d_small_calls",
    "load_buffer_2d_strided_calls",
    "store_buffer_2d_small_calls",
    "store_buffer_2d_strided_calls",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def topology_id_from_candidate(candidate_id):
    return "__".join(re.sub(r"_t[0-9]+$", "", part) for part in candidate_id.split("__"))


def topology_diverse(rows):
    """Keep the best deterministic thread realization for each topology."""
    ordered = sorted(rows, key=lambda row: (float(row["m1_score_ms"]), row["candidate_id"]))
    seen = set()
    result = []
    for row in ordered:
        topology_id = row.get("topology_id") or topology_id_from_candidate(row["candidate_id"])
        if topology_id in seen:
            continue
        seen.add(topology_id)
        copied = dict(row)
        copied["topology_id"] = topology_id
        result.append(copied)
    return result


def validate_topology(row, edge_contexts):
    required = {
        "candidate_id",
        "topology_id",
        "m1_score_ms",
        "edge_ids",
        "boundary_payload_bytes",
        "theoretical_k2_slot_bytes",
        "vta_logical_dma",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError("topology is missing fields: {}".format(", ".join(missing)))
    if not math.isfinite(float(row["m1_score_ms"])) or float(row["m1_score_ms"]) <= 0:
        raise ValueError("invalid M1 score")
    if int(row["boundary_payload_bytes"]) < 0 or int(row["theoretical_k2_slot_bytes"]) < 0:
        raise ValueError("negative boundary or slot bytes")
    if int(row["theoretical_k2_slot_bytes"]) < int(row["boundary_payload_bytes"]):
        raise ValueError("K2 slot capacity is smaller than logical payload")
    missing_edges = sorted(set(row["edge_ids"]) - set(edge_contexts))
    if missing_edges:
        raise ValueError("topology references unknown edges: {}".format(missing_edges))
    edge_bytes = sum(int(edge_contexts[edge_id]["total_payload_bytes"]) for edge_id in row["edge_ids"])
    if edge_bytes != int(row["boundary_payload_bytes"]):
        raise ValueError("topology boundary payload does not match edge contracts")


def validate_certificate(certificate, topology):
    if certificate["candidate_id"] != topology["candidate_id"]:
        raise ValueError("certificate candidate mismatch")
    if certificate["topology_id"] != topology["topology_id"]:
        raise ValueError("certificate topology mismatch")
    if set(certificate["edge_ids"]) != set(topology["edge_ids"]):
        raise ValueError("certificate edge-set mismatch")
    if certificate["policy_id"] not in (POLICY_EXTERNAL_K2, POLICY_POOL_ANCHOR_K2):
        raise ValueError("unsupported shared-memory policy")
    if int(certificate["logical_boundary_bytes"]) != int(topology["boundary_payload_bytes"]):
        raise ValueError("certificate logical-byte mismatch")
    if not certificate.get("functional_gate_passed", False):
        raise ValueError("certificate functional gate did not pass")
    if int(certificate.get("correctness_boot_count", 0)) < 1:
        raise ValueError("certificate has no correctness boot")
    slot_bytes = int(certificate["additional_slot_bytes"])
    if slot_bytes < 0:
        raise ValueError("certificate has negative slot bytes")
    if certificate["policy_id"] == POLICY_EXTERNAL_K2 and slot_bytes != int(
        topology["theoretical_k2_slot_bytes"]
    ):
        raise ValueError("external-K2 certificate slot bytes disagree with topology")
    if certificate["policy_id"] == POLICY_POOL_ANCHOR_K2 and slot_bytes > int(
        topology["theoretical_k2_slot_bytes"]
    ):
        raise ValueError("pool-anchor certificate uses more slots than external K2")


def copy_realization(topology, copy_multiplicity=2):
    if int(copy_multiplicity) <= 0:
        raise ValueError("copy multiplicity must be positive")
    logical = int(topology["boundary_payload_bytes"])
    return {
        "requested_policy": POLICY_COPY,
        "realized_policy": POLICY_COPY,
        "fallback_to_copy": False,
        "binding_status": "exact_frozen_runner_copy_contract",
        "performance_status": "m1_copy_baseline",
        "logical_boundary_bytes": logical,
        "framework_materialization_bytes": logical * int(copy_multiplicity),
        "additional_slot_bytes": 0,
        "physical_ddr_bytes": None,
        "sync_service_ms": None,
        "throughput_delta_ms": None,
        "throughput_delta_ci95_ms": None,
        "reason": "ordinary set/get materialization; physical DDR traffic is not inferred",
    }


def shared_realization(topology, certificate, capacity_bytes=None):
    if certificate is None:
        row = copy_realization(topology)
        row.update({
            "requested_policy": POLICY_EXTERNAL_K2,
            "fallback_to_copy": True,
            "binding_status": "unqualified_shared_edge",
            "reason": "no exact candidate/edge policy certificate; fail-closed copy fallback",
        })
        return row
    validate_certificate(certificate, topology)
    slot_bytes = int(certificate["additional_slot_bytes"])
    if capacity_bytes is not None and slot_bytes > int(capacity_bytes):
        row = copy_realization(topology)
        row.update({
            "requested_policy": certificate["policy_id"],
            "fallback_to_copy": True,
            "binding_status": "shared_slot_capacity_exceeded",
            "reason": "certified policy requires {} bytes but budget is {}".format(
                slot_bytes, int(capacity_bytes)
            ),
        })
        return row
    ci = certificate.get("throughput_improvement_ci95_ms")
    performance_allowed = bool(certificate.get("performance_claim_allowed", False))
    return {
        "requested_policy": certificate["policy_id"],
        "realized_policy": certificate["policy_id"],
        "fallback_to_copy": False,
        "binding_status": certificate["binding_status"],
        "performance_status": (
            "qualified_latency_delta" if performance_allowed else "latency_abstention"
        ),
        "logical_boundary_bytes": int(topology["boundary_payload_bytes"]),
        "framework_materialization_bytes": 0,
        "additional_slot_bytes": slot_bytes,
        "physical_ddr_bytes": None,
        "sync_service_ms": certificate.get("boundary_api_service_ms"),
        "throughput_delta_ms": (
            certificate.get("throughput_improvement_ms") if performance_allowed else None
        ),
        "throughput_delta_ci95_ms": ci if performance_allowed else None,
        "reason": (
            "exact certified binding; no throughput delta used because its confidence interval crosses zero"
            if not performance_allowed
            else "exact certified binding and qualified throughput delta"
        ),
    }


def policy_feature_row(topology, realization):
    dma = topology["vta_logical_dma"]
    load_calls = int(dma.get("load_buffer_2d_calls", 0))
    store_calls = int(dma.get("store_buffer_2d_calls", 0))
    load_bytes = int(dma.get("load_buffer_2d_bytes", 0))
    store_bytes = int(dma.get("store_buffer_2d_bytes", 0))
    if (load_calls and not load_bytes) or (store_calls and not store_bytes):
        raise ValueError("nonzero DMA calls require nonzero logical bytes")
    helper = topology.get("vta_internal_cpu_helper_logical_access", {})
    return {
        "candidate_id": topology["candidate_id"],
        "topology_id": topology["topology_id"],
        "path": topology.get("path", []),
        "m1_score_ms": float(topology["m1_score_ms"]),
        "vta_islands": int(topology.get("vta_islands", 0)),
        "logical_boundary_bytes": int(realization["logical_boundary_bytes"]),
        "framework_materialization_bytes": int(realization["framework_materialization_bytes"]),
        "additional_slot_bytes": int(realization["additional_slot_bytes"]),
        "vta_dma_calls": load_calls + store_calls,
        "vta_internal_dma_bytes": load_bytes + store_bytes,
        "vta_load_calls": load_calls,
        "vta_store_calls": store_calls,
        "vta_load_bytes": load_bytes,
        "vta_store_bytes": store_bytes,
        "vta_load_inp_bytes": int(dma.get("load_buffer_2d_inp_bytes", 0)),
        "vta_load_wgt_bytes": int(dma.get("load_buffer_2d_wgt_bytes", 0)),
        "vta_load_acc_bytes": int(dma.get("load_buffer_2d_acc_bytes", 0)),
        "vta_store_out_bytes": int(dma.get("store_buffer_2d_out_bytes", 0)),
        "vta_load_inp_calls": int(dma.get("load_buffer_2d_inp_calls", 0)),
        "vta_load_wgt_calls": int(dma.get("load_buffer_2d_wgt_calls", 0)),
        "vta_load_acc_calls": int(dma.get("load_buffer_2d_acc_calls", 0)),
        "vta_store_out_calls": int(dma.get("store_buffer_2d_out_calls", 0)),
        "vta_load_small_calls": int(dma.get("load_buffer_2d_small_calls", 0)),
        "vta_load_strided_calls": int(dma.get("load_buffer_2d_strided_calls", 0)),
        "vta_store_small_calls": int(dma.get("store_buffer_2d_small_calls", 0)),
        "vta_store_strided_calls": int(dma.get("store_buffer_2d_strided_calls", 0)),
        "vta_fragmented_dma_calls": int(dma.get("load_buffer_2d_small_calls", 0))
        + int(dma.get("load_buffer_2d_strided_calls", 0))
        + int(dma.get("store_buffer_2d_small_calls", 0))
        + int(dma.get("store_buffer_2d_strided_calls", 0)),
        "vta_average_load_request_bytes": load_bytes / load_calls if load_calls else 0.0,
        "vta_average_store_request_bytes": store_bytes / store_calls if store_calls else 0.0,
        "vta_load_request_density_per_mib": (
            load_calls * (1 << 20) / load_bytes if load_bytes else 0.0
        ),
        "vta_store_request_density_per_mib": (
            store_calls * (1 << 20) / store_bytes if store_bytes else 0.0
        ),
        "detailed_dma_complete": all(name in dma for name in DETAILED_DMA_FIELDS),
        "cpu_helper_logical_bytes": int(helper.get("read_bytes", 0))
        + int(helper.get("write_bytes", 0)),
        "realization": realization,
    }


def dominates(left, right, dimensions):
    lhs = tuple(float(left[name]) for name in dimensions)
    rhs = tuple(float(right[name]) for name in dimensions)
    return all(a <= b + EPSILON for a, b in zip(lhs, rhs)) and any(
        a < b - EPSILON for a, b in zip(lhs, rhs)
    )


def pareto_layers(rows, dimensions):
    remaining = list(rows)
    result = []
    while remaining:
        layer = [
            row for row in remaining
            if not any(dominates(other, row, dimensions) for other in remaining if other is not row)
        ]
        layer.sort(
            key=lambda row: tuple(row[name] for name in dimensions)
            + (row["m1_score_ms"], row["topology_id"], row["candidate_id"])
        )
        result.append(layer)
        layer_ids = {id(row) for row in layer}
        remaining = [row for row in remaining if id(row) not in layer_ids]
    return result


def order_within_m1_band(rows, band_fraction, dimensions):
    if not rows:
        return []
    if float(band_fraction) < 0:
        raise ValueError("band fraction must be nonnegative")
    rows = sorted(
        rows,
        key=lambda row: (row["m1_score_ms"], row["topology_id"], row["candidate_id"]),
    )
    best = rows[0]
    limit = best["m1_score_ms"] * (1.0 + float(band_fraction))
    band = [row for row in rows[1:] if row["m1_score_ms"] <= limit + EPSILON]
    outside = [row for row in rows[1:] if row["m1_score_ms"] > limit + EPSILON]
    ordered = [best]
    for layer_index, layer in enumerate(pareto_layers(band, dimensions), 1):
        for row in layer:
            copied = dict(row)
            copied["pareto_layer"] = layer_index
            ordered.append(copied)
    ordered.extend(outside)
    return ordered


def make_exact_certificate(
    topology,
    source_row,
    source_scope,
    policy_id=POLICY_EXTERNAL_K2,
    evidence_identity=None,
):
    functional = source_row.get("functional_gate", {}).get("passed", False)
    if not functional:
        raise ValueError("source row did not pass its functional gate")
    return {
        "schema": "vta_shared_edge_policy_certificate_v1",
        "candidate_id": topology["candidate_id"],
        "topology_id": topology["topology_id"],
        "edge_ids": list(topology["edge_ids"]),
        "policy_id": policy_id,
        "logical_boundary_bytes": int(topology["boundary_payload_bytes"]),
        "additional_slot_bytes": int(topology["theoretical_k2_slot_bytes"]),
        "functional_gate_passed": True,
        "correctness_boot_count": 1,
        "binding_status": "exact_candidate_single_boot_functional",
        "performance_claim_allowed": False,
        "boundary_api_service_ms": None,
        "throughput_improvement_ms": None,
        "throughput_improvement_ci95_ms": None,
        "source_scope": source_scope,
        "evidence_identity": evidence_identity or {},
        "deployment_hash_revalidation_required": True,
    }


def promote_cross_boot_certificate(certificate, cross_boot):
    if cross_boot["candidate_id"] != certificate["candidate_id"]:
        raise ValueError("cross-boot evidence candidate mismatch")
    if not cross_boot["gates"]["all_boot_functional_gates_passed"]:
        raise ValueError("cross-boot functional gate failed")
    aggregate = cross_boot["aggregate"]
    promoted = dict(certificate)
    promoted.update({
        "correctness_boot_count": int(cross_boot["independent_boot_count"]),
        "binding_status": "exact_candidate_three_boot_functional",
        "boundary_api_service_ms": float(aggregate["B2_boundary_api_service_ms_mean"]),
        "throughput_improvement_ms": float(aggregate["delta_ii_ms_mean"]),
        "throughput_improvement_ci95_ms": dict(aggregate["delta_ii_ms_mean_ci95"]),
        "performance_claim_allowed": bool(cross_boot["gates"]["formal_performance_claim_allowed"]),
        "source_scope": "three independent boots; boundary service qualified, throughput abstains",
    })
    return promoted


def make_anchor_certificate(
    topology, external_certificate, result, source_scope, evidence_identity=None
):
    validate_certificate(external_certificate, topology)
    external = result["modes"]["external"]
    anchor = result["modes"]["anchor"]
    if int(external["slot_actual"]) != int(topology["theoretical_k2_slot_bytes"]):
        raise ValueError("measured external slot bytes disagree with topology")
    if not external.get("byte_exact") or not anchor.get("byte_exact"):
        raise ValueError("anchor evidence failed byte-exact output gate")
    if int(anchor.get("steady_growth", -1)) != 0:
        raise ValueError("anchor evidence has steady-state allocation growth")
    saved = int(external["slot_actual"]) - int(anchor["slot_actual"])
    if saved != int(result["predicted_saved_bytes"]) or saved <= 0:
        raise ValueError("anchor saving does not match frozen prediction")
    return {
        "schema": "vta_shared_edge_policy_certificate_v1",
        "candidate_id": topology["candidate_id"],
        "topology_id": topology["topology_id"],
        "edge_ids": list(topology["edge_ids"]),
        "policy_id": POLICY_POOL_ANCHOR_K2,
        "logical_boundary_bytes": int(topology["boundary_payload_bytes"]),
        "additional_slot_bytes": int(anchor["slot_actual"]),
        "external_slot_bytes": int(external["slot_actual"]),
        "pool_anchor_saved_bytes": saved,
        "functional_gate_passed": True,
        "correctness_boot_count": 1,
        "binding_status": "exact_candidate_single_boot_anchor_functional",
        "performance_claim_allowed": False,
        "boundary_api_service_ms": None,
        "throughput_improvement_ms": None,
        "throughput_improvement_ci95_ms": None,
        "source_scope": source_scope,
        "evidence_identity": evidence_identity or {},
        "deployment_hash_revalidation_required": True,
    }
