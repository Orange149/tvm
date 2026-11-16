#!/usr/bin/env python3
"""Freeze a post-label diagnostic of the unified C1 structured Top-K order.

The ongoing S0/S1/S2 board protocol is immutable.  This script combines the
already specified fusion-aware dimensions with the shared-policy realization
and fine-grained DMA fields, but marks the result ineligible for the ongoing
prospective ranking claim.  It never reads board timing artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from shared_memory_topk import (
    POLICY_COPY,
    POLICY_EXTERNAL_K2,
    POLICY_POOL_ANCHOR_K2,
    order_within_m1_band,
    policy_feature_row,
    read_json,
    sha256_file,
    shared_realization,
    topology_diverse,
    validate_certificate,
    validate_topology,
)


FRONTEND = Path(__file__).resolve().parent
REPORT = FRONTEND / "report_out" / "stage_tile_cotuning"
DEFAULT_TOPOLOGIES = REPORT / "shared_edge_audit/domain_run2/topologies.json"
DEFAULT_EDGE_CONTEXTS = REPORT / "shared_edge_audit/domain_run2/edge_contexts.json"
DEFAULT_CERTIFICATES = REPORT / (
    "compile_context_audit/c1_shared_memory_topk_run5/policy_certificates.json"
)
DEFAULT_LEGACY_CONTRACT = REPORT / (
    "compile_context_audit/c1_shared_memory_topk_run5/selection_contract.json"
)
DEFAULT_FUSION_ESTIMATES = REPORT / (
    "compile_context_audit/c1_fusion_transfer_freeze_run1/estimates.json"
)
DEFAULT_FUSION_CONTRACT = REPORT / (
    "compile_context_audit/c1_fusion_selection_freeze_run3/selection_contract.json"
)


# All dimensions are minimized. Request density is the monotone inverse of
# average request size, so larger requests are preferred without reversing the
# generic Pareto implementation's direction.
UNIFIED_DIMENSIONS = (
    "framework_materialization_bytes",
    "additional_slot_bytes",
    "vta_load_inp_bytes",
    "vta_load_wgt_bytes",
    "vta_load_acc_bytes",
    "vta_store_out_bytes",
    "vta_load_inp_calls",
    "vta_load_wgt_calls",
    "vta_load_acc_calls",
    "vta_store_out_calls",
    "vta_fragmented_dma_calls",
    "vta_load_request_density_per_mib",
    "vta_store_request_density_per_mib",
    "cpu_helper_logical_bytes",
    "convert_calls",
    "vta_islands",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def seal(payload):
    result = dict(payload)
    result["artifact_sha256"] = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()
    return result


def select_certificate(candidate_id, certificates):
    options = [row for row in certificates if row["candidate_id"] == candidate_id]
    if not options:
        return None
    return min(
        options,
        key=lambda row: (
            int(row["additional_slot_bytes"]),
            -int(row["correctness_boot_count"]),
            row["policy_id"],
        ),
    )


def fusion_summary(topology, estimates):
    aggregate = Counter()
    statuses = []
    missing = []
    for segment_id, _threads in topology["path"]:
        if not segment_id.startswith("vta:"):
            continue
        estimate = estimates.get(segment_id)
        if estimate is None:
            missing.append(segment_id)
            continue
        statuses.append(estimate["status"])
        aggregate.update(estimate["structured_vector"])
    exact = (
        not missing
        and bool(statuses)
        and set(statuses) == {"exact_reuse"}
        and all(estimates[segment_id]["usable_for_ranking"] for segment_id, _ in topology["path"] if segment_id.startswith("vta:"))
    )
    return {
        "cpu_read_bytes": int(aggregate["cpu_read_bytes"]),
        "cpu_write_bytes": int(aggregate["cpu_write_bytes"]),
        "cpu_helper_calls": int(aggregate["cpu_helper_calls"]),
        "convert_calls": int(aggregate["convert_calls"]),
        "statuses": sorted(set(statuses)),
        "missing_segments": sorted(missing),
        "exact_reuse": exact,
    }


def feature_row(topology, certificates, estimates):
    certificate = select_certificate(topology["candidate_id"], certificates)
    realization = shared_realization(topology, certificate)
    row = policy_feature_row(topology, realization)
    fusion = fusion_summary(topology, estimates)
    helper = topology["vta_internal_cpu_helper_logical_access"]
    if fusion["cpu_read_bytes"] != int(helper.get("read_bytes", 0)):
        raise ValueError("fusion/domain CPU read-byte mismatch for {}".format(topology["candidate_id"]))
    if fusion["cpu_write_bytes"] != int(helper.get("write_bytes", 0)):
        raise ValueError("fusion/domain CPU write-byte mismatch for {}".format(topology["candidate_id"]))
    row.update(
        {
            "cpu_helper_calls": fusion["cpu_helper_calls"],
            "convert_calls": fusion["convert_calls"],
            "fusion_statuses": fusion["statuses"],
            "missing_fusion_segments": fusion["missing_segments"],
            "unified_feature_exact": bool(
                fusion["exact_reuse"] and row["detailed_dma_complete"]
            ),
        }
    )
    return row


def configuration_id(row):
    return "{}::{}".format(row["candidate_id"], row["realization"]["realized_policy"])


def source(path):
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def analyze(
    topologies_path,
    edge_contexts_path,
    certificates_path,
    legacy_contract_path,
    fusion_estimates_path,
    fusion_contract_path,
    band_fraction,
    budget,
):
    raw_topologies = read_json(topologies_path)
    edge_contexts = read_json(edge_contexts_path)
    certificate_payload = read_json(certificates_path)
    certificates = certificate_payload["certificates"]
    legacy_contract = read_json(legacy_contract_path)
    estimates_payload = read_json(fusion_estimates_path)
    fusion_contract = read_json(fusion_contract_path)
    estimates = {row["segment_id"]: row for row in estimates_payload["rows"]}

    topologies = topology_diverse(raw_topologies)
    for topology in topologies:
        validate_topology(topology, edge_contexts)
    by_candidate = {row["candidate_id"]: row for row in topologies}
    for certificate in certificates:
        topology = by_candidate.get(certificate["candidate_id"])
        if topology is None:
            raise ValueError("certificate references an unknown candidate")
        if certificate["policy_id"] not in (
            POLICY_EXTERNAL_K2,
            POLICY_POOL_ANCHOR_K2,
        ):
            raise ValueError("unexpected policy in certificate set")
        validate_certificate(certificate, topology)

    rows = [feature_row(row, certificates, estimates) for row in topologies]
    exact = [row for row in rows if row["unified_feature_exact"]]
    inexact = [row for row in rows if not row["unified_feature_exact"]]
    if inexact:
        raise ValueError("{} topology rows lack exact unified features".format(len(inexact)))
    ordered = order_within_m1_band(exact, band_fraction, UNIFIED_DIMENSIONS)
    selected = ordered[:budget]
    selected_configs = [configuration_id(row) for row in selected]
    legacy_configs = legacy_contract["strategies"]["S2_m1_certified_shared"]
    prior_dimensions = fusion_contract["pareto_dimensions"]

    return seal(
        {
            "schema": "vta_c1_unified_structured_order_diagnostic_v1",
            "status": "post_two_boot_diagnostic_freeze",
            "label_free_inputs_only": True,
            "board_artifacts_read": False,
            "eligible_for_ongoing_s0_s1_s2_claim": False,
            "reason": (
                "The ongoing S0/S1/S2 selection contract is immutable and already has "
                "two boot labels. This artifact checks implementation completeness only."
            ),
            "domain": {
                "raw_candidates": len(raw_topologies),
                "topology_diverse_candidates": len(topologies),
                "exact_unified_features": len(exact),
                "inexact_unified_features": len(inexact),
            },
            "method": {
                "m1_winner_always_first": True,
                "m1_band_fraction": band_fraction,
                "pareto_dimensions": list(UNIFIED_DIMENSIONS),
                "average_request_size_direction": (
                    "reported directly; inverse request density per MiB is minimized"
                ),
                "unlike_units_summed": False,
                "physical_ddr_inferred": False,
                "unknown_feature_action": "fail closed; retain M1 order",
                "preexisting_fusion_dimensions": prior_dimensions,
            },
            "selected_candidate_ids": [row["candidate_id"] for row in selected],
            "selected_configuration_ids": selected_configs,
            "selected_rows": selected,
            "comparison_to_frozen_s2": {
                "legacy_configuration_ids": legacy_configs,
                "same_order": selected_configs == legacy_configs,
                "same_set": set(selected_configs) == set(legacy_configs),
                "existing_boots_reclassified_as_prospective_s3_evidence": False,
            },
            "full_order": ordered,
            "sources": {
                "topologies": source(topologies_path),
                "edge_contexts": source(edge_contexts_path),
                "policy_certificates": source(certificates_path),
                "legacy_selection_contract": source(legacy_contract_path),
                "fusion_estimates": source(fusion_estimates_path),
                "fusion_selection_contract": source(fusion_contract_path),
                "script": source(Path(__file__).resolve()),
                "library": source(Path(__file__).with_name("shared_memory_topk.py")),
            },
        }
    )


def write_readme(path, result):
    comparison = result["comparison_to_frozen_s2"]
    lines = [
        "# C1 unified structured-order diagnostic",
        "",
        "This is a post-two-boot implementation-completeness audit. It reads no board "
        "artifact and cannot replace or extend the frozen S0/S1/S2 claim.",
        "",
        "- Topology-diverse rows with exact features: {}/{}.".format(
            result["domain"]["exact_unified_features"],
            result["domain"]["topology_diverse_candidates"],
        ),
        "- Dimensions: `{}`.".format(
            json.dumps(result["method"]["pareto_dimensions"], separators=(",", ":"))
        ),
        "- Unified Top-5: `{}`.".format(
            json.dumps(result["selected_configuration_ids"], separators=(",", ":"))
        ),
        "- Same order/set as frozen S2: `{}/{}`.".format(
            str(comparison["same_order"]).lower(), str(comparison["same_set"]).lower()
        ),
        "- Existing boots reclassified as prospective S3 evidence: `false`.",
        "- Artifact SHA-256: `{}`.".format(result["artifact_sha256"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies", type=Path, default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--edge-contexts", type=Path, default=DEFAULT_EDGE_CONTEXTS)
    parser.add_argument("--certificates", type=Path, default=DEFAULT_CERTIFICATES)
    parser.add_argument("--legacy-contract", type=Path, default=DEFAULT_LEGACY_CONTRACT)
    parser.add_argument("--fusion-estimates", type=Path, default=DEFAULT_FUSION_ESTIMATES)
    parser.add_argument("--fusion-contract", type=Path, default=DEFAULT_FUSION_CONTRACT)
    parser.add_argument("--band-fraction", type=float, default=0.02)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    if args.budget <= 0:
        raise ValueError("budget must be positive")
    result = analyze(
        args.topologies,
        args.edge_contexts,
        args.certificates,
        args.legacy_contract,
        args.fusion_estimates,
        args.fusion_contract,
        args.band_fraction,
        args.budget,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_readme(args.output_dir / "README.md", result)
    print(json.dumps({
        "artifact_sha256": result["artifact_sha256"],
        "selected_configuration_ids": result["selected_configuration_ids"],
        "comparison_to_frozen_s2": result["comparison_to_frozen_s2"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
