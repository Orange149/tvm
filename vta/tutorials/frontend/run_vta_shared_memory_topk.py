#!/usr/bin/env python3
"""Freeze a label-free, shared-memory-realization-aware topology Top-K study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from shared_memory_topk import (
    POLICY_EXTERNAL_K2,
    POLICY_POOL_ANCHOR_K2,
    copy_realization,
    make_anchor_certificate,
    make_exact_certificate,
    order_within_m1_band,
    policy_feature_row,
    promote_cross_boot_certificate,
    read_json,
    sha256_file,
    shared_realization,
    topology_diverse,
    topology_id_from_candidate,
    validate_topology,
)


BASE = Path(__file__).resolve().parent / "report_out"
DEFAULT_DOMAIN = BASE / "stage_tile_cotuning/shared_edge_audit/domain_run2"
DEFAULT_P8 = BASE / "resource_aware_maxplus/v1_p8_top20_zero_copy/v1_p8_top20_summary.json"
DEFAULT_CROSS_BOOT = BASE / "resource_aware_maxplus/v1_p8c_cross_boot_summary.json"
DEFAULT_ANCHOR = BASE / "stage_tile_cotuning/c3s_buffer_reuse/s1_board_run1/summary.json"
DEFAULT_CONTENTION = BASE / "resource_aware_maxplus/v1_p7c_qualification_summary.json"


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_multiboot_contention(payload, single_boot_summary):
    if payload.get("kind") != "cpu_vta_pipeline_v1_p7c_three_boot_reproducibility":
        raise ValueError("unexpected P7C multiboot artifact kind")
    expected = payload.get("artifact_sha256")
    unsealed = dict(payload)
    unsealed.pop("artifact_sha256", None)
    observed = hashlib.sha256(
        json.dumps(
            unsealed, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    if expected != observed:
        raise ValueError("P7C multiboot artifact_sha256 mismatch")
    source_hashes = {
        item["artifact_sha256"] for item in payload.get("source_sessions", [])
    }
    if single_boot_summary["source_session_artifact_sha256"] not in source_hashes:
        raise ValueError("P7C multiboot artifact does not include the frozen single boot")
    if len(set(payload.get("board_boot_ids", []))) != 3:
        raise ValueError("P7C multiboot artifact lacks three unique boots")
    return {
        "independent_boot_count": 3,
        "board_boot_ids": payload["board_boot_ids"],
        "ddr_specific_contrast": payload["ddr_specific_contrast"],
        "formal_shared_ddr_parameter_admitted": bool(
            payload["exact_protocol_contention_profile_ready"]
        ),
        "generalized_candidate_contention_surface_ready": bool(
            payload["generalized_candidate_contention_surface_ready"]
        ),
        "model_action": (
            "record exact-protocol coefficients only; do not alter candidate ranking "
            "without a candidate classifier and grouped holdout"
        ),
        "remaining_gate": (
            "measure request/stride and boundary-policy groups, then validate a "
            "candidate-level contention surface with grouped holdout"
        ),
        "source_artifact_sha256": payload["artifact_sha256"],
    }


def certificates_from_evidence(topologies, p8_summary, cross_boot, anchor_summary):
    by_candidate = {row["candidate_id"]: row for row in topologies}
    certificates = {}
    for source_row in p8_summary["rows"]:
        candidate_id = source_row["candidate_id"]
        topology = by_candidate.get(candidate_id)
        if topology is None:
            continue
        certificate = make_exact_certificate(
            topology,
            source_row,
            "P8 Top-20 single-boot exact candidate functional qualification",
            evidence_identity={
                "summary_artifact_sha256": p8_summary["artifact_sha256"],
                "protocol_sha256": p8_summary["protocol_sha256"],
                "board_properties": p8_summary["board_properties"],
                "raw_artifacts": source_row["raw_artifacts"],
            },
        )
        certificates[(candidate_id, POLICY_EXTERNAL_K2)] = certificate
    promoted_id = cross_boot["candidate_id"]
    promoted_key = (promoted_id, POLICY_EXTERNAL_K2)
    if promoted_key not in certificates:
        raise RuntimeError("cross-boot candidate lacks an exact single-boot certificate")
    certificates[promoted_key] = promote_cross_boot_certificate(
        certificates[promoted_key], cross_boot
    )
    rank_by_label = {"A": 1, "B": 5, "C": 7, "D": 13}
    anchor_results = {row["topology"]: row for row in anchor_summary["results"]}
    for label, rank in rank_by_label.items():
        candidate_id = p8_summary["rows"][rank - 1]["candidate_id"]
        topology = by_candidate.get(candidate_id)
        external_key = (candidate_id, POLICY_EXTERNAL_K2)
        if topology is None or external_key not in certificates:
            raise RuntimeError("anchor representative {} is absent from the topology domain".format(label))
        certificates[(candidate_id, POLICY_POOL_ANCHOR_K2)] = make_anchor_certificate(
            topology,
            certificates[external_key],
            anchor_results[label],
            "S1 single-boot exact candidate input-pool anchor qualification",
            evidence_identity={
                "board_boot_id": anchor_summary["boot"],
                "runner_sha256": anchor_summary["build"]["artifacts"][
                    "vta_stage_pipeline_runner"
                ],
                "driver_sha256": anchor_summary["build"]["artifacts"]["libvta.so"],
                "planner_sha256": anchor_summary["planner_sha256"],
            },
        )
    return certificates


def select_shared_certificate(candidate_id, certificates):
    options = [
        certificate for (owner, _policy), certificate in certificates.items()
        if owner == candidate_id
    ]
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


def configuration_id(row):
    return "{}::{}".format(row["candidate_id"], row["realization"]["realized_policy"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies", type=Path, default=DEFAULT_DOMAIN / "topologies.json")
    parser.add_argument("--edge-contexts", type=Path, default=DEFAULT_DOMAIN / "edge_contexts.json")
    parser.add_argument("--p8-summary", type=Path, default=DEFAULT_P8)
    parser.add_argument("--cross-boot", type=Path, default=DEFAULT_CROSS_BOOT)
    parser.add_argument("--anchor-summary", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--contention-summary", type=Path, default=DEFAULT_CONTENTION)
    parser.add_argument("--contention-multiboot-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--band-fraction", type=float, default=0.02)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--shared-capacity-bytes", type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    if args.budget <= 0:
        raise ValueError("budget must be positive")

    raw_topologies = read_json(args.topologies)
    edge_contexts = read_json(args.edge_contexts)
    topologies = topology_diverse(raw_topologies)
    for row in topologies:
        validate_topology(row, edge_contexts)
    p8_summary = read_json(args.p8_summary)
    cross_boot = read_json(args.cross_boot)
    anchor_summary = read_json(args.anchor_summary)
    contention_summary = read_json(args.contention_summary)
    contention_multiboot = (
        read_json(args.contention_multiboot_summary)
        if args.contention_multiboot_summary is not None
        else None
    )
    certificates = certificates_from_evidence(
        topologies, p8_summary, cross_boot, anchor_summary
    )

    copy_rows = [policy_feature_row(row, copy_realization(row)) for row in topologies]
    shared_rows = [
        policy_feature_row(
            row,
            shared_realization(
                row,
                select_shared_certificate(row["candidate_id"], certificates),
                capacity_bytes=args.shared_capacity_bytes,
            ),
        )
        for row in topologies
    ]
    s0 = sorted(copy_rows, key=lambda row: (row["m1_score_ms"], row["topology_id"]))
    s1 = order_within_m1_band(
        copy_rows,
        args.band_fraction,
        ("framework_materialization_bytes", "vta_dma_calls", "vta_islands"),
    )
    s2 = order_within_m1_band(
        shared_rows,
        args.band_fraction,
        (
            "framework_materialization_bytes",
            "additional_slot_bytes",
            "vta_dma_calls",
            "vta_islands",
        ),
    )
    strategies = {
        "S0_m1_copy": [row["candidate_id"] for row in s0[: args.budget]],
        "S1_m1_payload": [row["candidate_id"] for row in s1[: args.budget]],
        "S2_m1_certified_shared": [row["candidate_id"] for row in s2[: args.budget]],
    }
    strategy_rows = {
        "S0_m1_copy": s0[: args.budget],
        "S1_m1_payload": s1[: args.budget],
        "S2_m1_certified_shared": s2[: args.budget],
    }
    strategy_configurations = {
        name: [configuration_id(row) for row in rows]
        for name, rows in strategy_rows.items()
    }
    orders_differ = len({tuple(value) for value in strategies.values()}) > 1
    union = []
    for name in strategies:
        for candidate_id in strategies[name]:
            if candidate_id not in union:
                union.append(candidate_id)
    union_configurations = []
    for name in strategy_configurations:
        for config_id in strategy_configurations[name]:
            if config_id not in union_configurations:
                union_configurations.append(config_id)

    legacy_unique = {
        topology_id_from_candidate(row["candidate_id"])
        for row in p8_summary["rows"]
    }
    certified_rows = [row for row in shared_rows if not row["realization"]["fallback_to_copy"]]
    cross_boot_certificates = [
        row for row in certificates.values()
        if row["binding_status"] == "exact_candidate_three_boot_functional"
    ]
    capacity_sweep = []
    for capacity in (1 << 20, 2 << 20, 3 << 20, 4 << 20, 8 << 20, 16 << 20):
        feasible = sum(
            int(row["theoretical_k2_slot_bytes"]) <= capacity for row in topologies
        )
        capacity_sweep.append({
            "capacity_bytes": capacity,
            "topologies_fitting_external_k2_necessary_capacity": feasible,
            "fraction": feasible / len(topologies),
            "interpretation": "capacity-only upper bound; not a zero-copy qualification",
        })

    sources = {}
    for name, path in (
        ("topologies", args.topologies),
        ("edge_contexts", args.edge_contexts),
        ("p8_summary", args.p8_summary),
        ("cross_boot", args.cross_boot),
        ("anchor_summary", args.anchor_summary),
        ("contention_summary", args.contention_summary),
        ("script", Path(__file__)),
        ("library", Path(__file__).with_name("shared_memory_topk.py")),
    ):
        sources[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    if args.contention_multiboot_summary is not None:
        sources["contention_multiboot_summary"] = {
            "path": str(args.contention_multiboot_summary.resolve()),
            "sha256": sha256_file(args.contention_multiboot_summary),
        }

    if contention_multiboot is None:
        contention_qualification = {
            "independent_boot_count": 1,
            "board_boot_id": contention_summary["board_boot_id"],
            "observations": contention_summary["observations"],
            "streaming_excess_over_cache_control": contention_summary[
                "streaming_excess_over_cache_control"
            ],
            "formal_shared_ddr_parameter_admitted": bool(
                contention_summary["interpretation"][
                    "formal_shared_ddr_parameter_admitted"
                ]
            ),
            "model_action": (
                "retain VTA bytes/calls as separate ranking features; do not fit a "
                "contention coefficient"
            ),
            "remaining_gate": contention_summary["interpretation"]["remaining_gate"],
        }
    else:
        contention_qualification = validate_multiboot_contention(
            contention_multiboot, contention_summary
        )

    analysis = {
        "schema": "vta_shared_memory_topk_analysis_v1",
        "label_free": True,
        "board_contacted": False,
        "domain": {
            "raw_rows": len(raw_topologies),
            "topology_diverse_rows": len(topologies),
            "edge_contexts": len(edge_contexts),
            "legacy_p8_candidate_rows": len(p8_summary["rows"]),
            "legacy_p8_unique_topologies": len(legacy_unique),
        },
        "certification": {
            "exact_candidate_shared_certificates": len(certificates),
            "exact_candidate_topologies": len({candidate_id for candidate_id, _ in certificates}),
            "three_boot_functional_certificates": len(cross_boot_certificates),
            "performance_qualified_certificates": sum(
                bool(row.get("performance_claim_allowed")) for row in certificates.values()
            ),
            "rule": "unknown candidate/edge/policy or capacity overflow falls back to ordinary copy",
        },
        "contention_qualification": contention_qualification,
        "model_semantics": {
            "m1_is_primary": True,
            "m1_band_fraction": args.band_fraction,
            "shared_capacity_bytes": args.shared_capacity_bytes,
            "logical_boundary_invariant": True,
            "framework_materialization_policy_dependent": True,
            "vta_internal_dma_policy_invariant": True,
            "physical_ddr_inferred_from_logical_bytes": False,
            "latency_delta_used_when_ci_crosses_zero": False,
            "selection_is_measurement_order_not_safe_pruning": True,
        },
        "strategies": strategies,
        "strategy_configurations": strategy_configurations,
        "orders_differ": orders_differ,
        "union_candidate_ids": union,
        "union_configuration_ids": union_configurations,
        "capacity_sweep": capacity_sweep,
        "top_orders": {
            "S0_m1_copy": s0[:20],
            "S1_m1_payload": s1[:20],
            "S2_m1_certified_shared": s2[:20],
        },
        "sources": sources,
    }
    contract = {
        "schema": "vta_shared_memory_topk_selection_contract_v1",
        "status": "pre_board_frozen" if orders_differ else "pre_board_frozen_inactive_ablation",
        "label_free": True,
        "board_contacted": False,
        "budget_per_strategy": args.budget,
        "strategies": strategy_configurations,
        "strategy_candidate_ids": strategies,
        "union_candidate_ids": union,
        "union_configuration_ids": union_configurations,
        "configuration_identity": "candidate_id plus realized boundary policy",
        "orders_differ": orders_differ,
        "board_action": (
            "materialize exact shared policies and measure the frozen union with three independent boots"
            if orders_differ
            else "do not run an order-comparison experiment because all frozen orders are identical"
        ),
        "correctness_gate": "three deterministic full-output inputs; any mismatch rejects timing",
        "statistics": "cold boot is the experimental unit; interleave strategies within each boot",
        "primary_metric": "first index reaching union-oracle plus 2 percent",
        "secondary_metrics": [
            "Regret@K against union oracle",
            "framework materialization bytes per frame",
            "additional slot bytes and pool high-water",
            "boundary API service and slot wait",
            "VTA internal LOAD/STORE bytes and calls",
            "wall time to first qualifying candidate",
        ],
        "claim_gate": "S2 must hit oracle+2% no later than S0/S1 and have Regret@K no worse than the best baseline",
    }

    args.output_dir.mkdir(parents=True)
    policy_path = args.output_dir / "policy_certificates.json"
    write_json(policy_path, {
        "schema": "vta_shared_edge_policy_certificate_set_v1",
        "certificates": list(certificates.values()),
        "sources": sources,
    })
    analysis["policy_certificate_set_sha256"] = sha256_file(policy_path)
    analysis_path = args.output_dir / "analysis.json"
    write_json(analysis_path, analysis)
    contract["frozen_sources"] = {
        "policy_certificates": {
            "path": str(policy_path.resolve()),
            "sha256": sha256_file(policy_path),
        },
        "analysis": {
            "path": str(analysis_path.resolve()),
            "sha256": sha256_file(analysis_path),
        },
    }
    write_json(args.output_dir / "selection_contract.json", contract)
    (args.output_dir / "STATUS.md").write_text(
        "# C1 shared-memory-aware Top-K freeze\n\n"
        "- Domain: {} topology-diverse candidates and {} edge contexts.\n"
        "- Legacy P8 Top-20 contains only {} unique topologies.\n"
        "- Exact shared-policy certificates: {} across {} topologies; three-boot: {}; performance-qualified: 0.\n"
        "- Shared-DDR contention qualification: one boot; formal coefficient not admitted.\n"
        "- S0/S1/S2 frozen orders differ: {}.\n"
        "- Frozen union: {} topologies / {} candidate-policy configurations. Board contacted: no.\n"
        "- Unknown policies fail closed to copy; no physical DDR bytes or throughput benefit are inferred.\n"
        "- Board action: {}.\n".format(
            len(topologies),
            len(edge_contexts),
            len(legacy_unique),
            len(certificates),
            len({candidate_id for candidate_id, _ in certificates}),
            len(cross_boot_certificates),
            str(orders_differ).lower(),
            len(union),
            len(union_configurations),
            contract["board_action"],
        ),
        encoding="utf-8",
    )
    write_json(args.output_dir / "artifact_hashes.json", {
        "artifacts": {
            path.name: sha256_file(path)
            for path in sorted(args.output_dir.iterdir())
            if path.is_file()
        }
    })
    print(json.dumps({
        "topologies": len(topologies),
        "certificates": len(certificates),
        "strategies": strategies,
        "union_topologies": len(union),
        "union_configurations": len(union_configurations),
        "orders_differ": orders_differ,
    }, indent=2))


if __name__ == "__main__":
    main()
