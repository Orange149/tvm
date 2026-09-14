#!/usr/bin/env python3
"""Freeze correctness-driven Pareto peeling before target qualification or labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-contract", type=Path, required=True)
    parser.add_argument("--development-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    target = read(args.target_contract / "contract.json")
    development = read(args.development_audit / "summary.json")
    verified = {
        "target": len(verify_artifacts_compatible(args.target_contract)),
        "development": len(verify_artifacts_compatible(args.development_audit)),
    }
    if target.get("status") != "frozen_before_static_fsim_fpga_or_performance_observation":
        raise ValueError("target identities are not pristine")
    if target.get("exposed_performance_label_collision") is not False:
        raise ValueError("target geometry collides with exposed labels")
    if development.get("status") != "correctness_stable_quality_band_stable_exact_oracle_not_stable":
        raise ValueError("development audit does not establish the invalid-dominator gap")
    if not development.get("peeling_would_admit_current_oracle"):
        raise ValueError("development audit does not support peeling")

    policy = {
        "schema": "c3_invalid_dominator_peeling_holdout_v1",
        "status": "frozen_before_target_lowering_fsim_fullgraph_fpga_or_latency",
        "workload_id": target["workload_id"],
        "source_model": target["source_model"],
        "source_layer": target["source_layer"],
        "geometry_signature": target["geometry_signature"],
        "candidate_commitment_sha256": target["selection"]["candidate_commitment_sha256"],
        "eligibility": "real operator lowering and three-seed FSim; retain every failure",
        "feature_visibility": "final lowered operator TIR only; no target latency or FPGA label",
        "axes": [
            "operator_expanded_LOAD_plus_STORE_bytes",
            "operator_expanded_LOAD_plus_STORE_calls",
            "weight_resident_barrier_indicator",
        ],
        "dominance": "no worse on all three axes and strictly better on at least one",
        "peeling_algorithm": {
            "wave_0": "compute the non-dominated eligible operator-TIR front",
            "execution": "build one final graph at a time; three-seed FPGA correctness fail-fast; measure latency only after correctness passes",
            "update": "remove FPGA-invalid candidates, recompute the front over all non-invalid eligible identities, and dispatch only newly exposed identities",
            "stop": "stop when a completed wave has no FPGA-invalid candidate, no new identity is exposed, or the 600-second complete-outer-process budget is exhausted",
            "latency_is_not_used_for_peeling": True,
            "already_correct_front_points_remain_as_dominators": True,
            "candidate_replacement_outside_recomputed_front": False,
        },
        "controls": {
            "fixed_front": "wave_0 only",
            "bytes_lazy": "ascending bytes then candidate_id",
            "calls_lazy": "ascending calls then candidate_id",
            "random_lazy": "SHA256(c3-invalid-peeling-random-v1|workload_id|candidate_id)",
            "budget": "same 600-second complete outer process wall; a running action drains and overrun is charged",
        },
        "cost_accounting": {
            "included": [
                "candidate preparation", "operator lowering", "FSim", "selection bookkeeping",
                "one-at-a-time full-graph build", "clean board start", "upload/allocation",
                "FPGA correctness", "FPGA timing", "hashing and artifact finalization",
            ],
            "common_qualification_reported_separately": True,
            "front_filter_and_internal_order_reported_separately": True,
        },
        "oracle_completion": "after the online result is immutable, build and measure all remaining eligible candidates to reveal the complete FPGA-correct pool oracle",
        "label_visibility": {
            "target_lowering": False, "target_fsim": False,
            "target_final_graph": False, "target_fpga_correctness": False,
            "target_operator_latency": False, "target_full_graph_latency": False,
            "tophub": False,
        },
        "bindings": {
            "target_contract_sha256": sha256(args.target_contract / "contract.json"),
            "target_candidates_sha256": sha256(args.target_contract / "candidates.jsonl"),
            "target_artifact_manifest_sha256": sha256(args.target_contract / "artifact_hashes.json"),
            "development_summary_sha256": sha256(args.development_audit / "summary.json"),
            "development_artifact_manifest_sha256": sha256(args.development_audit / "artifact_hashes.json"),
        },
        "verified_artifact_counts": verified,
        "source_sha256": sha256(Path(__file__).resolve()),
        "claim_boundary": "Prospective policy specification only. Peeling was developed after R50D labels and has no R50F correctness, latency, oracle-retention, or cost result yet.",
    }
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "policy.json", policy)
    (args.output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    write_json(args.output_dir / "artifact_hashes.json", {"artifacts": {
        path.name: sha256(path) for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "status": policy["status"], "workload_id": policy["workload_id"],
        "geometry_signature": policy["geometry_signature"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
