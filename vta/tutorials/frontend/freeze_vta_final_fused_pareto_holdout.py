#!/usr/bin/env python3
"""Freeze a final-fused Pareto holdout before any target qualification or latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(args):
    target_dir = Path(args.target_contract)
    development_dir = Path(args.pareto_development)
    verified = {
        "target_contract": len(verify_artifacts_compatible(target_dir)),
        "pareto_development": len(verify_artifacts_compatible(development_dir)),
    }
    target = read_json(target_dir / "contract.json")
    development = read_json(development_dir / "analysis.json")
    if target.get("status") != "frozen_before_static_fsim_fpga_or_performance_observation":
        raise RuntimeError("target candidate domain is not pristine")
    if target.get("exposed_performance_label_collision") is not False:
        raise RuntimeError("target geometry collides with exposed performance labels")
    if development.get("status") != "posthoc_pareto_front_retains_four_oracles_and_exposes_conflict_cost":
        raise RuntimeError("Pareto development rule is incomplete")

    result = {
        "schema": "c3_vta_final_fused_pareto_holdout_contract_v1",
        "status": "frozen_before_target_lower_fsim_fpga_or_latency",
        "workload_id": target["workload_id"],
        "source_model": target["source_model"],
        "source_layer": target["source_layer"],
        "geometry_signature": target["geometry_signature"],
        "candidate_count_legacy_four_modes": len(
            (target_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        ),
        "candidate_commitment_sha256": target["selection"]["candidate_commitment_sha256"],
        "policy": {
            "eligibility": "real lowering then three-seed FSim; all failures retained",
            "final_program_feature": (
                "cross-build final pretrained ResNet50 graph and aggregate exact fused-TIR "
                "logical LOAD+STORE over every matching Graph JSON occurrence"
            ),
            "axes": ["load_plus_store_bytes", "load_plus_store_calls", "extra_submissions"],
            "dominance": (
                "A dominates B iff A is no worse on every axis and strictly better on at "
                "least one axis"
            ),
            "action": (
                "FPGA correctness and latency are required for every non-dominated program; "
                "dominated programs are measured only in the later oracle-completion phase"
            ),
            "tie_break": "candidate_id only for execution order, never for elimination",
            "scalar_request_penalty": None,
            "stop": (
                "after the prospective Pareto wave, complete every remaining eligible program "
                "to construct an unbiased FPGA-correct pool oracle"
            ),
        },
        "cost_accounting": {
            "common_qualification": "report all lowering and FSim wall time separately",
            "final_fused_feature_cost": "charge every full-graph build and fused-TIR audit",
            "prospective_wave": "charge cross-compile, FPGA correctness, timing and invocations",
            "oracle_completion": "report separately after prospective labels are exposed",
        },
        "label_visibility": {
            "target_lowering": False,
            "target_fsim": False,
            "target_fpga_correctness": False,
            "target_operator_latency": False,
            "target_full_graph_latency": False,
            "tophub_used": False,
        },
        "bound_inputs": {
            "target_contract_sha256": sha256(target_dir / "contract.json"),
            "target_candidates_sha256": sha256(target_dir / "candidates.jsonl"),
            "target_artifact_manifest_sha256": sha256(target_dir / "artifact_hashes.json"),
            "pareto_development_analysis_sha256": sha256(development_dir / "analysis.json"),
            "pareto_development_artifact_manifest_sha256": sha256(
                development_dir / "artifact_hashes.json"
            ),
        },
        "verified_artifact_counts": verified,
        "freezer_source_sha256": sha256(Path(__file__).resolve()),
        "board_contacted": False,
        "claim_boundary": (
            "Prospective policy and identity freeze only. Full local qualification is common "
            "cost, and Pareto dominance is a search rule to test rather than an assumed "
            "latency theorem."
        ),
    }
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    write_json(output / "contract.json", result)
    (output / "command.txt").write_text(
        " ".join(__import__("sys").argv) + "\n", encoding="utf-8"
    )
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-contract", required=True)
    parser.add_argument("--pareto-development", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
