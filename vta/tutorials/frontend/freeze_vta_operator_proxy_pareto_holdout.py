#!/usr/bin/env python3
"""Freeze the cheap operator-TIR Pareto policy before target qualification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(args):
    target_dir = Path(args.target_contract)
    development_dir = Path(args.proxy_development)
    verified = {
        "target_contract": len(verify_artifacts_compatible(target_dir)),
        "proxy_development": len(verify_artifacts_compatible(development_dir)),
    }
    target = read_json(target_dir / "contract.json")
    development = read_json(development_dir / "analysis.json")
    if target.get("status") != "frozen_before_static_fsim_fpga_or_performance_observation":
        raise RuntimeError("target candidate domain is not pristine")
    if target.get("exposed_performance_label_collision") is not False:
        raise RuntimeError("target geometry collides with exposed performance labels")
    expected = "posthoc_operator_proxy_retains_r50d_oracle_rule_ready_for_new_holdout"
    if development.get("status") != expected:
        raise RuntimeError("operator-proxy development rule is incomplete")
    if development.get("proxy_retains_oracle") is not True:
        raise RuntimeError("development proxy did not retain its exposed oracle")

    result = {
        "schema": "c3_vta_operator_proxy_pareto_holdout_contract_v1",
        "status": "frozen_before_target_lower_fsim_fullgraph_fpga_or_latency",
        "workload_id": target["workload_id"],
        "source_model": target["source_model"],
        "source_layer": target["source_layer"],
        "geometry_signature": target["geometry_signature"],
        "candidate_count_legacy_four_modes": len(
            (target_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        ),
        "candidate_commitment_sha256": target["selection"]["candidate_commitment_sha256"],
        "policy": {
            "eligibility": "real operator lowering then three-seed FSim; retain all failures",
            "proxy_source": "final lowered operator TIR before any full-graph build",
            "axes": [
                "operator_lowered_tir_expanded_load_plus_store_bytes",
                "operator_lowered_tir_expanded_load_plus_store_calls",
                "weight_barrier_indicator",
            ],
            "dominance": (
                "A dominates B iff A is no worse on every axis and strictly better on at "
                "least one axis"
            ),
            "prospective_action": (
                "form the non-dominated proxy front without target performance labels; build "
                "a stock ResNet50 reference and final fused graphs only for that front; run "
                "three-seed FPGA correctness and balanced latency only for the front"
            ),
            "oracle_completion": (
                "after the prospective front outcomes are immutable, build and measure every "
                "remaining eligible candidate to reveal the complete FPGA-correct oracle"
            ),
            "tie_break": "candidate_id only controls deterministic execution order",
            "scalarization": None,
        },
        "cost_accounting": {
            "operator_qualification": "charge lowering and FSim wall time to every policy",
            "prospective_build": "measure actual stock-plus-front full-graph build wall time",
            "oracle_completion_build": "report separately after front labels are exposed",
            "board": "charge candidate dispatches, RPC/API calls, driver invocations and host wall",
        },
        "label_visibility": {
            "target_lowering": False,
            "target_fsim": False,
            "target_full_graph_features": False,
            "target_fpga_correctness": False,
            "target_operator_latency": False,
            "target_full_graph_latency": False,
            "tophub_used": False,
        },
        "bound_inputs": {
            "target_contract_sha256": sha256(target_dir / "contract.json"),
            "target_candidates_sha256": sha256(target_dir / "candidates.jsonl"),
            "target_artifact_manifest_sha256": sha256(target_dir / "artifact_hashes.json"),
            "proxy_development_analysis_sha256": sha256(development_dir / "analysis.json"),
            "proxy_development_artifact_manifest_sha256": sha256(
                development_dir / "artifact_hashes.json"
            ),
        },
        "verified_artifact_counts": verified,
        "freezer_source_sha256": sha256(Path(__file__).resolve()),
        "board_contacted": False,
        "claim_boundary": (
            "Prospective identity and policy freeze only. R50D selected this rule post hoc; "
            "R50E is the first independent test of full-graph build-cost reduction and oracle "
            "retention."
        ),
    }
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    write_json(output / "contract.json", result)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-contract", required=True)
    parser.add_argument("--proxy-development", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
