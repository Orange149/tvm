#!/usr/bin/env python3
"""Audit the prospective fused-program reranker chain and preserve its outcome."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from run_vta_resnet50_fused_program_reranker_board import profile_delta
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def ratio_is_exact_double(actual, expected):
    return all(float(actual[key]) == 2.0 * float(expected[key]) for key in expected)


def run(args):
    contract_dir = Path(args.contract)
    board_dir = Path(args.board_result)
    failed_dir = Path(args.failed_attempt)
    verified = {
        "contract": len(verify_artifacts_compatible(contract_dir)),
        "board_result": len(verify_artifacts_compatible(board_dir)),
    }
    contract = read_json(contract_dir / "contract.json")
    board_contract = read_json(board_dir / "contract.json")
    summary = read_json(board_dir / "summary.json")
    if contract.get("status") != "fused_service_choice_frozen_before_fpga_latency":
        raise RuntimeError("prospective choice contract is incomplete")
    if any(contract["target_label_exposure"][key] for key in (
        "full_graph_latency_observed", "partial_mask_latency_observed",
        "operator_latency_used_for_selection",
    )):
        raise RuntimeError("target latency leaked into prospective selection")
    if board_contract["reranker_contract_sha256"] != sha256(contract_dir / "contract.json"):
        raise RuntimeError("board result does not bind the prospective contract")
    if summary.get("status") != "prospective_fused_service_reranker_board_evaluation_complete":
        raise RuntimeError("board evaluation is incomplete")
    if not (summary["all_outputs_equal"] and summary["all_outputs_nonzero"]
            and summary["fused_tir_prediction_exact"]):
        raise RuntimeError("board correctness/profile contract failed")
    if summary["expected_profile_delta_b_minus_a"] != contract["expected_profile_delta_b_minus_a"]:
        raise RuntimeError("board expected profile differs from frozen contract")

    service_selected = min(
        contract["programs"],
        key=lambda row: (row["service_score_byte_equivalent"], row["program_id"]),
    )["program_id"]
    bytes_selected = min(
        contract["programs"], key=lambda row: (row["dma_bytes"], row["program_id"])
    )["program_id"]
    if service_selected != contract["selected_program"]:
        raise RuntimeError("frozen service selection is internally inconsistent")
    if service_selected == bytes_selected:
        raise RuntimeError("this audit expects the byte/request conflict pair")
    if summary["selected_program"] != service_selected:
        raise RuntimeError("board evaluated a different frozen selection")
    oracle = summary["oracle_program"]

    # Preserve the first failed execution as an implementation diagnosis. It
    # reached all timing calls but compared an unnormalized two-execution timed
    # profile against a one-execution contract.
    failed_files = {
        name: sha256(failed_dir / name)
        for name in (
            "contract.json", "correctness.json", "parameter_equivalence.json", "timing.jsonl"
        )
    }
    failed_contract = read_json(failed_dir / "contract.json")
    failed_correctness = read_json(failed_dir / "correctness.json")
    failed_timing = read_jsonl(failed_dir / "timing.jsonl")
    program_ids = failed_contract["program_ids"]
    failed_correctness_delta = profile_delta(failed_correctness, program_ids)
    failed_timed_raw_delta = profile_delta(failed_timing, program_ids)
    expected = failed_contract["expected_profile_delta_b_minus_a"]
    if failed_correctness_delta != expected or not ratio_is_exact_double(
        failed_timed_raw_delta, expected
    ):
        raise RuntimeError("failed-attempt diagnosis is not exact two-execution scaling")

    result = {
        "schema": "c3_vta_fused_program_reranker_holdout_audit_v1",
        "status": "prospective_fixed_proxy_failure_supports_conflict_escalation",
        "selection": {
            "frozen_service_proxy": service_selected,
            "bytes_only": bytes_selected,
            "fpga_oracle": oracle,
            "service_proxy_correct": service_selected == oracle,
            "bytes_only_correct": bytes_selected == oracle,
            "service_proxy_regret_percent": summary["selected_regret_percent"],
            "service_proxy_paired_wins": summary["selected_paired_wins"],
            "paired_rounds": summary["paired_rounds"],
        },
        "median_latency_ms": summary["median_latency_ms"],
        "program_static_metrics": [{
            key: row[key] for key in (
                "program_id", "candidate_id", "dma_bytes", "dma_calls",
                "service_score_byte_equivalent",
            )
        } for row in contract["programs"]],
        "board_evidence": {
            "correctness_calls": summary["correctness_calls"],
            "timing_calls": summary["timing_calls"],
            "all_outputs_equal": summary["all_outputs_equal"],
            "all_outputs_nonzero": summary["all_outputs_nonzero"],
            "fused_tir_prediction_exact": summary["fused_tir_prediction_exact"],
            "boot_id": summary["boot_id"],
        },
        "failed_attempt_diagnosis": {
            "failed_artifact_sha256": failed_files,
            "correctness_delta_matches_one_execution": True,
            "timed_raw_delta_is_exactly_two_executions": True,
            "repair": "divide timed profiler counts by two; selection rule unchanged",
        },
        "method_consequence": (
            "Do not treat a frozen scalar request penalty as universally calibrated. "
            "When bytes and calls disagree, retain both Pareto-incomparable programs and "
            "escalate fidelity; static elimination is safe only under joint dominance."
        ),
        "bound_inputs": {
            "contract_sha256": sha256(contract_dir / "contract.json"),
            "contract_artifact_manifest_sha256": sha256(contract_dir / "artifact_hashes.json"),
            "board_summary_sha256": sha256(board_dir / "summary.json"),
            "board_artifact_manifest_sha256": sha256(board_dir / "artifact_hashes.json"),
        },
        "verified_artifact_counts": verified,
        "auditor_source_sha256": sha256(Path(__file__).resolve()),
        "claim_boundary": (
            "One prospective conflict pair on one ResNet50 workload/model/boot. It rejects "
            "universal calibration of the P7R166 scalar but does not prove Pareto escalation "
            "optimal or characterize physical AXI service time."
        ),
    }
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    write_json(output / "audit.json", result)
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
    parser.add_argument("--contract", required=True)
    parser.add_argument("--board-result", required=True)
    parser.add_argument("--failed-attempt", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
