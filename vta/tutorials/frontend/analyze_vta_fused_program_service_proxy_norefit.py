#!/usr/bin/env python3
"""Check a frozen request-cost proxy on final fused-program traffic without refitting."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_artifacts_compatible(directory):
    directory = Path(directory)
    manifest = read_json(directory / "artifact_hashes.json")
    recorded = manifest.get("artifacts", manifest.get("files"))
    if not isinstance(recorded, dict):
        raise RuntimeError("unsupported artifact hash manifest schema")
    for relative, expected in recorded.items():
        if sha256(directory / relative) != expected:
            raise RuntimeError("artifact hash mismatch: " + relative)
    return recorded


def total_dma(traffic):
    return {
        "bytes": int(traffic["load_buffer_2d_bytes"] + traffic["store_buffer_2d_bytes"]),
        "calls": int(traffic["load_buffer_2d_calls"] + traffic["store_buffer_2d_calls"]),
    }


def service_score(traffic, request_equivalent_bytes, extra_submissions=0,
                  extra_submission_equivalent_bytes=0):
    total = total_dma(traffic)
    return (
        total["bytes"]
        + request_equivalent_bytes * total["calls"]
        + extra_submission_equivalent_bytes * extra_submissions
    )


def select(programs, metric):
    return min(programs, key=lambda row: (row[metric], row["program_id"]))


def pairwise_order_accuracy(programs, metric):
    rows = []
    for left, right in itertools.combinations(programs, 2):
        metric_delta = left[metric] - right[metric]
        latency_delta = left["latency_ms"] - right["latency_ms"]
        concordant = metric_delta * latency_delta > 0
        tied = metric_delta == 0 or latency_delta == 0
        rows.append({
            "left": left["program_id"],
            "right": right["program_id"],
            "metric_delta": metric_delta,
            "latency_delta_ms": latency_delta,
            "concordant": concordant,
            "tied": tied,
        })
    eligible = [row for row in rows if not row["tied"]]
    correct = sum(row["concordant"] for row in eligible)
    return {
        "correct": correct,
        "total": len(eligible),
        "accuracy": correct / len(eligible) if eligible else None,
        "discordant_pairs": [row for row in eligible if not row["concordant"]],
    }


def run(args):
    contract_dir = Path(args.contract)
    service_dir = Path(args.service_development)
    comparison_dir = Path(args.tile_comparison)
    verified = {
        "contract": len(verify_artifacts_compatible(contract_dir)),
        "service_development": len(verify_artifacts_compatible(service_dir)),
        "tile_comparison": len(verify_artifacts_compatible(comparison_dir)),
    }
    contract = read_json(contract_dir / "contract.json")
    service = read_json(service_dir / "analysis.json")
    comparison = read_json(comparison_dir / "summary.json")
    if contract.get("status") != "posthoc_test_spec_frozen_before_analysis_execution":
        raise RuntimeError("analysis contract is not frozen")
    if not contract["label_exposure"]["target_labels_already_exposed"]:
        raise RuntimeError("post-hoc label exposure must be explicit")
    bound = contract["bound_inputs"]
    checks = {
        "p7r166_analysis_sha256": sha256(service_dir / "analysis.json"),
        "p7r166_artifact_manifest_sha256": sha256(service_dir / "artifact_hashes.json"),
        "p7r296_summary_sha256": sha256(comparison_dir / "summary.json"),
        "p7r296_artifact_manifest_sha256": sha256(comparison_dir / "artifact_hashes.json"),
    }
    if checks != bound:
        raise RuntimeError("contract-bound input hash mismatch")
    if service.get("status") != "development_frozen_before_y01_latency_recovery_confirmation":
        raise RuntimeError("service proxy source is not the frozen development result")
    if comparison.get("status") != "two_callsite_latency_holdouts_expose_tile_dependent_dma_fragmentation":
        raise RuntimeError("tile comparison is incomplete")

    frozen = service["frozen_proxy"]
    request_cost = int(frozen["request_equivalent_bytes"])
    submission_cost = int(frozen["extra_submission_equivalent_bytes"])
    if request_cost != contract["frozen_weights"]["request_equivalent_bytes"]:
        raise RuntimeError("request coefficient drift")
    if submission_cost != contract["frozen_weights"]["extra_submission_equivalent_bytes"]:
        raise RuntimeError("submission coefficient drift")

    programs = []
    for tile in comparison["tiles"]:
        for mode, latency_key in (("original", "incumbent"), ("residency", "proposal")):
            traffic = tile[mode]
            total = total_dma(traffic)
            programs.append({
                "program_id": tile["family"] + ":" + mode,
                "family": tile["family"],
                "mode": mode,
                "dma_bytes": total["bytes"],
                "dma_calls": total["calls"],
                "service_score_byte_equivalent": service_score(
                    traffic, request_cost, 0, submission_cost
                ),
                "latency_ms": float(tile["latency_ms"][latency_key]),
                "extra_residency_submissions": 0,
            })

    by_mode = []
    for mode in ("original", "residency"):
        subset = [row for row in programs if row["mode"] == mode]
        oracle = select(subset, "latency_ms")
        policies = {}
        for name, metric in (
            ("bytes_only", "dma_bytes"),
            ("fused_service_proxy", "service_score_byte_equivalent"),
        ):
            chosen = select(subset, metric)
            policies[name] = {
                "selected_program": chosen["program_id"],
                "selected_latency_ms": chosen["latency_ms"],
                "oracle_hit": chosen["program_id"] == oracle["program_id"],
                "latency_regret_percent": (
                    chosen["latency_ms"] / oracle["latency_ms"] - 1.0
                ) * 100.0,
            }
        by_mode.append({
            "mode": mode,
            "oracle_program": oracle["program_id"],
            "oracle_latency_ms": oracle["latency_ms"],
            "policies": policies,
        })

    accuracy = {
        name: sum(row["policies"][name]["oracle_hit"] for row in by_mode) / len(by_mode)
        for name in ("bytes_only", "fused_service_proxy")
    }
    pairwise = {
        "bytes_only": pairwise_order_accuracy(programs, "dma_bytes"),
        "fused_service_proxy": pairwise_order_accuracy(
            programs, "service_score_byte_equivalent"
        ),
    }
    if accuracy != {"bytes_only": 0.0, "fused_service_proxy": 1.0}:
        raise RuntimeError("frozen expected no-refit comparison outcome changed")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    result = {
        "schema": "c3_vta_fused_program_service_proxy_norefit_analysis_v1",
        "status": "frozen_request_penalty_repairs_two_byte_only_tile_inversions",
        "frozen_formula": frozen["formula"],
        "request_equivalent_bytes": request_cost,
        "extra_submission_equivalent_bytes": submission_cost,
        "submission_term_scope": (
            "zero in this input-stationary comparison; no barrier-induced extra submission"
        ),
        "programs": programs,
        "within_mode_selection": by_mode,
        "selection_accuracy": accuracy,
        "all_four_program_pairwise_ordering": pairwise,
        "bound_input_hashes": checks,
        "verified_artifact_counts": verified,
        "analyzer_source_sha256": sha256(Path(__file__).resolve()),
        "board_contacted": False,
        "claim_boundary": (
            "No-refit post-hoc check on two tiles of one workload/model/boot. The 64 KiB "
            "coefficient predates the target labels, but this comparison was specified after "
            "label exposure; it is not a prospective holdout or a physical request-latency fit."
        ),
    }
    write_json(output / "analysis.json", result)
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--service-development", required=True)
    parser.add_argument("--tile-comparison", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
