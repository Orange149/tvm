#!/usr/bin/env python3
"""Summarize the frozen W05 hybrid search funnel and protected outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-scan", required=True)
    parser.add_argument("--pure-dispatch", required=True)
    parser.add_argument("--hybrid-scan", required=True)
    parser.add_argument("--hybrid-contract", required=True)
    parser.add_argument("--hybrid-fsim", required=True)
    parser.add_argument("--cross-summary", required=True)
    parser.add_argument("--board-correctness", required=True)
    parser.add_argument("--timing", required=True)
    parser.add_argument("--final-dispatch", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    paths = {name: Path(value).resolve() for name, value in vars(args).items()
             if name != "output_dir"}
    data = {name: load_json(path) for name, path in paths.items()}

    pure_scan = data["pure_scan"]
    pure_dispatch = data["pure_dispatch"]
    hybrid_scan = data["hybrid_scan"]
    contract = data["hybrid_contract"]
    fsim = data["hybrid_fsim"]
    cross = data["cross_summary"]
    correctness = data["board_correctness"]
    timing = data["timing"]
    final_dispatch = data["final_dispatch"]
    if pure_scan["attempted"] != hybrid_scan["attempted"] or hybrid_scan["attempted"] != 400:
        raise ValueError("expected the same 400-config W05 t2 space")
    if pure_dispatch["counts"].get("abstain_keep_original") != 27:
        raise ValueError("pure input-stationary Pareto outcome changed")
    if contract["selection_pool"]["frozen_for_dynamic_gates"] != 2:
        raise ValueError("dynamic shortlist changed")
    if fsim["passed_all_seeds"] != 2 or cross["passed"] != 4:
        raise ValueError("local correctness/deployability gate failed")
    if correctness["status"] != "passed" or correctness["passed_seed_checks"] != 18:
        raise ValueError("FPGA correctness gate failed")
    if final_dispatch["performance_labels_used"]:
        raise ValueError("pre-measure dispatch unexpectedly used latency labels")
    if final_dispatch["counts"].get("allow_timing") != 2:
        raise ValueError("exact hardware allowlist changed")

    comparison574 = timing["comparisons"]["574"]
    comparison494 = timing["comparisons"]["494"]
    gap = -comparison574["hybrid_vs_tophub_fraction"]
    result = {
        "schema": "c3_p7r_w05_hybrid_budget_analysis_v1",
        "status": "completed",
        "scope": "single-boot prospective operator experiment; no stage/FPS or full-pool oracle claim",
        "search_funnel": {
            "raw_t2_configs": hybrid_scan["attempted"],
            "pure_input_stationary_input_reducing": pure_scan["locally_eligible"],
            "pure_input_stationary_dma_pareto": 0,
            "bounded_hybrid_input_reducing_and_dma_pareto": hybrid_scan["locally_eligible"],
            "frozen_dynamic_candidates": 2,
            "fpga_correct_candidates": 2,
            "exact_allow_timing_candidates": final_dispatch["counts"]["allow_timing"],
            "candidate_dispatch_reduction_vs_raw_fraction": 1.0 - 2.0 / hybrid_scan["attempted"],
            "interpretation": "counterfactual candidate-space reduction; the 398 unmeasured points have no latency/oracle labels",
        },
        "performance": {
            "config574_same_tile_paired_median_speedup_fraction":
                comparison574["paired_median_speedup_fraction"],
            "config574_same_tile_wins": comparison574["paired_wins"],
            "config494_same_tile_paired_median_speedup_fraction":
                comparison494["paired_median_speedup_fraction"],
            "config494_same_tile_wins": comparison494["paired_wins"],
            "nearest_tophub_prediction_correct":
                timing["prediction_config574_best_new_candidate"],
            "best_new_gap_to_tophub_fraction": gap,
            "best_new_within_2pct_of_tophub": gap <= 0.02,
            "incumbent_sentinel_range_over_median_fraction":
                timing["incumbent_sentinel_drift_fraction"],
            "safe_dispatch_choice": timing["safe_dispatch_choice"],
        },
        "claim_boundary": {
            "supported": [
                "hardware-derived filters reduce the candidate dispatch set from 400 to 2",
                "both preselected candidates are FPGA-correct and improve exact same-tile originals",
                "the nearest-TopHub candidate is within 2% of the protected incumbent",
                "strict fallback prevents a final performance regression",
            ],
            "not_supported": [
                "the new candidate beats TopHub",
                "the unmeasured 400-point pool oracle is known",
                "stage latency or FPS improves",
                "same-boot blocks are independent replications",
            ],
        },
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)}
                   for name, path in paths.items()},
    }
    analysis = output / "analysis.json"
    analysis.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "STATUS.md").write_text(
        "# W05 bounded-hybrid budget analysis\n\n"
        "- Raw t2 space -> FPGA performance candidates: 400 -> 2 (99.50% fewer)\n"
        "- Config574 same-tile: {:+.2%}, {}/7 wins\n"
        "- Config494 same-tile: {:+.2%}, {}/7 wins\n"
        "- Best new candidate gap to TopHub: {:.2%}; within 2%: `{}`\n"
        "- Final safe dispatch: `{}`\n"
        "- Boundary: no full-pool oracle, stage, FPS, or independent-boot claim\n".format(
            comparison574["paired_median_speedup_fraction"], comparison574["paired_wins"],
            comparison494["paired_median_speedup_fraction"], comparison494["paired_wins"],
            gap, gap <= 0.02, timing["safe_dispatch_choice"],
        )
    )
    hashes = {path.name: sha256_file(path) for path in output.iterdir()
              if path.is_file() and path.name != "artifact_hashes.json"}
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes}, indent=2) + "\n"
    )
    print(json.dumps({"search_funnel": result["search_funnel"],
                      "performance": result["performance"]}, indent=2))


if __name__ == "__main__":
    main()
