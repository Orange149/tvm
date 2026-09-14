#!/usr/bin/env python3
"""Audit a prospective ResNet50 full-graph residency latency holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from run_vta_resnet50_callsite_composed_graph_board import verify_build_artifacts
from run_vta_resnet50_fused_tir_pair_board_v2 import PROFILE_KEYS, profile_delta
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def run(args):
    roots = {
        "holdout": Path(args.holdout_contract),
        "pair_build": Path(args.pair_build),
        "tir_audit": Path(args.tir_audit),
        "fused_contract": Path(args.fused_contract),
        "board_run": Path(args.board_run),
    }
    verified = {name: verify_build_artifacts(path) for name, path in roots.items()}

    holdout = read_json(roots["holdout"] / "contract.json")
    pair = read_json(roots["pair_build"] / "summary.json")
    tir = read_json(roots["tir_audit"] / "summary.json")
    fused = read_json(roots["fused_contract"] / "fused_tir_occurrence_delta.json")
    board_contract = read_json(roots["board_run"] / "contract.json")
    board = read_json(roots["board_run"] / "summary.json")
    correctness = read_json(roots["board_run"] / "correctness.json")
    timings = load_jsonl(roots["board_run"] / "timing.jsonl")

    require(
        holdout.get("status") == "frozen_before_new_full_graph_build_or_callsite_latency",
        "holdout was not prospectively frozen",
    )
    require(not holdout.get("full_graph_latency_labels_used"), "holdout used full-graph labels")
    require(not holdout.get("partial_mask_latency_labels_used"), "holdout used mask labels")
    require(not holdout.get("operator_latency_labels_used"), "holdout used operator latency")
    require(pair.get("status") == "resnet50_generic_pair_cross_build_dispatch_verified", "pair build incomplete")
    require(tir.get("status") == "full_graph_fused_tir_pair_captured", "TIR audit incomplete")
    require(
        fused.get("status") == "compiler_fused_tir_prediction_frozen_before_fpga",
        "fused contract was not prospective",
    )
    require(fused.get("full_graph_latency_observed") is False, "fused contract saw graph latency")
    require(fused.get("partial_mask_latency_observed") is False, "fused contract saw mask latency")
    require(
        board.get("status") == "fused_tir_predicted_pair_correctness_dma_and_timing_complete",
        "board run incomplete",
    )

    candidate_ids = [row["candidate_id"] for row in holdout["selected_pair"]]
    modes = [row["public_mode"] for row in holdout["selected_pair"]]
    require(candidate_ids == [row["candidate_id"] for row in pair["builds"]], "pair identity drift")
    require(candidate_ids == [row["candidate_id"] for row in tir["builds"]], "TIR identity drift")
    require(candidate_ids == fused["candidate_ids"], "fused identity drift")
    require(modes == pair["public_modes"] == tir["public_modes"] == fused["modes"] == board["modes"], "mode drift")
    require(pair["builds"][0]["graph_sha256"] == pair["builds"][1]["graph_sha256"], "A/B graph differs")

    require(
        fused["tir_audit_summary_sha256"] == sha256(roots["tir_audit"] / "summary.json"),
        "fused-to-TIR hash link failed",
    )
    require(
        fused["pair_build_summary_sha256"] == sha256(roots["pair_build"] / "summary.json"),
        "fused-to-build hash link failed",
    )
    require(
        fused["holdout_contract_sha256"] == sha256(roots["holdout"] / "contract.json"),
        "fused-to-holdout hash link failed",
    )
    require(
        board_contract["build_manifest_sha256"] == sha256(roots["pair_build"] / "summary.json"),
        "board-to-build hash link failed",
    )
    require(
        board_contract["fused_analysis_sha256"]
        == sha256(roots["fused_contract"] / "fused_tir_occurrence_delta.json"),
        "board-to-fused hash link failed",
    )

    occurrence_count = fused["graph_occurrence_count"]
    require(occurrence_count > 0, "empty graph occurrence set")
    require(len(fused["graph_occurrences"]) == occurrence_count, "occurrence list incomplete")
    require(board["graph_occurrence_count"] == occurrence_count, "board occurrence count drift")
    expected = fused["full_graph_fused_tir_delta"]
    require(set(expected) == set(PROFILE_KEYS), "profile field set drift")
    require(board_contract["expected_full_graph_fused_tir_delta"] == expected, "board contract delta drift")
    require(board["expected_full_graph_fused_tir_delta"] == expected, "board summary delta drift")

    seeds = board_contract["seeds"]
    rounds = board_contract["rounds"]
    require(len(correctness) == len(seeds) * len(modes), "correctness row count drift")
    require(len(timings) == rounds * len(modes), "timing row count drift")
    require(
        {(row["seed"], row["public_mode"]) for row in correctness}
        == {(seed, mode) for seed in seeds for mode in modes},
        "correctness seed/mode coverage drift",
    )
    require(
        {(row["round"], row["public_mode"]) for row in timings}
        == {(index, mode) for index in range(rounds) for mode in modes},
        "timing round/mode coverage drift",
    )
    require(all(row["paired_equal"] and row["paired_mismatch_count"] == 0 for row in correctness + timings), "A/B output mismatch")
    require(all(row["output_nonzero"] > 0 for row in correctness + timings), "zero output observed")

    correctness_delta = profile_delta(correctness, modes)
    timing_delta = profile_delta(timings, modes, divisor=2.0)
    require(correctness_delta == expected, "correctness profile differs from frozen TIR")
    require(timing_delta == expected, "timing profile differs from frozen TIR")
    medians = {
        mode: statistics.median(row["latency_ms"] for row in timings if row["public_mode"] == mode)
        for mode in modes
    }
    wins = sum(
        next(row["latency_ms"] for row in timings if row["round"] == index and row["public_mode"] == modes[1])
        < next(row["latency_ms"] for row in timings if row["round"] == index and row["public_mode"] == modes[0])
        for index in range(rounds)
    )
    require(medians == board["median_latency_ms"], "latency median drift")
    require(wins == board["residency_paired_wins"], "paired-win count drift")
    require(board["all_outputs_equal"] and board["all_outputs_nonzero"], "summary output gate failed")
    require(board["fused_tir_prediction_exact"], "summary exact-profile gate failed")
    require(board["board_after"].get("FPGA") == "operating", "FPGA not operating after run")
    require(board["board_after"].get("UDMABUF") == "201326592", "u-dma-buf size drift")
    require(not board["board_after"].get("STORAGE_ERRORS"), "storage error recorded")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    summary = {
        "schema": "c3_vta_resnet50_fullgraph_latency_holdout_audit_v1",
        "status": "prospective_fullgraph_holdout_chain_verified",
        "selected_family_id": holdout["selected_family_id"],
        "candidate_ids": candidate_ids,
        "modes": modes,
        "graph_occurrence_count": occurrence_count,
        "graph_nodes": [row["graph_node"] for row in fused["graph_occurrences"]],
        "expected_and_observed_dma_delta": expected,
        "correctness_calls": len(correctness),
        "timing_calls": len(timings),
        "paired_rounds": rounds,
        "residency_paired_wins": wins,
        "median_latency_ms": medians,
        "residency_speedup_percent": (medians[modes[0]] / medians[modes[1]] - 1.0) * 100.0,
        "boot_id": board["boot_id"],
        "verified_artifact_counts": {name: len(rows) for name, rows in verified.items()},
        "auditor_source_sha256": sha256(Path(__file__).resolve()),
        "claim_boundary_correction": (
            "The board contract inherited an obsolete P7R273/R50A prose string from the genericized runner. "
            "The hash-linked evidence is a prospective R50CF00 five-occurrence full-graph latency holdout."
        ),
        "claim_boundary": (
            "One new ResNet50 workload and one fixed tile, five full-graph call sites, one model and one boot; "
            "logical VTA DMA rather than physical AXI, and no ImageNet accuracy claim"
        ),
    }
    write_json(output / "audit_summary.json", summary)
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-contract", required=True)
    parser.add_argument("--pair-build", required=True)
    parser.add_argument("--tir-audit", required=True)
    parser.add_argument("--fused-contract", required=True)
    parser.add_argument("--board-run", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
