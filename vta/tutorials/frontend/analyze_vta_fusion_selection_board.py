#!/usr/bin/env python3
"""Evaluate a frozen F0/F1/F2 contract after independent board observations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import statistics
from pathlib import Path

from fusion_aware_transfer import read_json, sha256_file


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summarize_candidates(contract, observations):
    expected = set(contract["union_candidate_ids"])
    by_candidate = defaultdict(lambda: defaultdict(list))
    correctness = defaultdict(list)
    for row in observations:
        if row["candidate_id"] not in expected:
            raise ValueError("observation outside frozen union: {}".format(row["candidate_id"]))
        correctness[row["candidate_id"]].append(bool(row["correct"]))
        if row["correct"] and row.get("latency_ms") is not None:
            by_candidate[row["candidate_id"]][row["boot_id"]].append(float(row["latency_ms"]))
    result = {}
    for candidate_id in contract["union_candidate_ids"]:
        boots = by_candidate[candidate_id]
        valid = bool(correctness[candidate_id]) and all(correctness[candidate_id]) and len(boots) >= 3
        boot_medians = {boot: statistics.median(values) for boot, values in sorted(boots.items())}
        result[candidate_id] = {
            "correct": bool(correctness[candidate_id]) and all(correctness[candidate_id]),
            "boots": len(boots),
            "boot_medians_ms": boot_medians,
            "valid_for_timing": valid,
            "latency_ms": statistics.median(boot_medians.values()) if valid else None,
        }
    return result


def evaluate_strategy(order, candidates, oracle):
    valid = [(index, candidates[candidate_id]) for index, candidate_id in enumerate(order, 1)
             if candidates[candidate_id]["valid_for_timing"]]
    if not valid:
        return {"best_latency_ms": None, "regret_at_5": None, "evaluations_to_union_oracle_2pct": None}
    best = min(row["latency_ms"] for _, row in valid)
    hit = next((index for index, row in valid if row["latency_ms"] <= oracle * 1.02), None)
    return {
        "best_latency_ms": best,
        "regret_at_5": (best - oracle) / oracle,
        "evaluations_to_union_oracle_2pct": hit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    contract = read_json(args.contract)
    if contract.get("status") == "pre_board_frozen_inactive_ablation":
        raise ValueError(
            "frozen F0/F1/F2 orders are identical; board timing cannot test a ranking advantage"
        )
    if contract.get("status") != "pre_board_frozen" or not contract.get("label_free"):
        raise ValueError("selection contract was not frozen label-free")
    args.output_dir.mkdir(parents=True)
    candidates = summarize_candidates(contract, load_jsonl(args.observations))
    valid_latencies = [row["latency_ms"] for row in candidates.values() if row["valid_for_timing"]]
    if not valid_latencies:
        raise ValueError("no candidate has three valid boots")
    oracle = min(valid_latencies)
    strategies = {
        name: evaluate_strategy(order, candidates, oracle)
        for name, order in contract["strategies"].items()
    }
    f2 = strategies["F2"]
    baselines = [strategies["F0"], strategies["F1"]]
    gate = (
        f2["evaluations_to_union_oracle_2pct"] is not None
        and all(base["evaluations_to_union_oracle_2pct"] is not None for base in baselines)
        and f2["evaluations_to_union_oracle_2pct"] <= min(base["evaluations_to_union_oracle_2pct"] for base in baselines)
        and f2["regret_at_5"] <= min(base["regret_at_5"] for base in baselines) + 1e-12
    )
    result = {
        "schema": "vta_fusion_aware_selection_board_analysis_v1",
        "union_oracle_latency_ms": oracle,
        "candidate_results": candidates,
        "strategies": strategies,
        "f2_efficiency_claim_gate": gate,
        "interpretation": "claim selection improvement only if the F2 gate is true; otherwise report a negative ranking result",
        "sources": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in (
            ("script", Path(__file__)), ("contract", args.contract), ("observations", args.observations)
        )},
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"union_oracle_latency_ms": oracle, "strategies": strategies, "f2_gate": gate}, indent=2))


if __name__ == "__main__":
    main()
