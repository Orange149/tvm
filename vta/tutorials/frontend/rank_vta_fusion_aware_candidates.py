#!/usr/bin/env python3
"""Freeze F0/F1/F2 Top-5 orders without reading new board labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from fusion_aware_transfer import pareto_layers, read_json, sha256_file


PARETO_DIMENSIONS = (
    "vta_service_ms",
    "cpu_helper_logical_bytes",
    "boundary_materialized_bytes",
    "fragmented_dma_calls",
    "convert_calls",
)


def candidate_features(candidate, estimates):
    aggregate = Counter()
    statuses = []
    for segment_id, _threads in candidate["path"]:
        if not segment_id.startswith("vta:"):
            continue
        estimate = estimates.get(segment_id)
        if estimate is None:
            statuses.append("missing_segment_estimate")
            continue
        statuses.append(estimate["status"])
        aggregate.update(estimate["structured_vector"])
    return {
        "candidate_id": candidate["candidate_id"],
        "topology_id": candidate["topology_id"],
        "path": candidate["path"],
        "m1_score_ms": float(candidate["m1_score_ms"]),
        "vta_service_ms": float(candidate["m1_components"]["single_vta_plus_mutex_boundary_ms"]),
        "cpu_helper_logical_bytes": int(aggregate["cpu_read_bytes"] + aggregate["cpu_write_bytes"]),
        "boundary_materialized_bytes": int(candidate["boundary_bytes"]),
        "fragmented_dma_calls": int(aggregate["small_load_calls"] + aggregate["strided_load_calls"] + aggregate["strided_store_calls"]),
        "convert_calls": int(aggregate["convert_calls"]),
        "transfer_vector": dict(aggregate),
        "transfer_statuses": sorted(set(statuses)),
        "transfer_exact_reuse": bool(statuses) and set(statuses) == {"exact_reuse"},
    }


def fusion_order(m1_rows, estimates, band_fraction=0.02):
    features = [candidate_features(row, estimates) for row in m1_rows]
    best = features[0]
    limit = best["m1_score_ms"] * (1.0 + band_fraction)
    band = [row for row in features[1:] if row["m1_score_ms"] <= limit]
    outside = [row for row in features[1:] if row["m1_score_ms"] > limit]
    exact = [row for row in band if row["transfer_exact_reuse"]]
    abstained = [row for row in band if not row["transfer_exact_reuse"]]
    layers = pareto_layers(exact, PARETO_DIMENSIONS)
    ordered = [best]
    for layer_index, layer in enumerate(layers, 1):
        for row in layer:
            row["pareto_layer"] = layer_index
            ordered.append(row)
    ordered.extend(sorted(abstained, key=lambda row: (row["m1_score_ms"], row["candidate_id"])))
    ordered.extend(sorted(outside, key=lambda row: (row["m1_score_ms"], row["candidate_id"])))
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--estimates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--band-fraction", type=float, default=0.02)
    parser.add_argument("--budget", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    if args.budget <= 0:
        raise ValueError("budget must be positive")
    args.output_dir.mkdir(parents=True)
    ablation = read_json(args.ablation)
    estimates_payload = read_json(args.estimates)
    estimates = {row["segment_id"]: row for row in estimates_payload["rows"]}
    f0_rows = ablation["models"]["m1"]["top20"]
    f1_rows = ablation["models"]["m2"]["top20"]
    f2_rows = fusion_order(f0_rows, estimates, args.band_fraction)
    strategies = {
        "F0": [row["candidate_id"] for row in f0_rows[:args.budget]],
        "F1": [row["candidate_id"] for row in f1_rows[:args.budget]],
        "F2": [row["candidate_id"] for row in f2_rows[:args.budget]],
    }
    strategy_orders_identical = len({tuple(order) for order in strategies.values()}) == 1
    board_experiment_informative = not strategy_orders_identical
    status = (
        "pre_board_frozen"
        if board_experiment_informative
        else "pre_board_frozen_inactive_ablation"
    )
    candidates = {row["candidate_id"]: row for row in f0_rows + f1_rows}
    candidates.update({row["candidate_id"]: row for row in f2_rows})
    union_ids = []
    for strategy in ("F0", "F1", "F2"):
        for candidate_id in strategies[strategy]:
            if candidate_id not in union_ids:
                union_ids.append(candidate_id)
    contract = {
        "schema": "vta_fusion_aware_selection_contract_v1",
        "status": status,
        "label_free": True,
        "board_contacted": False,
        "strategy_orders_identical": strategy_orders_identical,
        "board_experiment_informative": board_experiment_informative,
        "board_action": (
            "measure frozen union under the registered three-boot protocol"
            if board_experiment_informative
            else "do not launch E3 board timing: identical frozen orders cannot test a ranking advantage"
        ),
        "band_fraction": args.band_fraction,
        "budget_per_strategy": args.budget,
        "f2_semantics": "orders candidates only; no pruning or global-optimum guarantee",
        "pareto_dimensions": list(PARETO_DIMENSIONS),
        "strategies": strategies,
        "union_candidate_ids": union_ids,
        "union_candidates": [candidates[candidate_id] for candidate_id in union_ids],
        "f2_full_order": f2_rows,
        "sources": {
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
            "ablation": {"path": str(args.ablation.resolve()), "sha256": sha256_file(args.ablation)},
            "estimates": {"path": str(args.estimates.resolve()), "sha256": sha256_file(args.estimates)},
        },
    }
    output = args.output_dir / "selection_contract.json"
    output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "STATUS.md").write_text(
        "# C1 F0/F1/F2 selection freeze\n\n"
        "- Status: `{}`.\n"
        "- Budget: {} candidates per strategy.\n"
        "- Union: {} unique candidates.\n"
        "- Strategy orders identical: {}.\n"
        "- Board experiment informative: {}.\n"
        "- Board contacted: no.\n"
        "- F2 is a deterministic measurement order, not safe pruning.\n"
        "- Board action: {}.\n".format(
            status,
            args.budget,
            len(union_ids),
            str(strategy_orders_identical).lower(),
            str(board_experiment_informative).lower(),
            (
                "run the registered three-boot comparison"
                if board_experiment_informative
                else "skip this inactive ablation and report no ranking-improvement claim"
            ),
        )
    )
    hashes = {path.name: sha256_file(path) for path in args.output_dir.iterdir() if path.is_file()}
    (args.output_dir / "artifact_hashes.json").write_text(json.dumps({"artifacts": hashes}, indent=2) + "\n")
    print(json.dumps({
        "status": status,
        "strategies": strategies,
        "union_candidates": len(union_ids),
        "board_experiment_informative": board_experiment_informative,
    }, indent=2))


if __name__ == "__main__":
    main()
