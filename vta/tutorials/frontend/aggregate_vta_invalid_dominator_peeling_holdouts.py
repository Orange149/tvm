#!/usr/bin/env python3
"""Aggregate the three prospectively frozen cheap-proxy/peeling holdouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reduction(value, baseline):
    return 100.0 * (1.0 - value / baseline)


def normalize(path):
    data = read(path / "analysis.json")
    if data["workload_id"] == "R50G":
        exact = bool(data["front_retains_full_fpga_correct_pool_oracle"])
        return {
            "workload_id": "R50G",
            "pool_size": int(data["pool_size"]),
            "fpga_correct": int(data["fpga_correct_count"]),
            "fpga_invalid": int(data["fpga_rejected_count"]),
            "online_dispatches": int(data["operator_proxy_front_size"]),
            "exact_oracle": exact,
            "oracle_plus_2pct": bool(data["front_best_within_oracle_2_percent"]),
            "regret_percent": float(data["front_best_regret_percent"]),
            "peeling_expanded": bool(
                data["invalid_dominator_peeling"].get("prospective_peeling_triggered", False)
            ),
            "outer_process_seconds": None,
            "component_savings": None,
        }
    pool = data["pool"]
    online = data["online"]
    replay = data["frozen_order_replay"]["online_peeling"]
    return {
        "workload_id": data["workload_id"],
        "pool_size": int(pool["eligible"]),
        "fpga_correct": int(pool["fpga_correct"]),
        "fpga_invalid": int(pool["fpga_invalid"]),
        "online_dispatches": int(online["dispatch_count"]),
        "exact_oracle": bool(online["best_is_exact_oracle"]),
        "oracle_plus_2pct": replay["targets"]["oracle_plus_2pct"] is not None,
        "regret_percent": float(online["best_regret_percent"]),
        "peeling_expanded": len(online["waves"]) > 1,
        "outer_process_seconds": float(online["outer_process_seconds_before_summary_write"]),
        "component_savings": data["online_vs_exhaustive_savings"],
    }


def run(args):
    paths = [Path(value) for value in args.analysis]
    verified = {path.name: len(verify_artifacts_compatible(path)) for path in paths}
    rows = [normalize(path) for path in paths]
    if [row["workload_id"] for row in rows] != ["R50G", "R50H", "R50I"]:
        raise RuntimeError("expected ordered R50G/R50H/R50I inputs")
    total_pool = sum(row["pool_size"] for row in rows)
    total_dispatch = sum(row["online_dispatches"] for row in rows)
    result = {
        "schema": "c3_vta_invalid_dominator_peeling_three_holdout_aggregate_v1",
        "status": "three_prospective_peeling_holdouts_aggregate_complete",
        "holdouts": rows,
        "aggregate": {
            "holdout_count": len(rows),
            "pool_candidates": total_pool,
            "fpga_correct_candidates": sum(row["fpga_correct"] for row in rows),
            "fpga_invalid_candidates": sum(row["fpga_invalid"] for row in rows),
            "online_candidate_dispatches": total_dispatch,
            "candidate_dispatch_reduction_percent": reduction(total_dispatch, total_pool),
            "oracle_plus_2pct_holdouts": sum(row["oracle_plus_2pct"] for row in rows),
            "exact_oracle_holdouts": sum(row["exact_oracle"] for row in rows),
            "actual_peeling_expansion_holdouts": sum(row["peeling_expanded"] for row in rows),
            "complete_outer_wall_available_holdouts": sum(
                row["outer_process_seconds"] is not None for row in rows
            ),
            "available_complete_outer_wall_seconds": sum(
                row["outer_process_seconds"] or 0.0 for row in rows
            ),
        },
        "verified_artifact_counts": verified,
        "bound_analysis_manifests": {
            path.name: sha256(path / "artifact_hashes.json") for path in paths
        },
        "aggregator_source_sha256": sha256(Path(__file__).resolve()),
        "claim_boundary": (
            "Three source-consistent ResNet50-v2 layer holdouts selected and frozen before "
            "their target labels. R50G lacks serialized complete outer-process wall. R50H "
            "is the only holdout whose initial front contains an FPGA-invalid dominator and "
            "therefore the only direct peeling-expansion validation. This is three workloads "
            "within one model family, not cross-model or statistical universal validity."
        ),
    }
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    write_json(output / "summary.json", result)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
