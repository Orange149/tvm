#!/usr/bin/env python3
"""Summarize same-tile residency pairs in a canonical completed pool."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from run_vta_p7r119_yolo_barrier_pilot import sha256_file, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def change(new, old):
    return 100.0 * (float(new) - float(old)) / float(old)


def percentile(values, q):
    ordered = sorted(values)
    return ordered[int(round(q * (len(ordered) - 1)))]


def bootstrap_median_ci(values, seed=20260912, samples=10000):
    rng = random.Random(seed)
    boot = [statistics.median(rng.choices(values, k=len(values))) for _ in range(samples)]
    return {"lower_2_5": percentile(boot, 0.025),
            "upper_97_5": percentile(boot, 0.975),
            "samples": samples, "seed": seed,
            "interpretation": "descriptive family-cluster bootstrap; not a population CI"}


def two_sided_sign_p(wins, losses):
    n = wins + losses
    if not n:
        return None
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def measured_latency(row):
    value = row.get("oracle", {}).get("latency_ms")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    value = float(value)
    return value if value > 0 and math.isfinite(value) else None


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    source = Path(args.completed_pool).resolve()
    pool = read_json(source)
    candidates = pool["workloads"][args.workload_id]["candidates"]
    by_id = {row["candidate_id"]: row for row in candidates}
    pairs = []
    for row in candidates:
        if row["residence_mode"] == "original":
            continue
        control_id = row.get("same_tile_control_id")
        if not control_id or control_id not in by_id:
            continue
        control = by_id[control_id]
        if control["residence_mode"] != "original":
            raise ValueError("same-tile control is not original")
        control_latency = measured_latency(control)
        latency = measured_latency(row)
        # A canonical completed pool retains lowering/FSim/FPGA-invalid identities.
        # They establish failure boundaries but cannot form a latency pair.
        if control_latency is None or latency is None:
            continue
        old = control["predispatch"]["validity"]
        new = row["predispatch"]["validity"]
        pairs.append({
            "mode": row["residence_mode"],
            "candidate_id": row["candidate_id"],
            "control_candidate_id": control_id,
            "knobs": row["predispatch"]["knobs"],
            "control_latency_ms": control_latency,
            "latency_ms": latency,
            "speedup_percent": 100.0 * (control_latency - latency) / control_latency,
            "input_dma_bytes_change_percent": change(
                new["static_input_dma_bytes"], old["static_input_dma_bytes"]
            ),
            "total_dma_bytes_change_percent": change(
                new["static_dma_total_bytes"], old["static_dma_total_bytes"]
            ),
            "total_dma_calls_change_percent": change(
                new["static_dma_total_calls"], old["static_dma_total_calls"]
            ),
            "instruction_peak_change_percent": change(
                new["static_insn_peak_bytes"], old["static_insn_peak_bytes"]
            ),
            "uop_peak_change_percent": change(
                new["static_uop_peak_bytes"], old["static_uop_peak_bytes"]
            ),
        })
    by_mode = {}
    for mode in sorted({row["mode"] for row in pairs}):
        selected = [row for row in pairs if row["mode"] == mode]
        speedups = [row["speedup_percent"] for row in selected]
        wins = sum(value > 0 for value in speedups)
        losses = sum(value < 0 for value in speedups)
        by_mode[mode] = {
            "pairs": len(selected), "wins": wins, "losses": losses,
            "ties": len(selected) - wins - losses,
            "median_speedup_percent": statistics.median(speedups),
            "min_speedup_percent": min(speedups),
            "max_speedup_percent": max(speedups),
            "median_input_dma_bytes_change_percent": statistics.median(
                row["input_dma_bytes_change_percent"] for row in selected
            ),
            "median_total_dma_bytes_change_percent": statistics.median(
                row["total_dma_bytes_change_percent"] for row in selected
            ),
            "median_total_dma_calls_change_percent": statistics.median(
                row["total_dma_calls_change_percent"] for row in selected
            ),
            "two_sided_exact_sign_test_p": two_sided_sign_p(wins, losses),
            "median_speedup_bootstrap_95_percent": bootstrap_median_ci(speedups),
        }
    result = {
        "schema": "c3_same_tile_completed_pool_analysis_v1",
        "workload_id": args.workload_id,
        "completed_pool": {"path": str(source), "sha256": sha256_file(source)},
        "pair_count": len(pairs),
        "by_mode": by_mode,
        "pairs": pairs,
        "claim_boundary": (
            "same-tile descriptive evidence; families share one geometry/boot and are not "
            "independent workload replications; logical DMA is not physical AXI traffic"
        ),
    }
    output.mkdir(parents=True)
    write_json(output / "same_tile_analysis.json", result)
    write_json(output / "artifact_hashes.json", {
        "artifacts": {"same_tile_analysis.json": sha256_file(
            output / "same_tile_analysis.json"
        )},
        "source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__)},
    })
    print(json.dumps({"workload_id": args.workload_id, "by_mode": by_mode},
                     indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-pool", required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
