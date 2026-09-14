#!/usr/bin/env python3
"""Freeze W05 bounded-hybrid/virtual-thread candidates before dynamic labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dma_pareto(row):
    original = row["original"]["dma"]
    hybrid = row["paper_inspired_hybrid"]["dma"]
    before_bytes = original["load_buffer_2d_bytes"] + original["store_buffer_2d_bytes"]
    after_bytes = hybrid["load_buffer_2d_bytes"] + hybrid["store_buffer_2d_bytes"]
    before_calls = original["load_buffer_2d_calls"] + original["store_buffer_2d_calls"]
    after_calls = hybrid["load_buffer_2d_calls"] + hybrid["store_buffer_2d_calls"]
    result = {
        "input_byte_reduction_fraction": row["input_dma_reduction_fraction"],
        "total_byte_reduction_fraction": (before_bytes - after_bytes) / before_bytes,
        "total_call_reduction_fraction": (before_calls - after_calls) / before_calls,
    }
    result["certified"] = (
        result["input_byte_reduction_fraction"] > 0
        and result["total_byte_reduction_fraction"] >= 0
        and result["total_call_reduction_fraction"] >= 0
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    scan = Path(args.scan).resolve()
    scan_hashes = json.loads((scan.parent / "artifact_hashes.json").read_text())["artifacts"]
    if scan_hashes.get(scan.name) != sha256_file(scan):
        raise ValueError("scan artifact hash mismatch")

    rows = [
        row for row in load_jsonl(scan)
        if row["workload_id"] == "W05"
        and row["residence_mode"] == "paper_inspired_hybrid"
        and row.get("locally_eligible")
    ]
    for row in rows:
        row["dma_pareto"] = dma_pareto(row)
    certified_pool = [row for row in rows if row["dma_pareto"]["certified"]]
    incumbent_knobs = {"tile_h": 14, "tile_w": 14, "tile_ci": 1, "tile_co": 4}
    for row in certified_pool:
        row["incumbent_log2_distance"] = sum(
            abs(math.log2(row["knobs"][name] / incumbent_knobs[name]))
            for name in incumbent_knobs
        )
    certified_pool.sort(
        key=lambda row: (
            row["incumbent_log2_distance"],
            -row["dma_pareto"]["total_byte_reduction_fraction"],
            -row["dma_pareto"]["total_call_reduction_fraction"],
            row["config_index"],
        )
    )
    if len(certified_pool) < 2:
        raise ValueError("W05 hybrid experiment requires two Pareto candidates")
    certified = certified_pool[:2]
    for ordinal, row in enumerate(certified):
        row["candidate_role"] = "p7r_hybrid_confirmation"
        row["p7r_selection"] = {
            "stratum": "nearest_tophub" if ordinal == 0 else "next_nearest_tophub",
            "prediction_frozen_before_fsim_fpga_or_latency": (
                "best_final_latency" if ordinal == 0 else "second_best_final_latency"
            ),
            "features": row["dma_pareto"],
            "hardware_rule": "bounded co/width reuse with oc_nthread=2",
            "incumbent_log2_distance": row["incumbent_log2_distance"],
        }

    shortlist = output / "shortlist.jsonl"
    shortlist.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in certified),
        encoding="utf-8",
    )
    protocol = {
        "schema": "c3_p7r_w05_hybrid_confirmation_v1",
        "status": "frozen_before_hybrid_fsim_fpga_or_latency_labels",
        "workload_id": "W05",
        "mode": "paper_inspired_hybrid",
        "oc_nthread": 2,
        "selection_pool": {
            "joint_space": 400,
            "both_paths_lower_and_input_reduces": len(rows),
            "dma_pareto_certified": len(certified_pool),
            "frozen_for_dynamic_gates": len(certified),
        },
        "selection_rule": "DMA-Pareto first; then minimum log2 tile distance to TopHub config575, total-byte reduction tie-break",
        "primary_hypothesis": "both candidates improve over exact same-tile original",
        "secondary_hypothesis": "nearest-TopHub config574 has the best final latency",
        "protected_hypothesis": "only a candidate faster than TopHub config575 may replace it",
        "candidates": [
            {
                "candidate_id": row["candidate_id"],
                "config_index": row["config_index"],
                "selection": row["p7r_selection"],
            }
            for row in certified
        ],
        "source_scan": {"path": str(scan), "sha256": sha256_file(scan)},
        "script_sha256": sha256_file(__file__),
        "performance_labels_used": False,
        "board_contacted": False,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "STATUS.md").write_text(
        "# W05 bounded-hybrid t2 confirmation\n\n"
        "- Status: `frozen_before_hybrid_fsim_fpga_or_latency_labels`\n"
        "- Joint space: 400; local input-reducing: {}; DMA-Pareto: {}\n"
        "- Frozen configs: {}\n"
        "- No FSim, FPGA correctness, or latency label used for selection.\n".format(
            len(rows), len(certified_pool),
            ", ".join(str(row["config_index"]) for row in certified)
        )
    )
    hashes = {
        path.name: sha256_file(path) for path in output.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes}, indent=2) + "\n"
    )
    print(json.dumps(protocol["candidates"], indent=2))


if __name__ == "__main__":
    main()
