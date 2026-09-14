#!/usr/bin/env python3
"""Freeze a prospective E03 confirmation pair using the DMA Pareto policy."""

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


def load_verified(path):
    path = Path(path).resolve()
    hashes = json.loads((path.parent / "artifact_hashes.json").read_text())["artifacts"]
    if hashes.get(path.name) != sha256_file(path):
        raise ValueError("scan artifact hash mismatch")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def annotate(row):
    original, residency = row["original"]["dma"], row["input_stationary"]["dma"]
    before_bytes = original["load_buffer_2d_bytes"] + original["store_buffer_2d_bytes"]
    after_bytes = residency["load_buffer_2d_bytes"] + residency["store_buffer_2d_bytes"]
    before_calls = original["load_buffer_2d_calls"] + original["store_buffer_2d_calls"]
    after_calls = residency["load_buffer_2d_calls"] + residency["store_buffer_2d_calls"]
    result = json.loads(json.dumps(row))
    result["dma_pareto"] = {
        "input_byte_reduction_fraction": row["input_dma_reduction_fraction"],
        "total_byte_reduction_fraction": (before_bytes - after_bytes) / before_bytes,
        "total_call_reduction_fraction": (before_calls - after_calls) / before_calls,
    }
    result["dma_pareto"]["certified"] = (
        result["dma_pareto"]["input_byte_reduction_fraction"] > 0
        and result["dma_pareto"]["total_byte_reduction_fraction"] >= 0
        and result["dma_pareto"]["total_call_reduction_fraction"] >= 0
    )
    return result


def select(rows):
    eligible = [annotate(row) for row in rows if row["workload_id"] == "E03"
                and row["locally_eligible"]]
    certified = [row for row in eligible if row["dma_pareto"]["certified"]]
    if len(certified) < 2:
        raise ValueError("E03 requires at least two Pareto-certified candidates")
    order = sorted(
        certified,
        key=lambda row: (
            -row["dma_pareto"]["total_byte_reduction_fraction"],
            -row["dma_pareto"]["total_call_reduction_fraction"],
            row["config_index"],
        ),
    )
    priority, boundary = order[0], order[-1]
    if priority["candidate_id"] == boundary["candidate_id"]:
        raise ValueError("confirmation strata collapsed")
    for row, stratum, prediction in (
        (priority, "pareto_priority", "larger_same_tile_speedup"),
        (boundary, "pareto_boundary", "smaller_same_tile_speedup"),
    ):
        row["candidate_role"] = "p7r_e03_confirmation"
        row["p7r_selection"] = {
            "stratum": stratum,
            "prediction_frozen_before_fsim_or_board": prediction,
            "features": row["dma_pareto"],
        }
    return [priority, boundary], len(eligible), len(certified)


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
    selected, eligible_count, certified_count = select(load_verified(scan))
    shortlist = output / "shortlist.jsonl"
    shortlist.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8"
    )
    protocol = {
        "schema": "c3_p7r_e03_pareto_confirmation_v1",
        "status": "frozen_before_e03_fsim_fpga_or_latency_labels",
        "workload_id": "E03",
        "selection_pool": {
            "joint_space": 864,
            "locally_eligible": eligible_count,
            "pareto_certified": certified_count,
        },
        "selection_rule": "maximize total-byte reduction, then total-call reduction; compare with minimum-margin certified boundary",
        "primary_hypothesis": "same-tile speedup(priority) > same-tile speedup(boundary)",
        "safety_policy": "measure original and residency correctness first; final choice is min(measured original, measured residency)",
        "candidates": [
            {
                "candidate_id": row["candidate_id"],
                "config_index": row["config_index"],
                "selection": row["p7r_selection"],
            }
            for row in selected
        ],
        "source_scan": {"path": str(scan), "sha256": sha256_file(scan)},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "STATUS.md").write_text(
        "# E03 DMA Pareto confirmation\n\n"
        "- Status: `frozen_before_e03_fsim_fpga_or_latency_labels`\n"
        "- Joint space: 864; locally eligible: {}; Pareto-certified: {}\n"
        "- Priority config: {}; boundary config: {}\n"
        "- Primary hypothesis: same-tile speedup(priority) > same-tile speedup(boundary)\n".format(
            eligible_count, certified_count, selected[0]["config_index"], selected[1]["config_index"]
        )
    )
    manifest = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__)),
        "board_contacted": False,
        "performance_labels_used": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    hashes = {path.name: sha256_file(path) for path in output.iterdir()
              if path.is_file() and path.name != "artifact_hashes.json"}
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes}, indent=2) + "\n"
    )
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
