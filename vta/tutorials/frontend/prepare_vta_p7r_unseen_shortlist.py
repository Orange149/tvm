#!/usr/bin/env python3
"""Freeze the P7R request-shape shortlist for the E00--E02 unseen geometries."""

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
    ledger = json.loads((path.parent / "artifact_hashes.json").read_text())["artifacts"]
    if ledger.get(path.name) != sha256_file(path):
        raise ValueError("source artifact hash mismatch: {}".format(path))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def features(row):
    workload = row["identity"]["workload"]
    knobs = row["knobs"]
    input_width = int(workload[1][1][3])
    kernel_width = int(workload[2][1][3])
    stride_width = int(workload[3][1])
    request_width = (knobs["tile_w"] - 1) * stride_width + kernel_width
    contiguity = min(1.0, request_width / float(input_width))
    tile_work = (
        knobs["tile_h"] * knobs["tile_w"] * knobs["tile_ci"] * knobs["tile_co"]
    )
    shape_score = row["input_dma_reduction_fraction"] * (1.0 - contiguity) * tile_work
    dma = row["input_stationary"]["dma"]
    return {
        "estimated_input_row_contiguity": contiguity,
        "tile_work_proxy": tile_work,
        "request_shape_score": shape_score,
        "residency_total_load_bytes": int(dma["load_buffer_2d_bytes"]),
        "residency_total_load_calls": int(dma["load_buffer_2d_calls"]),
    }


def select(rows, workload_id):
    eligible = [row for row in rows if row["workload_id"] == workload_id and row["locally_eligible"]]
    if not eligible:
        raise ValueError("no eligible candidates for {}".format(workload_id))
    annotated = [(row, features(row)) for row in eligible]
    shape = min(
        annotated,
        key=lambda item: (
            -item[1]["request_shape_score"],
            item[1]["residency_total_load_calls"],
            item[0]["config_index"],
        ),
    )
    contiguous = min(
        annotated,
        key=lambda item: (
            -item[1]["estimated_input_row_contiguity"],
            -item[1]["tile_work_proxy"],
            item[0]["config_index"],
        ),
    )
    if shape[0]["candidate_id"] == contiguous[0]["candidate_id"]:
        raise ValueError("shape and contiguous strata collapsed for {}".format(workload_id))
    result = []
    for stratum, prediction, (row, feature) in (
        ("request_shape", "larger_same_tile_speedup", shape),
        ("contiguous_control", "smaller_or_negative_same_tile_speedup", contiguous),
    ):
        copied = json.loads(json.dumps(row))
        copied["candidate_role"] = "p7r_unseen_candidate"
        copied["p7r_selection"] = {
            "stratum": stratum,
            "prediction_frozen_before_fsim_or_board": prediction,
            "features": feature,
        }
        result.append(copied)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e00", required=True)
    parser.add_argument("--e01", required=True)
    parser.add_argument("--e02", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    sources = {key.upper(): Path(getattr(args, key)) for key in ("e00", "e01", "e02")}
    rows = []
    for workload_id, path in sources.items():
        rows.extend(select(load_verified(path), workload_id))
    (output / "shortlist.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    protocol = {
        "schema": "c3_p7r_unseen_shortlist_v1",
        "status": "frozen_before_unseen_fsim_or_board_labels",
        "development_calibration": "W04 config455 positive and config461 negative orientation pair",
        "request_shape_score": "input_reduction * (1-estimated_input_row_contiguity) * tile_h*tile_w*tile_ci*tile_co",
        "strata_per_workload": ["maximum request_shape_score", "maximum contiguous-row control"],
        "candidate_count": len(rows),
        "workloads": {
            workload_id: [
                {
                    "candidate_id": row["candidate_id"],
                    "config_index": row["config_index"],
                    "selection": row["p7r_selection"],
                }
                for row in rows
                if row["workload_id"] == workload_id
            ]
            for workload_id in sources
        },
        "correctness_gate": "three fixed FSim seeds then three real-FPGA seeds for candidate and exact same-tile original",
        "timing_gate": "all correctness checks pass before paired timing",
        "claim_boundary": "prospective sign/ranking test on three unseen geometries; no ResNet stage or FPS claim",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    manifest = {
        "schema": "c3_p7r_unseen_shortlist_manifest_v1",
        "board_contacted": False,
        "source_results": {
            workload_id: {"path": str(path), "sha256": sha256_file(path)}
            for workload_id, path in sources.items()
        },
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "STATUS.md").write_text(
        "# P7R unseen request-shape shortlist\n\n"
        "- Status: `frozen_before_unseen_fsim_or_board_labels`\n"
        "- Workloads: E00, E01, E02\n"
        "- Candidates: 6 (request-shape prediction plus contiguous control per workload)\n"
        "- Latency/FPS labels used from E00--E02: none\n"
    )
    hashes = {
        path.name: sha256_file(path)
        for path in output.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_text(json.dumps({"artifacts": hashes}, indent=2) + "\n")
    print(json.dumps(protocol["workloads"], indent=2), flush=True)


if __name__ == "__main__":
    main()
