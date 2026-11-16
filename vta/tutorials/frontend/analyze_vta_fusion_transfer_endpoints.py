#!/usr/bin/env python3
"""Freeze E2 compile-visible deltas for the three endpoint triplets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fusion_aware_transfer import read_json, sha256_file, subtract_vectors


TRIPLETS = (
    ("vta:05:05", "vta:05:06", "vta:05:07"),
    ("vta:10:10", "vta:10:11", "vta:10:12"),
    ("vta:15:15", "vta:15:16", "vta:15:17"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    args.output_dir.mkdir(parents=True)
    source = read_json(args.estimates)
    estimates = {row["segment_id"]: row for row in source["rows"]}
    rows = []
    for base_id, projection_id, tail_id in TRIPLETS:
        base, projection, tail = (estimates[segment_id] for segment_id in (base_id, projection_id, tail_id))
        if any(row["status"] != "exact_reuse" for row in (base, projection, tail)):
            raise ValueError("endpoint triplet contains a non-reusable estimate")
        projection_dma = subtract_vectors(projection["total_dma"], base["total_dma"])
        tail_dma = subtract_vectors(tail["total_dma"], projection["total_dma"])
        projection_host = subtract_vectors(projection["components"]["host"], base["components"]["host"])
        tail_host = subtract_vectors(tail["components"]["host"], projection["components"]["host"])
        rows.append({
            "triplet": [base_id, projection_id, tail_id],
            "base_to_projection": {"dma_delta": projection_dma, "host_delta": projection_host},
            "projection_to_cpu_tail": {"dma_delta": tail_dma, "host_delta": tail_host},
            "cpu_tail_vta_dma_unchanged": not any(tail_dma.values()),
            "cpu_tail_host_changed": any(tail_host.values()),
        })
    result = {
        "schema": "vta_fixed_tile_fusion_endpoint_deltas_v1",
        "status": "compile_visible_mechanism_complete_board_service_pending",
        "claim_boundary": "logical DMA and TIR CPU helpers only; no physical traffic or FPS direction",
        "all_cpu_tails_leave_vta_dma_unchanged": all(row["cpu_tail_vta_dma_unchanged"] for row in rows),
        "all_cpu_tails_change_host_work": all(row["cpu_tail_host_changed"] for row in rows),
        "rows": rows,
        "source": {"path": str(args.estimates.resolve()), "sha256": sha256_file(args.estimates)},
    }
    (args.output_dir / "endpoint_deltas.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "STATUS.md").write_text(
        "# C1 E2 endpoint mechanisms\n\n"
        "- Compile-visible triplets: 3/3 complete.\n"
        "- CPU-tail VTA DMA unchanged: {}/3.\n"
        "- CPU-tail host work changed: {}/3.\n"
        "- Board stage-service qualification: pending.\n"
        "- No claim that fewer bytes must improve FPS.\n".format(
            sum(row["cpu_tail_vta_dma_unchanged"] for row in rows),
            sum(row["cpu_tail_host_changed"] for row in rows),
        )
    )
    print(json.dumps({key: result[key] for key in (
        "status", "all_cpu_tails_leave_vta_dma_unchanged", "all_cpu_tails_change_host_work"
    )}, indent=2))


if __name__ == "__main__":
    main()
