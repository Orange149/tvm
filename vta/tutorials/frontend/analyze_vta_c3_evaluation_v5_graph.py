#!/usr/bin/env python3
"""Analyze the two frozen C3 v5 YOLO-512 graph boots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from c3_evaluation_v5 import read_json, seal_flat, verify_flat, write_json

HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_OUTPUT = GROUPED / "20260926_c3_yolo512_evaluation_v5_graph_analysis_run01"


def run(args):
    dirs = [Path(args.boot1_dir).resolve(), Path(args.boot2_dir).resolve()]
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    manifests = [verify_flat(path) for path in dirs]
    boots = [read_json(path / "summary.json") for path in dirs]
    for index, boot in enumerate(boots, 1):
        if boot.get("status") != "C3_EVALUATION_V5_GRAPH_BOOT{}_COMPLETE".format(index):
            raise ValueError("graph boot {} is incomplete".format(index))
    if boots[0]["boot_id"] == boots[1]["boot_id"]:
        raise ValueError("graph boots do not have independent boot IDs")
    if boots[0]["build_manifest_sha256"] != boots[1]["build_manifest_sha256"]:
        raise ValueError("graph build changed between boots")
    if boots[0]["completion_manifest_sha256"] != boots[1]["completion_manifest_sha256"]:
        raise ValueError("completion policy changed between boots")
    if boots[0]["state_diagnostic_manifest_sha256"] != boots[1]["state_diagnostic_manifest_sha256"]:
        raise ValueError("clean-deployment isolation evidence changed between boots")
    per_boot = []
    all_blocks_positive = True
    both_boots_one_percent = True
    for index, boot in enumerate(boots, 1):
        stats = boot["statistics"]
        incumbent = float(stats["incumbent"]["median_ms"])
        selector = float(stats["selector_v4"]["median_ms"])
        gain = 100.0 * (1.0 - selector / incumbent)
        incumbent_fps = 1000.0 / incumbent
        selector_fps = 1000.0 / selector
        throughput_gain = 100.0 * (selector_fps / incumbent_fps - 1.0)
        blocks = {}
        for block in ("A", "B"):
            left = float(stats["incumbent"]["block_median_ms"][block])
            right = float(stats["selector_v4"]["block_median_ms"][block])
            blocks[block] = {"incumbent_median_ms": left, "selector_median_ms": right,
                             "improvement_pct": 100.0 * (1.0 - right / left),
                             "selector_lower": right < left}
        both_boots_one_percent = both_boots_one_percent and gain >= 1.0
        all_blocks_positive = all_blocks_positive and all(row["selector_lower"] for row in blocks.values())
        per_boot.append({
            "boot_index": index, "boot_id": boot["boot_id"],
            "incumbent_median_ms": incumbent, "selector_median_ms": selector,
            "incumbent_throughput_fps": incumbent_fps,
            "selector_throughput_fps": selector_fps,
            "latency_improvement_pct": gain,
            "throughput_improvement_pct": throughput_gain,
            "blocks": blocks,
        })
    passed = both_boots_one_percent and all_blocks_positive
    output.mkdir(parents=True)
    analysis = {
        "schema": "c3_evaluation_v5_graph_analysis_v1",
        "status": "C3_EVALUATION_V5_GRAPH_ANALYSIS_COMPLETE",
        "boot_manifests": manifests, "per_boot": per_boot,
        "both_boots_median_improvement_at_least_1pct": both_boots_one_percent,
        "all_four_blocks_selector_lower": all_blocks_positive,
        "stable_practical_graph_acceleration": passed,
        "thresholds_predeclared": True,
        "claim_boundary": "input-ready board-side graph execution; excludes external preprocessing/output transfer/postprocessing",
    }
    write_json(output / "analysis.json", analysis)
    write_json(output / "summary.json", {
        "status": analysis["status"], "stable_practical_graph_acceleration": passed,
        "boot_latency_improvements_pct": [row["latency_improvement_pct"] for row in per_boot],
        "boot_throughput_improvements_pct": [
            row["throughput_improvement_pct"] for row in per_boot
        ],
        "all_four_blocks_selector_lower": all_blocks_positive,
    })
    seal_flat(output, (Path(__file__).resolve(),))
    print(json.dumps(read_json(output / "summary.json"), indent=2, sort_keys=True), flush=True)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--boot1-dir", type=Path, required=True)
    p.add_argument("--boot2-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
