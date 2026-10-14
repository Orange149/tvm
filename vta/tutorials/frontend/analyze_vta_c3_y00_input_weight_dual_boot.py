#!/usr/bin/env python3
"""Combine two sealed Y00F06 input-plus-weight board sessions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import c3_y00_input_weight_protocol as protocol
import run_vta_c3_reuse_supplement as s


SCHEMA = "c3_y00_input_weight_dual_boot_analysis_v1"
SOURCE_SCHEMA = "c3_y00_input_weight_staged_v1"


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_board(directory, frozen_manifest):
    directory = directory.resolve()
    contract = s.verify(directory)
    if (directory / "invalid_session.json").exists():
        raise ValueError("invalid board session: {}".format(directory))
    summary = s.read(directory / "summary.json")
    if (
        contract.get("schema") != SOURCE_SCHEMA
        or contract.get("phase") != "board"
        or contract.get("frozen_manifest_sha256") != frozen_manifest
        or not summary.get("timing_valid")
    ):
        raise ValueError("board identity or timing validity mismatch: {}".format(directory))
    timing = _jsonl(directory / "timing.jsonl")
    protocol.validate_timing(timing, summary["accepted"])
    return {
        "directory": str(directory),
        "manifest_sha256": s.file_hash(directory / "artifact_hashes.json"),
        "boot_id": summary["boot_id"],
        "accepted": summary["accepted"],
        "timing": timing,
    }


def _classification(observations, matched):
    if not matched:
        return "control_not_matched"
    if len(observations) != 2:
        return "awaiting_two_distinct_boots"
    if all(
        row["paired_median_gain_pct"] >= 2 and row["wins"] >= 5
        for row in observations
    ):
        return "stable_improvement"
    if all(
        row["paired_median_gain_pct"] <= -2 and row["losses"] >= 5
        for row in observations
    ):
        return "stable_regression"
    return "no_stable_direction_threshold"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--boot-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.boot_dir) != 2:
        raise ValueError("exactly two --boot-dir arguments are required")

    frozen = args.contract.resolve().parent
    frozen_contract = s.verify(frozen)
    if (
        frozen_contract.get("schema") != SOURCE_SCHEMA
        or frozen_contract.get("phase") != "freeze"
    ):
        raise ValueError("analysis requires the frozen Y00 input+weight contract")
    prepared = Path(frozen_contract["prepared_directory"])
    s.verify(prepared)
    if (
        s.file_hash(prepared / "artifact_hashes.json")
        != frozen_contract["prepared_manifest_sha256"]
    ):
        raise ValueError("prepared artifacts changed")

    frozen_manifest = s.file_hash(frozen / "artifact_hashes.json")
    boots = [_load_board(path, frozen_manifest) for path in args.boot_dir]
    if len({row["boot_id"] for row in boots}) != 2:
        raise ValueError("board sessions must come from two distinct boot IDs")
    if len({tuple(row["accepted"]) for row in boots}) != 1:
        raise ValueError("the two boots accepted different candidate sets")

    pair_matched = s.read(prepared / "pair_audit.json")["eligible_program_comparison"]
    comparisons = []
    flat = []
    for baseline, candidate, kind in protocol.COMPARISONS:
        observations = []
        for boot in boots:
            observation = protocol.paired(boot["timing"], baseline, candidate)
            observation["boot_id"] = boot["boot_id"]
            observations.append(observation)
        matched = pair_matched if kind == "incremental_staged_weight_residency" else True
        status = _classification(observations, matched)
        comparisons.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "kind": kind,
                "program_comparison_eligible": matched,
                "status": status,
                "boots": observations,
            }
        )
        for observation in observations:
            flat.append(
                {
                    "baseline": baseline,
                    "candidate": candidate,
                    "kind": kind,
                    "program_comparison_eligible": matched,
                    "status": status,
                    "boot_id": observation["boot_id"],
                    "paired_median_gain_pct": observation["paired_median_gain_pct"],
                    "wins": observation["wins"],
                    "losses": observation["losses"],
                }
            )

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    s.write(out / "comparisons.json", comparisons)
    _write_csv(out / "paired_comparisons.csv", flat)
    lines = [
        "# YOLOv3-tiny Y00F06 输入＋首次使用权重驻留双启动分析",
        "",
        "固定 h16/w52 Tile、程序二进制和计时协议；两个会话来自不同 boot ID。",
        "",
        "| 基线→测试 | 含义 | 跨启动判定 | boot 1 | boot 2 |",
        "|---|---|---|---:|---:|",
    ]
    for row in comparisons:
        values = [
            "{:.3f}% ({}/6胜)".format(
                item["paired_median_gain_pct"], item["wins"]
            )
            for item in row["boots"]
        ]
        lines.append(
            "| {} → {} | {} | {} | {} | {} |".format(
                row["baseline"],
                row["candidate"],
                row["kind"],
                row["status"],
                values[0],
                values[1],
            )
        )
    lines += [
        "",
        "I→IW 保持输入请求、计算、静态 arena 与同步匹配，解释为增量程序级权重驻留效果。",
        "稳定改善门槛为两个 boot 各自中位改善至少 2%，且各至少 5/6 轮获胜。",
        "耗时为预热后的板端算子调用耗时；不声称纯硬件时间、逐周期等待或整网加速。",
        "逻辑请求量不是外部总线物理流量，静态 arena 不是动态 SRAM 利用率。",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))
    s.write(
        out / "contract.json",
        {
            "schema": SCHEMA,
            "phase": "dual_boot_analyze",
            "frozen_directory": str(frozen),
            "frozen_manifest_sha256": frozen_manifest,
            "board_sessions": [
                {
                    "directory": row["directory"],
                    "manifest_sha256": row["manifest_sha256"],
                    "boot_id": row["boot_id"],
                }
                for row in boots
            ],
        },
    )
    s.write(
        out / "summary.json",
        {
            "status": "TWO_BOOT_Y00_INPUT_WEIGHT_ANALYSIS",
            "valid_boots": 2,
            "boot_ids": [row["boot_id"] for row in boots],
            "timing_samples": sum(len(row["timing"]) for row in boots),
            "cross_boot_stability_claim": True,
            "whole_graph_claim": False,
            "hardware_overlap_cycles_claim": False,
        },
    )
    s.seal(out)


if __name__ == "__main__":
    main()
