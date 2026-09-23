"""Read-only analysis for the T8 eager/staged full-residency experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import c3_vthread_staged_protocol as protocol
import run_vta_c3_reuse_supplement as s


SCHEMA = "c3_vthread_staged_residency_v1"


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(args, out):
    frozen = Path(args.contract).resolve().parent
    contract = s.verify(frozen)
    if contract.get("phase") != "freeze" or contract.get("schema") != SCHEMA:
        raise ValueError("analysis requires the frozen staged-residency contract")
    prepared = Path(contract["prepared_directory"])
    s.verify(prepared)
    if s.file_hash(prepared / "artifact_hashes.json") != contract["prepared_manifest_sha256"]:
        raise ValueError("prepared artifacts changed")
    board_dir = Path(args.board_dir).resolve()
    board_contract = s.verify(board_dir)
    if (board_dir / "invalid_session.json").exists():
        raise ValueError("board session is invalid")
    summary = s.read(board_dir / "summary.json")
    if (
        board_contract.get("schema") != SCHEMA
        or board_contract.get("phase") != "board"
        or board_contract.get("frozen_manifest_sha256")
        != s.file_hash(frozen / "artifact_hashes.json")
        or not summary.get("timing_valid")
    ):
        raise ValueError("board session identity or validity mismatch")
    timing = _jsonl(board_dir / "timing.jsonl")
    protocol.validate_timing(timing, summary["accepted"])

    comparisons = []
    pair_eligible = s.read(prepared / "pair_audit.json")["eligible_program_comparison"]
    for baseline, candidate, kind in protocol.COMPARISONS:
        observation = protocol.paired(timing, baseline, candidate)
        comparisons.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "kind": kind,
                "program_comparison_eligible": pair_eligible
                if kind == "first_use_staging_effect"
                else True,
                "classification": protocol.development_classification(observation),
                **observation,
            }
        )
    s.write(out / "comparisons.json", comparisons)
    _csv(
        out / "paired_comparisons.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key not in ("round_gains_pct", "baseline_round_ms", "candidate_round_ms")
            }
            for row in comparisons
        ],
    )

    prepared_candidates = s.read(prepared / "candidates.json")
    candidates = []
    for row in prepared_candidates:
        item = {
            "candidate_id": row["candidate_id"],
            "variant": row["variant"],
            "status": row["status"],
        }
        item.update(row.get("arenas", {}))
        item.update(row["local_qualification"]["semantics"]["totals"])
        command = row["command_qualification"]["instruction_audits"]["0"]
        item["uop_load_bytes"] = command["uop_load_bytes"]
        item["submission_count"] = command["submission_count"]
        item["instruction_count"] = sum(
            block["instructions"] for block in command["submissions"]
        )
        values = [
            sample["latency_ms"]
            for sample in timing
            if sample["candidate_id"] == row["candidate_id"]
        ]
        item["board_operator_call_median_ms"] = statistics.median(values)
        candidates.append(item)
    _csv(out / "space_transfer_instruction_time.csv", candidates)
    _csv(out / "timing_samples.csv", timing)

    lines = [
        "# R18-H1 T8 首次使用分阶段全权重驻留开发实验",
        "",
        "固定同一 T8/O2 Tile 与双虚拟线程，对比原始 O2、前置全装载 P4-E、首次使用分阶段装载 P4-S。",
        "这是一次启动的开发测量，不构成跨启动稳定性或整网结论。",
        "",
        "| 基线→测试 | 含义 | 配对中位改善 | 胜/负轮 | 开发判定 |",
        "|---|---|---:|---:|---|",
    ]
    for row in comparisons:
        lines.append(
            "| {} → {} | {} | {:.3f}% | {}/{} | {} |".format(
                row["baseline"],
                row["candidate"],
                row["kind"],
                row["paired_median_gain_pct"],
                row["wins"],
                row["losses"],
                row["classification"],
            )
        )
    lines += [
        "",
        "PE→PS 是程序级首次使用分阶段装载效果；它包含派生的指令与 μop 组织变化，不能解释为纯 DMA 周期。",
        "耗时是统一预热后的板端算子调用耗时，不是纯硬件指令时间。",
        "静态 arena 不是逐周期有效 SRAM 利用率；累计 UOP 装载量也不是片上 UOP 峰值。",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))
    s.write(
        out / "contract.json",
        {
            "schema": SCHEMA,
            "phase": "analyze",
            "frozen_directory": str(frozen),
            "frozen_manifest_sha256": s.file_hash(frozen / "artifact_hashes.json"),
            "board_directory": str(board_dir),
            "board_manifest_sha256": s.file_hash(board_dir / "artifact_hashes.json"),
        },
    )
    s.write(
        out / "summary.json",
        {
            "status": "ONE_BOOT_STAGED_RESIDENCY_ANALYSIS",
            "valid_boots": 1,
            "boot_id": summary["boot_id"],
            "timing_samples": len(timing),
            "cross_boot_stability_claim": False,
            "whole_graph_claim": False,
            "hardware_overlap_cycles_claim": False,
        },
    )
