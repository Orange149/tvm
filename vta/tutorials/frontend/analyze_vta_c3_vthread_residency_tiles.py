#!/usr/bin/env python3
"""Combine sealed T8/T7 vthread-residency program and timing evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import run_vta_c3_reuse_supplement as s


SCHEMA = "c3_vthread_weight_residency_two_tile_analysis_v1"


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_tile(tile, prepared, board, analysis):
    prepare_contract = s.verify(prepared)
    board_contract = s.verify(board)
    analysis_contract = s.verify(analysis)
    board_summary = s.read(board / "summary.json")
    if not board_summary.get("timing_valid") or (board / "invalid_session.json").exists():
        raise ValueError(tile + " board session is invalid")
    if board_contract["frozen_manifest_sha256"] != analysis_contract["frozen_manifest_sha256"]:
        raise ValueError(tile + " board/analysis frozen identity differs")
    if Path(analysis_contract["board_directory"]).resolve() != board.resolve():
        raise ValueError(tile + " analysis does not bind the requested board session")
    rows = s.read(prepared / "candidates.json")
    timing = jsonl(board / "timing.jsonl")
    metrics = []
    for row in rows:
        candidate_id = row["candidate_id"]
        local = row["local_qualification"]
        totals = local["semantics"]["totals"]
        arenas = local["storage_arenas"]["scope_arena_span_sum_upper_bound_bytes"]
        command = row["fsim_qualification"]["command_features"]["values"]
        dump = row["command_qualification"]["instruction_audits"]["0"]
        address = s.read(prepared / "address_audit" / (candidate_id + ".json"))
        semantic = {}
        raw = {}
        for submission in dump["submissions"]:
            for name, value in submission["semantic"].items():
                semantic[name] = semantic.get(name, 0) + value
            for name, value in submission["raw_opcode"].items():
                raw[name] = raw.get(name, 0) + value
        samples = [item["latency_ms"] for item in timing if item["candidate_id"] == candidate_id]
        if len(samples) != 18:
            raise ValueError(candidate_id + " does not have 18 timing samples")
        metrics.append(
            {
                "tile": tile,
                "candidate_id": candidate_id,
                "variant": row["variant"],
                "baseline_provenance": row.get("baseline_provenance", "genuine TopHub O2"),
                "tile_h": row["visible_features"]["tile_h"],
                "oc_nthread": row["visible_features"]["oc_nthread"],
                "inp_arena_bytes": arenas["local.inp_buffer"],
                "wgt_arena_bytes": arenas["local.wgt_buffer"],
                "acc_arena_bytes": arenas["local.acc_buffer"],
                "inp_load_bytes": totals["load_buffer_2d_inp_bytes"],
                "inp_load_calls": totals["load_buffer_2d_inp_calls"],
                "wgt_load_bytes": totals["load_buffer_2d_wgt_bytes"],
                "wgt_load_calls": totals["load_buffer_2d_wgt_calls"],
                "out_store_bytes": totals["store_buffer_2d_out_bytes"],
                "out_store_calls": totals["store_buffer_2d_out_calls"],
                "sync_multiplicity": json.dumps(local["sync_multiplicity"]),
                "explicit_group_drains": local["explicit_residency_drains"],
                "submission_count": command["submissions"],
                "submit_reason_counts": json.dumps(command["submit_reason_counts"], sort_keys=True),
                "instruction_bytes": command["totals"]["insn_bytes"],
                "uop_load_bytes": command["totals"]["uop_bytes"],
                "max_single_uop_kernel_entries": address["max_kernel_uops"],
                "uop_effective_occupancy_peak": address["uop_effective_occupancy_peak"],
                "reset_gemm_instructions": semantic.get("RESET_GEMM", 0),
                "mac_gemm_instructions": semantic.get("MAC_GEMM", 0),
                "alu_instructions": semantic.get("ALU", 0),
                "nop_load_instructions": semantic.get("NOP-LOAD", 0),
                "nop_store_instructions": semantic.get("NOP-STORE", 0),
                "raw_load_instructions": raw.get("LOAD", 0),
                "raw_store_instructions": raw.get("STORE", 0),
                "compute_sha256": address["compute_sha256"],
                "ordered_event_sha256": address["event_sha256"],
                "lowered_tir_sha256": local["tir_sha256"],
                "binary_sha256": row["cross_qualification"]["binary_sha256"],
                "binary_bytes": row["cross_qualification"]["binary_bytes"],
                "board_operator_call_median_ms": statistics.median(samples),
                "board_operator_call_min_ms": min(samples),
                "board_operator_call_max_ms": max(samples),
                "timing_samples": len(samples),
                "boot_id": board_summary["boot_id"],
            }
        )
    return {
        "prepare_contract": prepare_contract,
        "metrics": metrics,
        "comparisons": s.read(analysis / "comparisons.json"),
        "boot_id": board_summary["boot_id"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t8-prepare", type=Path, required=True)
    parser.add_argument("--t8-board", type=Path, required=True)
    parser.add_argument("--t8-analysis", type=Path, required=True)
    parser.add_argument("--t7-prepare", type=Path, required=True)
    parser.add_argument("--t7-board", type=Path, required=True)
    parser.add_argument("--t7-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {
        "T8": tuple(path.resolve() for path in (args.t8_prepare, args.t8_board, args.t8_analysis)),
        "T7": tuple(path.resolve() for path in (args.t7_prepare, args.t7_board, args.t7_analysis)),
    }
    loaded = {
        tile: load_tile(tile, *paths)
        for tile, paths in inputs.items()
    }
    metrics = loaded["T8"]["metrics"] + loaded["T7"]["metrics"]
    comparisons = []
    for tile in ("T8", "T7"):
        for row in loaded[tile]["comparisons"]:
            comparisons.append(
                {
                    "tile": tile,
                    "baseline": row["baseline"],
                    "candidate": row["candidate"],
                    "kind": row["kind"],
                    "paired_median_gain_pct": row["paired_median_gain_pct"],
                    "wins": row["wins"],
                    "losses": row["losses"],
                    "classification": row["classification"],
                }
            )
    write_csv(out / "tile_variant_program_metrics.csv", metrics)
    write_csv(out / "tile_paired_comparisons.csv", comparisons)

    by_key = {(row["tile"], row["variant"]): row for row in metrics}
    lines = [
        "# R18-H1 T8/T7 双虚拟线程驻留统一对照",
        "",
        "两个 Tile 的开发测量来自同一 boot ID `{}`；这是 Tile 条件性重复，不是独立启动复现。".format(
            loaded["T8"]["boot_id"]
        ),
        "T8 O2 是 genuine TopHub；T7 O2 是历史 ALT_251 实体上的生产 mode 0，不能称 genuine TopHub。",
        "",
        "## 程序与命令组织",
        "",
        "| Tile/版本 | WGT arena | WGT LOAD B/calls | UOP LOAD B | 指令 B | 提交 | 分组 drain | 算子中位 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tile in ("T8", "T7"):
        for variant in ("O2", "VR", "VS"):
            row = by_key[tile, variant]
            lines.append(
                "| {}/{} | {} | {}/{} | {} | {} | {} | {} | {:.6f} |".format(
                    tile,
                    variant,
                    row["wgt_arena_bytes"],
                    row["wgt_load_bytes"],
                    row["wgt_load_calls"],
                    row["uop_load_bytes"],
                    row["instruction_bytes"],
                    row["submission_count"],
                    row["explicit_group_drains"],
                    row["board_operator_call_median_ms"],
                )
            )
    lines += [
        "",
        "VR/VS 在各自 Tile 内具有相同计算程序哈希、输入/输出请求和 arena；差异是权重生命周期及其派生的同步、提交和 μop。",
        "T8 有 7 个高度块，VR 因两个线程产生 14 个分组 drain/15 次总提交；T7 有 8 个高度块，对应 16/17。",
        "VS 在两个 Tile 上都只加载两段线程权重、只做函数结束提交；μop 累计装载为 T8 1624 B、T7 1424 B。",
        "单个 UOP kernel 最大条目数均经过最终地址审计；运行期有效 UOP 占用峰值仍为 unknown。",
        "",
        "## 配对时间",
        "",
        "| Tile | O2→VR | VR→VS | O2→VS |",
        "|---|---:|---:|---:|",
    ]
    for tile in ("T8", "T7"):
        values = {row["kind"]: row for row in comparisons if row["tile"] == tile}
        lines.append(
            "| {} | {:+.3f}% ({}/6胜) | {:+.3f}% ({}/6胜) | {:+.3f}% ({}/6胜) |".format(
                tile,
                values["new_framework_effect"]["paired_median_gain_pct"],
                values["new_framework_effect"]["wins"],
                values["residency_program_effect"]["paired_median_gain_pct"],
                values["residency_program_effect"]["wins"],
                values["net_vs_o2"]["paired_median_gain_pct"],
                values["net_vs_o2"]["wins"],
            )
        )
    lines += [
        "",
        "两个 Tile 的 VS 均为 6/6 轮快于各自 O2，但 +1.191%/+1.126% 都未达到预注册 2% 门槛。",
        "因此结论是：并发驻留机制跨两个 Tile 可执行且额外代价受控，并出现一致的轻微净改善；尚无独立启动稳定加速结论。",
        "命令字节和累计 UOP 是程序要求，不是物理总线流量或片上动态利用率；耗时也不是纯硬件周期。",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))
    s.write(
        out / "contract.json",
        {
            "schema": SCHEMA,
            "phase": "two_tile_analysis",
            "inputs": {
                tile: {
                    kind: {
                        "directory": str(path),
                        "manifest_sha256": s.file_hash(path / "artifact_hashes.json"),
                    }
                    for kind, path in zip(("prepare", "board", "analysis"), paths)
                }
                for tile, paths in inputs.items()
            },
        },
    )
    s.write(
        out / "summary.json",
        {
            "status": "TWO_TILE_ONE_BOOT_DEVELOPMENT_COMPLETE",
            "tiles": ["T8", "T7"],
            "variants_per_tile": 3,
            "same_boot": loaded["T8"]["boot_id"] == loaded["T7"]["boot_id"],
            "boot_id": loaded["T8"]["boot_id"],
            "independent_boot_confirmation": False,
            "whole_graph_claim": False,
        },
    )
    s.seal(out)


if __name__ == "__main__":
    main()

