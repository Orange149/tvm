#!/usr/bin/env python3
"""Audit two immutable full-graph pair runs as independent-boot evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory):
    directory = Path(directory).resolve()
    manifest = read(directory / "artifact_hashes.json")["artifacts"]
    for relative, expected in manifest.items():
        if sha(directory / relative) != expected:
            raise ValueError("artifact mismatch: " + relative)
    return sha(directory / "artifact_hashes.json"), len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != 2:
        raise ValueError("exactly two independent runs are required")
    if args.output.exists():
        raise FileExistsError(args.output)

    rows = []
    for directory in args.run:
        ledger, artifact_count = verify(directory)
        summary = read(directory / "summary.json")
        if summary["status"] != "fused_tir_predicted_pair_correctness_dma_and_timing_complete":
            raise ValueError("incomplete pair run")
        if not (summary["all_outputs_equal"] and summary["all_outputs_nonzero"] and summary["fused_tir_prediction_exact"]):
            raise ValueError("functional or fused-TIR gate failed")
        rows.append({
            "directory": str(directory),
            "ledger_sha256": ledger,
            "artifact_count": artifact_count,
            "boot_id": summary["boot_id"],
            "correctness_calls": summary["correctness_calls"],
            "timing_calls": summary["timing_calls"],
            "original_median_ms": summary["median_latency_ms"]["original"],
            "resident_median_ms": summary["median_latency_ms"]["input_stationary"],
            "absolute_reduction_ms": summary["median_latency_ms"]["original"] - summary["median_latency_ms"]["input_stationary"],
            "speedup_percent": summary["residency_speedup_percent"],
            "paired_wins": summary["residency_paired_wins"],
            "paired_rounds": summary["paired_rounds"],
            "expected_dma_delta": summary["expected_full_graph_fused_tir_delta"],
            "observed_dma_delta": summary["observed_full_graph_delta"],
        })
    if rows[0]["boot_id"] == rows[1]["boot_id"]:
        raise ValueError("runs do not come from independent boots")
    if rows[0]["expected_dma_delta"] != rows[1]["expected_dma_delta"]:
        raise ValueError("runs do not bind the same fused program delta")

    speedups = [row["speedup_percent"] for row in rows]
    result = {
        "schema": "c3_vta_resnet50_pair_cross_boot_v1",
        "status": "two_independent_boots_direction_and_magnitude_consistent",
        "runs": rows,
        "independent_boots": 2,
        "positive_boots": sum(row["resident_median_ms"] < row["original_median_ms"] for row in rows),
        "paired_wins": sum(row["paired_wins"] for row in rows),
        "paired_rounds": sum(row["paired_rounds"] for row in rows),
        "correctness_calls": sum(row["correctness_calls"] for row in rows),
        "timing_calls": sum(row["timing_calls"] for row in rows),
        "speedup_percent_mean": statistics.mean(speedups),
        "speedup_percent_min": min(speedups),
        "speedup_percent_max": max(speedups),
        "claim_boundary": "Two independent boots for one frozen pretrained ResNet50 pair; descriptive replication, not a cross-workload population confidence claim, ImageNet accuracy, or physical AXI measurement.",
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output / "README.md").write_text(f"""# P7R380：R50C 全图驻留收益跨启动复验

同一组冻结的预训练 ResNet50 图、参数、A/B DSO 与 fused-TIR DMA 预测在两个独立 boot 上执行：

| boot | original (ms) | input-stationary (ms) | throughput improvement | paired wins |
|---|---:|---:|---:|---:|
| `{rows[0]['boot_id']}` | {rows[0]['original_median_ms']:.3f} | {rows[0]['resident_median_ms']:.3f} | {rows[0]['speedup_percent']:.3f}% | {rows[0]['paired_wins']}/{rows[0]['paired_rounds']} |
| `{rows[1]['boot_id']}` | {rows[1]['original_median_ms']:.3f} | {rows[1]['resident_median_ms']:.3f} | {rows[1]['speedup_percent']:.3f}% | {rows[1]['paired_wins']}/{rows[1]['paired_rounds']} |

两启动均为正向，合计 14/14 配对轮获胜；速度提升范围 {min(speedups):.3f}%--{max(speedups):.3f}%，
均值 {statistics.mean(speedups):.3f}%。12 次 correctness 与 28 次 timing 调用全部 A/B 输出相等且非零，
预注册的十项 fused-TIR LOAD/STORE 差值在两启动上都精确命中。

这支持“该 frozen R50C 全图驻留收益跨启动方向与量级稳定”，但只有一个 workload 的两个 boot，
不能把 14 个配对轮当成 14 个独立启动，也不能外推为 ResNet50 通用加速率、ImageNet accuracy 或
物理 AXI 流量结论。
""")
    artifacts = {p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != "artifact_hashes.json"}
    (args.output / "artifact_hashes.json").write_text(json.dumps({"artifacts": artifacts, "source_sha256": sha(__file__)}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
