#!/usr/bin/env python3
"""Analyze the preregistered Y08 FPGA pool without changing its frozen orders."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify(directory):
    directory = Path(directory).resolve()
    ledger = read(directory / "artifact_hashes.json")["artifacts"]
    for name, expected in ledger.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or sha(path) != expected:
            raise ValueError("artifact mismatch: " + str(path))
    return sha(directory / "artifact_hashes.json")


def pct_drop(before, after):
    return 100.0 * (before - after) / before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--orders", type=Path, required=True)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    inputs = {
        "qualification_ledger": verify(args.qualification),
        "orders_ledger": verify(args.orders),
        "board_ledger": verify(args.board),
    }
    order_doc = read(args.orders / "orders.json")
    if order_doc.get("board_contacted") or order_doc.get("performance_labels_used"):
        raise ValueError("orders were not frozen label-free")
    board_summary = read(args.board / "summary.json")
    if board_summary["status"] != "completed_full_pool" or board_summary["fpga_correct"] != 6:
        raise ValueError("board result is not the completed six-candidate pool")

    programs = {item["candidate_id"]: item for item in order_doc["programs"]}
    static = {item["candidate_id"]: item for item in rows(args.qualification / "static_results.jsonl")}
    fsim = {item["candidate_id"]: item for item in rows(args.qualification / "fsim_results.jsonl")}
    correctness = {item["candidate_id"]: item for item in rows(args.board / "correctness.jsonl")}
    timing_rows = rows(args.board / "timing.jsonl")
    timing = read(args.board / "timing_summary.json")
    compiled = read(args.board / "cross_compile.json")
    oracle = timing["pool_oracle_latency_ms"]
    ids = list(programs)

    candidate_table = []
    for cid in ids:
        samples = [item for item in timing_rows if item["candidate_id"] == cid]
        stats = timing["candidate_stats"][cid]
        single_runs = len(correctness[cid]["seeds"])
        timed_runs = sum(item["time_evaluator_total_kernel_invocations"] for item in samples)
        candidate_table.append({
            **programs[cid],
            "latency_median_ms": stats["median_ms"],
            "latency_iqr_ms": stats["iqr_ms"],
            "regret_percent": 100.0 * (stats["median_ms"] / oracle - 1.0),
            "within_oracle_2pct": stats["median_ms"] <= oracle * 1.02,
            "within_oracle_5pct": stats["median_ms"] <= oracle * 1.05,
            "static_wall_seconds": static[cid]["diagnostic_wall_seconds"],
            "fsim_wall_seconds": fsim[cid]["diagnostic_wall_seconds"],
            "cross_compile_wall_ms": compiled[cid]["cross_compile_wall_ms"],
            "correctness_host_wall_ms": sum(item["host_wall_ms"] for item in correctness[cid]["seeds"]),
            "correctness_kernel_invocations": single_runs,
            "timing_kernel_invocations": timed_runs,
            "total_board_kernel_invocations": single_runs + timed_runs,
            "logical_dma_bytes_for_candidate_admission": programs[cid]["bytes"] * (single_runs + timed_runs),
            "logical_dma_calls_for_candidate_admission": programs[cid]["calls"] * (single_runs + timed_runs),
        })
    by_id = {item["candidate_id"]: item for item in candidate_table}

    paired = []
    for family in sorted({item["family_id"] for item in candidate_table}):
        family_rows = [item for item in candidate_table if item["family_id"] == family]
        original = next(item for item in family_rows if item["mode"] == "original")
        resident = next(item for item in family_rows if item["mode"] == "input_stationary")
        paired.append({
            "family_id": family,
            "original_candidate_id": original["candidate_id"],
            "resident_candidate_id": resident["candidate_id"],
            "original_latency_ms": original["latency_median_ms"],
            "resident_latency_ms": resident["latency_median_ms"],
            "latency_improvement_percent": pct_drop(original["latency_median_ms"], resident["latency_median_ms"]),
            "dma_bytes_reduction_percent": pct_drop(original["bytes"], resident["bytes"]),
            "dma_calls_reduction_percent": pct_drop(original["calls"], resident["calls"]),
            "resident_faster": resident["latency_median_ms"] < original["latency_median_ms"],
        })

    budgets = (1, 2, 4, 6)
    policies = []
    for policy, frozen_order in order_doc["orders"].items():
        traces = []
        best = float("inf")
        for position, cid in enumerate(frozen_order, 1):
            best = min(best, by_id[cid]["latency_median_ms"])
            traces.append({
                "trial": position,
                "candidate_id": cid,
                "candidate_latency_ms": by_id[cid]["latency_median_ms"],
                "best_latency_ms": best,
                "best_regret_percent": 100.0 * (best / oracle - 1.0),
                "hit_oracle_2pct": best <= oracle * 1.02,
                "hit_oracle_5pct": best <= oracle * 1.05,
            })
        at_budget = []
        for budget in budgets:
            used = min(budget, len(traces))
            last = traces[used - 1]
            prefix = frozen_order[:used]
            at_budget.append({
                "budget": budget,
                "effective_trials": used,
                "best_latency_ms": last["best_latency_ms"],
                "best_regret_percent": last["best_regret_percent"],
                "hit_oracle_2pct": last["hit_oracle_2pct"],
                "hit_oracle_5pct": last["hit_oracle_5pct"],
                "candidate_admission_fpga_invocations": sum(by_id[cid]["total_board_kernel_invocations"] for cid in prefix),
                "candidate_admission_logical_dma_bytes": sum(by_id[cid]["logical_dma_bytes_for_candidate_admission"] for cid in prefix),
                "candidate_admission_logical_dma_calls": sum(by_id[cid]["logical_dma_calls_for_candidate_admission"] for cid in prefix),
            })
        policies.append({
            "policy": policy,
            "frozen_order": frozen_order,
            "trials_to_exact_oracle": next((x["trial"] for x in traces if x["best_latency_ms"] <= oracle), None),
            "trials_to_oracle_2pct": next((x["trial"] for x in traces if x["hit_oracle_2pct"]), None),
            "trials_to_oracle_5pct": next((x["trial"] for x in traces if x["hit_oracle_5pct"]), None),
            "trace": traces,
            "at_budget": at_budget,
        })

    args.output.mkdir(parents=True)
    write(args.output / "candidate_table.json", candidate_table)
    write(args.output / "paired_ablation.json", paired)
    write(args.output / "policy_replay.json", {
        "schema": "c3_y08_frozen_order_replay_v1",
        "scope": "post-collection replay of preregistered orders; not a separately timed online tuner run",
        "oracle_candidate_id": board_summary["oracle_candidate_id"],
        "oracle_latency_ms": oracle,
        "budgets": list(budgets),
        "policies": policies,
    })
    write(args.output / "summary.json", {
        "schema": "c3_y08_frozen_pool_analysis_v1",
        "status": "completed",
        "input_bindings": inputs,
        "candidate_count": len(candidate_table),
        "fpga_correct": board_summary["fpga_correct"],
        "timing_samples": board_summary["timing_samples"],
        "oracle_candidate_id": board_summary["oracle_candidate_id"],
        "oracle_latency_ms": oracle,
        "same_tile_pairs": len(paired),
        "same_tile_resident_faster": sum(item["resident_faster"] for item in paired),
        "board_boot_id": board_summary["boot_id"],
        "full_pool_process_wall_ms": board_summary["process_wall_ms"],
        "important_limit": "The single frozen random seed also hits oracle+2% at trial 1; this pool supports a third-network mechanism result, not a random-baseline superiority claim.",
    })

    table = "\n".join(
        f"| {item['family_id']} | {item['mode']} | {item['bytes']:,} | {item['calls']:,} | {item['latency_median_ms']:.3f} | {item['regret_percent']:.2f}% |"
        for item in sorted(candidate_table, key=lambda x: (x["family_id"], x["mode"]))
    )
    pair_table = "\n".join(
        f"| {item['family_id']} | {item['dma_bytes_reduction_percent']:.2f}% | {item['dma_calls_reduction_percent']:.2f}% | {item['latency_improvement_percent']:.2f}% |"
        for item in paired
    )
    policy_table = "\n".join(
        f"| {item['policy']} | {item['trials_to_exact_oracle'] or 'not reached'} | {item['trials_to_oracle_2pct'] or 'not reached'} |"
        for item in policies
    )
    (args.output / "README.md").write_text(f"""# P7R378：Y08 冻结全池分析

Y08 是 YOLOv3-tiny conv12（CI=512、CO=1024、H=W=13、K=3）。P7R357 在读取任何 Y08
FPGA 正确性或 latency 前冻结了六候选和五个顺序；P7R377 在 boot
`{board_summary['boot_id']}` 上得到 6/6 三 seed 正确、42/42 计时正确，完整池 oracle 为
`{oracle:.6f} ms`。

| family | mode | logical DMA bytes | DMA calls | median latency (ms) | regret |
|---|---|---:|---:|---:|---:|
{table}

## 同 tile 驻留消融

| family | DMA bytes reduction | DMA calls reduction | latency improvement |
|---|---:|---:|---:|
{pair_table}

三个 input-stationary 配对全部加速（3/3），但加速幅度从 3.70% 到 36.22%。F02 与 F06 的
input-stationary 总 DMA bytes 都是 61,761,024 B，延迟仍为 133.240 ms 与 132.920 ms；因此
DMA bytes 是有效的冷启动优先级，不是精确 latency 模型。F01 只降低 1.30% bytes，却因请求数
下降 92.31% 得到 27.71% 加速，也说明请求粒度会影响 bytes 与时间之间的映射。

## 冻结顺序回放

| policy | trials to exact oracle | trials to oracle+2% |
|---|---:|---:|
{policy_table}

bytes、calls 和 Pareto 顺序都在第一次测量得到精确 oracle。不过，预注册的单个 Random 顺序首先
测到 F02 input-stationary，也已进入 oracle+2% 带，只是到第 4 次才得到精确 oracle。因此本池不能
宣称本文方法在 success@2% 上优于随机；它提供的是第三个 YOLO 网络几何上的机制复现和
`exact-oracle time-to-target` 证据。这里的策略成本是对完整全池测量所得逐候选成本的事后前缀累计，
不是五套分别执行并计时的在线 tuner，不能写成独立 online wall-clock 实验。

## 口径

- 逻辑 DMA 来自最终 lowered VTA LOAD/STORE，不等同于物理 AXI burst。
- 每个通过候选的 admission 成本含三次 correctness 与七轮计时；每轮 time evaluator 实际执行两次，
  因而是 17 次 FPGA kernel invocation。
- 全池外层进程墙钟为 `{board_summary['process_wall_ms']/1000:.3f} s`，包含交叉编译、clean start、
  健康门、上传、分配、正确性和计时；不能按候选前缀精确分摊。
""")
    write(args.output / "artifact_hashes.json", {
        "artifacts": {path.name: sha(path) for path in args.output.iterdir() if path.is_file() and path.name != "artifact_hashes.json"},
        "source_sha256": sha(__file__),
    })


if __name__ == "__main__":
    main()
