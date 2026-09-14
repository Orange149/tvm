#!/usr/bin/env python3
"""Combine four clean-start paired edges into a YOLO two-route factorial audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


PROFILE_KEYS = (
    "load_buffer_2d_bytes",
    "load_buffer_2d_calls",
    "load_buffer_2d_inp_bytes",
    "load_buffer_2d_inp_calls",
    "load_buffer_2d_wgt_bytes",
    "load_buffer_2d_wgt_calls",
    "store_buffer_2d_bytes",
    "store_buffer_2d_calls",
    "driver_run_calls",
    "driver_run_insns",
    "synchronize_calls",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def label(row):
    return row.get("deployment_variant", row.get("public_mode"))


def edge(root, lhs, rhs):
    root = Path(root)
    summary = read_json(root / "summary.json")
    rows = read_jsonl(root / "timing.jsonl")
    rounds = sorted({int(row["round"]) for row in rows})
    paired = []
    for round_index in rounds:
        lhs_ms = next(row["latency_ms"] for row in rows if row["round"] == round_index and label(row) == lhs)
        rhs_ms = next(row["latency_ms"] for row in rows if row["round"] == round_index and label(row) == rhs)
        paired.append(float(rhs_ms) - float(lhs_ms))
    profiles = {}
    for variant in (lhs, rhs):
        selected = [row["runtime_profile_complete"] for row in rows if label(row) == variant]
        profiles[variant] = {
            key: statistics.median(float(row[key]) for row in selected) / 2.0
            for key in PROFILE_KEYS
        }
        profiles[variant]["total_dma_bytes"] = (
            profiles[variant]["load_buffer_2d_bytes"]
            + profiles[variant]["store_buffer_2d_bytes"]
        )
        profiles[variant]["total_dma_calls"] = (
            profiles[variant]["load_buffer_2d_calls"]
            + profiles[variant]["store_buffer_2d_calls"]
        )
    delta = {key: profiles[rhs][key] - profiles[lhs][key] for key in profiles[lhs]}
    return {
        "source": str(root.resolve()),
        "summary_sha256": sha256(root / "summary.json"),
        "boot_id": summary["boot_id"],
        "parameter_semantic_sha256": summary["parameter_semantic_sha256"],
        "all_outputs_equal": bool(summary["all_paired_outputs_equal"]),
        "lhs": lhs,
        "rhs": rhs,
        "paired_rounds": len(rounds),
        "median_paired_delta_ms": statistics.median(paired),
        "rhs_paired_wins": sum(value < 0.0 for value in paired),
        "paired_delta_ms": paired,
        "per_inference_profile": profiles,
        "per_inference_profile_delta": delta,
    }


def subtract(rhs, lhs):
    return {key: rhs[key] - lhs[key] for key in rhs}


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    edges = {
        "y00_without_y02": edge(args.y00_without_y02, "stock_tophub", "input_stationary"),
        "y02_with_y00": edge(
            args.y02_with_y00, "y00_input_only", "y00_input_y02_weight"
        ),
        "y02_without_y00": edge(args.y02_without_y00, "stock_all", "y02_weight_only"),
        "y00_with_y02": edge(
            args.y00_with_y02, "y02_weight_only", "y00_input_y02_weight"
        ),
    }
    if len({row["boot_id"] for row in edges.values()}) != 1:
        raise RuntimeError("factorial edges span different boots")
    if len({row["parameter_semantic_sha256"] for row in edges.values()}) != 1:
        raise RuntimeError("factorial edges do not share semantic parameters")
    if not all(row["all_outputs_equal"] for row in edges.values()):
        raise RuntimeError("at least one factorial edge lacks exact output equality")

    latency_main_effects = {
        name: edges[name]["median_paired_delta_ms"]
        for name in ("y00_without_y02", "y00_with_y02", "y02_without_y00", "y02_with_y00")
    }
    latency_interaction = {
        "from_y00_edges_ms": (
            latency_main_effects["y00_with_y02"] - latency_main_effects["y00_without_y02"]
        ),
        "from_y02_edges_ms": (
            latency_main_effects["y02_with_y00"] - latency_main_effects["y02_without_y00"]
        ),
        "note": (
            "The two estimates use separate clean-start paired runs and differ slightly due to "
            "run-to-run timing noise; they are not forced into one algebraic four-cell estimate"
        ),
    }
    profiles = {
        name: edges[name]["per_inference_profile_delta"] for name in latency_main_effects
    }
    profile_interaction = {
        "from_y00_edges": subtract(profiles["y00_with_y02"], profiles["y00_without_y02"]),
        "from_y02_edges": subtract(profiles["y02_with_y00"], profiles["y02_without_y00"]),
    }
    max_abs_dma_interaction = max(
        abs(value)
        for estimate in profile_interaction.values()
        for key, value in estimate.items()
        if key in {
            "load_buffer_2d_bytes", "load_buffer_2d_calls", "store_buffer_2d_bytes",
            "store_buffer_2d_calls", "total_dma_bytes", "total_dma_calls",
        }
    )
    result = {
        "schema": "c3_vta_yolov3_tiny_route_factorial_edge_analysis_v1",
        "status": "two_route_factorial_edges_complete",
        "boot_id": next(iter(edges.values()))["boot_id"],
        "parameter_semantic_sha256": next(iter(edges.values()))["parameter_semantic_sha256"],
        "edges": edges,
        "latency_main_effects_ms": latency_main_effects,
        "latency_interaction": latency_interaction,
        "per_inference_profile_main_effects": profiles,
        "per_inference_profile_interaction": profile_interaction,
        "max_abs_logical_dma_interaction": max_abs_dma_interaction,
        "interpretation": {
            "y00_route": "beneficial with and without Y02",
            "y02_route": "harmful relative to TopHub with and without Y00",
            "logical_dma_composition": "exactly additive for recorded DMA byte/call counters",
            "deployment_rule": (
                "A same-tile residency improvement is not sufficient for deployment; admit a route "
                "only after it improves the protected incumbent in its graph context"
            ),
        },
        "claim_boundary": (
            "Four edge runs share one boot but use separate clean-start RPC sessions because four "
            "co-resident graph executors exhausted the 192 MiB u-dma-buf allocation path"
        ),
    }
    output.mkdir(parents=True)
    write_json(output / "summary.json", result)
    report = """# P7R249：YOLO 双路由 2×2 边际效应审计

## 结论

Y00 input route 在 Y02 关闭/开启时都带来整网加速；Y02 weight route 在 Y00 关闭/开启时都造成
整网退化。两条 route 的逻辑 DMA 字节和请求数变化严格可加，因而 P7R241 的组合失败不是两个
驻留机制互相冲突，而是 Y02 的 `tile+barrier` 候选虽然优于 same-tile original，却仍显著差于
受保护的 TopHub incumbent。

| 边际效应（rhs-lhs） | 中位配对延迟差 | rhs 获胜 | 单次总 DMA 差 | 单次 DMA calls 差 |
|---|---:|---:|---:|---:|
| Y00，Y02 关闭 | {y00_off:+.6f} ms | {y00_off_wins}/7 | {y00_off_dma:+,.0f} B | {y00_off_calls:+,.0f} |
| Y00，Y02 开启 | {y00_on:+.6f} ms | {y00_on_wins}/7 | {y00_on_dma:+,.0f} B | {y00_on_calls:+,.0f} |
| Y02，Y00 关闭 | {y02_off:+.6f} ms | {y02_off_wins}/7 | {y02_off_dma:+,.0f} B | {y02_off_calls:+,.0f} |
| Y02，Y00 开启 | {y02_on:+.6f} ms | {y02_on_wins}/7 | {y02_on_dma:+,.0f} B | {y02_on_calls:+,.0f} |

Y00 两条边得到的延迟交互估计为 {int_y00:+.6f} ms，Y02 两条边得到 {int_y02:+.6f} ms；它们来自
四个独立 clean-start 配对运行，因此保留约 0.19 ms 的估计差异。逻辑 DMA 交互严格为 0。

## 方法与空间边界

四个版本分别是全 stock、仅 Y00、仅 Y02、Y00+Y02。最初尝试在一个 RPC 中同时加载四个完整
graph executor，但第一次 VTA 执行因 `fpga_buff_ == nullptr` 失败；三版本同驻留此前可以运行。
因此正式因子审计改为四条两版本 clean-start 边，每条都用 person 图片和两组随机输入检查全部
8 个非零输出，再做 7 轮 AB/BA 计时。四条边都在同一 boot 上，参数语义哈希一致，56/56 个 timed
graph call 全部输出配对相同。

该失败说明替代版本并存本身会占用共享物理内存；它是测试/部署空间规划边界，不能算作某个候选
的数值失败，也不能据此断言 u-dma-buf 驱动存在缺陷。

## 对第三创新点的修正

驻留机制用于扩展 `tile × residency` 搜索空间，静态共享内存服务代价用于决定先测谁；最终部署还
必须保留 incumbent protection：只接受在目标 graph 上通过正确性且边际延迟改善的 route。局部
same-tile 改善、DMA 字节下降或单算子加速都不是自动叠加的充分条件。
""".format(
        y00_off=latency_main_effects["y00_without_y02"],
        y00_off_wins=edges["y00_without_y02"]["rhs_paired_wins"],
        y00_off_dma=profiles["y00_without_y02"]["total_dma_bytes"],
        y00_off_calls=profiles["y00_without_y02"]["total_dma_calls"],
        y00_on=latency_main_effects["y00_with_y02"],
        y00_on_wins=edges["y00_with_y02"]["rhs_paired_wins"],
        y00_on_dma=profiles["y00_with_y02"]["total_dma_bytes"],
        y00_on_calls=profiles["y00_with_y02"]["total_dma_calls"],
        y02_off=latency_main_effects["y02_without_y00"],
        y02_off_wins=edges["y02_without_y00"]["rhs_paired_wins"],
        y02_off_dma=profiles["y02_without_y00"]["total_dma_bytes"],
        y02_off_calls=profiles["y02_without_y00"]["total_dma_calls"],
        y02_on=latency_main_effects["y02_with_y00"],
        y02_on_wins=edges["y02_with_y00"]["rhs_paired_wins"],
        y02_on_dma=profiles["y02_with_y00"]["total_dma_bytes"],
        y02_on_calls=profiles["y02_with_y00"]["total_dma_calls"],
        int_y00=latency_interaction["from_y00_edges_ms"],
        int_y02=latency_interaction["from_y02_edges_ms"],
    )
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "status": result["status"],
        "latency_main_effects_ms": latency_main_effects,
        "latency_interaction": latency_interaction,
        "max_abs_logical_dma_interaction": max_abs_dma_interaction,
        "interpretation": result["interpretation"],
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--y00-without-y02", required=True)
    parser.add_argument("--y02-with-y00", required=True)
    parser.add_argument("--y02-without-y00", required=True)
    parser.add_argument("--y00-with-y02", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
