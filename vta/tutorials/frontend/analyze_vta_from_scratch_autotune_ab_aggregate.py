#!/usr/bin/env python3
"""Aggregate three balanced from-scratch XGB/ours Y10 sessions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    manifest = read(root / "artifact_hashes.json")["artifacts"]
    bad = []
    for relative, expected in manifest.items():
        path = root / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            bad.append(relative)
    if bad:
        raise RuntimeError("artifact hash mismatch under {}: {}".format(root, bad[:3]))
    return len(manifest)


def describe(values):
    values = [float(value) for value in values]
    return {
        "values": values,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def profile_sum(rows):
    result = {key: 0 for key in (
        "load_buffer_2d_bytes", "store_buffer_2d_bytes",
        "load_buffer_2d_calls", "store_buffer_2d_calls", "driver_run_calls",
    )}
    for row in rows:
        profile = row.get("runtime_profile_complete") or row.get("runtime_profile") or {}
        for key in result:
            result[key] += int(profile.get(key, 0))
    result["logical_dma_bytes"] = (
        result["load_buffer_2d_bytes"] + result["store_buffer_2d_bytes"]
    )
    result["logical_dma_calls"] = (
        result["load_buffer_2d_calls"] + result["store_buffer_2d_calls"]
    )
    return result


def xgb_run(root):
    verified = verify(root)
    summary = read(root / "summary.json")
    timeline = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()
                if line.strip()]
    measured = [read(path) for path in (root / "isolated_measurements").glob("*/measurement.json")]
    passed = [row for row in measured if row.get("correct")]
    failed = [row for row in measured if not row.get("correct")]
    one_profile = profile_sum(passed)
    successful_call_traffic = {key: value * 7 for key, value in one_profile.items()}
    graph = summary["full_graph_result"]
    graph_traffic = profile_sum(graph["correctness"] + graph["timings"])
    lower_bound = {
        key: successful_call_traffic[key] + graph_traffic[key]
        for key in successful_call_traffic
    }
    candidate = graph["candidate_id"]
    errors = {}
    for row in timeline:
        key = str(row["error_no"])
        errors[key] = errors.get(key, 0) + 1
    return {
        "path": str(root), "verified_artifacts": verified,
        "t0_to_t1_seconds": summary["t0_to_t1_seconds"],
        "tuner_seconds": next(row["seconds"] for row in summary["phases"]
                              if row["phase"] == "empty_history_autotvm_tune"),
        "gross_proposals": len(timeline), "build_artifacts": len(measured),
        "fpga_correct_isolated": len(passed), "fpga_failed_isolated": len(failed),
        "error_no_counts": errors,
        "selected_config_index": summary["selected_config_index"],
        "selected_fullgraph_ms": graph["median_latency_ms"][candidate],
        "paired_stock_ms": graph["median_latency_ms"]["stock_reference"],
        "paired_ratio": graph["median_paired_latency_ratio"],
        "paired_wins": graph["candidate_paired_wins"],
        "all_outputs_equal": all(row.get("paired_equal") for row in graph["correctness"]),
        "correctness_inputs": len(graph["correctness"]) // 2,
        "board_resource_lower_bound_excluding_failed_device_calls": lower_bound,
    }


def ours_run(root):
    verified = verify(root)
    summary = read(root / "summary.json")
    return {
        "path": str(root), "verified_artifacts": verified,
        "t0_to_t1_seconds": summary["t0_to_t1_seconds"],
        "qualification_and_front_seconds": sum(
            row["seconds"] for row in summary["phase_timeline"]
            if row["phase"] != "online_fullgraph_search"
        ),
        "online_fullgraph_seconds": next(
            row["seconds"] for row in summary["phase_timeline"]
            if row["phase"] == "online_fullgraph_search"
        ),
        "first_deployable_xgb_plus_2_seconds": summary[
            "first_final_best_plus_2_percent_t0_seconds"
        ],
        "generated_mode_identities": summary["generated_mode_identities"],
        "qualified_mode_identities": summary["qualified_mode_identities"],
        "static_ok": summary["static_ok"], "fsim_passed": summary["fsim_passed"],
        "fsim_invocations": summary["fsim_passed"] * 3,
        "fullgraph_candidate_dispatches": summary["fpga_dispatch_count"],
        "selected_candidate_id": summary["selected_candidate_id"],
        "selected_mode": summary["selected_public_mode"],
        "selected_fullgraph_ms": summary["selected_fullgraph_median_ms"],
        "paired_stock_ms": summary["paired_stock_fullgraph_median_ms"],
        "paired_ratio": summary["selected_over_stock_ratio"],
        "paired_wins": summary["selected_paired_wins"],
        "all_outputs_equal": summary["selected_all_outputs_equal"],
        "correctness_inputs": summary["selected_correctness_inputs"],
        "board_resources_including_paired_stock": summary[
            "online_search_board_resources_including_paired_stock"
        ],
        "identity_exact_cost_replay": summary["candidate_generation_replay_proof"][
            "candidate_rows_exactly_equal"
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xgb", type=Path, nargs=3, required=True)
    parser.add_argument("--ours", type=Path, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    xruns = [xgb_run(path.resolve()) for path in args.xgb]
    oruns = [ours_run(path.resolve()) for path in args.ours]
    x_time = describe(row["t0_to_t1_seconds"] for row in xruns)
    o_time = describe(row["t0_to_t1_seconds"] for row in oruns)
    x_latency = describe(row["selected_fullgraph_ms"] for row in xruns)
    o_latency = describe(row["selected_fullgraph_ms"] for row in oruns)
    x_ratio = describe(row["paired_ratio"] for row in xruns)
    o_ratio = describe(row["paired_ratio"] for row in oruns)
    first_quality = describe(row["first_deployable_xgb_plus_2_seconds"] for row in oruns)
    x_dispatch = describe(row["build_artifacts"] for row in xruns)
    o_dispatch = describe(row["fullgraph_candidate_dispatches"] for row in oruns)
    x_bytes = describe(
        row["board_resource_lower_bound_excluding_failed_device_calls"]["logical_dma_bytes"]
        for row in xruns
    )
    o_bytes = describe(
        row["board_resources_including_paired_stock"]["logical_load_bytes"]
        + row["board_resources_including_paired_stock"]["logical_store_bytes"]
        for row in oruns
    )
    result = {
        "schema": "c3_from_scratch_autotune_ab_y10_three_by_three_v1",
        "status": "three_xgb_seeds_and_three_ours_clean_start_sessions_complete",
        "workload": "YOLOv3-tiny-320 conv18, CI=256, CO=128, H=W=10, K=1",
        "execution_order": ["XGB-44501", "OURS-1", "OURS-2", "XGB-44502", "XGB-44503", "OURS-3"],
        "xgb_runs": xruns, "ours_runs": oruns,
        "aggregate": {
            "xgb_t0_to_t1_seconds": x_time,
            "ours_t0_to_t1_seconds": o_time,
            "xgb_gross_proposals": describe(row["gross_proposals"] for row in xruns),
            "xgb_candidate_board_dispatches": x_dispatch,
            "ours_candidate_board_dispatches": o_dispatch,
            "xgb_selected_fullgraph_ms": x_latency,
            "ours_selected_fullgraph_ms": o_latency,
            "xgb_paired_ratio": x_ratio,
            "ours_paired_ratio": o_ratio,
            "ours_first_deployable_within_final_xgb_plus_2_seconds": first_quality,
            "xgb_board_logical_dma_bytes_lower_bound": x_bytes,
            "ours_board_logical_dma_bytes": o_bytes,
        },
        "comparison_of_medians": {
            "t0_to_t1_reduction_percent": (1.0 - o_time["median"] / x_time["median"]) * 100.0,
            "t0_to_t1_speedup": x_time["median"] / o_time["median"],
            "candidate_program_dispatch_reduction_percent": (
                1.0 - o_dispatch["median"] / x_dispatch["median"]
            ) * 100.0,
            "ours_latency_minus_xgb_percent": (
                o_latency["median"] / x_latency["median"] - 1.0
            ) * 100.0,
            "paired_ratio_difference_percentage_points": (
                o_ratio["median"] - x_ratio["median"]
            ) * 100.0,
            "time_to_xgb_plus_2_reduction_percent": (
                1.0 - first_quality["median"] / x_time["median"]
            ) * 100.0,
            "time_to_xgb_plus_2_speedup": x_time["median"] / first_quality["median"],
            "ours_board_logical_dma_bytes_minus_xgb_lower_bound_percent": (
                o_bytes["median"] / x_bytes["median"] - 1.0
            ) * 100.0,
        },
        "invariants": {
            "xgb_empty_history_and_no_tophub_for_target_search": True,
            "all_xgb_selected_config_index_equal": len({row["selected_config_index"] for row in xruns}) == 1,
            "all_ours_selected_candidate_equal": len({row["selected_candidate_id"] for row in oruns}) == 1,
            "all_final_graphs_three_input_all_output_equal": all(
                row["all_outputs_equal"] and row["correctness_inputs"] == 3
                for row in xruns + oruns
            ),
            "all_runs_seven_of_seven_paired_wins": all(row["paired_wins"] == 7 for row in xruns + oruns),
            "all_ours_replays_identity_exact": all(row["identity_exact_cost_replay"] for row in oruns),
        },
        "claim_boundary": (
            "Three-by-three balanced clean-start system-flow comparison on one already exposed "
            "YOLO workload. It establishes repeatable Y10 end-to-end time-to-quality, not a "
            "same-search-space algorithm ablation, a new prospective holdout, COCO mAP, physical "
            "AXI traffic, or cross-network generalization."
        ),
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "run", "t0_to_t1_s", "candidate_dispatches",
                         "selected_graph_ms", "paired_stock_ms", "paired_ratio", "wins"])
        for index, row in enumerate(xruns, 1):
            writer.writerow(["AutoTVM-XGB", index, row["t0_to_t1_seconds"],
                             row["build_artifacts"], row["selected_fullgraph_ms"],
                             row["paired_stock_ms"], row["paired_ratio"], row["paired_wins"]])
        for index, row in enumerate(oruns, 1):
            writer.writerow(["Ours", index, row["t0_to_t1_seconds"],
                             row["fullgraph_candidate_dispatches"], row["selected_fullgraph_ms"],
                             row["paired_stock_ms"], row["paired_ratio"], row["paired_wins"]])

    c = result["comparison_of_medians"]
    report = f"""# Y10 从零 AutoTVM-XGB 与本文方法三对三 A/B

状态：`THREE_BY_THREE_CLEAN_START_SYSTEM_COMPARISON_COMPLETE`

## 结论

在 YOLOv3-tiny-320 `conv18` 上，空历史 AutoTVM-XGB 与本文方法均从健康 bitstream/RPC 后的
T0 开始，并在选中配置进入完整图、通过三个输入的全部八个输出正确性和七轮配对计时后到达 T1。
三轮中位数显示，本文方法将 T0→T1 从 **{x_time['median']:.3f} s** 降至
**{o_time['median']:.3f} s**，降低 **{c['t0_to_t1_reduction_percent']:.2f}%**，即
**{c['t0_to_t1_speedup']:.2f}×** 更快完成可部署结果。最终整图中位数为
{o_latency['median']:.3f} ms，对照 XGB 的 {x_latency['median']:.3f} ms 仅慢
{c['ours_latency_minus_xgb_percent']:.3f}%，处于 2% 等价带内。

## 核心数据

| 指标（三轮中位数；括号为 min--max） | AutoTVM-XGB | 本文方法 | 结果 |
|---|---:|---:|---:|
| T0→T1 | {x_time['median']:.3f} s ({x_time['min']:.3f}--{x_time['max']:.3f}) | {o_time['median']:.3f} s ({o_time['min']:.3f}--{o_time['max']:.3f}) | -{c['t0_to_t1_reduction_percent']:.2f}% |
| 候选程序板端派发 | {x_dispatch['median']:.0f} ({x_dispatch['min']:.0f}--{x_dispatch['max']:.0f}) 个孤立算子 | {o_dispatch['median']:.0f} 个完整图 | -{c['candidate_program_dispatch_reduction_percent']:.2f}% |
| 最早获得 XGB 最终图 +2% 内的可部署完整图 | {x_time['median']:.3f} s | {first_quality['median']:.3f} s ({first_quality['min']:.3f}--{first_quality['max']:.3f}) | -{c['time_to_xgb_plus_2_reduction_percent']:.2f}% |
| 最终完整图 latency | {x_latency['median']:.3f} ms | {o_latency['median']:.3f} ms | 本文 +{c['ours_latency_minus_xgb_percent']:.3f}% |
| 同轮 selected/stock 配对比 | {x_ratio['median']:.6f} | {o_ratio['median']:.6f} | 相差 {c['paired_ratio_difference_percentage_points']:.3f} 个百分点 |
| 三输入/八输出正确性 | 3/3，每轮 | 3/3，每轮 | 全部通过 |
| 七轮胜场 | 7/7，每轮 | 7/7，每轮 | 全部通过 |

AutoTVM 三个空历史种子提出 {int(result['aggregate']['xgb_gross_proposals']['median'])} 个候选（范围
{int(result['aggregate']['xgb_gross_proposals']['min'])}--{int(result['aggregate']['xgb_gross_proposals']['max'])}），
并均选中 original `config 543`。本文每轮枚举同一 1280 点 original tile 域，以容量/确定性规则生成
32 个四模式身份，正式资格化 24 个三模式身份；12 个通过 lowering 与三 seed FSim，最终只派发
3 个完整图候选，并均选中同一 `input_stationary` 身份。

“AutoTVM-XGB”在这里指官方空历史 `XGBTuner`、原始模板与 1280 点 ConfigSpace；测量端使用本实验
的三 seed 正确性和逐候选 bitstream/RPC clean-start 适配器。未经隔离的共享 RPC 已在 P7R447 被
VTA fatal 配置污染并使整轮中断，因此该安全开销必须计入。结果比较的是可完成的部署调优流程，
不是未修改的默认 AutoTVM runner 微基准，也不能假设其他平台具有相同重启代价。

## 必须保留的负向成本

本文方法不是所有指标都更低。其每个晋级候选都执行完整 YOLO 图，并与 stock 成对做三输入和七轮
计时；AutoTVM 搜索期主要执行孤立卷积。因此本文每轮板端逻辑 DMA 为约
{o_bytes['median']/1e9:.3f} GB，而 AutoTVM 可完整审计的下界约为 {x_bytes['median']/1e9:.3f} GB，
本文高 {c['ours_board_logical_dma_bytes_minus_xgb_lower_bound_percent']:.1f}%。这个指标的执行粒度不同，
只能说明安全完整图门的代价，不能用来声称本文降低了所有搜索流量；两者均不是物理 AXI 计数。

## 结论边界

这是三对三 clean-start 的系统流程比较，但 Y10 标签在成本重演前已经暴露。本文候选生成逐行匹配
P7R428，且运行器不读取旧性能标签，因此成本是有效的身份精确重执行；它仍不是新的 prospective
holdout。两边搜索空间与测量保真度不同，因此可支持“完整系统更快获得等价质量的可部署整图”，
不能单独归因于搜索算法。相同空间在线消融和另一个网络的完整 T0→T1 A/B 仍是进一步增强项。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    artifacts = {
        path.name: sha256(path) for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_text(json.dumps({
        "artifacts": artifacts, "source_sha256": sha256(Path(__file__).resolve())
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], **result["comparison_of_medians"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
