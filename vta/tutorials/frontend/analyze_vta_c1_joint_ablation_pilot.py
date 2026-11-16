#!/usr/bin/env python3
"""Recompute the pilot's completion intervals and report its limited contrasts."""

import argparse
import json
import math
from pathlib import Path
import statistics

from run_vta_c1_joint_ablation_pilot import read, ref, verify
from thesis_e2e_ablation_protocol import window_metrics
from thesis_supplement_run import RunStore, write_new_json


NAMES = {"paper_style_baseline": "论文思想适配基线", "without_cpu_pool": "去 CPU 核池",
         "without_boundary": "去边界成本", "without_internal_features": "去内部 DMA 特征", "full": "完整联合模型"}


def resource_summary(rows, path):
    cpu = [i for i, (sid, _) in enumerate(path) if sid.startswith("cpu:")]
    vta = [i for i, (sid, _) in enumerate(path) if sid.startswith("vta:")]
    waits = [[r["stage{}_vta_mutex_wait_ms".format(i)] for i in vta] for r in rows]
    return {
        "max_cpu_service_ms": statistics.median(max(r["stage{}_ms".format(i)] for i in cpu) for r in rows),
        "vta_run_sum_ms": statistics.median(sum(r["stage{}_run_ms".format(i)] for i in vta) for r in rows),
        "vta_mutex_wait_sum_ms": (statistics.median(sum(values) for values in waits)
                                  if all(value is not None for values in waits for value in values) else None),
        "boundary_api_sum_ms": statistics.median(sum(r["stage{}_get_ms".format(i)] + r["stage{}_set_ms".format(i + 1)]
                                                     for i in range(len(path) - 1)) for r in rows),
    }


def analyze(directory):
    store = RunStore(directory)
    c = store.contract
    result = read(directory / "analysis/result.json")
    if result["status"] != "completed_single_boot_pilot" or result["performance_windows"] != 15:
        raise ValueError("pilot incomplete")
    qualification = read(directory / "analysis/qualification.json")
    if len(qualification["sessions"]) != 10:
        raise ValueError("qualification coverage incomplete")
    for session in qualification["sessions"]:
        if session["boot_id"] != result["boot_id"]:
            raise ValueError("qualification boot differs")
        verify(session["raw_result"])
        if session["rows"] != 6 or session["full_frontier_checks"] <= 0:
            raise ValueError("missing complete frontier checks")
        if session["mode"] == "pipeline":
            audit = session["worker_completion_audit"]
            if not all(audit[k] for k in ("queue_worker_binding", "global_vta_mutex", "fresh_completion", "store_before_finish")):
                raise ValueError("worker completion audit failed")
    sessions, resources = [], {key: [] for key in c["programs"]}
    for index, order in enumerate(c["performance_orders"]):
        for key in order:
            tag = "r{}_{}".format(index, key)
            path = directory / "raw/sessions" / tag
            metadata = read(path / "metrics.json")
            if metadata["boot_id"] != result["boot_id"]:
                raise ValueError("performance boot differs")
            rows = [json.loads(line) for line in verify(metadata["raw_result"]).read_text().splitlines()]
            rows.sort(key=lambda row: row["completion_index"])
            if len(rows) != 307 or [r["completion_index"] for r in rows] != list(range(307)):
                raise ValueError("window frame coverage differs")
            if any(r["frame_id"] != i or r["input_index"] != i % 3 for i, r in enumerate(rows)):
                raise ValueError("frame/input identity differs")
            metrics = window_metrics([r["completion_ms"] / 1000 for r in rows[5:]])
            ii_ms = metrics["ii_seconds"] * 1000
            if not math.isclose(ii_ms, metadata["ii_ms"], abs_tol=1e-9, rel_tol=0):
                raise ValueError("independent completion interval recomputation differs")
            script = (directory / "raw" / ("session_" + tag + ".sh")).read_text()
            if "--c1-memory-audit" in script or "C3_COMPLETION_TRACE=0" not in script or "VTA_QUEUE_DIAGNOSTICS=0" not in script:
                raise ValueError("diagnostic instrumentation leaked into performance")
            sessions.append({"round": index, "program": key, "ii_ms": ii_ms,
                             "raw_result": metadata["raw_result"], "window": metrics})
            resources[key].append(resource_summary(rows[7:], c["programs"][key]["path"]))
    summaries = {key: {"ii_ms_by_round": [r["ii_ms"] for r in sessions if r["program"] == key],
                       "ii_ms_median": statistics.median(r["ii_ms"] for r in sessions if r["program"] == key),
                       "resource_medians_ms": {field: (statistics.median(r[field] for r in resources[key])
                                                       if all(r[field] is not None for r in resources[key]) else None)
                                               for field in resources[key][0]}}
                 for key in c["programs"]}
    full = summaries[c["groups"]["full"]["selected_program"]]
    comparisons = {}
    for name, group in c["groups"].items():
        baseline = summaries[group["selected_program"]]
        comparisons[name] = {
            "ii_reduction_fraction": 1 - full["ii_ms_median"] / baseline["ii_ms_median"],
            "same_choice_as_full": group["selected_program"] == c["groups"]["full"]["selected_program"],
            "per_round_ii_reduction_fraction": [1 - f / b for f, b in zip(full["ii_ms_by_round"], baseline["ii_ms_by_round"])],
        }
    report = {"schema": "c1_joint_ablation_pilot_analysis_v1", "boot_id": result["boot_id"],
              "independent_boots": 1, "qualification_sessions": 10, "performance_windows": 15,
              "full_frontier_checks": sum(s["full_frontier_checks"] for s in qualification["sessions"]),
              "programs": summaries, "contrasts_to_full": comparisons, "sessions": sessions,
              "source_refs": [ref(directory / "contract.json"), ref(directory / "analysis/result.json"),
                              ref(directory / "analysis/qualification.json"), ref(__file__)],
              "limitations": c["limitations"]}
    write_new_json(directory / "analysis/comparison.json", report)
    lines = ["# 联合资源模型：同板消融试运行", "",
             "本轮完整模型所选方案相对基线的完成间隔中位数下降 {:.2%}；相对去边界成本所选方案下降 {:.2%}。仅描述本次统一图、输入和运行策略下的五个冻结方案。".format(
                 comparisons["paper_style_baseline"]["ii_reduction_fraction"], comparisons["without_boundary"]["ii_reduction_fraction"]), "",
             "同一份一次量化图、Tile 历史、输入、运行库与 runner。五个方案全部先通过串行/流水完整阶段输出检查，以及流水 worker 的互斥、fresh completion、STORE 完成依赖审计。", "",
             "主评分沿用此前 CPU 核池/边界模型对照，额外 DDR 下界统一关闭。内部 DMA 沿用二级筛选特征，不在阶段 run 外重复追加 DMA 时间。", "",
             "一独立启动内三轮轮换顺序；每窗口 307 帧，去掉 5 帧预热，再按原规则去掉窗口前 2 帧，以 299 个完成间隔计算 II。完成间隔越小越好。", "",
             "## 首选方案实测", "", "| 模型 | VTA段数 | 三轮II/ms | II中位数/ms | 完整模型相对该组II下降 |",
             "|---|---:|---|---:|---:|"]
    for name, group in c["groups"].items():
        key = group["selected_program"]
        data = summaries[key]
        count = sum(sid.startswith("vta:") for sid, _ in c["programs"][key]["path"])
        lines.append("| {} | {} | {} | {:.6f} | {:.2%} |".format(
            NAMES[name], count, ", ".join("{:.4f}".format(x) for x in data["ii_ms_by_round"]),
            data["ii_ms_median"], comparisons[name]["ii_reduction_fraction"]))
    lines.extend(["", "同一所选方案共用测量。完整模型与去内部项选择相同，其差值为0，不能识别内部项的独立性能贡献。",
                  "基线与去核池组同样选中相同方案。这是各自模型选择的比较，不是运行时人为关闭真实传输或CPU约束。", "",
                  "## 15/16/17 权衡", "", "| VTA终点 | CPU线程 | 三轮II/ms | II中位数/ms |", "|---:|---|---|---:|"])
    for end, key in ((15, "endpoint_15"), (16, "endpoint_16"), (17, "full_one_island")):
        data = summaries[key]
        lines.append("| {} | 4,2 | {} | {:.6f} |".format(end, ", ".join("{:.4f}".format(x) for x in data["ii_ms_by_round"]), data["ii_ms_median"]))
    lines.extend(["", "内部搬运更少并不自动意味着全流水线更快；这里固定首尾线程，比较真实联合执行结果。", "",
                  "## 资源服务观测", "",
                  "以下先逐窗口取帧中位数，再跨三轮取中位数。CPU列为每帧最慢CPU阶段的服务时间，VTA列为每帧所有VTA段run之和；不把并行阶段简单相加解释为II。", "",
                  "| 程序 | 最慢CPU阶段/ms | VTA累计run/ms | 边界API合计/ms | VTA互斥等待合计/ms |",
                  "|---|---:|---:|---:|---:|"])
    for key, data in summaries.items():
        loads = data["resource_medians_ms"]
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
            key, loads["max_cpu_service_ms"], loads["vta_run_sum_ms"], loads["boundary_api_sum_ms"],
            "{:.4f}".format(loads["vta_mutex_wait_sum_ms"]) if loads["vta_mutex_wait_sum_ms"] is not None else "未记录"))
    lines.extend(["", "各阶段中位数与窗口平均完成间隔属于不同统计口径，不作为可加公式或严格下界。"])
    if (summaries["three_islands"]["ii_ms_median"] > full["ii_ms_median"]
            and summaries["three_islands"]["resource_medians_ms"]["vta_run_sum_ms"] < full["resource_medians_ms"]["vta_run_sum_ms"]):
        lines.extend(["", "三段VTA方案的累计VTA run更少，但完成间隔更差；完整方案承担更多VTA工作后改善了整体结果。这支持围绕流水线瓶颈联合选择的解释，不能仅以VTA时间或DMA字节单独选点。"])
    lines.extend(["", "## 验证和范围", "",
                  "- {} 个完整阶段张量检查；10 个正确性会话；15 个性能窗口全部按原始完成时间独立复算一致。".format(report["full_frontier_checks"]),
                  "- 三轮来自同一 kernel boot，不能视为三次独立启动，不计算跨启动置信区间。",
                  "- 只有一个真实图像及两个诊断输入循环，未验证任务准确率或输入分布泛化。",
                  "- 本轮是五个冻结方案的Top-1及邻近切点工程试运行，不是全候选域Top-5搜索收益实验。",
                  "- 原组件成本模型沿用既有校准；执行统一量化图的对应切点，不声称该校准已在新图上重新验证。",
                  "- 论文基线是最大负载思想的平台评分适配，不是完整原算法复现。", ""])
    (directory / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({name: {"ii_ms_median": summaries[group["selected_program"]]["ii_ms_median"],
                             **comparisons[name]} for name, group in c["groups"].items()}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    analyze(parser.parse_args().run_dir.resolve())
