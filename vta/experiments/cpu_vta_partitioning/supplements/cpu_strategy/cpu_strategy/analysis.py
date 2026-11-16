"""Evidence-backed reporting for fixed-partition CPU configuration experiments."""
import csv
import math
from pathlib import Path
import statistics

from cpu_vta_partitioning.common import read, write
from .domain import matrix, components
from .evidence import Evidence


NAMES = {"local_fastest": "隔离阶段各自最快", "dp_static": "静态DP第一名", "dp_budget5": "静态DP前五名实测优选",
         "profile_disjoint": "依据组件预测的不重叠核组",
         "current": "当前四线程／两线程配置", "single_thread": "单线程固定物理核心参照",
         "exploration_best": "探索阶段冻结的整体优选"}


def aggregate(rounds):
    """Average rounds within a boot before giving each boot equal weight."""
    grouped = {}
    for row in rounds:
        for key, summary in row["measurements"].items():
            grouped.setdefault(key, {}).setdefault(row["boot_id"], []).append(summary)
    answer = {}
    for key, boots in grouped.items():
        per_boot = {}
        for boot, samples in boots.items():
            per_boot[boot] = {field: statistics.mean(float(s[field]) for s in samples)
                              for field in ("ii_ms", "process_core_ms_per_completion", "average_cpu_cores")}
            per_boot[boot]["rounds"] = len(samples)
        mean_ii = statistics.mean(v["ii_ms"] for v in per_boot.values())
        answer[key] = {"boots": per_boot, "ii_ms": mean_ii, "fps": 1000 / mean_ii,
                       "core_ms_per_completion": statistics.mean(v["process_core_ms_per_completion"] for v in per_boot.values()),
                       "average_cpu_cores": statistics.mean(v["average_cpu_cores"] for v in per_boot.values())}
    return answer


def comparison(values, selected, reference):
    a, b = values[selected], values[reference]
    if set(a["boots"]) != set(b["boots"]):
        raise ValueError("paired comparison requires identical boot coverage")
    gains = {boot: b["boots"][boot]["ii_ms"] / a["boots"][boot]["ii_ms"] - 1
             for boot in sorted(a["boots"])}
    return {"selected": selected, "reference": reference, "same_configuration": selected == reference,
            "paired_throughput_gains": gains, "mean_paired_gain": statistics.mean(gains.values()),
            "range": [min(gains.values()), max(gains.values())],
            "ii_reduction": 1 - a["ii_ms"] / b["ii_ms"],
            "all_boots_faster": all(x > 0 for x in gains.values()), "formal_significance": False}


def checked_costs(root, state):
    path = root / "costs.json"
    if not path.exists():
        return None
    document = read(path)
    evidence = Evidence(root, state["bundle_id"], state["bundle"])
    for signature, blocks in document["blocks"].items():
        if not blocks:
            raise ValueError("component has no accepted blocks")
        for block in blocks:
            evidence.summary(block["summary"], block["document"], block["boot_id"])
        expected = document["costs"][signature]
        for field, value in expected.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                actual = statistics.mean(float(b["summary"][field]) for b in blocks)
                if not math.isclose(float(value), actual, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("component table differs from accepted native summaries")
    return document


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(destination, exploration, cost_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    table = matrix()
    if exploration:
        keys = sorted(exploration, key=lambda key:(exploration[key]["ii_ms"],key))
        labels = []
        for key in keys:
            stages = table["configs"][key]["stages"]
            labels.append(f"{stages[0]['threads']}/{stages[2]['threads']} "
                          f"{stages[0]['mask']}/{stages[2]['mask']}")
        fig, ax = plt.subplots(figsize=(9, max(3,len(keys)*.3)))
        ax.barh(range(len(keys)),[exploration[key]["fps"] for key in keys])
        ax.set(yticks=range(len(keys)),yticklabels=labels,
               xlabel="Frames/s",title="Measured candidates only: static DP screening")
        ax.invert_yaxis()
        fig.tight_layout()
        for extension in ("png", "svg"): fig.savefig(destination / ("measured_cpu_configurations." + extension), dpi=160)
        plt.close(fig)
    diagnostic = [r for r in cost_rows if r["role"] == "diagnostic"]
    if diagnostic:
        programs = sorted({r["program"] for r in diagnostic})
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for program in programs:
            rows = sorted((r for r in diagnostic if r["program"] == program), key=lambda r:r["threads"])
            for ax, field in zip(axes, ("run_ms", "run_core_ms")):
                base = rows[0][field]
                ax.plot([r["threads"] for r in rows], [r[field] / base for r in rows], marker="o", label=program)
                ax.set(xticks=range(1,5), xlabel="Configured threads", ylabel="Ratio to one thread")
                ax.grid(alpha=.3)
        axes[0].set_title("Isolated computation wall time")
        axes[1].set_title("Isolated process CPU work")
        axes[1].legend(fontsize=8)
        fig.suptitle("Current compiled units; diagnostic only")
        fig.tight_layout()
        for extension in ("png", "svg"):fig.savefig(destination / ("unit_wall_core_tradeoff." + extension), dpi=160)
        plt.close(fig)


def analyze(directory):
    from .campaign import validate_state, validated_rounds, validated_ranking
    root = Path(directory).resolve()
    state = validate_state(root)
    rounds = validated_rounds(root)
    dp_ranking = validated_ranking(root)
    costs = checked_costs(root, state)
    exploration = aggregate([r for r in rounds if r["phase"] == "exploration"])
    confirmation = aggregate([r for r in rounds if r["phase"] == "confirmation"])
    selection = read(root / "selection.json") if (root / "selection.json").exists() else None
    method_rows, comparisons, cost_rows = [], {}, []
    if costs:
        roles = {r["stage"]["signature"]:r for r in components()}
        for key, values in sorted(costs["costs"].items()):
            if key not in roles: raise ValueError("unexpected component signature")
            item = roles[key]
            cost_rows.append({"signature":key,"role":item["role"],"program":item["stage"]["program"],
                              "threads":item["stage"]["threads"],"cores":','.join(map(str,item["stage"]["mask"])),
                              **{k:values[k] for k in ("service_ms","run_ms","run_core_ms","boundary_core_ms")}})
    if selection:
        named = {**selection["methods"], **selection["references"], "exploration_best":selection["exploration_best"]}
        for name, key in named.items():
            if key in confirmation:
                v = confirmation[key]
                method_rows.append({"method":NAMES.get(name,name),"configuration":key,
                                    "boots":len(v["boots"]),"ii_ms":v["ii_ms"],"fps":v["fps"],
                                    "core_ms_per_completion":v["core_ms_per_completion"],"average_cpu_cores":v["average_cpu_cores"]})
        pairs = {"combined_strategy_vs_local":("dp_budget5","local_fastest"),
                 "profile_disjoint_vs_local":("profile_disjoint","local_fastest"),
                 "dp_vs_profile":("dp_budget5","profile_disjoint"),
                 "budget5_vs_static":("dp_budget5","dp_static"),
                 "combined_strategy_vs_single_thread":("dp_budget5","single_thread"),
                 "frozen_best_vs_current":("exploration_best","current")}
        for label, (a,b) in pairs.items():
            if named[a] in confirmation and named[b] in confirmation:
                comparisons[label] = comparison(confirmation,named[a],named[b])
        for orientation in ("low","high"):
            a,b=(selection["interventions"][k+"_"+orientation] for k in ("separate","overlap"))
            if a in confirmation and b in confirmation:
                comparisons["placement_"+orientation] = comparison(confirmation,a,b)
    complete = state["status"] == "COMPLETE"
    if complete and (not confirmation or any(len(v["boots"]) != 3 or any(b["rounds"] != 2 for b in v["boots"].values())
                                              for v in confirmation.values())):
        raise ValueError("completed campaign lacks three paired two-round confirmation boots")
    recommendation = "INCOMPLETE"
    if complete:
        point = comparisons["frozen_best_vs_current"]
        recommendation = "DIRECTIONAL_IMPROVEMENT" if point["all_boots_faster"] else "NO_CONSISTENT_GAIN"
        if point["same_configuration"]: recommendation = "SAME_CONFIGURATION"
    report = {"status":"COMPLETE" if complete else "INCOMPLETE", "recommendation":recommendation,
              "scope":"fixed_resnet18_partition_stage_level", "formal_significance":False,
              "exploration":exploration,"confirmation":confirmation,"methods":method_rows,
              "comparisons":comparisons,"components":cost_rows,
              "selection":selection,"dp_ranking":dp_ranking,
              "cpu_time_scope":"whole_process_between_scored_completions",
              "limits":["32项是离线配置库，默认仅对固定预算候选和必要对照实测，不报告全网格最优。",
                        "组合复用线程预算、核心分组和阶段负载平衡策略；采用静态DP，不使用贝叶斯调优。",
                        "profile_disjoint是固定划分下最大阶段负载准则的适配，未复现Pipe-it阶段合并或逐线程绑核。",
                        "三次确认仅报告方向和量级，不报告显著性或等效性。",
                        "单元核时变化不自动解释完整流水吞吐变化。"]}
    destination = root / "analysis"
    destination.mkdir(exist_ok=True)
    write(destination / "analysis.json", report, replace=True)
    write_csv(destination / "methods.csv", method_rows)
    write_csv(destination / "component_costs.csv", cost_rows)
    write_csv(destination / "per_boot.csv", [{"configuration":key,"boot_id":boot,**v}
              for key,item in confirmation.items() for boot,v in item["boots"].items()])
    plots(destination,exploration,cost_rows)
    text = ["ResNet18 固定划分 CPU 策略实验", "状态："+report["status"],
            "CPU单元0–2 → VTA单元3–17 → CPU单元18–20；线程按阶段配置。",
            "组件实测后，复用通用核心子集容量模型，以精确DP冻结Top-20；仅测前五名与必要对照。",
            "32项为离线配置库，默认不作全量上板扫描。", ""]
    for row in method_rows:
        text.append(f"{row['method']}：{row['fps']:.4f}张/秒，完成间隔{row['ii_ms']:.4f}ms，"
                    f"每完成间隔CPU核时{row['core_ms_per_completion']:.4f}ms，{row['boots']}次独立启动。")
    for label, point in comparisons.items():
        gains = list(point["paired_throughput_gains"].values())
        text.append(f"{label}：逐启动吞吐变化"+' / '.join(f"{100*g:+.3f}%" for g in gains)
                    +("；选择相同配置。" if point["same_configuration"] else "。"))
    text.extend(["",*report["limits"]])
    (destination / "REPORT_zh.txt").write_text('\n'.join(text)+'\n')
    return report
