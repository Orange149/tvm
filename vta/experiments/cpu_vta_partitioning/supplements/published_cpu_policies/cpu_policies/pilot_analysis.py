"""Descriptive, one-boot reporting for the short-window published-policy pilot."""
import csv
import math
from pathlib import Path
import statistics

from .pilot_campaign import _common, load_costs, validate_state, validated_screening


METHOD_NAMES = {"single_core": "Tarnawski式单线程物理核心映射适配",
                "prefix": "原前缀核组评分静态DP", "enhanced": "固定核组逐线程绑定增强DP"}
POLICY_NAMES = {"shared4": "四核共享", "reserved3": "三核计算、一核主机与控制"}


def _csv(path, rows):
    path = Path(path)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _configuration(config):
    stages = config["stages"]
    return {"configuration": config["id"],
            "resource_policy": "reserved3" if config["host_mask"] == [3] else "shared4",
            "resource_description": POLICY_NAMES["reserved3" if config["host_mask"] == [3] else "shared4"],
            "partition": " → ".join(f"{s['device'].upper()} {s['begin']}–{s['end'] - 1}" for s in stages),
            "cpu_threads": "/".join(str(s["threads"]) for s in stages if s["device"] == "cpu"),
            "cpu_groups": "/".join("{" + ",".join(map(str, s["mask"])) + "}" for s in stages if s["device"] == "cpu"),
            "host_cores": ",".join(map(str, config["host_mask"])),
            "control_cores": ",".join(map(str, config["control_mask"])),
            "affinity_mode": config["affinity_mode"], "execution_mode": config["execution_mode"]}


def _dispersion(root, item):
    api = _common()
    receipt = api.read(api.relative_file(root, item["summary"]["_receipt"]["path"]))
    files = [name for name in receipt["files"] if Path(name).name == "result.jsonl"]
    if len(files) != 1:
        raise ValueError("pilot report needs exactly one raw completion record")
    from cpu_vta_partitioning.measurement import frames
    doc = item["document"]
    rows = frames(api.relative_file(root, files[0]), doc["warmup"] + doc["scored"] + doc["tail"])
    kept = rows[doc["warmup"]:doc["warmup"] + doc["scored"]]
    gaps = [b["completion_ms"] - a["completion_ms"] for a, b in zip(kept, kept[1:])]
    half = len(gaps) // 2
    return {"gap_min_ms": min(gaps), "gap_max_ms": max(gaps),
            "gap_cv": statistics.pstdev(gaps) / statistics.mean(gaps),
            "first_part_ii_ms": statistics.mean(gaps[:half]),
            "last_part_ii_ms": statistics.mean(gaps[half:]),
            "performance_variation_is_rejection_gate": False}


def _spearman(rows):
    """Ranks have deterministic identity tie-breaking; no scipy dependency."""
    if len(rows) < 2:
        return None
    a = [float(row["static_rank"]) for row in rows]
    b = [float(row["measured_rank"]) for row in rows]
    av, bv = statistics.mean(a), statistics.mean(b)
    numerator = sum((x - av) * (y - bv) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - av) ** 2 for x in a) * sum((y - bv) ** 2 for y in b))
    return numerator / denominator if denominator else None


def analyze(root):
    root = Path(root).resolve()
    state = validate_state(root, validate_payload=False)
    api = _common(state["bundle"])
    costs = load_costs(root, state)
    screen = validated_screening(root, state)
    destination = root / "analysis"
    destination.mkdir(exist_ok=True)
    rankings = api.read(root / "rankings.json") if (root / "rankings.json").exists() else {"methods": {}}
    pool = api.read(root / "search_pool.json") if (root / "search_pool.json").exists() else {}
    measurements, measured_rows = {}, []
    if screen is not None:
        for key, item in screen["measurements"].items():
            summary = item["summary"]
            measurements[key] = {"measured_ms": summary["ii_ms"], "fps": 1000.0 / summary["ii_ms"],
                                 "process_core_ms_per_completion": summary["process_core_ms_per_completion"],
                                 "average_cpu_cores": summary["average_cpu_cores"],
                                 "p95_gap_ms": summary["p95_gap_ms"], "kept_timestamps": summary["kept_timestamps"],
                                 "completion_intervals": summary["gaps"], **_dispersion(root, item)}
            measured_rows.append({**_configuration(pool[key]), **measurements[key]})

    ranking_rows, top20, budget_rows, method_results, comparisons = [], [], [], {}, {}
    for method, document in rankings["methods"].items():
        rows = document["rankings"]
        available = [row for row in rows if row["id"] in measurements]
        actual_order = {row["id"]: i + 1 for i, row in enumerate(sorted(available, key=lambda row: (
                        measurements[row["id"]]["measured_ms"], row["id"])))}
        for i, row in enumerate(rows, 1):
            actual = measurements.get(row["id"])
            entry = {"method": METHOD_NAMES[method], "method_key": method, "static_rank": i,
                     **_configuration(row), "predicted_ms": row["score_ms"], "measured": actual is not None,
                     "measured_rank_within_measured_candidates": actual_order.get(row["id"], "")}
            if actual is not None:
                entry.update(actual)
                entry["relative_prediction_error"] = row["score_ms"] / actual["measured_ms"] - 1
                entry["rank_difference_within_measured_candidates"] = actual_order[row["id"]] - i
            ranking_rows.append(entry)
            if method == "enhanced" and actual is not None:
                top20.append({**entry, "measured_rank": actual_order[row["id"]]})
        selected = {}
        for budget in ((1, 5, 20) if method == "enhanced" else (1, 5)):
            candidates = rows[:budget]
            if not candidates or any(row["id"] not in measurements for row in candidates):
                continue
            best = min(candidates, key=lambda row: (measurements[row["id"]]["measured_ms"], row["id"]))
            entry = {"method": METHOD_NAMES[method], "method_key": method, "budget": budget,
                     "actual_candidates": len(candidates), "selected_static_rank": next(
                         i + 1 for i, row in enumerate(rows) if row["id"] == best["id"]),
                     **_configuration(best), "predicted_ms": best["score_ms"], **measurements[best["id"]]}
            selected[str(budget)] = entry
            budget_rows.append(entry)
        method_results[method] = {"budgets": selected,
                                  "static_top20_resource_counts": {policy: sum(
                                      ("reserved3" if r["host_mask"] == [3] else "shared4") == policy for r in rows)
                                      for policy in POLICY_NAMES}}
    for budget in ("1", "5"):
        enhanced = method_results.get("enhanced", {}).get("budgets", {}).get(budget)
        if enhanced:
            for baseline in ("single_core", "prefix"):
                other = method_results.get(baseline, {}).get("budgets", {}).get(budget)
                if other:
                    comparisons[baseline + "_budget" + budget] = {
                        "enhanced": enhanced["configuration"], "baseline": other["configuration"],
                        "budget": int(budget), "baseline_name": METHOD_NAMES[baseline],
                        "throughput_gain": other["measured_ms"] / enhanced["measured_ms"] - 1,
                        "same_configuration": enhanced["configuration"] == other["configuration"]}

    complete = state["status"] == "COMPLETE"
    if complete and (screen is None or set(measurements) != set(pool) or len(top20) != 20 or len(costs) != 514):
        raise ValueError("complete pilot lacks 514 costs or full measured enhanced Top-20")
    fastest = min(top20, key=lambda row: (row["measured_ms"], row["configuration"])) if top20 else None
    errors = sorted(top20, key=lambda row: (-abs(row["relative_prediction_error"]), row["configuration"]))
    report = {"status": "COMPLETE" if complete else "INCOMPLETE", "experiment": state["experiment"],
              "scope": "single_board_boot_single_fixed_input_short_window_descriptive_results",
              "boot_id": state["boot_id"], "independent_boots": 1 if state["boot_id"] else 0,
              "input_indices": [0], "rounds": 1, "significance_or_equivalence_claim": False,
              "performance_rejection_gates": [], "component_count": len(costs),
              "measured_pipeline_configurations": len(measurements), "methods": method_results,
              "comparisons_same_budget": comparisons,
              "enhanced_top20": top20, "enhanced_best": fastest,
              "enhanced_ranking": {"best_static_rank": fastest["static_rank"] if fastest else None,
                                   "spearman_with_identity_tie_breaking": _spearman(top20),
                                   "largest_absolute_relative_errors": errors[:5]},
              "model_assumptions": ["isolated_CPU_costs_reused_across_host_policies_not_concurrent_costs",
                                    "same_core_count_costs_inferred_for_other_masks",
                                    "producer_collector_queue_control_costs_not_separately_modeled"],
              "cpu_time_scope": "whole_process_between_16_scored_completions_divided_by_15_intervals",
              "budget20_scope": "additional_screening_budget_not_a_same_budget_advantage_claim"}
    components = []
    for key, item in sorted(costs.items()):
        config = item["configuration"]
        stage = config["stages"][0]
        components.append({"component": key, "program": stage["program"], "device": stage["device"],
                           "threads": stage["threads"], "mask": ",".join(map(str, stage["mask"])),
                           "host_cores": ",".join(map(str, config["host_mask"])),
                           "control_cores": ",".join(map(str, config["control_mask"])),
                           "source": item["kind"], **item["cost"]})
    api.write(destination / "analysis.json", report, replace=True)
    for name, rows in (("rankings.csv", ranking_rows), ("top20.csv", top20),
                       ("measured_configurations.csv", measured_rows), ("budgets.csv", budget_rows),
                       ("components.csv", components)):
        _csv(destination / name, rows)
    lines = ["ResNet18 成熟CPU执行机制与完整静态DP：首轮试跑",
             "状态：" + report["status"],
             f"同一次板子启动，固定输入0，各配置一轮；组件{len(costs)}项，完整流水{len(measurements)}个配置。",
             "组件预热2帧、计入8帧；流水预热8帧、保留16个完成时间戳、尾部4帧，计时15个完成间隔。",
             "未设置锚点漂移或时间波动拒收门槛；结果是本次短窗口观察，不作跨启动稳定收益、显著性或等效判断。"]
    for method in ("single_core", "prefix", "enhanced"):
        if method not in method_results:
            continue
        counts = method_results[method]["static_top20_resource_counts"]
        lines.append(f"{METHOD_NAMES[method]}静态前20：四核共享{counts['shared4']}项，三核计算、一核主机与控制{counts['reserved3']}项。")
        for budget, item in method_results[method]["budgets"].items():
            lines.append(f"{METHOD_NAMES[method]}，预算{budget}：{item['fps']:.3f} FPS，完成间隔{item['measured_ms']:.3f} ms；"
                         f"{item['resource_description']}；静态名次{item['selected_static_rank']}；{item['partition']}；"
                         f"CPU线程{item['cpu_threads']}，核组{item['cpu_groups']}。")
    for comparison in comparisons.values():
        suffix = "；选择同一配置，选点差异为零" if comparison["same_configuration"] else ""
        lines.append(f"预算{comparison['budget']}，增强DP相对{comparison['baseline_name']}的本轮吞吐变化："
                     f"{comparison['throughput_gain'] * 100:+.3f}%{suffix}。")
    if fastest:
        lines.append(f"增强方法20项全部实测；最快位于静态第{fastest['static_rank']}名，"
                     f"{fastest['fps']:.3f} FPS，采用{fastest['resource_description']}。")
    lines.extend(["前二十名优选是额外搜索预算，与对照前五名分开报告。",
                  "同核数不同核心集合采用同构核近似；控制线程编排开销未单独建模，整体进程核时包含这些工作。",
                  "原前缀模型保留门控关闭语义；增强模型同组串行，不能将两者全部差距归为评分公式。"])
    (destination / "REPORT_zh.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
