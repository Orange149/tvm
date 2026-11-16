#!/usr/bin/env python3
"""Six-cell static ablation with the existing solver and frozen calibration.

No board execution. Freeze all rankings before joining historical measurements.
"""

import argparse
import copy
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import solve_cpu_vta_pipeline_v1_p3 as p3
from compare_cpu_vta_pipeline_static_scores import INPUTS, comparison_record


CELLS = {
    "A": (False, "none"), "B": (False, "generic"), "C": (False, "actual"),
    "D": (True, "none"), "E": (True, "generic"), "F": (True, "actual"),
}
FIELDS = ("cpu_owner_ms", "vta_mutex_ms", "total_ms", "host_core_demand_ms")


def observations(measurement):
    stages = measurement["stages"]
    if [s["device"] for s in stages] != ["cpu", "vta", "cpu"]:
        raise ValueError("expected CPU-VTA-CPU calibration")
    return {
        "cpu_to_vta": {
            "logical_bytes": int(stages[0]["output_bytes"]),
            "cpu_owner_ms": float(stages[0]["get_ms"]),
            "vta_mutex_ms": float(stages[1]["set_ms"]),
            "host_core_demand_ms": float(stages[0]["get_process_cpu_ms"])
            + float(stages[1]["set_process_cpu_ms"]),
        },
        "vta_to_cpu": {
            "logical_bytes": int(stages[1]["output_bytes"]),
            "cpu_owner_ms": float(stages[2]["set_ms"]),
            "vta_mutex_ms": float(stages[1]["get_ms"]),
            "host_core_demand_ms": float(stages[1]["get_process_cpu_ms"])
            + float(stages[2]["set_process_cpu_ms"]),
        },
    }


def fit_generic(costs):
    """One pooled bytes/time slope, same grouped training data as actual model."""
    grouped = defaultdict(list)
    for measurement in costs["fit_stage_measurements"]:
        for direction, row in observations(measurement).items():
            grouped[(measurement["scheme"], direction)].append(row)
    points = []
    for (scheme, direction), rows in sorted(grouped.items()):
        if len({r["logical_bytes"] for r in rows}) != 1:
            raise ValueError("thread sweep changed tensor size")
        point = {field: statistics.median(r[field] for r in rows) for field in rows[0]}
        point.update(group=scheme, direction=direction)
        point["total_ms"] = point["cpu_owner_ms"] + point["vta_mutex_ms"]
        points.append(point)
    denominator = sum(p["logical_bytes"] ** 2 for p in points)
    if denominator <= 0:
        raise ValueError("singular generic model")
    return {
        "fit_policy": "through_origin_pooled_direction_group_median",
        "fit_points": points,
        "total_ms_per_byte": sum(p["logical_bytes"] * p["total_ms"] for p in points)
        / denominator,
        "host_core_ms_per_byte": sum(
            p["logical_bytes"] * p["host_core_demand_ms"] for p in points
        ) / denominator,
        "ownership": "single serialized communication service on accelerator; CPU owner zero",
        "holdout_consumed_for_fit": False,
    }


def transport_context(base, transport, generic):
    if transport not in ("none", "generic", "actual"):
        raise ValueError(transport)
    context = dict(base)
    if transport == "actual":
        return context
    overrides = {}
    for bid, boundary in base["boundary_costs"].items():
        size = boundary["logical_bytes"]
        total = size * generic["total_ms_per_byte"] if transport == "generic" else 0.0
        core = size * generic["host_core_ms_per_byte"] if transport == "generic" else 0.0
        row = dict(cpu_owner_ms=0.0, vta_mutex_ms=total, total_ms=total,
                   host_core_demand_ms=core)
        if not all(math.isfinite(v) and v >= 0 for v in row.values()):
            raise ValueError("invalid edge override")
        overrides[bid] = row
    context["boundary_component_overrides"] = overrides
    return context


def record(label, rank, context, cell):
    row = comparison_record(label, rank, context)
    row["cpu_pool_enabled"], row["transport_model"] = CELLS[cell]
    provenance = row["cost_provenance"]
    provenance["single_vta_policy"] = (
        "sum VTA run service and the selected transport model's serialized service"
    )
    provenance["boundary_ownership_models"] = (
        context["boundary_ownership_models"] if CELLS[cell][1] == "actual" else None
    )
    provenance["transport_model"] = CELLS[cell][1]
    for edge in row["boundaries"]:
        edge["ownership"] = "selected transport model; replacement, not additional cost"
    return row


def component_holdout(costs, base, generic):
    result = []
    for direction, observed in observations(costs["grouped_holdout_measurement"]).items():
        size = observed["logical_bytes"]
        model = base["boundary_ownership_models"][direction]
        # Core slopes of the actual model are in the frozen boundary table.
        reference = next(b for b in costs["boundary_costs"] if b["direction"] == direction)
        actual_core_rate = reference["host_core_demand_ms"] / reference["logical_bytes"]
        actual_cpu = size * model["cpu_owner_ms_per_byte"]
        actual_vta = size * model["vta_mutex_ms_per_byte"]
        observed["total_ms"] = observed["cpu_owner_ms"] + observed["vta_mutex_ms"]
        predictions = {
            "none": dict.fromkeys(FIELDS, 0.0),
            "generic": dict(cpu_owner_ms=0.0, vta_mutex_ms=size * generic["total_ms_per_byte"],
                            total_ms=size * generic["total_ms_per_byte"],
                            host_core_demand_ms=size * generic["host_core_ms_per_byte"]),
            "actual": dict(cpu_owner_ms=actual_cpu, vta_mutex_ms=actual_vta,
                           total_ms=actual_cpu + actual_vta,
                           host_core_demand_ms=size * actual_core_rate),
        }
        result.append({"direction": direction, "observed": observed, "predictions": predictions,
                       "absolute_percentage_error": {
                           mode: {field: abs(pred[field] - observed[field]) / observed[field]
                                  for field in FIELDS} for mode, pred in predictions.items()
                       }})
    return {"scheme": costs["grouped_holdout_measurement"]["scheme"], "edges": result,
            "scope": "one grouped heldout calibration scheme; component accuracy only"}


def freeze_historical_scores(input_dir, output_dir, contexts):
    """Only compatibility metadata is read here; no measured candidate outcomes."""
    import evaluate_cpu_vta_pipeline_v1_p4b as p4b
    compatibility = p3.load_sealed_artifact(
        input_dir / "v1_p4_measured_pool_compatibility.json",
        "cpu_vta_pipeline_v1_p4_measured_pool_compatibility",
    )
    rows = []
    for row in p4b._semantic_rows(compatibility):
        scores = {}
        for cell, (cpu, _) in CELLS.items():
            label = p3.label_from_path(row["path"], contexts[cell],
                                       scoring_mode="cpu_resource" if cpu else "baseline")
            scores[cell] = label.score
        rows.append(dict(candidate_id=row["execution_candidate_id"], scores=scores))
    frozen = p3.seal_artifact({
        "kind": "cpu_transport_historical_scores", "rows": rows,
        "compatibility_artifact_sha256": compatibility["artifact_sha256"],
        "candidate_outcomes_consumed": False,
        "current_runtime_policy_comparison_eligible": compatibility[
            "current_runtime_policy_comparison_eligible"],
    })
    p3.write_json(output_dir / "historical_scores_frozen.json", frozen)
    return frozen


def evaluate_historical(input_dir, frozen):
    import evaluate_cpu_vta_pipeline_v1_p4b as p4b
    labels = p3.load_sealed_artifact(input_dir / "v1_p4_historical_labels.json",
                                     "cpu_vta_pipeline_v1_p4_historical_labels")
    if labels["compatibility_artifact_sha256"] != frozen["compatibility_artifact_sha256"]:
        raise ValueError("historical compatibility hash mismatch")
    samples = defaultdict(list)
    for row in labels["rows"]:
        samples[row["mapped_execution_candidate_id"]].append(float(row["measured_cycle_ms"]))
    scores = {r["candidate_id"]: r["scores"] for r in frozen["rows"]}
    measured = {key: statistics.median(samples[key]) for key in scores}
    optimum = min(measured.values())
    results = {}
    for cell in CELLS:
        order = sorted(scores, key=lambda key: (scores[key][cell], key))
        results[cell] = {
            "ranked_candidate_ids": order, "selected_top1_measured_cycle_ms": measured[order[0]],
            "best_top5_measured_cycle_ms": min(measured[key] for key in order[:5]),
            "top1_ii_regret": measured[order[0]] / optimum - 1,
            "first_candidate_within_2_percent_of_pool_oracle": next(
                i for i, key in enumerate(order, 1) if measured[key] <= optimum * 1.02),
            "spearman": p4b._spearman([scores[key][cell] for key in sorted(scores)],
                                       [measured[key] for key in sorted(scores)]),
            "mae_ms": statistics.mean(abs(scores[key][cell] - measured[key]) for key in scores),
        }
    return {"kind": "cpu_transport_historical_diagnostic", "candidate_count": len(scores),
            "pool_oracle_cycle_ms": optimum, "cells": results,
            "frozen_scores_artifact_sha256": frozen["artifact_sha256"],
            "historical_labels_artifact_sha256": labels["artifact_sha256"],
            "formal_current_runtime_validation": False,
            "limitations": "Legacy affinity unrecorded; zero current-runtime-qualified samples. "
            "Posthoc historical diagnostic only, not a new board experiment or heldout ranking test."}


def render(report):
    names = {"none": "无显式传输", "generic": "通用通信", "actual": "平台传输模型"}
    lines = ["# CPU 核池 × 传输成本：六组消融", "",
             "六组沿用同一 DP、合法域、阶段成本和线程范围；每组完整枚举核对 Top-5。",
             "A/B/C 关闭 CPU 核池，D/E/F 开启。DDR 额外约束在六组中统一关闭，单独贡献见此前三模式对照。", "",
             "通用通信将两方向的同批训练样本合成一个每字节系数，并作为 VTA 串行通信服务；平台模型保留方向及 CPU/VTA 归属。",
             "两者互相替换；关闭通信同时去掉边界 CPU 核时。阶段 run 内已有 DMA 保留。平台模型仍是校准预测，不是逐切点实测。", "",
             "| 组 | CPU核池 | 传输模型 | islands | CPU线程 | CPU阶段/ms | VTA服务/ms | CPU核池/ms | 预测II/ms |",
             "|---|---|---|---:|---|---:|---:|---:|---:|"]
    for cell, result in report["cells"].items():
        row = result["ranked_candidates"][0]
        load = row["resource_load_ms"]
        cpu, transport = CELLS[cell]
        lines.append("| {} | {} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            cell, "开" if cpu else "关", names[transport], row["vta_island_count"],
            row["cpu_stage_thread_parameters"], load["max_cpu_stage"],
            load["single_vta_plus_mutex_boundaries"], load["cpu_core_pool_lower_bound"],
            row["predicted_ii_ms"]))
    lines.extend(["", "不同组的预测 II 使用不同模型，不能相除宣称真实加速。", "", "## 固定另一因素后的选择变化", ""])
    for pair in report["comparisons"]:
        lines.append("- {}→{}：Top-1 {}，Top-5 重合 {}/5，顺序{}。".format(
            pair["from"], pair["to"], "改变" if pair["top1_changed"] else "不变",
            pair["top5_overlap"], "改变" if pair["ranking_changed"] else "不变"))
    lines.extend(["", "## 切点及 Top-5", ""])
    for cell, result in report["cells"].items():
        lines.append("### {}".format(cell))
        lines.append("")
        for row in result["ranked_candidates"]:
            lines.append("{}. `{}`，预测 II {:.6f} ms；瓶颈 {}。".format(
                row["rank"], row["candidate_id"], row["predicted_ii_ms"],
                ", ".join(row["active_bottlenecks"])))
        lines.append("")
    lines.extend(["## 分组留出：传输组件预测", "",
                  "一个独立留出拓扑、两条边；没有参与系数拟合。它只验证组件模型，不能代替整条流水线实测。", "",
                  "| 方向 | 实测边界总时间/ms | 通用预测/ms | 平台预测/ms | 通用误差 | 平台误差 |",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in report["component_holdout"]["edges"]:
        lines.append("| {} | {:.6f} | {:.6f} | {:.6f} | {:.2%} | {:.2%} |".format(
            row["direction"], row["observed"]["total_ms"],
            row["predictions"]["generic"]["total_ms"], row["predictions"]["actual"]["total_ms"],
            row["absolute_percentage_error"]["generic"]["total_ms"],
            row["absolute_percentage_error"]["actual"]["total_ms"]))
    if report.get("historical_diagnostic"):
        diagnostic = report["historical_diagnostic"]
        lines.extend(["", "## 历史公共池回放（诊断）", "",
                      "{} 个历史配置先冻结六组排序，再读取实测标签；重复配置取中位数。".format(diagnostic["candidate_count"]),
                      "这些旧数据没有当前运行策略资格，亲和性未记录，且此前已被研究使用。仅作事后诊断，不能当作独立留出或本轮上板结果。", "",
                      "| 组 | 选中Top-1的历史II/ms | Top-5内最优历史II/ms | Spearman | 首次进入池最优2%范围的序号 |",
                      "|---|---:|---:|---:|---:|"])
        for cell, row in diagnostic["cells"].items():
            lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
                cell, row["selected_top1_measured_cycle_ms"], row["best_top5_measured_cycle_ms"],
                row["spearman"], row["first_candidate_within_2_percent_of_pool_oracle"]))
    lines.extend(["", "默认 Top-20 回归、输入未变及逐组完整枚举检查：`{}`。".format(report["gate"]),
                  "", "本轮完成静态决策消融与组件/历史诊断，尚未测量六组新选择的真实流水线吞吐。", ""])
    return "\n".join(lines)


def run(input_dir, output_dir, default_reference):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    inputs = {name: p3.load_sealed_artifact(input_dir / name, kind) for name, kind in INPUTS.items()}
    hashes = {name: p3.sha256_file(input_dir / name) for name in INPUTS}
    costs = inputs["v1_local_cost_table.json"]
    base = p3.build_context(inputs["v1_profile_manifest.json"], costs)
    generic = fit_generic(costs)
    contexts = {cell: transport_context(base, transport, generic)
                for cell, (_, transport) in CELLS.items()}
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = p3.seal_artifact({
        "kind": "cpu_transport_ablation_protocol", "cells": CELLS, "input_sha256": hashes,
        "generic_model": generic, "top_k": 5, "extra_ddr_constraint_enabled": False,
        "actual_model": base["boundary_ownership_models"],
        "candidate_outcomes_consumed_for_fit_or_selection": False,
        "sources_sha256": {Path(path).name: p3.sha256_file(path)
                           for path in (__file__, p3.__file__)},
        "limitations": "paper-style platform scoring adaptation, not full original algorithm; "
        "actual is directional resource-owned calibrated prediction, not per-cut measurement",
    })
    p3.write_json(output_dir / "protocol.json", protocol)
    results, winners = {}, {}
    for cell, (cpu, transport) in CELLS.items():
        mode, context = "cpu_resource" if cpu else "baseline", contexts[cell]
        print("{} CPU={} transport={}: DP".format(cell, cpu, transport), flush=True)
        labels, stats = p3.solve_k_best_label_setting(context, 5, scoring_mode=mode)
        winners[cell] = labels[0].path
        print("{}: enumerate every legal configuration".format(cell), flush=True)
        start = time.monotonic()
        oracle, _, topology_count, count = p3.enumerate_oracle(
            inputs["v1_unit_and_boundary_schema.json"], context, 5, scoring_mode=mode)
        matches = [(x.path, x.score) for x in labels] == [(x.path, x.score) for x in oracle]
        if not matches or len(labels) != 5:
            raise RuntimeError("DP/oracle mismatch in {}".format(cell))
        result = p3.seal_artifact({
            "kind": "cpu_transport_ablation_cell", "cell": cell,
            "protocol_artifact_sha256": protocol["artifact_sha256"],
            "exact_top5_matches_enumeration": matches,
            "legal_topology_count": topology_count, "legal_execution_candidate_count": count,
            "search": stats, "enumeration_seconds": time.monotonic() - start,
            "ranked_candidates": [record(x, i, context, cell) for i, x in enumerate(labels, 1)],
        })
        results[cell] = result
        p3.write_json(output_dir / (cell + ".json"), result)
        print("{}: verified {}; top1 {} ({:.6f} ms)".format(
            cell, count, p3.candidate_id(labels[0].path), labels[0].score), flush=True)
    default_labels, _ = p3.solve_k_best_label_setting(base, 20)
    reference = json.loads(Path(default_reference).read_text())
    p3.write_json(output_dir / "default_before.json", reference)
    gate = {
        "default_top20_unchanged": reference["candidate_ids"] == [p3.candidate_id(x.path) for x in default_labels]
        and reference["scores"] == [x.score for x in default_labels],
        "inputs_unchanged": hashes == {name: p3.sha256_file(input_dir / name) for name in INPUTS},
        "all_six_exact_top5_match_enumeration": all(r["exact_top5_matches_enumeration"] for r in results.values()),
        "same_candidate_domain": len({(r["legal_topology_count"], r["legal_execution_candidate_count"])
                                      for r in results.values()}) == 1,
    }
    if not all(gate.values()):
        raise RuntimeError(gate)
    comparisons = []
    for before, after in (("A", "B"), ("B", "C"), ("D", "E"), ("E", "F"),
                          ("A", "D"), ("B", "E"), ("C", "F")):
        old = [r["candidate_id"] for r in results[before]["ranked_candidates"]]
        new = [r["candidate_id"] for r in results[after]["ranked_candidates"]]
        comparisons.append(dict(from_=before, to=after, top1_changed=old[0] != new[0],
                                ranking_changed=old != new, top5_overlap=len(set(old) & set(new))))
        comparisons[-1]["from"] = comparisons[-1].pop("from_")
    rescored = {}
    for selected_by, path in winners.items():
        rescored[selected_by] = {cell: p3.label_from_path(
            path, contexts[cell], scoring_mode="cpu_resource" if cpu else "baseline").score
            for cell, (cpu, _) in CELLS.items()}
    report = {"kind": "cpu_transport_ablation", "protocol_artifact_sha256": protocol["artifact_sha256"],
              "cells": results, "comparisons": comparisons, "gate": gate,
              "top1_cross_scores_ms": rescored,
              "component_holdout": component_holdout(costs, base, generic)}
    # Freeze the six full-domain rankings before reading historical outcomes.
    p3.write_json(output_dir / "static_results_frozen.json", p3.seal_artifact(copy.deepcopy(report)))
    if (input_dir / "v1_p4_measured_pool_compatibility.json").exists():
        frozen = freeze_historical_scores(input_dir, output_dir, contexts)
        report["historical_diagnostic"] = evaluate_historical(input_dir, frozen)
    p3.write_json(output_dir / "ablation.json", p3.seal_artifact(report))
    (output_dir / "REPORT.md").write_text(render(report), encoding="utf-8")
    print("Completed: {}".format(output_dir), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=p3.DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--default-reference", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.default_reference)
