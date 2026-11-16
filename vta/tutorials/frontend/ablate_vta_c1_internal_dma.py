#!/usr/bin/env python3
"""Ablate exact internal DMA features with CPU and boundary cost models fixed.

Keep the existing M1 anchor, 2% band, Pareto implementation and Top-5 budget.
Do not remove DMA already implicit in run service or add it a second time.
"""

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path

import analyze_vta_c1_memory_features as memory
import freeze_vta_c1_unified_structured_order as unified
import shared_memory_topk as topk
import solve_cpu_vta_pipeline_v1_p3 as p3


BYTES = memory.BYTES
CALLS = ("vta_load_inp_calls", "vta_load_wgt_calls", "vta_load_acc_calls", "vta_store_out_calls")
SHAPE = ("vta_fragmented_dma_calls", "vta_load_request_density_per_mib",
         "vta_store_request_density_per_mib")
INTERNAL = BYTES + CALLS + SHAPE
CONTROL = tuple(d for d in unified.UNIFIED_DIMENSIONS if d not in INTERNAL)
GROUPS = {
    "off": (), "bytes": BYTES, "calls": CALLS, "shape": SHAPE,
    "bytes_calls": BYTES + CALLS, "internal_all": INTERNAL,
    "deployed_off": CONTROL, "deployed_full": unified.UNIFIED_DIMENSIONS,
}
NAMES = {"off": "无内部特征", "bytes": "内部字节", "calls": "内部请求数",
         "shape": "内部请求形态", "bytes_calls": "字节+请求数", "internal_all": "全部内部特征",
         "deployed_off": "现有完整规则去内部项", "deployed_full": "现有完整规则"}
ENDPOINTS = (15, 16, 17)
EPSILON = p3.EPSILON


def write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def validate_rows(rows):
    if not rows or len({r["candidate_id"] for r in rows}) != len(rows):
        raise ValueError("empty domain or duplicate identities")
    for row in rows:
        if not row["unified_feature_exact"]:
            raise ValueError("unknown compiler features cannot be zero filled")
        for dimension in unified.UNIFIED_DIMENSIONS:
            value = row[dimension]
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError("invalid feature: " + dimension)
        if not math.isfinite(row["m1_score_ms"]) or row["m1_score_ms"] <= 0:
            raise ValueError("invalid M1 service")


def frozen_orders(rows):
    validate_rows(rows)
    before = copy.deepcopy(rows)
    base = sorted(rows, key=lambda r: (r["m1_score_ms"], r["topology_id"], r["candidate_id"]))
    limit = base[0]["m1_score_ms"] * 1.02
    in_band = {r["candidate_id"] for r in base if r["m1_score_ms"] <= limit + EPSILON}
    results = {}
    for name, dimensions in GROUPS.items():
        ordered = topk.order_within_m1_band(rows, 0.02, dimensions)
        ids = [unified.configuration_id(r) for r in ordered]
        if len(ids) != len(set(ids)) or len(ids) != len(rows):
            raise ValueError("ranking changed candidate domain")
        if ordered[0]["candidate_id"] != base[0]["candidate_id"]:
            raise ValueError("M1 anchor changed")
        if [r["candidate_id"] for r in ordered if r["candidate_id"] not in in_band] != [
            r["candidate_id"] for r in base if r["candidate_id"] not in in_band
        ]:
            raise ValueError("out-of-band order changed")
        results[name] = {"dimensions": list(dimensions), "configuration_ids": ids,
                         "top5": ordered[:5], "selected_configuration_ids": ids[:5],
                         "pareto_layers_in_band": {
                             r["candidate_id"]: r.get("pareto_layer", 0)
                             for r in ordered if r["candidate_id"] in in_band},
                         "band_size_including_anchor": len(in_band)}
    if rows != before:
        raise ValueError("ranking mutated fixed cost/feature rows")
    return results


def validate_compiled_topology(topology, estimates):
    aggregate = Counter()
    for sid, _ in topology["path"]:
        if not sid.startswith("vta:"):
            continue
        row = estimates[sid]
        if row["status"] != "exact_reuse" or not row["usable_for_ranking"]:
            raise ValueError("unqualified DMA signature: " + sid)
        aggregate.update(row["total_dma"])
    for field, value in topology["vta_logical_dma"].items():
        if aggregate[field] != value:
            raise ValueError("compiled DMA/domain mismatch: " + field)


def endpoint_rows(topologies, estimates, context):
    by_topology = {t["topology_id"]: t for t in topologies}
    result = []
    for prefix_threads in p3.CPU_THREAD_CHOICES:
        for suffix_threads in p3.CPU_THREAD_CHOICES:
            rows = []
            for end in ENDPOINTS:
                path = (("cpu:00:02", prefix_threads), ("vta:03:{:02d}".format(end), 1),
                        ("cpu:{:02d}:20".format(end + 1), suffix_threads))
                label = p3.label_from_path(path, context, scoring_mode="cpu_resource")
                topology = copy.deepcopy(by_topology[p3.topology_id(path)])
                topology.update(path=path, candidate_id=p3.candidate_id(path), m1_score_ms=label.score)
                row = memory.make_row(topology, [], estimates)
                row.update(endpoint=end, resource_load_ms=p3.candidate_record(label, 0, context)["resource_load_ms"])
                rows.append(row)
            orders = frozen_orders(rows)
            # Dominance is diagnostic; it does not bypass M1 or its anchor.
            pairwise = []
            for left, right in ((rows[0], rows[1]), (rows[1], rows[2]), (rows[0], rows[2])):
                pairwise.append({
                    "from_endpoint": left["endpoint"], "to_endpoint": right["endpoint"],
                    "feature_delta_to_minus_from": {d: right[d] - left[d] for d in unified.UNIFIED_DIMENSIONS},
                    "m1_delta_ms": right["m1_score_ms"] - left["m1_score_ms"],
                    "logical_boundary_delta_bytes": right["logical_boundary_bytes"] - left["logical_boundary_bytes"],
                    "internal_from_dominates_to": topk.dominates(left, right, INTERNAL),
                    "control_to_dominates_from": topk.dominates(right, left, CONTROL),
                    "full_from_dominates_to": topk.dominates(left, right, unified.UNIFIED_DIMENSIONS),
                    "full_to_dominates_from": topk.dominates(right, left, unified.UNIFIED_DIMENSIONS),
                })
            result.append({"cpu_threads": [prefix_threads, suffix_threads], "rows": rows,
                           "orders": {name: [next(r["endpoint"] for r in rows if unified.configuration_id(r) == key)
                                            for key in order["configuration_ids"]]
                                      for name, order in orders.items()},
                           "pairwise": pairwise})
    return result


def summarize_changes(orders):
    comparisons = []
    for before, after in (("off", "bytes"), ("off", "calls"), ("off", "shape"),
                          ("off", "bytes_calls"), ("off", "internal_all"),
                          ("deployed_off", "deployed_full")):
        old, new = orders[before]["selected_configuration_ids"], orders[after]["selected_configuration_ids"]
        comparisons.append({"from": before, "to": after, "top5_changed": old != new,
                            "top5_overlap": len(set(old) & set(new)),
                            "full_order_changed": orders[before]["configuration_ids"] != orders[after]["configuration_ids"]})
    return comparisons


def render(report):
    lines = ["# VTA 内部 DMA 感知消融", "",
             "固定 CPU 核池、方向/归属感知边界模型、copy 策略、阶段运行成本与线程范围。额外 DDR 下界统一关闭。",
             "关闭内部项指精确 DMA 特征不参与二级排序；阶段 run 中隐含的 DMA 成本保留，未重复叠加 DMA 时间。",
             "同一 CPU-resource 主评分先选每个拓扑的最优线程，再按原有首选保留、2% 近优带、Pareto 分层、Top-5 预算筛选。",
             "首选保留是固定规则，因此 Top-1 不变不能用作否定内部感知的证据。所有候选均保留；不做物理不可行剪枝。", "",
             "前六组只切换内部特征。最后两组在现有完整规则中移除/恢复内部项，固定其余结构化特征。", "",
             "## 当前候选域", "",
             "完整遍历 {} 个配置、{} 个拓扑；每拓扑最优线程均重新计算。".format(
                 report["current_domain"]["execution_configurations"], report["current_domain"]["topologies"]),
             "全部编译 DMA 特征与冻结段模型逐项核对。DP 前五个配置与独立枚举排序一致：{}。".format(report["gate"]["dp_top5_matches_exhaustive"]), "",
             "| 对照 | Top-5顺序变化 | 重合数 | 全域顺序变化 |", "|---|---|---:|---|"]
    for row in report["current_comparisons"]:
        lines.append("| {} → {} | {} | {}/5 | {} |".format(
            NAMES[row["from"]], NAMES[row["to"]], "是" if row["top5_changed"] else "否",
            row["top5_overlap"], "是" if row["full_order_changed"] else "否"))
    for name, order in report["current_orders"].items():
        lines.extend(["", "### {}".format(NAMES[name]), ""])
        for rank, row in enumerate(order["top5"], 1):
            lines.append("{}. `{}`；M1 {:.6f} ms；边界 {} B；DMA {} B / {} 请求；Pareto 层 {}。".format(
                rank, row["candidate_id"], row["m1_score_ms"], row["framework_materialization_bytes"],
                row["vta_internal_dma_bytes"], row["vta_dma_calls"], row.get("pareto_layer", "首选")))
    focus = next(row for row in report["endpoint_checks"] if row["cpu_threads"] == [4, 2])
    lines.extend(["", "## 15/16/17 同线程对照", "",
                  "固定前缀 CPU 4 线程、后缀 CPU 2 线程；其余 15 组线程组合同时保存在 JSON 中，未按结果挑线程。", "",
                  "| VTA段 | M1/ms | 逻辑边界/B | 框架物化/B | LOAD/B | STORE/B | LOAD次数 | STORE次数 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in focus["rows"]:
        lines.append("| 03..{} | {:.6f} | {} | {} | {} | {} | {} | {} |".format(
            row["endpoint"], row["m1_score_ms"], row["logical_boundary_bytes"], row["framework_materialization_bytes"],
            row["vta_load_bytes"], row["vta_store_bytes"], row["vta_load_calls"], row["vta_store_calls"]))
    lines.extend(["", "| 排序规则 | 三候选顺序 |", "|---|---|"])
    for name, order in focus["orders"].items():
        lines.append("| {} | {} |".format(NAMES[name], " → ".join(map(str, order))))
    changes = sum(row["orders"]["off"] != row["orders"]["internal_all"] for row in report["endpoint_checks"])
    lines.extend(["", "16 组同线程对照中，全部内部项相对关闭内部项改变 {} 组顺序。".format(changes),
                  "15→16 将 projection 移到 VTA：边界更少，内部 LOAD/STORE 更多；16→17 内部 DMA 不变、边界及 host 工作变化。",
                  "内部特征可使15在DMA维度支配16，但完整向量还包含边界/host，不能据此断言最终一定舍弃16。", "",
                  "## 历史实测回放（辅助诊断）", "",
                  "固定历史普通复制策略，仅比较相同配置的测量顺序。先冻结本轮排序，再重算三启动原始窗口。",
                  "历史评分与当前评分分开，旧标签不转移给当前线程配置；该池此前已被分析使用。", "",
                  "| 规则 | 首次进入池最优+2%的位次 | Regret@3 | Regret@5 |", "|---|---:|---:|---:|"])
    evaluation = report["historical_evaluation"]
    for name, metrics in evaluation["metrics"].items():
        if metrics["status"] != "complete":
            lines.append("| {} | 标签不全 | — | — |".format(NAMES[name]))
            continue
        curve = metrics["curves"]
        lines.append("| {} | {} | {:.2%} | {:.2%} |".format(
            NAMES[name], metrics["first_oracle_plus_2pct_index"], curve[2]["regret"], curve[4]["regret"]))
    lines.extend(["", "各启动首次命中位次：", ""])
    for boot in evaluation["per_boot"]:
        lines.append("- boot {}：{}。".format(boot["boot_index"], "；".join(
            "{}={}".format(NAMES[name], metric["first_oracle_plus_2pct_index"])
            for name, metric in boot["metrics"].items())))
    lines.extend(["", "## 限制与验证", ""] + ["- " + item for item in report["limitations"]])
    lines.extend(["", "检查：`{}`。".format(report["gate"]), ""])
    return "\n".join(lines)


def run(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite " + str(output_dir))
    source_paths = [Path(__file__), Path(memory.__file__), Path(memory.board.__file__),
                    Path(unified.__file__), Path(topk.__file__),
                    Path(p3.__file__), unified.DEFAULT_TOPOLOGIES, unified.DEFAULT_EDGE_CONTEXTS,
                    unified.DEFAULT_FUSION_ESTIMATES, unified.DEFAULT_LEGACY_CONTRACT,
                    p3.DMA_QUALIFICATION] + [p3.DEFAULT_OUTPUT / name for name in (
                        "v1_unit_and_boundary_schema.json", "v1_profile_manifest.json", "v1_local_cost_table.json")]
    sources = [memory.snapshot(path) for path in source_paths]
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = unified.seal({
        "schema": "c1_internal_dma_ablation_protocol_v1", "groups": GROUPS, "band_fraction": 0.02,
        "budget": 5, "scoring_mode": "cpu_resource", "boundary_model": "directional_resource_owned",
        "extra_ddr_lower_bound": False, "realization": "ordinary_copy", "anchor_retained": True,
        "endpoint_threads": [[a, b] for a in p3.CPU_THREAD_CHOICES for b in p3.CPU_THREAD_CHOICES],
        "candidate_labels_consumed_for_current_selection": False, "source_hashes": sources,
    })
    write(output_dir / "protocol.json", protocol)
    topologies = topk.topology_diverse(topk.read_json(unified.DEFAULT_TOPOLOGIES))
    edges = topk.read_json(unified.DEFAULT_EDGE_CONTEXTS)
    estimates = {r["segment_id"]: r for r in topk.read_json(unified.DEFAULT_FUSION_ESTIMATES)["rows"]}
    for topology in topologies:
        topk.validate_topology(topology, edges)
        validate_compiled_topology(topology, estimates)
    print("Enumerating all CPU-resource thread configurations with fixed edge model", flush=True)
    rows, domain = memory.current_dp_rows(topologies, estimates, scoring_mode="cpu_resource")
    orders = frozen_orders(rows)
    profile = p3.load_sealed_artifact(p3.DEFAULT_OUTPUT / "v1_profile_manifest.json",
                                     "cpu_vta_pipeline_v1_profile_manifest")
    costs = p3.load_sealed_artifact(p3.DEFAULT_OUTPUT / "v1_local_cost_table.json",
                                   "cpu_vta_pipeline_v1_local_cost_table")
    context = p3.build_context(profile, costs)
    labels, _ = p3.solve_k_best_label_setting(context, 5, scoring_mode="cpu_resource")
    # Verify configuration Top-5 including thread variants, before deduplication.
    schema = p3.load_sealed_artifact(p3.DEFAULT_OUTPUT / "v1_unit_and_boundary_schema.json",
                                    "cpu_vta_pipeline_v1_unit_and_boundary_schema")
    print("Verifying DP Top-5 using the independent full enumeration oracle", flush=True)
    oracle, _, topology_count, execution_count = p3.enumerate_oracle(schema, context, 5,
                                                                   scoring_mode="cpu_resource")
    matches = [(x.path, x.score) for x in labels] == [(x.path, x.score) for x in oracle]
    if not matches or topology_count != domain["topologies"] or execution_count != domain["execution_configurations"]:
        raise ValueError("DP/domain enumeration mismatch")
    baseline_rows = sorted(rows, key=lambda r: (r["m1_score_ms"], r["topology_id"], r["candidate_id"]))
    if baseline_rows[0]["candidate_id"] != p3.candidate_id(labels[0].path):
        raise ValueError("topology winner differs from DP")
    endpoints = endpoint_rows(topologies, estimates, context)
    legacy_rows = [memory.make_row(t, [], estimates) for t in topologies]
    legacy_orders = frozen_orders(legacy_rows)
    frozen = unified.seal({"schema": "c1_internal_dma_orders_v1", "protocol_sha256": protocol["artifact_sha256"],
                           "current_domain": domain, "current_orders": orders,
                           "legacy_orders": legacy_orders, "endpoint_checks": endpoints,
                           "current_cost_and_feature_rows_sha256": unified.seal({"rows": rows})["artifact_sha256"],
                           "candidate_labels_joined": False})
    write(output_dir / "orders_frozen.json", frozen)
    write(output_dir / "current_rows.json", unified.seal({"rows": rows}))
    print("Orders frozen; replaying exactly matched archived board windows", flush=True)
    measurements, replay = memory.replay_board()
    evaluation = memory.evaluate_cohort(legacy_rows, legacy_orders, measurements, baseline_group="off")
    unchanged = all(topk.sha256_file(s["path"]) == s["sha256"] for s in sources + replay["sources"])
    gate = {"dp_top5_matches_exhaustive": matches, "source_hashes_unchanged": unchanged,
            "compiled_dma_vectors_match": True, "endpoint_thread_pairs_checked": len(endpoints) == 16,
            "no_dma_time_added_to_m1": all(
                row["m1_score_ms"] == next(original["m1_score_ms"] for original in rows
                                           if original["candidate_id"] == row["candidate_id"])
                for order in orders.values() for row in order["top5"])}
    if not all(gate.values()):
        raise ValueError(gate)
    report = unified.seal({
        "schema": "c1_internal_dma_ablation_v1", "new_board_calls": 0,
        "protocol_sha256": protocol["artifact_sha256"], "orders_file_sha256": topk.sha256_file(output_dir / "orders_frozen.json"),
        "current_domain": domain, "current_orders": orders, "current_comparisons": summarize_changes(orders),
        "endpoint_checks": endpoints, "historical_evaluation": evaluation, "historical_raw_replay": replay,
        "gate": gate, "limitations": [
            "本轮为静态筛选消融及历史回放；当前新线程配置没有匹配的上板时延，不报告吞吐提升。",
            "历史普通复制组只有5个已测拓扑，池最优不是全域最优，也不是新采集的前瞻验证。",
            "历史候选各自符合自己的串行参考，尚未证明来自同一份一次量化图；跨候选性能仅作描述性线索。",
            "精确DMA计数限定在既有冻结Tile/融合编译域，不能等同物理DDR流量或AXI burst。",
            "请求形态包含重叠的small/strided统计，是已有排序特征，不是额外唯一请求计数。",
            "Pareto层内沿用既有维度顺序做字典序排序，完整规则中的边界维度排在内部DMA之前。",
        ],
    })
    write(output_dir / "analysis.json", report)
    (output_dir / "REPORT.md").write_text(render(report), encoding="utf-8")
    print("Completed: " + str(output_dir), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
