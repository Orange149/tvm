#!/usr/bin/env python3
"""Compare three static objectives using the existing DP and enumeration oracle.

Read frozen P1/P2 inputs without advancing their execution state.  No compilation,
board execution, candidate performance labels, or Top-K postprocessing is used.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import time
from pathlib import Path

import solve_cpu_vta_pipeline_v1_p3 as p3


OBJECTIVES = {
    "baseline": ["max_cpu_stage", "single_vta_plus_mutex_boundaries"],
    "cpu_resource": [
        "max_cpu_stage", "single_vta_plus_mutex_boundaries", "cpu_core_pool_lower_bound",
    ],
    "current": [
        "max_cpu_stage", "single_vta_plus_mutex_boundaries", "cpu_core_pool_lower_bound",
        "shared_ddr_physical_demand",
    ],
}
INPUTS = {
    "v1_unit_and_boundary_schema.json": "cpu_vta_pipeline_v1_unit_and_boundary_schema",
    "v1_profile_manifest.json": "cpu_vta_pipeline_v1_profile_manifest",
    "v1_local_cost_table.json": "cpu_vta_pipeline_v1_local_cost_table",
}


def comparison_record(label, rank, context):
    record = p3.candidate_record(label, rank, context)
    record["scoring_mode"] = label.scoring_mode
    record["active_resource_terms"] = OBJECTIVES[label.scoring_mode]
    record["scores_by_mode_ms"] = {
        mode: dataclasses.replace(label, scoring_mode=mode).score for mode in p3.SCORING_MODES
    }
    record["active_bottlenecks"] = [
        name for name in OBJECTIVES[label.scoring_mode]
        if abs(record["resource_load_ms"][name] - label.score) <= p3.EPSILON
    ]
    return record


def render_report(report):
    lines = [
        "# 现有 DP 三种静态评分对照", "",
        "同一合法域、成本资料、切点与线程范围；只切换评分目标。",
        "未编译、未上板、未读取候选吞吐标签；未改变或执行后续拓扑/Pareto重排。", "",
        "baseline 是论文最大负载思想的平台评分适配，不是原论文通用算法的完整复现。",
        "DDR 项沿用已有混合流量估计和组件带宽，不是物理 DDR 计数器实测。", "",
        "## 逐模式 Top-K", "",
    ]
    for mode, result in report["modes"].items():
        lines.extend([
            "### {}".format(mode), "",
            "评分：max({})。".format(", ".join(OBJECTIVES[mode])), "",
            "| 排名 | 切点与线程 | islands | CPU阶段/ms | VTA累计/ms | CPU核池/ms | DDR/ms | 预测II/ms |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in result["ranked_candidates"]:
            loads = row["resource_load_ms"]
            lines.append(
                "| {} | `{}` | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                    row["rank"], row["candidate_id"], row["vta_island_count"],
                    loads["max_cpu_stage"], loads["single_vta_plus_mutex_boundaries"],
                    loads["cpu_core_pool_lower_bound"], loads["shared_ddr_physical_demand"],
                    row["predicted_ii_ms"],
                )
            )
        lines.extend([
            "", "DP/完整枚举一致：{}；拓扑 {}，线程配置 {}。".format(
                result["gate"]["exact_top_k_matches_enumeration"],
                result["legal_topology_count"], result["legal_execution_candidate_count"],
            ), "",
        ])
    lines.extend(["## 选择变化", ""])
    for pair in report["comparisons"]:
        lines.append("- {} → {}：Top-1 {}；Top-{} 顺序{}，候选重合 {}/{}。".format(
            pair["from"], pair["to"], "改变" if pair["top1_changed"] else "不变",
            report["top_k"], "改变" if pair["ranking_changed"] else "不变",
            pair["top_k_overlap_count"], report["top_k"],
        ))
        winner = report["modes"][pair["to"]]["ranked_candidates"][0]
        lines.append("  新模式首选的各模式评分/ms：{}；当前起作用的约束：{}。".format(
            json.dumps(winner["scores_by_mode_ms"], sort_keys=True),
            ", ".join(winner["active_bottlenecks"]),
        ))
    memory_pair = next(p for p in report["comparisons"] if p["from"] == "cpu_resource")
    if not memory_pair["ranking_changed"]:
        lines.extend(["", "在本次冻结域内，加入共享 DDR 项未改变所报告的 Top-K 决策。"])
    lines.extend([
        "", "静态预测不代表真实吞吐提升；CPU资源约束产生的变化单独归因，不能记作DDR收益。",
        "默认模式回归：{}。输入文件哈希保持不变：{}。".format(
            report["gate"]["default_mode_unchanged"], report["gate"]["inputs_unchanged"],
        ), "",
    ])
    return "\n".join(lines)


def compare(input_dir, output_dir, top_k=5, default_reference=None):
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("comparison output must be a new directory: {}".format(output_dir))
    inputs = {name: p3.load_sealed_artifact(input_dir / name, kind) for name, kind in INPUTS.items()}
    hashes = {name: p3.sha256_file(input_dir / name) for name in INPUTS}
    profile, costs = inputs["v1_profile_manifest.json"], inputs["v1_local_cost_table.json"]
    context = p3.build_context(profile, costs)
    output_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    current_labels = None
    for mode in p3.SCORING_MODES:
        print("{}: DP search".format(mode), flush=True)
        started = time.monotonic()
        labels, stats = p3.solve_k_best_label_setting(context, top_k, scoring_mode=mode)
        dp_seconds = time.monotonic() - started
        print("{}: full enumeration".format(mode), flush=True)
        started = time.monotonic()
        oracle, _, topology_count, execution_count = p3.enumerate_oracle(
            inputs["v1_unit_and_boundary_schema.json"], context, top_k, scoring_mode=mode
        )
        ids = [p3.candidate_id(item.path) for item in labels]
        oracle_ids = [p3.candidate_id(item.path) for item in oracle]
        gate = {
            "exact_top_k_matches_enumeration": ids == oracle_ids and all(
                abs(left.score - right.score) <= p3.EPSILON
                for left, right in zip(labels, oracle)
            ),
            "requested_top_k_returned": len(ids) == top_k,
            "no_duplicate_candidates": len(ids) == len(set(ids)),
        }
        if not all(gate.values()):
            raise RuntimeError("{} verification failed: {}".format(mode, gate))
        result = p3.seal_artifact({
            "kind": "cpu_vta_static_score_mode_comparison", "scoring_mode": mode,
            "objective": OBJECTIVES[mode], "gate": gate,
            "legal_topology_count": topology_count,
            "legal_execution_candidate_count": execution_count,
            "search": stats, "dp_wall_seconds": dp_seconds,
            "enumeration_wall_seconds": time.monotonic() - started,
            "enumeration_top_k_candidate_ids": oracle_ids,
            "ranked_candidates": [comparison_record(x, i, context) for i, x in enumerate(labels, 1)],
        })
        p3.write_json(output_dir / (mode + ".json"), result)
        results[mode] = result
        if mode == "current":
            current_labels = labels
        print("{}: verified {} configurations; top1 {}".format(mode, execution_count, ids[0]), flush=True)

    default_labels, _ = p3.solve_k_best_label_setting(context, top_k)
    default_unchanged = [(x.path, x.score) for x in default_labels] == [
        (x.path, x.score) for x in current_labels
    ]
    reference = None
    if default_reference is not None:
        payload = json.loads(Path(default_reference).read_text())
        p3.write_json(output_dir / "default_before.json", payload)
        reference = {"path": str(default_reference), "sha256": p3.sha256_file(default_reference)}
        reference["saved_snapshot"] = "default_before.json"
        reference["source_sha256"] = payload["source_sha256"]
        reference["matches_pre_change_top_k"] = (
            len(payload["candidate_ids"]) >= top_k
            and payload["candidate_ids"][:top_k] == [p3.candidate_id(x.path) for x in default_labels]
            and payload["scores"][:top_k] == [x.score for x in default_labels]
        )
        default_unchanged = default_unchanged and reference["matches_pre_change_top_k"]
    comparisons = []
    for before, after in itertools.combinations(p3.SCORING_MODES, 2):
        old = [row["candidate_id"] for row in results[before]["ranked_candidates"]]
        new = [row["candidate_id"] for row in results[after]["ranked_candidates"]]
        comparisons.append({
            "from": before, "to": after, "top1_changed": old[0] != new[0],
            "ranking_changed": old != new, "top_k_overlap_count": len(set(old) & set(new)),
        })
    gate = {
        "default_mode_unchanged": default_unchanged,
        "same_candidate_domain": len({
            (row["legal_topology_count"], row["legal_execution_candidate_count"])
            for row in results.values()
        }) == 1,
        "inputs_unchanged": hashes == {name: p3.sha256_file(input_dir / name) for name in INPUTS},
    }
    if not all(gate.values()):
        raise RuntimeError("comparison verification failed: {}".format(gate))
    report = p3.seal_artifact({
        "schema_version": 1, "kind": "cpu_vta_static_scoring_comparison", "top_k": top_k,
        "input_dir": str(input_dir.resolve()), "input_sha256": hashes,
        "input_artifact_sha256": {name: payload["artifact_sha256"] for name, payload in inputs.items()},
        "source_sha256": {"solver": p3.sha256_file(p3.__file__), "comparison": p3.sha256_file(__file__)},
        "dma_qualification": context["vta_dma_qualification"],
        "default_reference": reference, "gate": gate, "comparisons": comparisons, "modes": results,
        "scope": "existing DP and unchanged transitions; raw Top-K before any topology/Pareto "
        "postprocessing; baseline is a platform adaptation of the max-load objective; no board "
        "execution or measured candidate throughput; no runtime state updates",
    })
    p3.write_json(output_dir / "comparison.json", report)
    (output_dir / "REPORT.md").write_text(render_report(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=p3.DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--default-reference", type=Path, help="optional pre-change default DP snapshot")
    args = parser.parse_args()
    report = compare(args.input_dir, args.output_dir, args.top_k, args.default_reference)
    print(json.dumps({"gate": report["gate"], "comparisons": report["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
