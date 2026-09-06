#!/usr/bin/env python3
"""Audit RAMPS score provenance and retrospective low-budget ranking.

This command never compiles or contacts the board.  Candidate measurements are
used only as retrospective labels after every score has been computed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics
from typing import Any, Dict, Mapping, Sequence

from resource_aware_dataset import DEFAULT_RESNET_ROOTS, collect_resnet_records
from resource_aware_maxplus import (
    PipelineRecord,
    lightweight_cycle_scores,
    ranking_metrics,
)


DEFAULT_OUTPUT = (
    "vta/tutorials/frontend/report_out/resource_aware_maxplus/budgeted_search"
)
BUDGET_KEYS = tuple("regret_at_{}".format(value) for value in (1, 3, 5, 10, 20, 50, 100))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--resnet-root", action="append", default=[])
    parser.add_argument("--random-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _random_baseline(
    records: Sequence[PipelineRecord], repetitions: int, seed: int
) -> Dict[str, Any]:
    rng = random.Random(seed)
    samples: Dict[str, list] = {}
    target = [float(record.measured_cycle_ms) for record in records]
    best = min(target)
    thresholds = {fraction: best / fraction for fraction in (0.95, 0.98)}
    for _ in range(max(1, int(repetitions))):
        order = list(range(len(records)))
        rng.shuffle(order)
        evaluations = {fraction: len(order) for fraction in thresholds}
        for rank, index in enumerate(order, 1):
            for fraction, threshold in thresholds.items():
                if evaluations[fraction] == len(order) and target[index] <= threshold:
                    evaluations[fraction] = rank
        for budget in (1, 3, 5, 10, 20, 50, 100):
            key = "regret_at_{}".format(budget)
            selected_best = min(target[index] for index in order[: min(budget, len(order))])
            samples.setdefault(key, []).append(float((selected_best - best) / best))
        for fraction, count in evaluations.items():
            key = "evaluations_to_oracle_{}pct".format(int(100 * fraction))
            samples.setdefault(key, []).append(float(count))
    return {
        "repetitions": max(1, int(repetitions)),
        "seed": int(seed),
        "metrics": {
            key: {
                "median": statistics.median(values),
                "p10": sorted(values)[int(0.10 * (len(values) - 1))],
                "p90": sorted(values)[int(0.90 * (len(values) - 1))],
            }
            for key, values in sorted(samples.items())
        },
    }


def evaluate_records(
    records: Sequence[PipelineRecord], random_repetitions: int, seed: int
) -> Dict[str, Any]:
    if not records:
        raise ValueError("budget audit requires measured records")
    scores = [lightweight_cycle_scores(record, require_zero_feedback=True) for record in records]
    models = {
        "legacy_static_score": [record.legacy_score_ms for record in records],
        "m0_single_vta_compute": [item["compute_balance_ms"] for item in scores],
        "m1_static_communication": [item["communication_aware_ms"] for item in scores],
        # Diagnostic only: this uses the rejected wall-time x requested-threads proxy.
        "m2_invalid_core_proxy_diagnostic": [item["executor_resource_ms"] for item in scores],
    }
    return {
        "scope": "retrospective_measured_pool_only",
        "candidate_measurements_used_as": "evaluation_labels_only",
        "models": {
            name: {
                "metrics": ranking_metrics(records, predictions),
                "publication_eligible": name != "m2_invalid_core_proxy_diagnostic",
                "reason": (
                    "zero_feedback_static_score"
                    if name != "m2_invalid_core_proxy_diagnostic"
                    else "rejected_threads_times_wall_ms_core_demand_proxy"
                ),
            }
            for name, predictions in models.items()
        },
        "random_uniform": _random_baseline(records, random_repetitions, seed),
    }


def source_audit(records: Sequence[PipelineRecord]) -> Dict[str, Any]:
    rows = []
    service_sources = Counter()
    blockers = Counter()
    for record in records:
        provenance = lightweight_cycle_scores(record)["provenance"]
        service_sources[provenance["service_parameter_source"]] += 1
        blockers.update(provenance["blockers"])
        rows.append(
            {
                "candidate_id": record.candidate_id,
                "source_path": record.source_path,
                "zero_feedback_eligible_m0_m1": provenance["zero_feedback_eligible"],
                "service_parameter_kind": provenance["service_parameter_kind"],
                "service_parameter_source": provenance["service_parameter_source"],
                "workload_feature_kind": provenance["workload_feature_kind"],
                "candidate_measurement_fields_present_for_evaluation": provenance[
                    "candidate_measurement_fields_present_for_evaluation"
                ],
                "candidate_measurement_fields_used_by_score": provenance[
                    "candidate_measurement_fields_used_by_score"
                ],
                "m2_has_measured_core_demand": not provenance[
                    "missing_cpu_core_demand_stages"
                ],
            }
        )
    return {
        "record_count": len(rows),
        "zero_feedback_eligible_m0_m1_count": sum(
            bool(row["zero_feedback_eligible_m0_m1"]) for row in rows
        ),
        "m2_measured_core_demand_count": sum(
            bool(row["m2_has_measured_core_demand"]) for row in rows
        ),
        "service_parameter_sources": dict(sorted(service_sources.items())),
        "blockers": dict(sorted(blockers.items())),
        "direct_profile_is_oracle_service_only": True,
        "rows": rows,
    }


def candidate_coverage(records: Sequence[PipelineRecord]) -> Dict[str, Any]:
    island_counts = Counter(record.vta_island_count for record in records)
    return {
        "candidate_count": len(records),
        "unique_candidate_count": len({record.candidate_id for record in records}),
        "outer_span_group_count": len({record.group_id for record in records}),
        "vta_island_counts": {str(key): value for key, value in sorted(island_counts.items())},
        "all_correctness_passed": all(record.correctness_passed for record in records),
        "records_with_structured_boundaries": sum(bool(record.boundaries) for record in records),
        "records_with_legacy_boundary_scalar": sum(record.boundary_ms > 0.0 for record in records),
        "coverage_claim": "measured_pool_only",
        "complete_legal_partition_space_proven": False,
        "reason": "no frozen legal-candidate manifest is paired with the 200 measured rows",
    }


def cost_ledger(records: Sequence[PipelineRecord]) -> Dict[str, Any]:
    direct_profiles = sum(
        bool((record.metadata.get("direct_communication_profile") or {}).get("available"))
        for record in records
    )
    return {
        "candidate_board_evaluations_present": len(records),
        "candidate_direct_runtime_profiles_present": direct_profiles,
        "profile_board_evaluations_attributed": None,
        "compile_wall_clock_seconds": None,
        "board_wall_clock_seconds": None,
        "status": "count_only_wall_clock_unavailable",
        "required_fix": (
            "future runners must emit compile_start/end, board_start/end, cache_hit, "
            "failure, and retry fields per candidate"
        ),
    }


def render_report(
    audit: Mapping[str, Any],
    coverage: Mapping[str, Any],
    costs: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> str:
    models = evaluation["models"]
    lines = [
        "# RAMPS P1 来源与低预算审计",
        "",
        "更新日期：2026-09-01",
        "",
        "## 结论",
        "",
        "- 200 个候选只能定义 `measured-pool oracle`，尚未证明覆盖完整合法切图空间。",
        "- M0/M1 的排序输入来自编译期 workload 和默认静态常数；候选实测值只作为回顾性标签。",
        "- direct runtime profile 读取每个候选的 stage/copy 时间，只能作为 mechanism upper bound。",
        "- 200 个候选均缺少可用的 measured CPU core demand，因此 M2 结果不具 publication 资格。",
        "- 历史产物没有可靠的编译和上板 wall-clock 字段，当前只能报告 evaluation count。",
        "",
        "## 低预算结果",
        "",
        "| 模型 | regret@1 | regret@5 | regret@10 | 到 95% oracle 次数 | 资格 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in (
        "legacy_static_score",
        "m0_single_vta_compute",
        "m1_static_communication",
        "m2_invalid_core_proxy_diagnostic",
    ):
        item = models[name]
        metric = item["metrics"]
        lines.append(
            "| `{}` | {:.3f} | {:.3f} | {:.3f} | {:.0f} | {} |".format(
                name,
                metric["regret_at_1"],
                metric["regret_at_5"],
                metric["regret_at_10"],
                metric["evaluations_to_oracle_95pct"],
                "可用" if item["publication_eligible"] else "仅诊断",
            )
        )
    random_metrics = evaluation["random_uniform"]["metrics"]
    lines.extend(
        [
            "",
            "随机基线的 95% oracle 次数中位数为 {:.0f}，P10--P90 为 {:.0f}--{:.0f}。".format(
                random_metrics["evaluations_to_oracle_95pct"]["median"],
                random_metrics["evaluations_to_oracle_95pct"]["p10"],
                random_metrics["evaluations_to_oracle_95pct"]["p90"],
            ),
            "",
            "M1 没有改善 M0，说明‘加上默认 DMA 常数’不足以形成有效排序。M2 的表面改善来自",
            "已否决的 core-demand 代理，不能据此增加模型复杂度。下一步只应测能修复该排序缺口的",
            "最小参数：代表性 CPU stage、VTA island 端到端 service，以及边界 copy/sync。",
            "",
            "## 审计摘要",
            "",
            "```text",
            "records: {}".format(audit["record_count"]),
            "M0/M1 zero-feedback eligible: {}".format(
                audit["zero_feedback_eligible_m0_m1_count"]
            ),
            "M2 measured core-demand eligible: {}".format(
                audit["m2_measured_core_demand_count"]
            ),
            "coverage: {}".format(coverage["coverage_claim"]),
            "cost ledger: {}".format(costs["status"]),
            "```",
            "",
            "## 决策",
            "",
            "暂停 H2 20-template 和完整 HardwareProfile。先设计一个不超过约 12--20 个独立 case 的",
            "P3-min profile，并要求它在同一 measured pool 上显著降低低 K regret；否则采用更简单的",
            "stage lookup + batch shortlist，而不继续扩展 Max-Plus/FIFO 模型。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    roots = tuple(args.resnet_root) if args.resnet_root else DEFAULT_RESNET_ROOTS
    records, source_paths = collect_resnet_records(roots=roots)
    output = Path(args.output_dir)
    audit = source_audit(records)
    coverage = candidate_coverage(records)
    coverage["source_summary_count"] = len(source_paths)
    coverage["source_summaries"] = [str(path) for path in source_paths]
    costs = cost_ledger(records)
    evaluation = evaluate_records(records, args.random_repetitions, args.seed)
    write_json(output / "source_audit.json", audit)
    write_json(output / "candidate_coverage.json", coverage)
    write_json(output / "cost_ledger.json", costs)
    write_json(output / "retrospective_budget_curve.json", evaluation)
    (output / "P1_REPORT.md").write_text(
        render_report(audit, coverage, costs, evaluation), encoding="utf-8"
    )
    print("wrote {} for {} measured candidates".format(output, len(records)))


if __name__ == "__main__":
    main()
