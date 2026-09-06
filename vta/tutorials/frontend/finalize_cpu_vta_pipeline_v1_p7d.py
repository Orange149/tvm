#!/usr/bin/env python3
"""Freeze the admissible P7 component model and evaluate it without refitting.

P7D deliberately keeps component admission separate from candidate evaluation.
Only the three-boot streaming CPU-memory measurements replace an existing
parameter table.  Exact CPU-pair slowdowns and the single-boot P7B-2/P7C
observations remain evidence, but are not extrapolated across the search space.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from evaluate_resnet18_historical_affinity_models import (
    absolute_error_metrics,
    build_scores,
    evaluate_scores,
    score_path,
    _spearman,
)
from freeze_cpu_vta_pipeline_v1 import (
    load_sealed_artifact,
    repo_root,
    seal_artifact,
    validate_artifact_sha256,
)
from iterate_cpu_vta_pipeline_v1_p5b_iteration2 import (
    apply_atomic_cpu_costs,
    summarize_atomic_profiles,
)
from solve_cpu_vta_pipeline_v1_p3 import (
    build_context,
    candidate_id,
    candidate_record,
    enumerate_oracle,
    solve_k_best_label_setting,
)


DEFAULT_OUTPUT = repo_root() / "vta/tutorials/frontend/report_out/resource_aware_maxplus"
DEFAULT_PHYSICAL_PROFILE = DEFAULT_OUTPUT / "v1_p7_physical_profile.json"
DEFAULT_RANKED = DEFAULT_OUTPUT / "v1_p7d_ranked_candidates.json"
DEFAULT_VALIDATION = DEFAULT_OUTPUT / "v1_p7_formula_validation.json"
DEFAULT_REVIEW = DEFAULT_OUTPUT / "v1_p7d_review.md"
P7_PROTOCOL_ID = "cpu_vta_pipeline_v1_p7_physical_profile"
K_VALUES = (1, 3, 5, 10, 20)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_cpu_memory_profile(p7b1):
    if not p7b1.get("passed"):
        raise RuntimeError("P7B-1 three-boot reproducibility gate did not pass")
    if not p7b1.get("gate_checks", {}).get("three_unique_boots"):
        raise RuntimeError("P7B-1 does not contain three unique boots")

    rows = []
    keys = set()
    for result in p7b1["memory_results"]:
        case = result["case"]
        key = (
            str(case["pressure_class"]),
            str(case["operation"]),
            int(case["threads"]),
        )
        if key in keys:
            raise RuntimeError("duplicate P7B-1 memory key: {}".format(key))
        keys.add(key)
        rows.append(
            {
                "case_id": case["case_id"],
                "pressure_class": key[0],
                "operation": key[1],
                "threads": key[2],
                "target_active_working_set_bytes": int(
                    case["target_active_working_set_bytes"]
                ),
                "bandwidth_GBps_median": float(
                    result["three_boot_bandwidth_GBps_median"]
                ),
                "cross_boot_cv": float(result["cross_boot_cv"]),
                "source_boot_medians_GBps": result["per_boot_bandwidth_GBps_median"],
                "all_sessions_passed": bool(result["all_sessions_passed"]),
                "admitted_to_formula": key[0] == "streaming" and key[1] in {"read", "write"},
            }
        )
    expected = {
        (pressure, operation, threads)
        for pressure in ("cache_resident", "transition", "streaming")
        for operation in ("read", "write", "copy")
        for threads in (1, 2, 3, 4)
    }
    if keys != expected:
        raise RuntimeError(
            "P7B-1 memory matrix mismatch: missing={} extra={}".format(
                sorted(expected - keys), sorted(keys - expected)
            )
        )
    if not all(row["all_sessions_passed"] for row in rows):
        raise RuntimeError("P7B-1 contains a failed memory session")
    return sorted(rows, key=lambda row: (row["pressure_class"], row["operation"], row["threads"]))


def apply_streaming_memory_profile(context, memory_rows):
    revised = copy.deepcopy(context)
    admitted = [row for row in memory_rows if row["admitted_to_formula"]]
    if len(admitted) != 8:
        raise RuntimeError("expected 8 admitted streaming read/write models")
    for row in admitted:
        revised["cpu_memory_models"][(row["operation"], row["threads"])] = {
            "case_id": row["case_id"],
            "operation": row["operation"],
            "threads": row["threads"],
            "bandwidth_GBps_median": row["bandwidth_GBps_median"],
            "source": "P7B-1 three-boot 64 MiB streaming median",
        }
    return revised


def component_admission(p7b1, p7b2, p7c):
    return {
        "cpu_memory_streaming_bandwidth": {
            "admitted": bool(p7b1.get("passed")),
            "reason": "three independent boots passed the frozen component gate",
            "formula_role": "replace CPU logical read/write DDR lower-bound rates",
        },
        "cpu_pair_slowdown": {
            "admitted": False,
            "exact_observations_available": bool(
                p7b1.get("cpu_concurrency_exact_observations_ready")
            ),
            "reason": "generalized_slowdown_surface_ready is false; 16 selected pairs cannot be extrapolated to every DP transition",
            "formula_role": None,
        },
        "runtime_frame_or_stage_intercept": {
            "admitted": False,
            "reason": "P7B-2 found an unidentifiable/de-duplicated host queue floor below 0.1 ms; stage run_ms already owns invocation and wait",
            "formula_role": None,
        },
        "p7b2_boundary_slope": {
            "admitted": False,
            "reason": "single-boot observation and replacing the existing owner-split boundary model would increase the known underprediction",
            "formula_role": None,
        },
        "cpu_vta_contention": {
            "admitted": False,
            "qualification_passed": bool(p7c.get("passed")),
            "reason": "single-boot evidence is not a pure DDR parameter and has not passed the required three-boot gate",
            "formula_role": None,
        },
        "candidate_fitted_constant_or_scale": {
            "admitted": False,
            "reason": "+34 ms and proportional calibration consume complete-candidate outcomes and cannot enter a transferable score",
            "formula_role": None,
        },
    }


def build_physical_profile(output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir)
    p7b1 = load_sealed_artifact(
        output_dir / "v1_p7b1_three_boot_reproducibility.json",
        "cpu_vta_pipeline_v1_p7b1_three_boot_reproducibility",
    )
    p7b2 = load_sealed_artifact(
        output_dir / "v1_p7b2_runtime_qualification_summary.json",
        "cpu_vta_pipeline_v1_p7b2_runtime_qualification_summary",
    )
    p7c = load_sealed_artifact(
        output_dir / "v1_p7c_qualification_summary.json",
        "cpu_vta_pipeline_v1_p7c_shared_ddr_qualification_summary",
    )
    iteration2 = load_sealed_artifact(
        output_dir / "v1_p5b_iteration2_cpu_measurements.json",
        "cpu_vta_pipeline_v1_p5b_iteration2_cpu_measurements",
    )
    memory_rows = build_cpu_memory_profile(p7b1)
    admission = component_admission(p7b1, p7b2, p7c)
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7_physical_profile",
            "protocol_id": P7_PROTOCOL_ID,
            "status": "frozen_component_profile_with_explicit_exclusions",
            "candidate_throughput_used_for_parameter_fit": False,
            "source_artifacts": {
                "p7b1_three_boot_sha256": p7b1["artifact_sha256"],
                "p7b2_single_boot_sha256": p7b2["artifact_sha256"],
                "p7c_single_boot_sha256": p7c["artifact_sha256"],
                "iteration2_cpu_measurements_sha256": iteration2["artifact_sha256"],
            },
            "cpu_atomic_service": {
                "admitted": all(iteration2["grouped_holdout_gate"].values()),
                "updated_cpu_segment_count": iteration2["updated_cpu_segment_count"],
                "grouped_holdout_metrics": iteration2["grouped_holdout_metrics"],
                "composition_policy": "sum measured atomic-unit wall/core service within a CPU segment",
            },
            "cpu_memory_models": memory_rows,
            "component_admission": admission,
            "frozen_formula": {
                "cpu_stage": "isolated atomic-composition service + CPU-owned boundary",
                "vta": "sum isolated inclusive VTA service + VTA-mutex-owned boundary",
                "cpu_core_pool": "max(total host core-ms/4, every nested [0,k) prefix capacity bound)",
                "shared_ddr": "sum CPU logical IO at admitted streaming read/write rates and existing VTA physical LOAD/STORE estimate",
                "pipeline_ii": "max(max CPU stage, single VTA sum, CPU core pool, shared DDR)",
                "frame_intercept_ms": 0.0,
            },
            "limitations": [
                "P7B-1 exact CPU-pair slowdown observations are retained but not generalized.",
                "P7B-2 boundary and runtime measurements have only one boot and are not additive to inclusive stage service.",
                "P7C has one boot and mixes shared-memory and host-runtime effects, so no DDR contention coefficient is fitted.",
                "The score remains a resource lower-bound ranking model, not a calibrated absolute FPS predictor.",
            ],
        }
    )


def build_context_with_p7d(output_dir, physical_profile):
    output_dir = Path(output_dir)
    profile = load_sealed_artifact(
        output_dir / "v1_profile_manifest.json", "cpu_vta_pipeline_v1_profile_manifest"
    )
    local_cost = load_sealed_artifact(
        output_dir / "v1_local_cost_table.json", "cpu_vta_pipeline_v1_local_cost_table"
    )
    wall, core, _, _ = summarize_atomic_profiles(output_dir)
    context, _ = apply_atomic_cpu_costs(build_context(profile, local_cost), wall, core)
    return apply_streaming_memory_profile(context, physical_profile["cpu_memory_models"])


def build_ranked(output_dir, physical_profile, top_k=20):
    output_dir = Path(output_dir)
    unit_schema = load_sealed_artifact(
        output_dir / "v1_unit_and_boundary_schema.json",
        "cpu_vta_pipeline_v1_unit_and_boundary_schema",
    )
    context = build_context_with_p7d(output_dir, physical_profile)
    labels, stats = solve_k_best_label_setting(context, top_k=top_k)
    oracle, _, topology_count, execution_count = enumerate_oracle(
        unit_schema, context, top_k=top_k
    )
    ids = [candidate_id(label.path) for label in labels]
    oracle_ids = [candidate_id(label.path) for label in oracle]
    if ids != oracle_ids:
        raise RuntimeError("P7D DP Top-K differs from exact enumeration")
    rows = []
    for rank, label in enumerate(labels, 1):
        row = candidate_record(label, rank, context)
        row["cost_provenance"]["p7_physical_profile_artifact_sha256"] = physical_profile[
            "artifact_sha256"
        ]
        row["cost_provenance"]["candidate_measured_throughput_consumed"] = False
        rows.append(row)
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7d_ranked_candidates",
            "protocol_id": P7_PROTOCOL_ID,
            "physical_profile_artifact_sha256": physical_profile["artifact_sha256"],
            "candidate_throughput_used_for_ranking": False,
            "topology_count": topology_count,
            "execution_configuration_count": execution_count,
            "dp_top_k_equals_enumeration": ids == oracle_ids,
            "dp_stats": stats,
            "rows": rows,
        }
    ), context


def build_p7d_historical_scores(output_dir, context, ranked):
    base = build_scores(output_dir)
    rows = []
    for source in base["rows"]:
        path = tuple((str(item[0]), int(item[1])) for item in source["path"])
        scored = score_path(path, context)
        rows.append(
            {
                **source,
                **scored,
                "scores_ms": {"fluid_nested_prefix": scored["scores_ms"]["fluid_nested_prefix"]},
            }
        )
    scores = seal_artifact(
        {
            **{key: value for key, value in base.items() if key not in {"artifact_sha256", "rows", "models"}},
            "kind": "cpu_vta_pipeline_v1_p7d_historical_scores",
            "physical_profile_artifact_sha256": ranked["physical_profile_artifact_sha256"],
            "models": {
                "fluid_nested_prefix": "P7D frozen formula with three-boot streaming CPU-memory rates"
            },
            "rows": rows,
        }
    )
    evaluation = evaluate_scores(scores, output_dir, current_ranked=ranked)
    unsealed = copy.deepcopy(evaluation)
    unsealed.pop("artifact_sha256", None)
    unsealed["kind"] = "cpu_vta_pipeline_v1_p7d_historical_evaluation"
    return scores, seal_artifact(unsealed)


def _regret_at_k(rows, oracle_cycle_ms):
    result = {}
    for k in K_VALUES:
        selected = rows[: min(k, len(rows))]
        best_cycle = min(row["measured_cycle_ms"] for row in selected)
        result[str(k)] = 1.0 - oracle_cycle_ms / best_cycle
    return result


def evaluate_natural_top20(ranked, board_summary):
    measured = {
        row["candidate_id"]: 1000.0 / float(row["measurement"]["pipeline_fps"])
        for row in board_summary["rows"]
    }
    rows = []
    for candidate in ranked["rows"]:
        candidate_id_value = candidate["candidate_id"]
        if candidate_id_value not in measured:
            continue
        rows.append(
            {
                "static_rank": candidate["rank"],
                "candidate_id": candidate_id_value,
                "predicted_cycle_ms": float(candidate["predicted_ii_ms"]),
                "measured_cycle_ms": measured[candidate_id_value],
                "measured_fps": 1000.0 / measured[candidate_id_value],
            }
        )
    complete = len(rows) == len(ranked["rows"]) == 20
    if not complete:
        return {
            "complete_board_coverage": False,
            "covered_count": len(rows),
            "required_count": len(ranked["rows"]),
            "rows": rows,
        }
    predicted = [row["predicted_cycle_ms"] for row in rows]
    actual = [row["measured_cycle_ms"] for row in rows]
    static_order = [float(row["static_rank"]) for row in rows]
    oracle_cycle = min(actual)
    fixed34 = [value + 34.0 for value in predicted]
    return {
        "complete_board_coverage": True,
        "covered_count": len(rows),
        "correctness_pass_count": int(board_summary["summary"]["correctness_pass_count"]),
        "measured_pool_scope": "the frozen natural Top-20 only; not the 972528-configuration global oracle",
        "measured_pool_oracle_cycle_ms": oracle_cycle,
        "measured_pool_oracle_fps": 1000.0 / oracle_cycle,
        "cycle_error": absolute_error_metrics(predicted, actual),
        "fixed_34ms_cycle_error": absolute_error_metrics(fixed34, actual),
        # Keep the ordinal metric comparable with the frozen board summary.  The
        # score-tie-aware value is also reported because many thread variants
        # have exactly the same predicted II.
        "spearman": _spearman(static_order, actual),
        "score_tie_aware_spearman": _spearman(predicted, actual),
        "throughput_regret_at_k": _regret_at_k(rows, oracle_cycle),
        "rows": rows,
    }


def fixed_offset_mape(rows, offset_ms):
    predicted = [float(row["scores_ms"]["fluid_nested_prefix"]) + float(offset_ms) for row in rows]
    measured = [float(row["measured_cycle_ms"]) for row in rows]
    return absolute_error_metrics(predicted, measured)["mape"]


def build_formula_validation(output_dir, physical_profile, ranked, historical):
    output_dir = Path(output_dir)
    board_path = output_dir / "v1_p7_top20_board_20260904/top20_board_summary.json"
    board = json.loads(board_path.read_text(encoding="utf-8"))
    natural = evaluate_natural_top20(ranked, board)
    all_200 = historical["scopes"]["all_200_records_absolute_only"]["models"][
        "fluid_nested_prefix"
    ]
    unique_199 = historical["scopes"]["all_199_unique_layouts"]["models"][
        "fluid_nested_prefix"
    ]
    canonical_69 = historical["scopes"]["canonical_69_unique_layouts"]["models"][
        "fluid_nested_prefix"
    ]

    # The joined rows are retained inside the evaluation scopes only as metrics, so
    # reconstruct the historical +34 ms baseline from the separately sealed scores/labels.
    scores = load_sealed_artifact(
        output_dir / "v1_p7d_historical_scores.json",
        "cpu_vta_pipeline_v1_p7d_historical_scores",
    )
    labels = load_sealed_artifact(
        output_dir / "v1_p4_historical_labels.json",
        "cpu_vta_pipeline_v1_p4_historical_labels",
    )
    score_by_id = {row["execution_candidate_id"]: row for row in scores["rows"]}
    all_records = []
    for label in labels["rows"]:
        row = score_by_id[label["mapped_execution_candidate_id"]]
        all_records.append({**row, "measured_cycle_ms": float(label["measured_cycle_ms"])})
    historical_fixed34_mape = fixed_offset_mape(all_records, 34.0)

    gates = {
        "no_candidate_outcome_fitted_parameter": not physical_profile[
            "candidate_throughput_used_for_parameter_fit"
        ],
        "dp_top20_exact": bool(ranked["dp_top_k_equals_enumeration"]),
        "natural_top20_board_coverage_complete": bool(natural["complete_board_coverage"]),
        "historical_200_mape_below_fixed34_baseline": all_200["cycle_error"]["mape"]
        < historical_fixed34_mape,
        "natural_top20_mape_below_fixed34_baseline": natural.get("cycle_error", {}).get(
            "mape", float("inf")
        )
        < natural.get("fixed_34ms_cycle_error", {}).get("mape", float("inf")),
        "natural_top20_spearman_above_0_155": natural.get("spearman", float("-inf"))
        > 0.155,
        "natural_top20_regret_at_5_below_5_419pct": natural.get(
            "throughput_regret_at_k", {}
        ).get("5", float("inf"))
        < 0.05419,
    }
    quality_gates = {
        key: value
        for key, value in gates.items()
        if key
        not in {
            "no_candidate_outcome_fitted_parameter",
            "dp_top20_exact",
            "natural_top20_board_coverage_complete",
        }
    }
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7_formula_validation",
            "protocol_id": P7_PROTOCOL_ID,
            "physical_profile_artifact_sha256": physical_profile["artifact_sha256"],
            "ranked_artifact_sha256": ranked["artifact_sha256"],
            "historical_evaluation_artifact_sha256": historical["artifact_sha256"],
            "top20_board_summary_sha256": file_sha256(board_path),
            "candidate_outcomes_used_for_fit_or_ranking": False,
            "historical_200": {
                "raw": all_200,
                "fixed_34ms_cycle_mape_diagnostic_baseline": historical_fixed34_mape,
            },
            "historical_199_unique": unique_199,
            "historical_69_canonical": canonical_69,
            "natural_top20": natural,
            "acceptance_gates": gates,
            "quality_improvement_gate_passed": all(quality_gates.values()),
            "decision": (
                "freeze_as_ranking_only_v1_and_do_not_claim_absolute_calibration"
                if not all(quality_gates.values())
                else "admit_as_p7_physical_formula"
            ),
        }
    )


def write_review(path, profile, validation):
    natural = validation["natural_top20"]
    historical = validation["historical_200"]
    gates = validation["acceptance_gates"]
    failed = [name for name, passed in gates.items() if not passed]
    lines = [
        "# P7D 公式收口 Review",
        "",
        "更新日期：2026-09-06",
        "",
        "## 结论",
        "",
        "P7D 已完成参数准入审计、DP 重跑和冻结标签评价。三 boot 的 CPU streaming read/write",
        "带宽进入共享 DDR 下界；CPU pair slowdown 因尚无可泛化 surface 不进入全局搜索，P7B-2/P7C",
        "只有单 boot，也不进入。候选拟合的 `+34 ms` 与比例系数继续禁用。",
        "",
        "本轮没有强行生成新的修正项。质量改善 gate 未通过，因此 V1 冻结为高性能区域筛选模型，",
        "不能称为经过物理校准的绝对 FPS 预测器。P8A 可以把它作为不再变化的规划 baseline，独立验证",
        "共享 slot zero-copy；P8 不得反向修改本轮排名。",
        "",
        "## 参数准入",
        "",
        "| 参数 | 是否进入公式 | 原因 |",
        "|---|---:|---|",
    ]
    for name, item in profile["component_admission"].items():
        lines.append(
            "| `{}` | {} | {} |".format(name, "是" if item["admitted"] else "否", item["reason"])
        )
    lines.extend(
        [
            "",
            "## 冻结评价",
            "",
            "| 范围 | cycle MAPE | Spearman | regret@5 |",
            "|---|---:|---:|---:|",
            "| 历史 200 条（绝对误差） | {:.3%} | 不适用 | 不适用 |".format(
                historical["raw"]["cycle_error"]["mape"]
            ),
            "| 历史 199 unique | {:.3%} | {:.3f} | {:.3%} |".format(
                validation["historical_199_unique"]["cycle_error"]["mape"],
                validation["historical_199_unique"]["spearman"],
                validation["historical_199_unique"]["ranking"]["regret_at_5"],
            ),
            "| 历史 69 canonical | {:.3%} | {:.3f} | {:.3%} |".format(
                validation["historical_69_canonical"]["cycle_error"]["mape"],
                validation["historical_69_canonical"]["spearman"],
                validation["historical_69_canonical"]["ranking"]["regret_at_5"],
            ),
            "| 自然 Top-20 | {:.3%} | {:.3f} | {:.3%} |".format(
                natural["cycle_error"]["mape"],
                natural["spearman"],
                natural["throughput_regret_at_k"]["5"],
            ),
            "",
            "历史 200 条统一 `+34 ms` 的诊断 MAPE 为 `{:.3%}`；自然 Top-20 为 `{:.3%}`。".format(
                historical["fixed_34ms_cycle_mape_diagnostic_baseline"],
                natural["fixed_34ms_cycle_error"]["mape"],
            ),
            "自然 Top-20 的并列分数感知 Spearman 为 `{:.3f}`；表中 `{:.3f}` 沿用冻结上板报告的".format(
                natural["score_tie_aware_spearman"], natural["spearman"]
            ),
            "固定顺序口径，用于与 `0.155` 基线直接比较。新旧 Top-20 的候选、顺序和预测周期完全一致。",
            "更新 streaming 带宽后 Top-20 仍由计算/core/VTA资源决定，DDR不是其关键瓶颈，因此排名",
            "和现有误差没有得到实质改善。这是负结果，但避免把局部 slowdown 或单 boot 干扰错误外推。",
            "",
            "失败 gate：`{}`。".format("`, `".join(failed) if failed else "无"),
            "",
            "## 下一步",
            "",
            "进入 P8A，只验证单边界 zero-copy 可行性：审计普通 set/get 的真实复制字节，比较最快正确",
            "普通复制与 u-dma-buf 直接访问，并对一个 CPU->VTA 边界完成同址绑定和交替输入正确性检查。",
            "P8A 完成后停止 review，不自动进入双 slot Pipeline。",
            "",
            "## 产物",
            "",
            "- `v1_p7_physical_profile.json`: `{}`".format(profile["artifact_sha256"]),
            "- `v1_p7d_ranked_candidates.json`: `{}`".format(
                validation["ranked_artifact_sha256"]
            ),
            "- `v1_p7_formula_validation.json`: `{}`".format(
                validation["artifact_sha256"]
            ),
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run(output_dir=DEFAULT_OUTPUT, top_k=20):
    output_dir = Path(output_dir)
    profile = build_physical_profile(output_dir)
    validate_artifact_sha256(profile)
    write_json(output_dir / "v1_p7_physical_profile.json", profile)

    ranked, context = build_ranked(output_dir, profile, top_k=top_k)
    validate_artifact_sha256(ranked)
    write_json(output_dir / "v1_p7d_ranked_candidates.json", ranked)

    scores, historical = build_p7d_historical_scores(output_dir, context, ranked)
    validate_artifact_sha256(scores)
    validate_artifact_sha256(historical)
    write_json(output_dir / "v1_p7d_historical_scores.json", scores)
    write_json(output_dir / "v1_p7d_historical_evaluation.json", historical)

    validation = build_formula_validation(output_dir, profile, ranked, historical)
    validate_artifact_sha256(validation)
    write_json(output_dir / "v1_p7_formula_validation.json", validation)
    write_review(output_dir / "v1_p7d_review.md", profile, validation)
    return profile, ranked, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    profile, ranked, validation = run(args.output_dir, args.top_k)
    print(
        json.dumps(
            {
                "profile_sha256": profile["artifact_sha256"],
                "ranked_sha256": ranked["artifact_sha256"],
                "validation_sha256": validation["artifact_sha256"],
                "top1": ranked["rows"][0]["candidate_id"],
                "quality_improvement_gate_passed": validation[
                    "quality_improvement_gate_passed"
                ],
                "decision": validation["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
