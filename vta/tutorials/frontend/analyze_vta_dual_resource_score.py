#!/usr/bin/env python3
"""Develop and audit a two-resource concurrency score for the VTA pipeline.

This tool is deliberately offline.  It consumes the sealed concurrency-score
campaign, fits only the calibration cases, evaluates the already observed
holdout cases, and emits a separate development artifact.  It never changes
the frozen production score or its ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np


SCHEMA = "vta_dual_resource_score_development_v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and values[order[end]] == values[order[begin]]:
            end += 1
        rank = (begin + end - 1) / 2.0 + 1.0
        for position in range(begin, end):
            ranks[order[position]] = rank
        begin = end
    return ranks


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("pearson requires equal sequences with at least two values")
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_norm = sum((x - left_mean) ** 2 for x in left)
    right_norm = sum((y - right_mean) ** 2 for y in right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / math.sqrt(left_norm * right_norm)


def spearman(predictions: Sequence[float], observations: Sequence[float]) -> float:
    return pearson(average_ranks(predictions), average_ranks(observations))


def static_terms(model: dict[str, Any]) -> dict[str, Any]:
    """Build pre-execution CPU and VTA pressure terms from a frozen candidate."""
    loads = model["resource_load_ms"]
    total_host_core_work = float(model["total_host_core_work_ms"])
    cpu_terms = []
    cpu_core_work = 0.0
    for stage in model["cpu_stages"]:
        run_ms = float(stage["run_ms"])
        if run_ms <= 0.0:
            raise ValueError(f"non-positive CPU run time for {stage['segment_id']}")
        core_work_ms = float(stage["core_work_ms"])
        effective_cores = core_work_ms / run_ms
        other_host_per_core_ms = max(0.0, total_host_core_work - core_work_ms) / 4.0
        cpu_terms.append(
            {
                "segment_id": stage["segment_id"],
                "service_ms": float(stage["inclusive_service_ms"]),
                "core_work_ms": core_work_ms,
                "effective_cores": effective_cores,
                "pressure_ms": other_host_per_core_ms * effective_cores,
            }
        )
        cpu_core_work += core_work_ms

    # These are the resource bounds used by the frozen score other than the
    # CPU-stage maximum and the single serialized VTA resource.  Diagnostic
    # uniform-per-core loads are intentionally excluded: they were not an
    # active term of S_static (K07 is the observable counterexample).
    other_bound_ms = max(
        float(loads["direct_boundary_communication"]),
        float(loads["fluid_prefix_max"]),
        float(loads["shared_ddr_physical_demand"]),
        float(loads["total_host_physical_pool"]),
    )
    vta_stage_count = int(model["vta_stage_count"])
    return {
        "static_score_ms": float(model["static_score_ms"]),
        "cpu_terms": cpu_terms,
        "vta_base_ms": float(loads["single_vta_plus_mutex_boundaries"]),
        "vta_pressure_ms": cpu_core_work / 4.0 * vta_stage_count,
        "other_bound_ms": other_bound_ms,
        "cpu_core_work_ms": cpu_core_work,
        "vta_stage_count": vta_stage_count,
    }


def predict_dual(
    terms: dict[str, Any], beta_cpu: float, beta_vta: float
) -> dict[str, float | str]:
    cpu_values = [
        float(stage["service_ms"]) + beta_cpu * float(stage["pressure_ms"])
        for stage in terms["cpu_terms"]
    ]
    cpu_index = max(range(len(cpu_values)), key=cpu_values.__getitem__)
    cpu_ms = cpu_values[cpu_index]
    vta_ms = float(terms["vta_base_ms"]) + beta_vta * float(
        terms["vta_pressure_ms"]
    )
    other_ms = float(terms["other_bound_ms"])
    score_ms = max(cpu_ms, vta_ms, other_ms)
    if score_ms == cpu_ms:
        active = "cpu"
    elif score_ms == vta_ms:
        active = "vta"
    else:
        active = "other"
    return {
        "score_ms": score_ms,
        "cpu_ms": cpu_ms,
        "vta_ms": vta_ms,
        "other_ms": other_ms,
        "active_resource": active,
        "active_cpu_stage": terms["cpu_terms"][cpu_index]["segment_id"],
    }


def fit_dual(
    records: Sequence[dict[str, Any]],
    lower: float = 0.0,
    upper: float = 1.0,
    step: float = 0.001,
) -> dict[str, float]:
    if not records:
        raise ValueError("cannot fit an empty record set")
    count = int(round((upper - lower) / step))
    grid = lower + np.arange(count + 1, dtype=np.float64) * step
    loss = np.zeros((len(grid), len(grid)), dtype=np.float64)
    for record in records:
        terms = record["static_terms"]
        cpu = np.maximum.reduce(
            [
                float(stage["service_ms"])
                + grid * float(stage["pressure_ms"])
                for stage in terms["cpu_terms"]
            ]
        )
        vta = float(terms["vta_base_ms"]) + grid * float(
            terms["vta_pressure_ms"]
        )
        prediction = np.maximum(
            np.maximum(cpu[:, np.newaxis], vta[np.newaxis, :]),
            float(terms["other_bound_ms"]),
        )
        loss += np.abs(prediction - float(record["observed_ii_ms"])) / float(
            record["observed_ii_ms"]
        )
    # np.argmin is row-major.  It therefore implements the preregistered
    # deterministic tie break: smaller beta_cpu, then smaller beta_vta.
    cpu_index, vta_index = np.unravel_index(np.argmin(loss), loss.shape)
    return {
        "beta_cpu": float(grid[cpu_index]),
        "beta_vta": float(grid[vta_index]),
        "mape": float(loss[cpu_index, vta_index] / len(records)),
        "lower": lower,
        "upper": upper,
        "step": step,
    }


def metrics(records: Sequence[dict[str, Any]], predictor: Callable[[dict[str, Any]], float]) -> dict[str, float]:
    observed = [float(record["observed_ii_ms"]) for record in records]
    predicted = [float(predictor(record)) for record in records]
    errors = [prediction - actual for prediction, actual in zip(predicted, observed)]
    return {
        "count": len(records),
        "mae_ms": mean(abs(error) for error in errors),
        "mape_percent": 100.0
        * mean(abs(error) / actual for error, actual in zip(errors, observed)),
        "signed_bias_ms": mean(errors),
        "spearman": spearman(predicted, observed) if len(records) >= 2 else 0.0,
    }


def bootstrap_mape_improvement(
    records: Sequence[dict[str, Any]],
    baseline: Callable[[dict[str, Any]], float],
    candidate: Callable[[dict[str, Any]], float],
    iterations: int = 20000,
) -> dict[str, float]:
    rng = random.Random(20260921)
    deltas = []
    for _ in range(iterations):
        sampled = [records[rng.randrange(len(records))] for _ in records]
        baseline_mape = mean(
            abs(baseline(record) - record["observed_ii_ms"])
            / record["observed_ii_ms"]
            for record in sampled
        )
        candidate_mape = mean(
            abs(candidate(record) - record["observed_ii_ms"])
            / record["observed_ii_ms"]
            for record in sampled
        )
        deltas.append(100.0 * (baseline_mape - candidate_mape))
    deltas.sort()
    return {
        "mape_reduction_percentage_points_mean": mean(deltas),
        "ci95_low": deltas[int(0.025 * iterations)],
        "ci95_high": deltas[int(0.975 * iterations) - 1],
        "iterations": iterations,
    }


def load_records(run_dir: Path) -> tuple[list[dict[str, Any]], list[Path], dict[str, Any]]:
    contract_path = run_dir / "contract.json"
    contract = read_json(contract_path)
    analysis_dir = run_dir / "analysis"
    source_paths = [contract_path]
    records = []
    for case in contract["cases"]:
        case_id = case["case_id"]
        role = case["role"]
        boot_index = 1 if role == "calibration" else 2
        blocks = []
        for block_index in (0, 1):
            path = analysis_dir / (
                f"boot{boot_index}_{role}_{case_id}_block{block_index}.json"
            )
            block = read_json(path)
            source_paths.append(path)
            if block.get("status") != "complete":
                raise ValueError(f"incomplete input block: {path}")
            if not block.get("local_baseline_guard", {}).get("passed", False):
                raise ValueError(f"failed local baseline guard: {path}")
            blocks.append(block)

        devices = [stage["device"] for stage in case["model"]["manifest"]["stages"]]
        stage_count = len(devices)
        concurrent_services = [
            mean(
                float(block["main_window"]["stage_service_mean_ms"][stage])
                for block in blocks
            )
            for stage in range(stage_count)
        ]
        isolated_windows = [
            block[window]
            for block in blocks
            for window in ("before_window", "after_window")
        ]
        isolated_services = [
            mean(
                float(window["stage_service_mean_ms"][stage])
                for window in isolated_windows
            )
            for stage in range(stage_count)
        ]
        record = {
            "case_id": case_id,
            "candidate_id": case["candidate_id"],
            "role": role,
            "source_boot_index": boot_index,
            "source_boot_id": blocks[0]["boot_id"],
            "cell": list(case["model"]["cell"]),
            "observed_ii_ms": mean(float(block["completion_II_ms"]) for block in blocks),
            "block_ii_ms": [float(block["completion_II_ms"]) for block in blocks],
            "devices": devices,
            "concurrent_stage_service_ms": concurrent_services,
            "isolated_stage_service_ms": isolated_services,
            "observed_concurrent_cpu_max_ms": max(
                value for value, device in zip(concurrent_services, devices) if device == "cpu"
            ),
            "observed_concurrent_vta_sum_ms": sum(
                value for value, device in zip(concurrent_services, devices) if device == "vta"
            ),
            "observed_isolated_cpu_max_ms": max(
                value for value, device in zip(isolated_services, devices) if device == "cpu"
            ),
            "observed_isolated_vta_sum_ms": sum(
                value for value, device in zip(isolated_services, devices) if device == "vta"
            ),
            "static_terms": static_terms(case["model"]),
        }
        records.append(record)
    records.sort(key=lambda record: record["case_id"])
    return records, source_paths, contract


def cross_validated_predictions(
    records: Sequence[dict[str, Any]], groups: Sequence[Sequence[str]]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    predictions: dict[str, float] = {}
    fits = []
    for group in groups:
        test_ids = set(group)
        train = [record for record in records if record["case_id"] not in test_ids]
        test = [record for record in records if record["case_id"] in test_ids]
        fit = fit_dual(train)
        fits.append({**fit, "held_out": sorted(test_ids)})
        for record in test:
            predictions[record["case_id"]] = float(
                predict_dual(
                    record["static_terms"], fit["beta_cpu"], fit["beta_vta"]
                )["score_ms"]
            )
    if set(predictions) != {record["case_id"] for record in records}:
        raise ValueError("cross-validation groups do not partition all records")
    return predictions, fits


def build_report(result: dict[str, Any]) -> str:
    fit = result["calibration_fit"]
    development_fit = result["development_fit"]
    holdout = result["metrics"]["holdout"]
    lines = [
        "# CPU–VTA 双资源并发评分：离线开发报告",
        "",
        "## 结论",
        "",
        (
            f"只使用原校准组拟合得到 `β_CPU={fit['beta_cpu']:.3f}`、"
            f"`β_VTA={fit['beta_vta']:.3f}`。在原 8 个 holdout 候选上，"
            f"双资源模型 MAPE 为 **{holdout['M3_dual_calibration_fit']['mape_percent']:.2f}%**，"
            f"低于静态模型的 **{holdout['M0_static']['mape_percent']:.2f}%**、"
            f"统一常数的 **{holdout['M1_constant']['mape_percent']:.2f}%** 和"
            f"旧并发模型的 **{holdout['M2_old_concurrency']['mape_percent']:.2f}%**。"
        ),
        "",
        (
            f"holdout 排序相关系数从 M0 的 {holdout['M0_static']['spearman']:.3f}、"
            f"旧 M2 的 {holdout['M2_old_concurrency']['spearman']:.3f} 提高到 "
            f"**{holdout['M3_dual_calibration_fit']['spearman']:.3f}**。"
        ),
        "",
        "这是一项模型开发结果。K07、K05 和 K08 的结果已经用于诊断公式，"
        "因此原 holdout 此后不能再称为新模型的盲测。正式替换评分公式前，"
        "需要冻结模型后选择未参与开发的新候选验证。",
        "",
        (
            f"用于下一轮新候选验证的开发集拟合值为 "
            f"`β_CPU={development_fit['beta_cpu']:.3f}`、"
            f"`β_VTA={development_fit['beta_vta']:.3f}`。这两个系数使用了全部 "
            "16 个开发候选；下面的原 holdout 指标仍使用只由 8 个校准候选得到的系数。"
        ),
        "",
        "## 模型",
        "",
        "对每个 CPU 阶段 `i`：",
        "",
        "```text",
        "U_i = CPU核时_i / run时间_i",
        "P_CPU_i = (总host核时 − CPU核时_i) / 4 × U_i",
        "C_i = 静态阶段服务_i + β_CPU × P_CPU_i",
        "```",
        "",
        "VTA 资源：",
        "",
        "```text",
        "P_VTA = CPU阶段总核时 / 4 × VTA阶段数",
        "V = 静态VTA串行资源 + β_VTA × P_VTA",
        "M3 = max(max_i(C_i), V, 其余原静态资源下界)",
        "```",
        "",
        "`U_i` 是关键修正：它区分单线程 CPU 阶段和多线程 CPU 阶段，"
        "从而避免旧模型把 K07 的单线程主导阶段过度放大。VTA 项用于补上"
        " K05/K08 中旧模型完全看不到的 VTA 串行服务膨胀。",
        "",
        "## 候选结果",
        "",
        "|候选|角色|实测II|M0|旧M2|M3校准式|M3冻结式|冻结式误差|实测瓶颈|冻结式瓶颈|",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in result["candidate_results"]:
        lines.append(
            "|{case_id}|{role}|{observed_ii_ms:.3f}|{M0_static_ms:.3f}|"
            "{M2_old_concurrency_ms:.3f}|{M3_calibration_fit_ms:.3f}|"
            "{M3_frozen_ms:.3f}|{M3_frozen_signed_error_ms:+.3f}|"
            "{observed_active_resource}|{M3_frozen_active_resource}|".format(**row)
        )
    loo = result["cross_validation"]["leave_one_candidate_out"]
    cell = result["cross_validation"]["leave_one_cell_out"]
    lines.extend(
        [
            "",
            "## 稳健性检查",
            "",
            (
                f"16 个候选留一预测：MAPE **{loo['metrics']['mape_percent']:.2f}%**，"
                f"Spearman **{loo['metrics']['spearman']:.3f}**。"
            ),
            (
                f"按 `(CPU阶段数, 总阶段数)` 整组留出：MAPE "
                f"**{cell['metrics']['mape_percent']:.2f}%**，Spearman "
                f"**{cell['metrics']['spearman']:.3f}**。"
            ),
            "",
            "最明显的剩余异常是 K09（留出预测高约 11%）；K14/K15 仍低估约 3–6%。"
            "它们说明两个系数尚未覆盖所有拓扑状态，但已经不再表现为所有候选统一"
            "增加一个常数。",
            "",
            "## 下一步",
            "",
            "1. 冻结当前公式、开发集系数以及代码哈希。",
            "2. 从当前资产完整且未参与开发的 179 个候选中只按静态特征选择新验证样本，禁止读取旧 II。",
            "3. 新样本先做一个 boot；若 M3 同时改善误差和排序，再补两个独立 boot。",
            "4. 通过新样本验证后，才在独立分支重排 199 个候选；当前正式评分保持不变。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records, source_paths, contract = load_records(run_dir)
    calibration = [record for record in records if record["role"] == "calibration"]
    holdout = [record for record in records if record["role"] == "holdout"]
    if len(calibration) != 8 or len(holdout) != 8:
        raise ValueError("expected eight calibration and eight holdout candidates")

    model_freeze_path = run_dir / "analysis" / "model_freeze.json"
    model_freeze = read_json(model_freeze_path)
    source_paths.append(model_freeze_path)
    frozen_prediction = {
        case["case_id"]: model_freeze["predictions"][case["candidate_id"]]
        for case in contract["cases"]
    }

    fit = fit_dual(calibration)
    development_fit = fit_dual(records)
    dual_predictions = {
        record["case_id"]: predict_dual(
            record["static_terms"], fit["beta_cpu"], fit["beta_vta"]
        )
        for record in records
    }
    frozen_predictions = {
        record["case_id"]: predict_dual(
            record["static_terms"],
            development_fit["beta_cpu"],
            development_fit["beta_vta"],
        )
        for record in records
    }

    predictors = {
        "M0_static": lambda record: float(record["static_terms"]["static_score_ms"]),
        "M1_constant": lambda record: float(
            frozen_prediction[record["case_id"]]["M1_constant_ms"]
        ),
        "M2_old_concurrency": lambda record: float(
            frozen_prediction[record["case_id"]]["M2_concurrency_ms"]
        ),
        "M3_dual_calibration_fit": lambda record: float(
            dual_predictions[record["case_id"]]["score_ms"]
        ),
        "M3_dual_development_fit": lambda record: float(
            frozen_predictions[record["case_id"]]["score_ms"]
        ),
    }
    metric_groups = {"calibration": calibration, "holdout": holdout, "all": records}
    metric_table = {
        group_name: {
            model_name: metrics(group, predictor)
            for model_name, predictor in predictors.items()
        }
        for group_name, group in metric_groups.items()
    }

    loo_groups = [[record["case_id"]] for record in records]
    loo_predictions, loo_fits = cross_validated_predictions(records, loo_groups)
    cells = sorted({tuple(record["cell"]) for record in records})
    cell_groups = [
        [record["case_id"] for record in records if tuple(record["cell"]) == cell]
        for cell in cells
    ]
    cell_predictions, cell_fits = cross_validated_predictions(records, cell_groups)

    def prediction_metrics(prediction_table: dict[str, float]) -> dict[str, float]:
        return metrics(records, lambda record: prediction_table[record["case_id"]])

    candidate_results = []
    for record in records:
        case_id = record["case_id"]
        dual = dual_predictions[case_id]
        frozen = frozen_predictions[case_id]
        actual_active = (
            "cpu"
            if record["observed_concurrent_cpu_max_ms"]
            >= record["observed_concurrent_vta_sum_ms"]
            else "vta"
        )
        candidate_results.append(
            {
                "case_id": case_id,
                "role": record["role"],
                "cell": record["cell"],
                "observed_ii_ms": record["observed_ii_ms"],
                "M0_static_ms": predictors["M0_static"](record),
                "M1_constant_ms": predictors["M1_constant"](record),
                "M2_old_concurrency_ms": predictors["M2_old_concurrency"](record),
                "M3_calibration_fit_ms": predictors["M3_dual_calibration_fit"](record),
                "M3_frozen_ms": predictors["M3_dual_development_fit"](record),
                "M3_frozen_signed_error_ms": predictors["M3_dual_development_fit"](record)
                - record["observed_ii_ms"],
                "M3_frozen_absolute_percentage_error": 100.0
                * abs(predictors["M3_dual_development_fit"](record) - record["observed_ii_ms"])
                / record["observed_ii_ms"],
                "M3_calibration_fit_cpu_ms": dual["cpu_ms"],
                "M3_calibration_fit_vta_ms": dual["vta_ms"],
                "M3_frozen_cpu_ms": frozen["cpu_ms"],
                "M3_frozen_vta_ms": frozen["vta_ms"],
                "M3_frozen_other_ms": frozen["other_ms"],
                "M3_frozen_active_resource": frozen["active_resource"],
                "M3_frozen_active_cpu_stage": frozen["active_cpu_stage"],
                "observed_concurrent_cpu_max_ms": record[
                    "observed_concurrent_cpu_max_ms"
                ],
                "observed_concurrent_vta_sum_ms": record[
                    "observed_concurrent_vta_sum_ms"
                ],
                "observed_active_resource": actual_active,
                "isolated_cpu_max_ms": record["observed_isolated_cpu_max_ms"],
                "isolated_vta_sum_ms": record["observed_isolated_vta_sum_ms"],
            }
        )

    result = {
        "schema": SCHEMA,
        "status": "development_only_not_blind_validation",
        "source_run": str(run_dir.resolve()),
        "formula": {
            "cpu": "C_i=T_i+beta_cpu*((W_host_total-W_i)/4)*(W_i/run_i)",
            "vta": "V=V_static+beta_vta*(sum(W_cpu)/4)*N_vta",
            "score": "max(max_i(C_i),V,other_static_resource_bounds)",
        },
        "calibration_fit": fit,
        "development_fit": development_fit,
        "metrics": metric_table,
        "holdout_bootstrap": {
            "M3_vs_M0": bootstrap_mape_improvement(
                holdout, predictors["M0_static"], predictors["M3_dual_calibration_fit"]
            ),
            "M3_vs_M1": bootstrap_mape_improvement(
                holdout, predictors["M1_constant"], predictors["M3_dual_calibration_fit"]
            ),
            "M3_vs_old_M2": bootstrap_mape_improvement(
                holdout, predictors["M2_old_concurrency"], predictors["M3_dual_calibration_fit"]
            ),
        },
        "cross_validation": {
            "leave_one_candidate_out": {
                "metrics": prediction_metrics(loo_predictions),
                "predictions_ms": loo_predictions,
                "fits": loo_fits,
            },
            "leave_one_cell_out": {
                "metrics": prediction_metrics(cell_predictions),
                "predictions_ms": cell_predictions,
                "fits": cell_fits,
            },
        },
        "candidate_results": candidate_results,
        "records": records,
        "limitations": [
            "The old holdout cases informed the new formula diagnosis and are development data.",
            "The model has only been evaluated on one observed boot per role.",
            "DDR and VTA-internal DMA/waiting remain conditional static bounds.",
            "The formal score and candidate ranking are unchanged.",
        ],
    }

    result_path = output_dir / "dual_resource_development.json"
    table_path = output_dir / "development_table.json"
    report_path = output_dir / "REPORT_ZH.md"
    freeze_path = output_dir / "model_freeze_candidate.json"
    write_json(result_path, result)
    write_json(table_path, candidate_results)
    report_path.write_text(build_report(result), encoding="utf-8")
    write_json(
        freeze_path,
        {
            "schema": "vta_dual_resource_score_model_freeze_candidate_v1",
            "status": "frozen_for_fresh_candidate_validation_not_formal_score",
            "formula": result["formula"],
            "beta_cpu": development_fit["beta_cpu"],
            "beta_vta": development_fit["beta_vta"],
            "grid": {
                key: development_fit[key] for key in ("lower", "upper", "step")
            },
            "fit_target": "MAPE on all 16 development candidates",
            "development_case_ids": [record["case_id"] for record in records],
            "development_candidate_ids": [record["candidate_id"] for record in records],
            "fresh_validation_required": True,
            "formal_score_changed": False,
            "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        },
    )

    source_paths.append(Path(__file__).resolve())
    artifacts = [result_path, table_path, report_path, freeze_path]
    manifest = {
        "schema": "vta_dual_resource_score_hash_manifest_v1",
        "sources": {
            str(path.resolve()): sha256_file(path) for path in sorted(set(source_paths))
        },
        "artifacts": {
            path.name: sha256_file(path) for path in sorted(artifacts)
        },
    }
    manifest_path = output_dir / "manifest.sha256.json"
    write_json(manifest_path, manifest)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args.run_dir.resolve(), args.output_dir.resolve())
    summary = {
        "output_dir": str(args.output_dir.resolve()),
        "fit": result["calibration_fit"],
        "development_fit": result["development_fit"],
        "holdout_metrics": result["metrics"]["holdout"],
        "leave_one_out": result["cross_validation"]["leave_one_candidate_out"][
            "metrics"
        ],
        "leave_one_cell_out": result["cross_validation"]["leave_one_cell_out"][
            "metrics"
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
