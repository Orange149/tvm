#!/usr/bin/env python3
"""Replay literature-inspired validity and residency-gain policies on frozen C3 data.

This is deliberately an offline development analysis.  It never treats a lowering
label as an FPGA-correctness label and never claims that a replay is a prospective
board experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20250901
VALIDITY_BUDGETS = (4, 8, 16)
PERFORMANCE_BUDGETS = (1, 2, 4, 8)
MODES = (
    "original",
    "input_stationary",
    "weight_stationary",
    "weight_stationary_barrier",
    "paper_inspired_hybrid",
)
KNOBS = ("tile_b", "tile_h", "tile_w", "tile_ci", "tile_co", "oc_nthread", "h_nthread")
DMA_BYTES = ("dma_total_bytes", "input_dma_bytes", "weight_dma_bytes", "output_dma_bytes")
REQUEST_SHAPE = (
    "dma_total_calls",
    "input_dma_calls",
    "weight_dma_calls",
    "output_dma_calls",
    "padded_dma_calls",
    "small_dma_calls",
    "strided_dma_calls",
    "input_reload",
    "weight_reload",
    "output_reload",
)
COMMAND = ("peak_insn_bytes", "peak_uop_bytes", "total_insn_bytes", "total_uop_bytes")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def knob_values(row):
    values = {}
    entity = row["identity"]["complete_config_entity"] if "identity" in row else row["complete_config_entity"]
    for name, kind, value in entity["entity"]:
        values[name] = float(value[-1] if kind == "sp" else value)
    return values


def geometry(row):
    workload = row["identity"]["workload"] if "identity" in row else row["workload"]
    inp = workload[1][1]
    wgt = workload[2][1]
    stride = workload[3]
    padding = workload[4]
    batch = inp[0] * inp[4]
    ci = inp[1] * inp[5]
    height, width = inp[2], inp[3]
    co = wgt[0] * wgt[4]
    kh, kw = wgt[2], wgt[3]
    return {
        "batch": float(batch), "ci": float(ci), "height": float(height), "width": float(width),
        "co": float(co), "kh": float(kh), "kw": float(kw),
        "stride_h": float(stride[0]), "stride_w": float(stride[1]),
        "padding_total": float(sum(padding)),
    }


def base_features(row):
    geom = geometry(row)
    knobs = knob_values(row)
    values = [math.log2(1.0 + geom[name]) for name in sorted(geom)]
    values.extend(math.log2(1.0 + knobs[name]) for name in KNOBS)
    mode = row["residence_mode"]
    values.extend(float(mode == candidate) for candidate in MODES)
    return values


def hardware_features(row):
    """Label-free VTA working-set/parallelism features derived before lowering."""
    geom = geometry(row)
    k = knob_values(row)
    tile_b = max(1.0, k["tile_b"])
    tile_h = max(1.0, k["tile_h"])
    tile_w = max(1.0, k["tile_w"])
    tile_ci = max(1.0, k["tile_ci"]) * 16.0
    tile_co = max(1.0, k["tile_co"]) * 16.0
    input_h = tile_h * geom["stride_h"] + geom["kh"] - 1.0
    input_w = tile_w * geom["stride_w"] + geom["kw"] - 1.0
    input_bytes = tile_b * input_h * input_w * tile_ci
    weight_bytes = tile_co * tile_ci * geom["kh"] * geom["kw"]
    acc_bytes = tile_b * tile_h * tile_w * tile_co * 4.0
    waves = (
        math.ceil(geom["height"] / tile_h)
        * math.ceil(geom["width"] / tile_w)
        * math.ceil(geom["ci"] / tile_ci)
        * math.ceil(geom["co"] / tile_co)
    )
    return base_features(row) + [
        input_bytes / 32768.0,
        weight_bytes / 262144.0,
        acc_bytes / 131072.0,
        math.log2(1.0 + waves),
        math.log2(1.0 + tile_b * tile_h * tile_w * tile_ci * tile_co),
        k["oc_nthread"] * k["h_nthread"],
    ]


def deterministic_order(rows, seed_suffix):
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{SEED}:{seed_suffix}:{row.get('candidate_id', row.get('mechanism_candidate_id'))}".encode()
        ).hexdigest(),
    )


def normalized_manhattan(left, right, scales):
    return sum(abs(a - b) / scale for a, b, scale in zip(left, right, scales))


def neighbor_order(rows, workload_id):
    """Rieber-inspired online order: random seed, then nearest to observed valid."""
    remaining = deterministic_order(rows, f"neighbor:{workload_id}")
    vectors = {row["candidate_id"]: hardware_features(row) for row in rows}
    columns = list(zip(*(vectors[row["candidate_id"]] for row in rows)))
    scales = [max(column) - min(column) or 1.0 for column in columns]
    selected = []
    valid_vectors = []
    while remaining:
        if not valid_vectors:
            row = remaining.pop(0)
        else:
            row = min(
                remaining,
                key=lambda item: (
                    min(normalized_manhattan(vectors[item["candidate_id"]], v, scales) for v in valid_vectors),
                    item["candidate_id"],
                ),
            )
            remaining.remove(row)
        selected.append(row)
        if row["status"] == "ok":
            valid_vectors.append(vectors[row["candidate_id"]])
    return selected


def summarize_validity_order(order):
    return {
        str(budget): {
            "valid": sum(row["status"] == "ok" for row in order[:budget]),
            "invalid": sum(row["status"] != "ok" for row in order[:budget]),
        }
        for budget in VALIDITY_BUDGETS
    }


def validity_analysis(rows):
    by_workload = defaultdict(list)
    for row in rows:
        by_workload[row["workload_id"]].append(row)
    result = {"protocol": "leave-one-workload-out; local lowering eligibility only", "workloads": {}}
    aggregate = defaultdict(lambda: defaultdict(list))
    for workload_id in sorted(by_workload):
        test = by_workload[workload_id]
        train = [row for other, group in by_workload.items() if other != workload_id for row in group]
        policies = {"r1_neighbor": neighbor_order(test, workload_id)}
        for feature_name, feature_fn in (("ml2_base", base_features), ("ml2_hardware", hardware_features)):
            model = RandomForestClassifier(
                n_estimators=256, max_depth=6, min_samples_leaf=2,
                class_weight="balanced", random_state=SEED, n_jobs=1,
            )
            model.fit(np.asarray([feature_fn(row) for row in train]), np.asarray([row["status"] == "ok" for row in train]))
            probabilities = model.predict_proba(np.asarray([feature_fn(row) for row in test]))[:, 1]
            policies[feature_name] = [
                row for _, _, row in sorted(
                    zip((-probabilities).tolist(), [row["candidate_id"] for row in test], test),
                    key=lambda item: (item[0], item[1]),
                )
            ]
        random_summaries = []
        for replicate in range(1000):
            random_summaries.append(summarize_validity_order(deterministic_order(test, f"random:{workload_id}:{replicate}")))
        workload_result = {name: summarize_validity_order(order) for name, order in policies.items()}
        workload_result["random"] = {
            str(budget): {
                "valid_mean": statistics.fmean(item[str(budget)]["valid"] for item in random_summaries),
                "invalid_mean": statistics.fmean(item[str(budget)]["invalid"] for item in random_summaries),
            }
            for budget in VALIDITY_BUDGETS
        }
        result["workloads"][workload_id] = workload_result
        for name, summary in workload_result.items():
            if name == "random":
                for budget in VALIDITY_BUDGETS:
                    aggregate[name][str(budget)].append(summary[str(budget)]["valid_mean"])
            else:
                for budget in VALIDITY_BUDGETS:
                    aggregate[name][str(budget)].append(summary[str(budget)]["valid"])
    result["aggregate_mean_valid"] = {
        name: {budget: statistics.fmean(values) for budget, values in budgets.items()}
        for name, budgets in aggregate.items()
    }
    result["candidate_count"] = len(rows)
    result["valid_count"] = sum(row["status"] == "ok" for row in rows)
    return result


def relative_delta(mechanism, control, names):
    return [
        (float(mechanism.get(name, 0.0)) - float(control.get(name, 0.0)))
        / max(abs(float(control.get(name, 0.0))), 1.0)
        for name in names
    ]


def request_signature_vector(row):
    signature = row["transfer_signature"]
    totals = signature["totals"]
    maxima = signature["max_request_bytes_by_memory"]
    histogram = signature["descriptor_aggregates"]["exact_request_bytes_histogram"]
    return [
        float(totals.get("load_average_request_bytes", 0.0)),
        float(totals.get("store_average_request_bytes", 0.0)),
        float(maxima.get("inp", 0.0)),
        float(maxima.get("wgt", 0.0)),
        float(maxima.get("out", 0.0)),
        float(signature["descriptor_aggregates"].get("descriptor_rows", 0.0)),
        float(signature["descriptor_aggregates"].get("expanded_calls", 0.0)),
        float(signature["descriptor_aggregates"]["padded_calls"].get("load", 0.0)),
        float(signature["descriptor_aggregates"]["padded_calls"].get("store", 0.0)),
        float(len(histogram.get("load", {}))),
        float(len(histogram.get("store", {}))),
    ]


def normalized_vector_delta(mechanism, control):
    return [(a - b) / max(abs(b), 1.0) for a, b in zip(mechanism, control)]


def performance_features(pair, entries, lowered_entries, feature_set):
    mechanism = entries[pair["mechanism_candidate_id"]]
    control = entries[pair["control_candidate_id"]]
    features = base_features(mechanism)
    if feature_set in ("dma_bytes", "request_shape", "joint"):
        features += relative_delta(mechanism["static_metrics"], control["static_metrics"], DMA_BYTES)
    if feature_set in ("request_shape", "joint"):
        features += relative_delta(mechanism["static_metrics"], control["static_metrics"], REQUEST_SHAPE)
        features += normalized_vector_delta(
            request_signature_vector(lowered_entries[pair["mechanism_candidate_id"]]),
            request_signature_vector(lowered_entries[pair["control_candidate_id"]]),
        )
    if feature_set == "joint":
        features += relative_delta(mechanism["static_metrics"], control["static_metrics"], COMMAND)
        features += hardware_features(mechanism)[len(base_features(mechanism)):]
    return features


def ranked_metrics(order, budget):
    prefix = order[: min(budget, len(order))]
    best = max(float(pair["improvement_percent"]) for pair in prefix)
    oracle = max(float(pair["improvement_percent"]) for pair in order)
    return {
        "best_improvement_percent": best,
        "regret_percentage_points": oracle - best,
        "all_selected_slow_down": all(float(pair["improvement_percent"]) < 0.0 for pair in prefix),
    }


def performance_analysis(pairs, contract, lowered_rows):
    entries = {
        row["candidate_id"]: row
        for workload in contract["workloads"].values()
        for row in workload["gross_candidates"]
    }
    lowered_entries = {row["candidate_id"]: row for row in lowered_rows if row["status"] == "ok"}
    missing = {
        pair[key]
        for pair in pairs
        for key in ("mechanism_candidate_id", "control_candidate_id")
        if pair[key] not in lowered_entries
    }
    if missing:
        raise ValueError(f"missing lowered request signatures for {len(missing)} same-tile identities")
    by_workload = defaultdict(list)
    for pair in pairs:
        by_workload[pair["workload_id"]].append(pair)
    result = {
        "protocol": "leave-one-workload-out development replay on same-tile FPGA pairs",
        "target": "mechanism improvement percent versus identical ConfigEntity original",
        "workloads": {},
    }
    feature_sets = ("mode_context", "dma_bytes", "request_shape", "joint")
    aggregates = defaultdict(lambda: defaultdict(list))
    sign_accuracy = defaultdict(list)
    for workload_id in sorted(by_workload):
        test = by_workload[workload_id]
        train = [pair for other, group in by_workload.items() if other != workload_id for pair in group]
        workload_result = {"oracle_improvement_percent": max(pair["improvement_percent"] for pair in test)}
        for feature_set in feature_sets:
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            model.fit(
                np.asarray([performance_features(pair, entries, lowered_entries, feature_set) for pair in train]),
                np.asarray([pair["improvement_percent"] for pair in train]),
            )
            predictions = model.predict(np.asarray([performance_features(pair, entries, lowered_entries, feature_set) for pair in test]))
            order = [
                pair for _, _, pair in sorted(
                    zip((-predictions).tolist(), [pair["mechanism_candidate_id"] for pair in test], test),
                    key=lambda item: (item[0], item[1]),
                )
            ]
            metrics = {str(budget): ranked_metrics(order, budget) for budget in PERFORMANCE_BUDGETS}
            workload_result[feature_set] = {
                "sign_accuracy": statistics.fmean(
                    float((prediction >= 0.0) == (pair["improvement_percent"] >= 0.0))
                    for prediction, pair in zip(predictions, test)
                ),
                "budgets": metrics,
            }
            sign_accuracy[feature_set].append(workload_result[feature_set]["sign_accuracy"])
            for budget in PERFORMANCE_BUDGETS:
                aggregates[feature_set][str(budget)].append(metrics[str(budget)]["regret_percentage_points"])
        random_runs = []
        for replicate in range(1000):
            order = deterministic_order(test, f"perf:{workload_id}:{replicate}")
            random_runs.append({str(b): ranked_metrics(order, b) for b in PERFORMANCE_BUDGETS})
        workload_result["random"] = {
            str(budget): {
                "mean_regret_percentage_points": statistics.fmean(run[str(budget)]["regret_percentage_points"] for run in random_runs),
                "slowdown_only_rate": statistics.fmean(float(run[str(budget)]["all_selected_slow_down"]) for run in random_runs),
            }
            for budget in PERFORMANCE_BUDGETS
        }
        result["workloads"][workload_id] = workload_result
    result["aggregate"] = {
        feature_set: {
            "mean_sign_accuracy": statistics.fmean(sign_accuracy[feature_set]),
            "mean_regret_percentage_points": {
                budget: statistics.fmean(values) for budget, values in aggregates[feature_set].items()
            },
        }
        for feature_set in feature_sets
    }
    result["pair_count"] = len(pairs)
    return result


def write_summary(path, result):
    validity = result["validity"]
    performance = result["performance"]
    lines = [
        "# 论文启发式搜索离线消融", "",
        "> 性质：冻结历史数据上的开发性 replay；不是论文原样复现，不是新的 FPGA 确认实验。", "",
        "## 合法性初始化（310 个候选，按 workload 留一）", "",
        "| 方法 | budget=4 合法数 | budget=8 合法数 | budget=16 合法数 |", "|---|---:|---:|---:|",
    ]
    names = (("random", "Random"), ("r1_neighbor", "Rieber-inspired 邻域"), ("ml2_base", "ML²-inspired 基础 V 模型"), ("ml2_hardware", "ML²-inspired 硬件 V 模型"))
    for key, label in names:
        values = validity["aggregate_mean_valid"][key]
        lines.append(f"| {label} | {values['4']:.2f} | {values['8']:.2f} | {values['16']:.2f} |")
    lines += ["", "这里的 label 仅为本地 lowering 成功/失败，不能外推为真实 FPGA correctness。", "", "## 驻留增益排序（56 个真实 FPGA same-tile 配对，按 workload 留一）", "", "| 特征 | 符号准确率 | regret@1 | regret@2 | regret@4 | regret@8 |", "|---|---:|---:|---:|---:|---:|"]
    labels = (("mode_context", "模式+计算上下文"), ("dma_bytes", "+总 DMA bytes"), ("request_shape", "+完整请求形态"), ("joint", "+命令/硬件联合"))
    for key, label in labels:
        row = performance["aggregate"][key]
        regret = row["mean_regret_percentage_points"]
        lines.append(f"| {label} | {row['mean_sign_accuracy'] * 100:.1f}% | {regret['1']:.2f} | {regret['2']:.2f} | {regret['4']:.2f} | {regret['8']:.2f} |")
    req = performance["aggregate"]["request_shape"]["mean_regret_percentage_points"]
    dma = performance["aggregate"]["dma_bytes"]["mean_regret_percentage_points"]
    wins = sum(req[str(b)] < dma[str(b)] for b in PERFORMANCE_BUDGETS)
    lines += ["", f"完整请求形态在 {wins}/{len(PERFORMANCE_BUDGETS)} 个预算点严格优于 bytes-only；这只是跨四个已曝光 workload 的开发证据。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validity-pool", action="append", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--same-tile-effects", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite analysis directory")
    pool_paths = [Path(path).resolve() for path in args.validity_pool]
    contract_path = Path(args.contract).resolve()
    pairs_path = Path(args.same_tile_effects).resolve()
    rows = [row for path in pool_paths for row in read_jsonl(path)]
    candidate_ids = [row["candidate_id"] for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("duplicate candidate IDs across validity pools")
    contract = json.loads(contract_path.read_text())
    effects = json.loads(pairs_path.read_text())
    result = {
        "schema": "c3_literature_guided_offline_replay_v1",
        "status": "development_only_not_prospective_confirmation",
        "seed": SEED,
        "inputs": {
            "validity_pools": [{"path": str(path), "sha256": sha256(path)} for path in pool_paths],
            "contract": {"path": str(contract_path), "sha256": sha256(contract_path)},
            "same_tile_effects": {"path": str(pairs_path), "sha256": sha256(pairs_path)},
        },
        "method_boundaries": {
            "r1_neighbor": "paper-inspired normalized Manhattan valid-neighborhood replay; not exact reproduction",
            "ml2": "paper-inspired separate validity model; no accuracy model and no compiler-hidden-feature stage",
            "performance": "four-workload leave-one-out replay; cannot establish unseen-workload generalization",
        },
        "validity": validity_analysis(rows),
        "performance": performance_analysis(effects["pairs"], contract, rows),
    }
    output.mkdir(parents=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(output / "RESULTS.md", result)


if __name__ == "__main__":
    main()
