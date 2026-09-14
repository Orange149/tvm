#!/usr/bin/env python3
"""Calibrate and evaluate explicit same-tile VTA latency-delta proxies.

This is an offline, leave-one-workload-out analysis.  It models the measured
latency difference ``mechanism - same-tile original``.  Physical feature
coefficients are constrained to be non-negative, while per-mode intercepts
remain unconstrained and absorb schedule overheads that the counters do not
describe.  The calibrated proxies are neither theoretical bounds nor board
performance measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr


SCHEMA = "c3_p7r_analytic_delta_model_v1"
RIDGE_ALPHA = 1e-6
RANKING_BUDGETS = (1, 2, 4)

BYTE_CALL_FEATURES = (
    "load_bytes",
    "store_bytes",
    "load_calls",
    "store_calls",
)
REQUEST_FEATURES = (
    "small_dma_calls",
    "strided_dma_calls",
    "padded_dma_calls",
    "input_reload",
    "weight_reload",
    "output_reload",
)
COMMAND_FEATURES = (
    "total_insn_bytes",
    "total_uop_bytes",
    "submissions",
    "finish_count",
)
MODEL_FEATURES = {
    "constant_mean": (),
    "mode_mean": (),
    "bytes_calls_nnls": BYTE_CALL_FEATURES,
    "request_shape_nnls": BYTE_CALL_FEATURES + REQUEST_FEATURES,
    "command_nnls": BYTE_CALL_FEATURES + REQUEST_FEATURES + COMMAND_FEATURES,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def candidate_metrics(entry, command_entry):
    static = entry["static_metrics"]
    structural = command_entry["command_signature"]["structural"]
    totals = structural["totals"]
    values = {
        "load_bytes": float(static["input_dma_bytes"] + static["weight_dma_bytes"]),
        "store_bytes": float(static["output_dma_bytes"]),
        "load_calls": float(static["input_dma_calls"] + static["weight_dma_calls"]),
        "store_calls": float(static["output_dma_calls"]),
        "small_dma_calls": float(static["small_dma_calls"]),
        "strided_dma_calls": float(static["strided_dma_calls"]),
        "padded_dma_calls": float(static["padded_dma_calls"]),
        "input_reload": float(static["input_reload"]),
        "weight_reload": float(static["weight_reload"]),
        "output_reload": float(static["output_reload"]),
        "total_insn_bytes": float(totals["insn_bytes"]),
        "total_uop_bytes": float(totals["uop_bytes"]),
        "submissions": float(structural["submissions"]),
        "finish_count": float(structural["finish"]["source_derived_count"]),
    }
    checks = {
        "total_insn_bytes": static["total_insn_bytes"],
        "total_uop_bytes": static["total_uop_bytes"],
        "submissions": static["submissions"],
        "finish_count": static["source_derived_finish_count"],
    }
    for name, expected in checks.items():
        if not math.isclose(values[name], float(expected), rel_tol=0.0, abs_tol=0.0):
            raise ValueError("contract/P4j command mismatch for {}".format(name))
    return values


def build_pairs(effects, contract, command_rows):
    if int(effects["overall"]["pair_count"]) != len(effects["pairs"]):
        raise ValueError("same-tile pair count mismatch")
    entries = {}
    for workload in contract["workloads"].values():
        for entry in workload["gross_candidates"]:
            candidate_id = entry["candidate_id"]
            if candidate_id in entries:
                raise ValueError("duplicate contract candidate_id")
            entries[candidate_id] = entry
    commands = {}
    for entry in command_rows:
        candidate_id = entry["candidate_id"]
        if candidate_id in commands:
            raise ValueError("duplicate P4j candidate_id")
        commands[candidate_id] = entry

    pairs = []
    for raw in effects["pairs"]:
        control_id = raw["control_candidate_id"]
        mechanism_id = raw["mechanism_candidate_id"]
        if control_id not in entries or mechanism_id not in entries:
            raise ValueError("pair candidate absent from timing contract")
        if control_id not in commands or mechanism_id not in commands:
            raise ValueError("pair candidate absent from P4j command signatures")
        control = entries[control_id]
        mechanism = entries[mechanism_id]
        if control["workload_id"] != mechanism["workload_id"]:
            raise ValueError("cross-workload pair")
        if control["residence_mode"] != "original":
            raise ValueError("same-tile control is not original")
        if mechanism["residence_mode"] != raw["residence_mode"]:
            raise ValueError("mechanism mode mismatch")
        control_entity = control["complete_config_entity"]["entity"]
        mechanism_entity = mechanism["complete_config_entity"]["entity"]
        if control_entity != mechanism_entity:
            raise ValueError("same-tile ConfigEntity mismatch")
        control_values = candidate_metrics(control, commands[control_id])
        mechanism_values = candidate_metrics(mechanism, commands[mechanism_id])
        delta = {
            name: mechanism_values[name] - control_values[name]
            for name in BYTE_CALL_FEATURES + REQUEST_FEATURES + COMMAND_FEATURES
        }
        target = float(raw["mechanism_median_ms"]) - float(raw["control_median_ms"])
        reconstructed = -float(raw["improvement_percent"]) * float(raw["control_median_ms"]) / 100.0
        if not math.isclose(target, reconstructed, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("latency delta/improvement mismatch")
        pairs.append(
            {
                "workload_id": raw["workload_id"],
                "residence_mode": raw["residence_mode"],
                "control_candidate_id": control_id,
                "mechanism_candidate_id": mechanism_id,
                "config_index": int(raw["mechanism_config_index"]),
                "control_median_ms": float(raw["control_median_ms"]),
                "mechanism_median_ms": float(raw["mechanism_median_ms"]),
                "delta_t_ms": target,
                "delta_features": delta,
            }
        )
    return pairs


def feature_scales(rows, feature_names):
    scales = {}
    for name in feature_names:
        nonzero = [abs(row["delta_features"][name]) for row in rows if row["delta_features"][name] != 0]
        scales[name] = statistics.median(nonzero) if nonzero else 1.0
    return scales


def fit_proxy(train, modes, feature_names):
    scales = feature_scales(train, feature_names)
    design = []
    target = []
    for row in train:
        design.append(
            [float(row["residence_mode"] == mode) for mode in modes]
            + [row["delta_features"][name] / scales[name] for name in feature_names]
        )
        target.append(row["delta_t_ms"])
    matrix = np.asarray(design, dtype="float64")
    values = np.asarray(target, dtype="float64")
    lower = np.asarray([-np.inf] * len(modes) + [0.0] * len(feature_names))
    upper = np.full(matrix.shape[1], np.inf)
    if feature_names:
        penalty = np.zeros((len(feature_names), matrix.shape[1]), dtype="float64")
        penalty[:, len(modes) :] = np.eye(len(feature_names)) * math.sqrt(RIDGE_ALPHA)
        matrix = np.vstack([matrix, penalty])
        values = np.concatenate([values, np.zeros(len(feature_names))])
    solution = lsq_linear(matrix, values, bounds=(lower, upper), method="trf", lsmr_tol="auto")
    if not solution.success:
        raise RuntimeError("bounded least-squares fit failed: {}".format(solution.message))
    coefficients = solution.x
    return {
        "modes": list(modes),
        "features": list(feature_names),
        "scales": scales,
        "mode_intercepts_ms": {
            mode: float(coefficients[index]) for index, mode in enumerate(modes)
        },
        "scaled_nonnegative_coefficients": {
            name: float(coefficients[len(modes) + index])
            for index, name in enumerate(feature_names)
        },
        "raw_unit_coefficients_ms": {
            name: float(coefficients[len(modes) + index] / scales[name])
            for index, name in enumerate(feature_names)
        },
    }


def predict_proxy(model, row):
    value = model["mode_intercepts_ms"][row["residence_mode"]]
    for name in model["features"]:
        value += (
            model["scaled_nonnegative_coefficients"][name]
            * row["delta_features"][name]
            / model["scales"][name]
        )
    return float(value)


def dcg(relevances):
    return sum(value / math.log2(index + 2.0) for index, value in enumerate(relevances))


def ranking_metrics(rows, prediction_key):
    ordered = sorted(rows, key=lambda row: (row[prediction_key], row["mechanism_candidate_id"]))
    ideal = sorted(rows, key=lambda row: (row["delta_t_ms"], row["mechanism_candidate_id"]))
    worst = max(row["delta_t_ms"] for row in rows)
    gains = [max(0.0, worst - row["delta_t_ms"]) for row in ordered[:4]]
    ideal_gains = [max(0.0, worst - row["delta_t_ms"]) for row in ideal[:4]]
    ideal_dcg = dcg(ideal_gains)
    oracle = ideal[0]["delta_t_ms"]
    metrics = {
        "candidate_count": len(rows),
        "oracle_delta_t_ms": oracle,
        "ndcg_at_4": 1.0 if ideal_dcg == 0 else dcg(gains) / ideal_dcg,
    }
    for budget in RANKING_BUDGETS:
        selected = ordered[: min(budget, len(ordered))]
        best = min(row["delta_t_ms"] for row in selected)
        metrics["regret_at_{}_ms".format(budget)] = best - oracle
    return metrics


def mean(values):
    return statistics.fmean(float(value) for value in values)


def evaluate(pairs):
    workloads = sorted({row["workload_id"] for row in pairs})
    modes = sorted({row["residence_mode"] for row in pairs})
    if len(workloads) < 2:
        raise ValueError("leave-one-workload-out needs at least two workloads")
    predictions = []
    folds = {}
    for held_out in workloads:
        train = [row for row in pairs if row["workload_id"] != held_out]
        test = [row for row in pairs if row["workload_id"] == held_out]
        global_mean = mean(row["delta_t_ms"] for row in train)
        mode_means = {}
        for mode in modes:
            values = [row["delta_t_ms"] for row in train if row["residence_mode"] == mode]
            mode_means[mode] = mean(values) if values else global_mean
        fold_models = {}
        for model_name, feature_names in MODEL_FEATURES.items():
            if model_name in ("constant_mean", "mode_mean"):
                continue
            fold_models[model_name] = fit_proxy(train, modes, feature_names)
        folds[held_out] = {
            "train_workloads": sorted(set(workloads) - {held_out}),
            "train_pairs": len(train),
            "test_pairs": len(test),
            "constant_mean_ms": global_mean,
            "mode_means_ms": mode_means,
            "proxy_fits": fold_models,
        }
        for source in test:
            row = dict(source)
            row["predictions_ms"] = {
                "constant_mean": global_mean,
                "mode_mean": mode_means[source["residence_mode"]],
            }
            for model_name, model in fold_models.items():
                row["predictions_ms"][model_name] = predict_proxy(model, source)
            predictions.append(row)

    summary = {}
    for model_name in MODEL_FEATURES:
        actual = [row["delta_t_ms"] for row in predictions]
        predicted = [row["predictions_ms"][model_name] for row in predictions]
        correlation = spearmanr(actual, predicted).statistic
        if not math.isfinite(float(correlation)):
            correlation = None
        per_workload = {}
        for workload in workloads:
            subset = [dict(row, predicted=row["predictions_ms"][model_name]) for row in predictions if row["workload_id"] == workload]
            per_workload[workload] = ranking_metrics(subset, "predicted")
        summary[model_name] = {
            "feature_names": list(MODEL_FEATURES[model_name]),
            "mae_ms": mean(abs(a - b) for a, b in zip(actual, predicted)),
            "spearman_oof": correlation,
            "ndcg_at_4_macro": mean(value["ndcg_at_4"] for value in per_workload.values()),
            "regret_at_1_ms_macro": mean(value["regret_at_1_ms"] for value in per_workload.values()),
            "regret_at_2_ms_macro": mean(value["regret_at_2_ms"] for value in per_workload.values()),
            "regret_at_4_ms_macro": mean(value["regret_at_4_ms"] for value in per_workload.values()),
            "per_workload": per_workload,
        }
    return workloads, modes, folds, predictions, summary


def render_results(result):
    lines = [
        "# P7R113 explicit analytic ΔT proxy",
        "",
        "This is a post-hoc, fully offline leave-one-workload-out analysis of 56 P7Q same-tile pairs.",
        "The fitted models are calibrated linear proxies with non-negative coefficients on physical cost deltas; they are not theoretical upper/lower bounds.",
        "Because the paired ConfigEntity is identical, MAC count, tile shape, and the compute lower bound cancel from ΔT.",
        "",
        "| Model | MAE (ms) | Spearman | nDCG@4 | regret@1 (ms) | regret@2 (ms) | regret@4 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "constant_mean": "constant mean",
        "mode_mean": "mode mean",
        "bytes_calls_nnls": "mode + bytes/calls",
        "request_shape_nnls": "+ request shape",
        "command_nnls": "+ command",
    }
    for name in MODEL_FEATURES:
        row = result["model_summary"][name]
        rho = "NA" if row["spearman_oof"] is None else "{:.4f}".format(row["spearman_oof"])
        lines.append(
            "| {} | {:.6f} | {} | {:.4f} | {:.6f} | {:.6f} | {:.6f} |".format(
                labels[name], row["mae_ms"], rho, row["ndcg_at_4_macro"],
                row["regret_at_1_ms_macro"], row["regret_at_2_ms_macro"],
                row["regret_at_4_ms_macro"],
            )
        )
    lines += [
        "",
        "Ranking is within each held-out workload; lower predicted ΔT is better. Regret@k is the best observed ΔT among the first k predictions minus the held-out workload's ΔT oracle. nDCG uses linear gain `worst ΔT - candidate ΔT`. Exact prediction ties use stable candidate ID.",
        "",
        "Claim boundary: all P7Q labels were already exposed, all measurements come from one boot, and only four ResNet18 workload groups are present. These results are model-development evidence, not prospective cross-workload or cross-boot validation.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(output_dir, inputs, result, predictions):
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite immutable output directory: {}".format(output_dir))
    output_dir.mkdir(parents=True)
    preregistered = {
        "schema": "c3_p7r113_delta_proxy_protocol_v1",
        "analysis_kind": "post_hoc_offline_development",
        "split": "leave_one_workload_out",
        "target": "delta_t_ms = mechanism_median_ms - same_tile_original_median_ms",
        "models": {name: list(features) for name, features in MODEL_FEATURES.items()},
        "physical_coefficient_constraint": "nonnegative",
        "mode_intercepts": "unconstrained",
        "ranking_budgets": list(RANKING_BUDGETS),
        "ridge_alpha_scaled_features": RIDGE_ALPHA,
        "not_a_theoretical_bound": True,
    }
    (output_dir / "preregistered.json").write_text(
        json.dumps(preregistered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in sorted(predictions, key=lambda item: (item["workload_id"], item["mechanism_candidate_id"])):
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "RESULTS.md").write_text(render_results(result), encoding="utf-8")
    (output_dir / "STATUS.md").write_text(
        "# P7R113 status\n\n- Status: `completed_offline_development`\n"
        "- Board/RPC/SSH: not used\n- Pairs: 56; split: four-fold leave-one-workload-out\n"
        "- Claim: calibrated non-negative linear physical proxy, not a theoretical bound\n",
        encoding="utf-8",
    )
    (output_dir / "command.txt").write_text(
        " ".join(json.dumps(argument) for argument in sys.argv) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "c3_p7r113_manifest_v1",
                "analysis_script": str(Path(__file__).resolve()),
                "numpy_version": np.__version__,
                "scipy_solver": "scipy.optimize.lsq_linear",
                "board_contacted": False,
                "immutable_output_policy": "refuse_if_output_directory_exists",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    hashes = {
        "schema": "c3_p7r113_artifact_hashes_v1",
        "inputs": {str(path.resolve()): sha256(path) for path in inputs},
    }
    hashes["outputs"] = {
        path.name: sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.name != "artifact_hashes.json"
    }
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--same-tile-effects", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--command-signatures", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    effects = load_json(args.same_tile_effects)
    contract = load_json(args.contract)
    expected_contract_hash = effects.get("contract", {}).get("sha256")
    if expected_contract_hash != sha256(args.contract):
        raise ValueError("same-tile effects contract hash mismatch")
    pairs = build_pairs(effects, contract, load_jsonl(args.command_signatures))
    if len(pairs) != 56:
        raise ValueError("P7R113 requires the frozen 56 same-tile pairs")
    workloads, modes, folds, predictions, model_summary = evaluate(pairs)
    result = {
        "schema": SCHEMA,
        "status": "completed_offline_development",
        "scope": "post-hoc P7Q single-boot labels; no board contact",
        "claim_boundary": [
            "calibrated linear proxy, not a theoretical upper or lower bound",
            "leave-one-ResNet18-workload-out development evidence, not an unseen-network claim",
            "same-tile delta model only; it does not rank absolute tile/thread configurations",
            "command fields are structural P4j FSim observations, not physical AXI timing",
        ],
        "pair_count": len(pairs),
        "workloads": workloads,
        "modes": modes,
        "target": "mechanism_median_ms - same_tile_original_median_ms",
        "compute_cancellation": "identical workload and ConfigEntity make MAC/tile compute lower bound identical within each pair",
        "feature_groups": {
            "bytes_calls": list(BYTE_CALL_FEATURES),
            "request_shape": list(REQUEST_FEATURES),
            "command": list(COMMAND_FEATURES),
        },
        "model_summary": model_summary,
        "folds": folds,
        "input_sha256": {
            "same_tile_effects": sha256(args.same_tile_effects),
            "contract": sha256(args.contract),
            "command_signatures": sha256(args.command_signatures),
        },
    }
    write_outputs(
        args.output_dir,
        [args.same_tile_effects, args.contract, args.command_signatures, Path(__file__).resolve()],
        result,
        predictions,
    )


if __name__ == "__main__":
    main()
