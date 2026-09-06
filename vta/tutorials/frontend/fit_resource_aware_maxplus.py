#!/usr/bin/env python3
"""Freeze, fit, evaluate, and prospectively apply the RAMPS model."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from resource_aware_dataset import (
    DEFAULT_CONTROLLED_RESNET_SUMMARY,
    DEFAULT_RESNET_ROOTS,
    DEFAULT_YOLO_SUMMARIES,
    calibration_is_measured,
    collect_controlled_resnet_records,
    collect_resnet_records,
    collect_yolo_records,
    load_json,
    yolo_record_from_candidate,
)
from resource_aware_maxplus import (
    BASE_PARAMETER_BOUNDS,
    PipelineRecord,
    build_cycle_constraints,
    fit_ramps_model,
    grouped_bootstrap_metrics,
    maximum_cycle_mean,
    predict_record,
    ranking_metrics,
    residual_feature_values,
    scaled_stage_service,
    sha256_file,
    timed_event_graph_payload,
    write_json,
)


DEFAULT_OUTPUT = (
    "vta/tutorials/frontend/report_out/resource_aware_maxplus/20260713_v1"
)
DEFAULT_CALIBRATION = DEFAULT_OUTPUT + "/calibration/cost_model.json"
DEFAULT_OLD_HEURISTIC = (
    "vta/tutorials/frontend/report_out/yolov3_tiny_heuristic_search/"
    "heuristic_params_resnet18_v23_200.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--resnet-root", action="append", default=[])
    parser.add_argument("--yolo-summary", action="append", default=[])
    parser.add_argument("--calibration-json", default=DEFAULT_CALIBRATION)
    parser.add_argument("--controlled-summary", default=DEFAULT_CONTROLLED_RESNET_SUMMARY)
    parser.add_argument("--require-controlled", action="store_true")
    parser.add_argument("--require-measured-calibration", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--rank-weight", type=float, default=0.20)
    parser.add_argument("--expected-resnet", type=int, default=200)
    parser.add_argument("--expected-controlled", type=int, default=72)
    parser.add_argument("--expected-yolo", type=int, default=169)
    parser.add_argument("--select-prospective", action="store_true")
    parser.add_argument("--prospective-count", type=int, default=30)
    parser.add_argument("--force-protocol", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def record_csv_row(record: PipelineRecord) -> Dict[str, Any]:
    features = residual_feature_values(record)
    row = {
        "model": record.model,
        "candidate_id": record.candidate_id,
        "group_id": record.group_id,
        "source_path": record.source_path,
        "measured_cycle_ms": record.measured_cycle_ms,
        "measured_fps": 1000.0 / record.measured_cycle_ms
        if record.measured_cycle_ms
        else "",
        "queue_depth": record.queue_depth,
        "stage_count": len(record.stages),
        "stage_devices": "/".join(item.device for item in record.stages),
        "stage_services_json": json.dumps([item.__dict__ for item in record.stages], sort_keys=True),
        "boundary_ms": record.boundary_ms,
        "fragmentation_ms": record.fragmentation_ms,
        "spill_ratio": record.spill_ratio,
        "avg_bytes_per_call": record.avg_bytes_per_call,
        "effective_segment_count": record.effective_segment_count,
        "vta_island_count": record.vta_island_count,
        "small_island_count": record.small_island_count,
        "legacy_score_ms": record.legacy_score_ms,
        "correctness_passed": record.correctness_passed,
        "metadata_json": json.dumps(record.metadata, sort_keys=True),
    }
    row.update(features)
    return row


def protocol_payload(
    args: argparse.Namespace,
    resnet_paths: Sequence[Path],
    yolo_paths: Sequence[Path],
    calibration: Mapping[str, Any],
    controlled_paths: Sequence[Path] = (),
) -> Dict[str, Any]:
    files = []
    for path in list(resnet_paths) + list(yolo_paths) + list(controlled_paths):
        files.append({"path": str(path), "sha256": sha256_file(path)})
    calibration_path = Path(args.calibration_json)
    if calibration_path.exists():
        files.append({"path": str(calibration_path), "sha256": sha256_file(calibration_path)})
    implementation_files = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("resource_aware_maxplus.py"),
        Path(__file__).resolve().with_name("resource_aware_dataset.py"),
    ]
    for path in implementation_files:
        files.append({"path": str(path), "sha256": sha256_file(path), "role": "implementation"})
    return {
        "protocol_version": 1,
        "frozen": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "board": "root@192.168.1.228",
        "rpc_port": 9090,
        "training_models": ["resnet18"],
        "zero_shot_test_models": ["yolov3_tiny"],
        "no_yolo_label_tuning": True,
        "label_access_policy": {
            "yolo_source_files_hashed_before_fit": True,
            "yolo_throughput_parsed_only_after_protocol_and_model_freeze": True,
            "yolo_labels_used_for_fitting_or_feature_selection": False,
        },
        "folds": int(args.folds),
        "bootstrap_repetitions": int(args.bootstrap),
        "seed": int(args.seed),
        "rank_weight": float(args.rank_weight),
        "model_specification": {
            "base_parameter_names": list(BASE_PARAMETER_BOUNDS),
            "base_parameter_bounds": {
                name: list(bounds) for name, bounds in BASE_PARAMETER_BOUNDS.items()
            },
            "residual": "nonnegative monotonic hinge GAM",
            "residual_knots": "training-only quartiles 0.25,0.50,0.75",
            "conformal": "grouped split absolute residual, 20 percent groups",
            "risk_quantile_multiplier": 1.645,
            "forbidden_features": [
                "model_name",
                "layer_name",
                "candidate_id",
                "measured_stage_time",
            ],
        },
        "expected_samples": {
            "resnet18": int(args.expected_resnet),
            "resnet18_controlled": int(args.expected_controlled),
            "yolov3_tiny": int(args.expected_yolo),
        },
        "calibration_measured": calibration_is_measured(calibration),
        "calibration_source": calibration.get("source", "missing"),
        "source_files": files,
        "prospective_policy": {
            "ramps": 10,
            "balance_only": 10,
            "stratified_random": 10,
            "labels_frozen_until_all_complete": True,
        },
    }


def freeze_protocol(
    path: Path, payload: Mapping[str, Any], force: bool
) -> Dict[str, Any]:
    if path.exists() and not force:
        previous = load_json(path)
        previous_files = previous.get("source_files", [])
        current_files = payload.get("source_files", [])
        if previous_files != current_files:
            raise RuntimeError(
                "frozen protocol source hashes changed; use a new output directory, "
                "or --force-protocol only for an intentional new protocol"
            )
        return dict(previous)
    write_json(path, payload)
    return dict(payload)


def group_folds(records: Sequence[PipelineRecord], folds: int, seed: int) -> List[List[int]]:
    groups: Dict[str, List[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.group_id, []).append(index)
    names = sorted(
        groups,
        key=lambda name: hashlib.sha256("{}:{}".format(seed, name).encode()).hexdigest(),
    )
    result = [[] for _ in range(max(2, min(int(folds), len(names))))]
    loads = [0 for _ in result]
    for name in sorted(names, key=lambda item: len(groups[item]), reverse=True):
        fold = min(range(len(result)), key=lambda index: loads[index])
        result[fold].extend(groups[name])
        loads[fold] += len(groups[name])
    return result


def cross_validate(
    records: Sequence[PipelineRecord],
    folds: int,
    seed: int,
    rank_weight: float,
    identification_records: Sequence[PipelineRecord] = (),
) -> Dict[str, Any]:
    fold_indices = group_folds(records, folds, seed)
    prediction = np.zeros(len(records), dtype="float64")
    interval90 = np.zeros((len(records), 2), dtype="float64")
    interval95 = np.zeros((len(records), 2), dtype="float64")
    rows = []
    for fold, test_indices in enumerate(fold_indices):
        test_set = set(test_indices)
        train = [item for index, item in enumerate(records) if index not in test_set]
        test = [records[index] for index in test_indices]
        model = fit_ramps_model(
            train,
            rank_weight=rank_weight,
            seed=seed + fold,
            identification_records=identification_records,
        )
        predicted = [predict_record(item, model) for item in test]
        values = [item["predicted_cycle_ms"] for item in predicted]
        for index, value, item in zip(test_indices, values, predicted):
            prediction[index] = value
            interval90[index] = item["prediction_interval_90_ms"]
            interval95[index] = item["prediction_interval_95_ms"]
        rows.append(
            {
                "fold": fold,
                "train_count": len(train),
                "test_count": len(test),
                "test_groups": sorted(set(item.group_id for item in test)),
                "metrics": ranking_metrics(test, values),
            }
        )
    return {
        "folds": rows,
        "aggregate_metrics": {
            **ranking_metrics(records, prediction.tolist()),
            **interval_coverage_metrics(records, interval90, interval95),
        },
        "predictions": prediction.tolist(),
        "prediction_intervals_90_ms": interval90.tolist(),
        "prediction_intervals_95_ms": interval95.tolist(),
    }


def interval_coverage_metrics(
    records: Sequence[PipelineRecord],
    interval90: Sequence[Sequence[float]],
    interval95: Sequence[Sequence[float]],
) -> Dict[str, float]:
    target = np.asarray([item.measured_cycle_ms for item in records], dtype="float64")
    result: Dict[str, float] = {}
    for label, values in (("90", interval90), ("95", interval95)):
        interval = np.asarray(values, dtype="float64")
        if interval.shape != (len(records), 2):
            raise ValueError("prediction interval shape mismatch")
        covered = (target >= interval[:, 0]) & (target <= interval[:, 1])
        result["prediction_interval_{}_coverage".format(label)] = float(np.mean(covered))
        result["prediction_interval_{}_mean_width_ms".format(label)] = float(
            np.mean(interval[:, 1] - interval[:, 0])
        )
    return result


def frozen_model_interval_metrics(
    records: Sequence[PipelineRecord], model: Mapping[str, Any]
) -> Dict[str, float]:
    predicted = [predict_record(item, model) for item in records]
    return interval_coverage_metrics(
        records,
        [item["prediction_interval_90_ms"] for item in predicted],
        [item["prediction_interval_95_ms"] for item in predicted],
    )


def stage_service_metrics(
    records: Sequence[PipelineRecord], model: Mapping[str, Any]
) -> Dict[str, float]:
    parameters = model.get("base_parameters", {}) or {}
    measured = []
    predicted = []
    for record in records:
        for stage in record.stages:
            if stage.measured_ms is None or float(stage.measured_ms) <= 0.0:
                continue
            measured.append(float(stage.measured_ms))
            predicted.append(scaled_stage_service(stage, parameters))
    if not measured:
        return {"stage_sample_count": 0.0}
    target = np.asarray(measured, dtype="float64")
    estimate = np.asarray(predicted, dtype="float64")
    return {
        "stage_sample_count": float(len(target)),
        "stage_service_mae_ms": float(np.mean(np.abs(estimate - target))),
        "stage_service_rmse_ms": float(np.sqrt(np.mean((estimate - target) ** 2))),
        "stage_service_mape": float(
            np.mean(np.abs(estimate - target) / np.maximum(target, 1.0e-9))
        ),
    }


def controlled_queue_depth_metrics(
    records: Sequence[PipelineRecord], model: Mapping[str, Any]
) -> Dict[str, float]:
    groups: Dict[Tuple[str, str], List[PipelineRecord]] = {}
    for record in records:
        key = (
            str(record.metadata.get("base_candidate_id", "")),
            str(record.metadata.get("cpu_thread_policy", "")),
        )
        groups.setdefault(key, []).append(record)
    pair_count = 0
    directional = 0
    measured_nonincreasing = 0
    predicted_nonincreasing = 0
    for values in groups.values():
        ordered = sorted(values, key=lambda item: item.queue_depth)
        for left_index in range(len(ordered)):
            for right_index in range(left_index + 1, len(ordered)):
                left = ordered[left_index]
                right = ordered[right_index]
                if left.queue_depth == right.queue_depth:
                    continue
                measured_delta = float(right.measured_cycle_ms) - float(left.measured_cycle_ms)
                predicted_delta = (
                    predict_record(right, model)["predicted_cycle_ms"]
                    - predict_record(left, model)["predicted_cycle_ms"]
                )
                pair_count += 1
                directional += int(
                    abs(measured_delta) <= 1.0
                    or (measured_delta < 0.0) == (predicted_delta < 0.0)
                )
                measured_nonincreasing += int(measured_delta <= 1.0)
                predicted_nonincreasing += int(predicted_delta <= 1.0)
    return {
        "queue_depth_group_count": float(len(groups)),
        "queue_depth_pair_count": float(pair_count),
        "queue_depth_directional_accuracy": directional / max(1, pair_count),
        "measured_nonincreasing_fraction": measured_nonincreasing / max(1, pair_count),
        "predicted_nonincreasing_fraction": predicted_nonincreasing / max(1, pair_count),
    }


def compute_only_score(record: PipelineRecord) -> float:
    cpu = max(
        [max(stage.compute_ms, stage.memory_ms) for stage in record.stages if stage.device == "cpu"]
        or [0.0]
    )
    vta = sum(stage.compute_ms for stage in record.stages if stage.device == "vta")
    return max(cpu, vta, 1.0e-9)


def max_stage_score(record: PipelineRecord) -> float:
    return max([stage.unscaled_service_ms for stage in record.stages] or [1.0e-9])


def resource_lower_bound(record: PipelineRecord) -> float:
    constraints = build_cycle_constraints(record)
    resources = [item for item in constraints if item.resource not in ("fifo_backpressure", "finite_buffers")]
    return maximum_cycle_mean(resources)[0]


def ridge_features(record: PipelineRecord) -> List[float]:
    cpu = [stage for stage in record.stages if stage.device == "cpu"]
    vta = [stage for stage in record.stages if stage.device == "vta"]
    residual = residual_feature_values(record)
    return [
        1.0,
        max([stage.compute_ms for stage in cpu] or [0.0]),
        sum(stage.compute_ms for stage in vta),
        sum(stage.dma_ms for stage in vta),
        record.boundary_ms,
        residual["fragmentation_ms"],
        residual["spill_ratio"],
        residual["cpu_oversubscription"],
        residual["inverse_transaction_kib"],
        residual["extra_segment_count"],
        residual["small_island_count"],
    ]


def ridge_predict(train: Sequence[PipelineRecord], test: Sequence[PipelineRecord]) -> List[float]:
    x = np.asarray([ridge_features(item) for item in train], dtype="float64")
    y = np.asarray([item.measured_cycle_ms for item in train], dtype="float64")
    xt = np.asarray([ridge_features(item) for item in test], dtype="float64")
    regularizer = np.eye(x.shape[1])
    regularizer[0, 0] = 0.0
    coefficient = np.linalg.solve(x.T.dot(x) + regularizer, x.T.dot(y))
    return np.maximum(1.0e-9, xt.dot(coefficient)).tolist()


def stratified_random_scores(
    records: Sequence[PipelineRecord], seed: int
) -> List[float]:
    """Return a randomized ordering balanced over islands and boundary quartiles."""

    boundaries = np.asarray(
        [float(item.metadata.get("boundary_bytes") or item.boundary_ms) for item in records],
        dtype="float64",
    )
    cuts = np.quantile(boundaries, [0.25, 0.50, 0.75]) if len(boundaries) else []
    strata: Dict[Tuple[int, int], List[int]] = {}
    for index, record in enumerate(records):
        boundary_bin = int(np.searchsorted(cuts, boundaries[index], side="right"))
        strata.setdefault((int(record.vta_island_count), boundary_bin), []).append(index)
    rng = random.Random(seed)
    for values in strata.values():
        rng.shuffle(values)
    keys = sorted(strata)
    rng.shuffle(keys)
    order = []
    while any(strata.values()):
        for key in keys:
            if strata[key]:
                order.append(strata[key].pop())
    scores = np.empty(len(records), dtype="float64")
    for rank, index in enumerate(order):
        scores[index] = float(rank)
    return scores.tolist()


def ablation_predictions(
    train: Sequence[PipelineRecord],
    test: Sequence[PipelineRecord],
    model: Mapping[str, Any],
    seed: int,
) -> Dict[str, List[float]]:
    return {
        "random_stratified": stratified_random_scores(test, seed),
        "flops_compute_only": [compute_only_score(item) for item in test],
        "legacy_static_score": [
            item.legacy_score_ms if item.legacy_score_ms > 0.0 else 1.0e9 for item in test
        ],
        "max_stage": [max_stage_score(item) for item in test],
        "resource_lower_bound": [resource_lower_bound(item) for item in test],
        "ramps": [
            predict_record(item, model, include_residual=False)["predicted_cycle_ms"]
            for item in test
        ],
        "ramps_residual_ablation": [
            predict_record(item, model, include_residual=True)["predicted_cycle_ms"]
            for item in test
        ],
        "unconstrained_ridge": ridge_predict(train, test),
    }


def evaluate_ablations(
    train: Sequence[PipelineRecord],
    test: Sequence[PipelineRecord],
    model: Mapping[str, Any],
    bootstrap: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], List[Dict[str, Any]]]:
    predictions = ablation_predictions(train, test, model, seed)
    rows = []
    for name, values in predictions.items():
        metrics = ranking_metrics(test, values)
        intervals = grouped_bootstrap_metrics(test, values, bootstrap, seed)
        rows.append({"method": name, **metrics, "bootstrap_json": json.dumps(intervals)})
    paired = paired_bootstrap_against_ramps(test, predictions, bootstrap, seed + 101)
    return rows, predictions, paired


def paired_bootstrap_against_ramps(
    records: Sequence[PipelineRecord],
    predictions: Mapping[str, Sequence[float]],
    repetitions: int,
    seed: int,
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.group_id, []).append(index)
    names = sorted(groups)
    rng = random.Random(seed)
    metrics = {
        "mae_ms": "lower",
        "spearman": "higher",
        "regret_at_10": "lower",
        "evaluations_to_oracle_95pct": "lower",
    }
    differences: Dict[Tuple[str, str], List[float]] = {}
    baselines = [name for name in predictions if name != "ramps"]
    for _ in range(max(1, int(repetitions))):
        selected = []
        for group_name in (rng.choice(names) for _ in names):
            selected.extend(groups[group_name])
        subset = [records[index] for index in selected]
        reference = ranking_metrics(subset, [predictions["ramps"][index] for index in selected])
        for baseline in baselines:
            measured = ranking_metrics(
                subset, [predictions[baseline][index] for index in selected]
            )
            for metric, direction in metrics.items():
                if direction == "lower":
                    improvement = float(measured[metric]) - float(reference[metric])
                else:
                    improvement = float(reference[metric]) - float(measured[metric])
                differences.setdefault((baseline, metric), []).append(improvement)
    rows = []
    for (baseline, metric), values in differences.items():
        nonpositive = sum(1 for value in values if value <= 0.0)
        rows.append(
            {
                "reference": "ramps",
                "baseline": baseline,
                "metric": metric,
                "positive_means_ramps_better": True,
                "mean_improvement": float(np.mean(values)),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "p_one_sided": float((nonpositive + 1) / (len(values) + 1)),
                "bootstrap_repetitions": len(values),
            }
        )
    ordered = sorted(range(len(rows)), key=lambda index: rows[index]["p_one_sided"])
    running = 0.0
    test_count = len(rows)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (test_count - rank) * rows[index]["p_one_sided"])
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running
        rows[index]["significant_at_0_05"] = running < 0.05
    return sorted(rows, key=lambda row: (row["metric"], row["baseline"]))


def prediction_rows(
    records: Sequence[PipelineRecord], model: Mapping[str, Any], label: str
) -> List[Dict[str, Any]]:
    rows = []
    for record in records:
        prediction = predict_record(record, model)
        rows.append(
            {
                "split": label,
                "model": record.model,
                "candidate_id": record.candidate_id,
                "group_id": record.group_id,
                "measured_cycle_ms": record.measured_cycle_ms,
                "measured_fps": 1000.0 / record.measured_cycle_ms,
                "predicted_cycle_ms": prediction["predicted_cycle_ms"],
                "predicted_fps": prediction["predicted_fps"],
                "risk_score_ms": prediction["risk_score_ms"],
                "predicted_std_ms": prediction["predicted_std_ms"],
                "bottleneck_cycle": (prediction.get("bottleneck_cycle") or {}).get("name", ""),
                "prediction_interval_90_json": json.dumps(prediction["prediction_interval_90_ms"]),
                "prediction_interval_95_json": json.dumps(prediction["prediction_interval_95_ms"]),
                "cycle_constraints_json": json.dumps(prediction["cycle_constraints"], sort_keys=True),
            }
        )
    return rows


def write_event_artifacts(
    output_dir: Path,
    records: Sequence[PipelineRecord],
    model: Mapping[str, Any],
) -> None:
    matrix_rows = []
    index_rows = []
    parameters = model.get("base_parameters", {}) or {}
    for record in records:
        payload = timed_event_graph_payload(record, parameters)
        if float(payload["spectral_validation_abs_error_ms"]) > 1.0e-8:
            raise RuntimeError(
                "max-plus spectral validation failed for {}: {}".format(
                    record.candidate_id, payload["spectral_validation_abs_error_ms"]
                )
            )
        digest = hashlib.sha256(record.candidate_id.encode()).hexdigest()[:12]
        filename = "{}_{}.json".format(
            "".join(
                value if value.isalnum() or value in "_.-" else "_"
                for value in record.candidate_id
            )[:100],
            digest,
        )
        relative = Path("event_graphs") / record.model / filename
        write_json(output_dir / relative, payload)
        index_rows.append(
            {
                "model": record.model,
                "candidate_id": record.candidate_id,
                "group_id": record.group_id,
                "event_graph_json": str(relative),
                "karp_spectral_radius_ms": payload["karp_spectral_radius_ms"],
                "bottleneck_cycle": (payload.get("bottleneck_cycle") or {}).get("name", ""),
            }
        )
        for demand in payload["resource_demand_matrix"]["rows"]:
            matrix_rows.append(
                {
                    "model": record.model,
                    "candidate_id": record.candidate_id,
                    "queue_depth": record.queue_depth,
                    **demand,
                }
            )
    write_csv(output_dir / "event_graphs" / "index.csv", index_rows)
    write_csv(output_dir / "resource_matrices" / "stage_resource_demands.csv", matrix_rows)


def select_prospective(
    output_dir: Path,
    model: Mapping[str, Any],
    measured_yolo: Sequence[PipelineRecord],
    old_heuristic_path: Path,
    seed: int,
    calibration: Mapping[str, Any],
) -> Dict[str, Any]:
    frontend = Path(__file__).resolve().parent
    if str(frontend) not in sys.path:
        sys.path.insert(0, str(frontend))
    import search_yolov3_tiny_stage_splits as yolo_search  # pylint: disable=import-outside-toplevel

    old_heuristic = load_json(old_heuristic_path)
    candidates = []
    for islands in yolo_search.enumerate_island_sets(3):
        candidate = yolo_search.make_multisplit_candidate(islands)
        yolo_search.score_multisplit_candidate(candidate, old_heuristic)
        record = yolo_record_from_candidate(
            candidate, calibration=calibration, legacy_analysis=True
        )
        prediction = predict_record(record, model)
        candidate.update(
            {
                "ramps_predicted_cycle_ms": prediction["predicted_cycle_ms"],
                "ramps_risk_score_ms": prediction["risk_score_ms"],
                "ramps_predicted_fps": prediction["predicted_fps"],
                "ramps_bottleneck_cycle": (prediction.get("bottleneck_cycle") or {}).get(
                    "name", ""
                ),
            }
        )
        candidates.append(candidate)
    measured = {item.candidate_id for item in measured_yolo}
    remaining = [item for item in candidates if item["candidate_id"] not in measured]
    remaining.sort(key=lambda item: (item["ramps_risk_score_ms"], item["candidate_id"]))
    selected = []
    used = set()

    def add(rows: Sequence[Mapping[str, Any]], group: str, count: int) -> None:
        for item in rows:
            candidate_id = str(item["candidate_id"])
            if candidate_id in used:
                continue
            row = dict(item)
            row["prospective_group"] = group
            selected.append(row)
            used.add(candidate_id)
            if sum(1 for value in selected if value["prospective_group"] == group) >= count:
                return

    add(remaining, "ramps_top", 10)
    balance = sorted(
        remaining,
        key=lambda item: (float(item.get("balanced_cycle_ms_est") or 1.0e9), item["candidate_id"]),
    )
    add(balance, "balance_only", 10)
    rng = random.Random(seed)
    random_pool = [item for item in remaining if item["candidate_id"] not in used]
    by_islands: Dict[int, List[Mapping[str, Any]]] = {}
    for item in random_pool:
        by_islands.setdefault(int(item.get("island_count") or 0), []).append(item)
    targets = {1: 3, 2: 4, 3: 3}
    random_rows = []
    for island_count, count in targets.items():
        pool = by_islands.get(island_count, [])
        rng.shuffle(pool)
        random_rows.extend(pool[:count])
    add(random_rows + random_pool, "stratified_random", 10)
    if len(selected) != 30:
        raise RuntimeError("expected 30 prospective candidates, selected {}".format(len(selected)))
    measurement_order = list(selected)
    rng.shuffle(measurement_order)
    prospective_dir = output_dir / "prospective30"
    write_csv(prospective_dir / "ranked_candidates.csv", selected)
    write_csv(prospective_dir / "measurement_order.csv", measurement_order)
    (prospective_dir / "candidate_prior.txt").write_text(
        "\n".join(item["candidate_id"] for item in selected) + "\n", encoding="utf-8"
    )
    (prospective_dir / "measurement_order.txt").write_text(
        "\n".join(item["candidate_id"] for item in measurement_order) + "\n", encoding="utf-8"
    )
    write_json(
        prospective_dir / "manifest.json",
        {
            "frozen": True,
            "seed": seed,
            "enumerated_count": len(candidates),
            "historically_measured_count": len(measured),
            "unmeasured_count": len(remaining),
            "selected_count": len(selected),
            "groups": {name: sum(1 for item in selected if item["prospective_group"] == name) for name in ("ramps_top", "balance_only", "stratified_random")},
            "candidate_ids": [item["candidate_id"] for item in selected],
            "measurement_order": [item["candidate_id"] for item in measurement_order],
        },
    )
    return {"selected": selected, "measurement_order": measurement_order}


def render_results(
    protocol: Mapping[str, Any],
    cv: Mapping[str, Any],
    resnet_metrics: Mapping[str, Any],
    yolo_metrics: Mapping[str, Any],
    ablations: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# RAMPS Experiment Results",
        "",
        "- Protocol frozen: `{}`".format(protocol.get("frozen")),
        "- Calibration measured: `{}`".format(protocol.get("calibration_measured")),
        "- Training: ResNet18 only",
        "- Zero-shot test: YOLOv3-tiny",
        "",
        "## Grouped Cross-Validation",
        "",
        "```json",
        json.dumps(cv.get("aggregate_metrics", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Frozen Model Metrics",
        "",
        "- ResNet18: `{}`".format(json.dumps(resnet_metrics, sort_keys=True)),
        "- YOLOv3-tiny zero-shot: `{}`".format(json.dumps(yolo_metrics, sort_keys=True)),
        "",
        "## YOLO Ablation",
        "",
        "| method | MAE ms | Spearman | Top10 R@20 | regret@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ablations:
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                row["method"],
                float(row["mae_ms"]),
                float(row["spearman"]),
                float(row["top10_recall_at_20"]),
                float(row["regret_at_10"]),
            )
        )
    lines.extend(
        [
            "",
            "Publication claims remain blocked until measured calibration and prospective30 are complete.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.folds < 2 or args.bootstrap <= 0:
        raise RuntimeError("--folds must be >=2 and --bootstrap must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = {}
    if Path(args.calibration_json).exists():
        calibration = load_json(Path(args.calibration_json))
    if args.require_measured_calibration and not calibration_is_measured(calibration):
        raise RuntimeError("measured calibration is required and fallback/missing data was found")

    resnet_roots = args.resnet_root or list(DEFAULT_RESNET_ROOTS)
    yolo_summaries = args.yolo_summary or list(DEFAULT_YOLO_SUMMARIES)
    resnet, resnet_paths = collect_resnet_records(resnet_roots, calibration)
    controlled, controlled_paths = collect_controlled_resnet_records(
        args.controlled_summary, resnet, calibration
    )
    # Before the model is frozen, YOLO files are only existence-checked and hashed.
    # Their throughput labels are intentionally not parsed by collect_yolo_records.
    yolo_paths = [Path(item) for item in yolo_summaries]
    missing_yolo = [str(path) for path in yolo_paths if not path.is_file()]
    if missing_yolo:
        raise RuntimeError("missing frozen YOLO test sources: {}".format(missing_yolo))
    if len(resnet) != args.expected_resnet:
        raise RuntimeError(
            "dataset count mismatch: ResNet {} (expected {})".format(
                len(resnet), args.expected_resnet
            )
        )
    if args.require_controlled and len(controlled) != args.expected_controlled:
        raise RuntimeError(
            "controlled identification count mismatch: {} (expected {})".format(
                len(controlled), args.expected_controlled
            )
        )

    protocol = freeze_protocol(
        output_dir / "protocol.json",
        protocol_payload(args, resnet_paths, yolo_paths, calibration, controlled_paths),
        args.force_protocol,
    )

    cv = cross_validate(
        resnet,
        args.folds,
        args.seed,
        args.rank_weight,
        identification_records=controlled,
    )
    write_json(output_dir / "metrics" / "resnet_grouped_cv.json", cv)
    model = fit_ramps_model(
        resnet,
        args.rank_weight,
        args.seed,
        identification_records=controlled,
    )
    model.update(
        {
            "frozen": True,
            "training_model": "resnet18",
            "zero_shot_test_model": "yolov3_tiny",
            "calibration_source": calibration.get("source", "historical_static"),
            "calibration_measured": calibration_is_measured(calibration),
            "hardware_calibration": calibration if calibration_is_measured(calibration) else {},
            "protocol_sha256": hashlib.sha256(
                json.dumps(protocol, sort_keys=True).encode()
            ).hexdigest(),
        }
    )
    write_json(output_dir / "models" / "resnet_to_yolo_frozen.json", model)

    # This receipt is written before the first YOLO throughput label is parsed.
    write_json(
        output_dir / "models" / "freeze_receipt.json",
        {
            "model_path": "models/resnet_to_yolo_frozen.json",
            "model_sha256": sha256_file(
                output_dir / "models" / "resnet_to_yolo_frozen.json"
            ),
            "protocol_sha256": model["protocol_sha256"],
            "training_record_count": len(resnet),
            "identification_record_count": len(controlled),
            "yolo_labels_parsed": False,
            "frozen_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    # Zero-shot evaluation begins only after both protocol and model are durable.
    yolo, parsed_yolo_paths = collect_yolo_records(yolo_summaries, calibration)
    if parsed_yolo_paths != yolo_paths:
        raise RuntimeError("YOLO source path order changed after model freeze")
    if len(yolo) != args.expected_yolo:
        raise RuntimeError(
            "dataset count mismatch: YOLO {} (expected {})".format(
                len(yolo), args.expected_yolo
            )
        )

    dataset_rows = [record_csv_row(item) for item in resnet + controlled + yolo]
    write_csv(output_dir / "dataset" / "resource_features.csv", dataset_rows)
    write_json(
        output_dir / "dataset" / "dataset.json",
        {"records": [item.to_dict() for item in resnet + controlled + yolo]},
    )
    write_json(
        output_dir / "dataset" / "data_manifest.json",
        {
            "resnet_count": len(resnet),
            "controlled_resnet_count": len(controlled),
            "yolo_count": len(yolo),
            "resnet_group_count": len(set(item.group_id for item in resnet)),
            "yolo_group_count": len(set(item.group_id for item in yolo)),
            "source_files": protocol["source_files"],
            "label_access_order": [
                "freeze_protocol",
                "fit_resnet_model",
                "write_frozen_model",
                "parse_yolo_throughput_labels",
                "evaluate_zero_shot",
            ],
        },
    )
    write_event_artifacts(output_dir, resnet + controlled + yolo, model)
    resnet_prediction = [predict_record(item, model)["predicted_cycle_ms"] for item in resnet]
    yolo_prediction = [predict_record(item, model)["predicted_cycle_ms"] for item in yolo]
    resnet_metrics = {
        **ranking_metrics(resnet, resnet_prediction),
        **frozen_model_interval_metrics(resnet, model),
        **stage_service_metrics(resnet, model),
    }
    yolo_metrics = {
        **ranking_metrics(yolo, yolo_prediction),
        **frozen_model_interval_metrics(yolo, model),
        **stage_service_metrics(yolo, model),
    }
    controlled_metrics = {}
    if controlled:
        controlled_prediction = [
            predict_record(item, model)["predicted_cycle_ms"] for item in controlled
        ]
        controlled_metrics = {
            **ranking_metrics(controlled, controlled_prediction),
            **stage_service_metrics(controlled, model),
            **controlled_queue_depth_metrics(controlled, model),
        }
    write_json(
        output_dir / "metrics" / "cross_model_metrics.json",
        {
            "resnet_train": resnet_metrics,
            "resnet_controlled_identification": controlled_metrics,
            "yolo_zero_shot": yolo_metrics,
            "resnet_bootstrap": grouped_bootstrap_metrics(
                resnet, resnet_prediction, args.bootstrap, args.seed
            ),
            "yolo_bootstrap": grouped_bootstrap_metrics(
                yolo, yolo_prediction, args.bootstrap, args.seed + 1
            ),
        },
    )
    write_csv(output_dir / "predictions" / "resnet.csv", prediction_rows(resnet, model, "train"))
    write_csv(output_dir / "predictions" / "yolo.csv", prediction_rows(yolo, model, "zero_shot"))
    ablation_rows, _, paired_rows = evaluate_ablations(
        resnet, yolo, model, args.bootstrap, args.seed
    )
    write_csv(output_dir / "metrics" / "ablation.csv", ablation_rows)
    write_json(output_dir / "metrics" / "ablation.json", {"rows": ablation_rows})
    write_csv(output_dir / "metrics" / "paired_bootstrap_holm.csv", paired_rows)
    write_json(
        output_dir / "metrics" / "paired_bootstrap_holm.json",
        {"rows": paired_rows},
    )

    if args.select_prospective:
        if args.prospective_count != 30:
            raise RuntimeError("the frozen prospective protocol requires exactly 30 candidates")
        select_prospective(
            output_dir,
            model,
            yolo,
            Path(DEFAULT_OLD_HEURISTIC),
            args.seed,
            calibration,
        )

    (output_dir / "PAPER_RESULTS.md").write_text(
        render_results(protocol, cv, resnet_metrics, yolo_metrics, ablation_rows),
        encoding="utf-8",
    )
    print("[RAMPS] ResNet records:", len(resnet))
    print("[RAMPS] controlled identification records:", len(controlled))
    print("[RAMPS] YOLO zero-shot records:", len(yolo))
    print("[RAMPS] calibration measured:", calibration_is_measured(calibration))
    print("[RAMPS] ResNet CV:", json.dumps(cv["aggregate_metrics"], sort_keys=True))
    print("[RAMPS] YOLO zero-shot:", json.dumps(yolo_metrics, sort_keys=True))
    print("[RAMPS] output:", output_dir)


if __name__ == "__main__":
    main()
