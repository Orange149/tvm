#!/usr/bin/env python3
"""Run the RAMPS communication-awareness go/no-go pre-experiment.

This is an oracle-service mechanism experiment, not a fitted static cost model.
All four ablations consume the same measured stage set/run/get timings and differ
only in which communication and shared-resource constraints are enabled.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import glob
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr

from resource_aware_dataset import DEFAULT_RESNET_ROOTS, collect_resnet_records
from resource_aware_maxplus import as_float


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    "vta/tutorials/frontend/report_out/resource_aware_maxplus/"
    "20260828_communication_preexperiment"
)
DEFAULT_BUILD_CACHE = (
    "vta/tutorials/frontend/report_out/native_stage_pipeline_build_cache/v23_20260506"
)
DEFAULT_RPC_BASELINE = (
    "vta/tutorials/frontend/report_out/native_stage_pipeline_searches/"
    "20260507_v23_warm100/artifacts/rpc_all_vta_baseline.jsonl"
)
DEFAULT_REMOTE_DIR = "/var/volatile/ramps_comm_preexperiment"
CAT_EQUIVALENT_TOP1 = {281, 282, 283, 284, 285}
MODEL_NAMES = ("b0_compute_only", "b1_communication", "b2_shared", "b3_full")
PRIMARY_EFFECTS = ("b1_minus_b0", "b3_minus_b2")


@dataclass
class TimingRecord:
    candidate_id: str
    group_id: str
    pair_id: str
    session: str
    source_path: str
    measured_cycle_ms: float
    stage_devices: List[str]
    stage_threads: List[int]
    stage_run_ms: List[float]
    stage_communication_ms: List[float]
    stage_service_ms: List[float]
    queue_depth: int
    b0_compute_only_ms: float
    b1_communication_ms: float
    b2_shared_ms: float
    b3_full_ms: float
    b3_bottleneck: str
    correctness_passed: bool

    def prediction(self, name: str) -> float:
        return float(getattr(self, name + "_ms"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        default="historical",
        help="Comma-separated: historical,plan,measure,analyze.",
    )
    parser.add_argument(
        "--allow-deprecated-set-get-board-protocol",
        action="store_true",
        help=(
            "Reproduce the archived set/get proxy protocol. Its board data is pilot-only "
            "and must not be used for communication inference or model fitting."
        ),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--board", default="root@192.168.1.150")
    parser.add_argument("--build-cache-dir", default=DEFAULT_BUILD_CACHE)
    parser.add_argument("--rpc-baseline-result", default=DEFAULT_RPC_BASELINE)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--resnet-root", action="append", default=[])
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--skip-first", type=int, default=5)
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--pair-count-per-transition", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--minimum-valid-candidates", type=int, default=20)
    parser.add_argument("--minimum-complete-pairs", type=int, default=10)
    parser.add_argument("--serial-timeout-s", type=int, default=180)
    parser.add_argument("--pipeline-timeout-s", type=int, default=240)
    parser.add_argument("--fetch-timeout-s", type=int, default=120)
    parser.add_argument("--scp-timeout-s", type=int, default=600)
    parser.add_argument("--ssh-timeout-s", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as inp:
        for line_number, line in enumerate(inp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise RuntimeError("{}:{} invalid JSON: {}".format(path, line_number, err)) from err
    return rows


def parse_json_value(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def median_positive(values: Iterable[float]) -> float:
    usable = [float(value) for value in values if math.isfinite(float(value)) and float(value) >= 0.0]
    return float(statistics.median(usable)) if usable else 0.0


def outer_span_from_row(row: Mapping[str, Any]) -> str:
    islands = parse_json_value(row.get("vta_islands"), []) or []
    if not islands:
        return "no_vta"
    starts = [int(item["start_idx"]) for item in islands]
    ends = [int(item["end_idx"]) for item in islands]
    return "{}_{}".format(min(starts), max(ends))


def completion_cycle_ms(rows: Sequence[Mapping[str, Any]], skip_first: int) -> float:
    pipeline = [row for row in rows if row.get("mode") == "pipeline"]
    keep = pipeline[int(skip_first) :]
    if len(keep) < 2:
        raise RuntimeError("need at least two measured pipeline frames after warmup")
    stage_count = int(keep[0].get("stage_count") or 0)
    if stage_count <= 0:
        raise RuntimeError("pipeline row has no stage_count")
    end_key = "stage{}_end_ms".format(stage_count - 1)
    completion = [float(row[end_key]) for row in keep]
    intervals = [right - left for left, right in zip(completion, completion[1:])]
    if any(value <= 0.0 for value in intervals):
        raise RuntimeError("non-positive completion interval found")
    return float((completion[-1] - completion[0]) / (len(completion) - 1))


def model_cycles(
    run_ms: Sequence[float],
    communication_ms: Sequence[float],
    devices: Sequence[str],
    threads: Sequence[int],
    queue_depth: int,
) -> Tuple[Dict[str, float], str]:
    """Evaluate the archived set/get proxy ablation.

    This is retained only to reproduce the 37-process pilot. In particular,
    thread_count * run_ms is not an identified CPU-core service demand, and the
    FIFO expressions below are approximations rather than a constructed
    max-plus timed event graph.
    """
    if not run_ms or not (
        len(run_ms) == len(communication_ms) == len(devices) == len(threads)
    ):
        raise ValueError("stage timing/device/thread vectors must be non-empty and equal length")
    run = [max(0.0, float(value)) for value in run_ms]
    comm = [max(0.0, float(value)) for value in communication_ms]
    service = [left + right for left, right in zip(run, comm)]

    b0 = max(run)
    b1 = max(service)
    b2_constraints = {
        "stage_worker": max(run),
        "vta_mutex": sum(value for value, device in zip(run, devices) if device == "vta"),
        "cpu_core_pool": sum(
            value * max(1, int(thread))
            for value, device, thread in zip(run, devices, threads)
            if device == "cpu"
        )
        / 4.0,
    }
    b2 = max(b2_constraints.values())

    depth = max(1, int(queue_depth))
    b3_constraints: Dict[str, float] = {
        "stage_worker": max(service),
        "vta_mutex": sum(value for value, device in zip(service, devices) if device == "vta"),
        "cpu_core_pool": sum(
            value * max(1, int(thread))
            for value, device, thread in zip(service, devices, threads)
            if device == "cpu"
        )
        / 4.0,
        "bridge": sum(comm),
        "finite_buffers": sum(service) / (1 + depth * max(0, len(service) - 1)),
    }
    for index in range(len(service) - 1):
        b3_constraints["fifo_{}_{}".format(index, index + 1)] = (
            service[index] + service[index + 1]
        ) / (depth + 1)
    bottleneck, b3 = max(b3_constraints.items(), key=lambda item: item[1])
    return {
        "b0_compute_only": max(b0, 1.0e-9),
        "b1_communication": max(b1, 1.0e-9),
        "b2_shared": max(b2, 1.0e-9),
        "b3_full": max(b3, 1.0e-9),
    }, bottleneck


def timing_record_from_result(
    candidate_id: str,
    group_id: str,
    pair_id: str,
    session: str,
    source_path: Path,
    devices: Sequence[str],
    threads: Sequence[int],
    queue_depth: int,
    skip_first: int,
    correctness_passed: bool,
) -> TimingRecord:
    rows = read_jsonl(source_path)
    pipeline = [row for row in rows if row.get("mode") == "pipeline"]
    keep = pipeline[int(skip_first) :]
    if not keep:
        raise RuntimeError("{} has no measured pipeline rows".format(source_path))
    stage_count = int(keep[0].get("stage_count") or 0)
    if stage_count != len(devices) or stage_count != len(threads):
        raise RuntimeError(
            "{} stage count {} does not match metadata {}/{}".format(
                candidate_id, stage_count, len(devices), len(threads)
            )
        )
    stage_run = [
        median_positive(row["stage{}_run_ms".format(index)] for row in keep)
        for index in range(stage_count)
    ]
    stage_comm = [
        median_positive(
            float(row["stage{}_set_ms".format(index)])
            + float(row["stage{}_get_ms".format(index)])
            for row in keep
        )
        for index in range(stage_count)
    ]
    predictions, bottleneck = model_cycles(
        stage_run, stage_comm, devices, threads, queue_depth
    )
    return TimingRecord(
        candidate_id=candidate_id,
        group_id=group_id,
        pair_id=pair_id,
        session=session,
        source_path=str(source_path),
        measured_cycle_ms=completion_cycle_ms(rows, skip_first),
        stage_devices=list(devices),
        stage_threads=[int(value) for value in threads],
        stage_run_ms=stage_run,
        stage_communication_ms=stage_comm,
        stage_service_ms=[left + right for left, right in zip(stage_run, stage_comm)],
        queue_depth=max(1, int(queue_depth)),
        b0_compute_only_ms=predictions["b0_compute_only"],
        b1_communication_ms=predictions["b1_communication"],
        b2_shared_ms=predictions["b2_shared"],
        b3_full_ms=predictions["b3_full"],
        b3_bottleneck=bottleneck,
        correctness_passed=bool(correctness_passed),
    )


def historical_summary_rows(roots: Sequence[str]) -> List[Tuple[Path, Mapping[str, Any]]]:
    result = []
    for root in roots:
        pattern = str(Path(root) / "batch[0-9][0-9][0-9]" / "summary.json")
        for path_string in sorted(glob.glob(pattern)):
            path = Path(path_string)
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload.get("rows", []) or []:
                result.append((path, row))
    return result


def collect_historical_records(roots: Sequence[str], skip_first: int) -> List[TimingRecord]:
    records = []
    seen = set()
    for summary_path, row in historical_summary_rows(roots):
        candidate_id = str(row.get("candidate_id") or row.get("scheme_name") or "")
        if (
            not candidate_id
            or candidate_id in seen
            or row.get("status") != "ok"
            or row.get("run_kind") != "default"
            or not bool(row.get("passes_correctness_gate", False))
        ):
            continue
        result_path = Path(str(row.get("output_dir") or "")) / "native_result.jsonl"
        if not result_path.is_file():
            continue
        stage_count = int(row.get("stage_count") or row.get("manifest_stage_count") or 0)
        devices = [str(row.get("stage{}_device".format(index)) or "") for index in range(stage_count)]
        threads = [int(row.get("stage{}_threads".format(index)) or 1) for index in range(stage_count)]
        record = timing_record_from_result(
            candidate_id=candidate_id,
            group_id=outer_span_from_row(row),
            pair_id="",
            session="historical",
            source_path=result_path,
            devices=devices,
            threads=threads,
            queue_depth=int(row.get("queue_depth") or 2),
            skip_first=skip_first,
            correctness_passed=True,
        )
        records.append(record)
        seen.add(candidate_id)
    return records


def static_b0(record: Any) -> float:
    cpu = max(
        [max(stage.compute_ms, stage.memory_ms) for stage in record.stages if stage.device == "cpu"]
        or [0.0]
    )
    vta = sum(stage.compute_ms for stage in record.stages if stage.device == "vta")
    return max(cpu, vta, 1.0e-9)


def record_outer_span(record: Any) -> str:
    parts = str(record.group_id).split("_")
    if len(parts) < 3:
        raise RuntimeError("invalid ResNet group id: {}".format(record.group_id))
    return "{}_{}".format(parts[-2], parts[-1])


def select_candidate_pairs(
    roots: Sequence[str], build_cache: Path, quota: int
) -> Dict[str, Any]:
    records, source_paths = collect_resnet_records(roots, {})
    package_root = build_cache / "buildability"
    candidates = [
        record
        for record in records
        if (package_root / record.candidate_id / "package.tar.gz").is_file()
    ]
    by_transition: Dict[str, List[Dict[str, Any]]] = {"1-3": [], "1-2": [], "2-3": []}
    for left, right in itertools.combinations(candidates, 2):
        if record_outer_span(left) != record_outer_span(right):
            continue
        transition = "{}-{}".format(
            min(left.vta_island_count, right.vta_island_count),
            max(left.vta_island_count, right.vta_island_count),
        )
        if transition not in by_transition:
            continue
        left_b0 = static_b0(left)
        right_b0 = static_b0(right)
        relative_delta = abs(left_b0 - right_b0) / max(left_b0, right_b0)
        if relative_delta > 0.01:
            continue
        low, high = (left, right) if left.boundary_ms <= right.boundary_ms else (right, left)
        ratio = (high.boundary_ms + 1.0e-9) / (low.boundary_ms + 1.0e-9)
        if ratio < 1.25:
            continue
        by_transition[transition].append(
            {
                "transition": transition,
                "outer_span": record_outer_span(left),
                "candidate_low_boundary": low.candidate_id,
                "candidate_high_boundary": high.candidate_id,
                "low_boundary_bytes": int(low.metadata.get("boundary_bytes") or 0),
                "high_boundary_bytes": int(high.metadata.get("boundary_bytes") or 0),
                "boundary_ratio": float(ratio),
                "static_b0_relative_delta": float(relative_delta),
                "candidate_ids": [low.candidate_id, high.candidate_id],
            }
        )
    selected = []
    used_spans = set()
    used_candidates = set()
    # Allocate the least common 1-3 transition first; tie breaking is label-free.
    for transition in ("1-3", "1-2", "2-3"):
        ranked = sorted(
            by_transition[transition],
            key=lambda row: (
                -row["boundary_ratio"],
                -row["high_boundary_bytes"] + row["low_boundary_bytes"],
                row["static_b0_relative_delta"],
                row["candidate_low_boundary"],
                row["candidate_high_boundary"],
            ),
        )
        added = 0
        for row in ranked:
            if row["outer_span"] in used_spans:
                continue
            if any(item in used_candidates for item in row["candidate_ids"]):
                continue
            selected.append(row)
            used_spans.add(row["outer_span"])
            used_candidates.update(row["candidate_ids"])
            added += 1
            if added >= quota:
                break
        if added != quota:
            raise RuntimeError(
                "could only select {} of {} pairs for transition {}".format(
                    added, quota, transition
                )
            )
    for index, row in enumerate(selected, 1):
        row["pair_id"] = "pair{:02d}_{}_span{}".format(
            index, row["transition"].replace("-", "v"), row["outer_span"]
        )
        for candidate_id in row["candidate_ids"]:
            package = package_root / candidate_id / "package.tar.gz"
            row.setdefault("packages", {})[candidate_id] = {
                "path": str(package),
                "sha256": sha256_file(package),
            }
    return {
        "selection_policy": {
            "uses_throughput_labels": False,
            "same_outer_span": True,
            "maximum_static_b0_relative_delta": 0.01,
            "minimum_boundary_ratio": 1.25,
            "pairs_per_transition": int(quota),
            "transitions": ["1-2", "1-3", "2-3"],
            "tie_break": "boundary_ratio_desc,boundary_delta_desc,b0_delta_asc,candidate_id",
        },
        "source_summaries": [str(path) for path in source_paths],
        "pair_count": len(selected),
        "candidate_count": len(used_candidates),
        "pairs": selected,
    }


def metric_values(records: Sequence[TimingRecord], model_name: str) -> Dict[str, float]:
    if not records:
        raise ValueError("metrics require records")
    target = np.asarray([record.measured_cycle_ms for record in records], dtype="float64")
    predicted = np.asarray([record.prediction(model_name) for record in records], dtype="float64")
    actual_order = np.argsort(target)
    predicted_order = np.argsort(predicted)
    best = float(target.min())
    selected = target[predicted_order[: min(10, len(target))]]
    threshold = best / 0.95
    evaluations = len(target)
    for index, candidate_index in enumerate(predicted_order, 1):
        if target[candidate_index] <= threshold:
            evaluations = index
            break
    pair_correct = 0
    pair_total = 0
    for left in range(len(target)):
        for right in range(left + 1, len(target)):
            if target[left] == target[right] or predicted[left] == predicted[right]:
                continue
            pair_total += 1
            pair_correct += int(
                (target[left] < target[right]) == (predicted[left] < predicted[right])
            )
    return {
        "sample_count": float(len(records)),
        "mae_ms": float(np.mean(np.abs(predicted - target))),
        "rmse_ms": float(np.sqrt(np.mean((predicted - target) ** 2))),
        "mape": float(np.mean(np.abs(predicted - target) / np.maximum(target, 1.0e-9))),
        "spearman": float(spearmanr(target, predicted).statistic),
        "kendall": float(kendalltau(target, predicted).statistic),
        "pairwise_accuracy": float(pair_correct / max(1, pair_total)),
        "regret_at_10": float((selected.min() - best) / best),
        "evaluations_to_oracle_95pct": float(evaluations),
    }


def effect_value(records: Sequence[TimingRecord], baseline: str, enhanced: str) -> float:
    target = np.asarray([record.measured_cycle_ms for record in records], dtype="float64")
    before = np.asarray([record.prediction(baseline) for record in records], dtype="float64")
    after = np.asarray([record.prediction(enhanced) for record in records], dtype="float64")
    return float(np.mean(np.abs(before - target)) - np.mean(np.abs(after - target)))


def grouped_effect_bootstrap(
    records: Sequence[TimingRecord], repetitions: int, seed: int
) -> Dict[str, Any]:
    groups: Dict[str, List[TimingRecord]] = {}
    for record in records:
        groups.setdefault(record.group_id, []).append(record)
    names = sorted(groups)
    rng = random.Random(seed)
    definitions = {
        "b1_minus_b0": ("b0_compute_only", "b1_communication"),
        "b3_minus_b2": ("b2_shared", "b3_full"),
        "b3_minus_b0": ("b0_compute_only", "b3_full"),
    }
    samples = {name: [] for name in definitions}
    for _ in range(int(repetitions)):
        sample = []
        for group_name in (rng.choice(names) for _ in names):
            sample.extend(groups[group_name])
        for name, (baseline, enhanced) in definitions.items():
            samples[name].append(effect_value(sample, baseline, enhanced))
    result = {}
    for name, values in samples.items():
        array = np.asarray(values, dtype="float64")
        result[name] = {
            "effect_definition": "MAE(baseline)-MAE(enhanced); positive favors enhanced",
            "point_estimate_ms": effect_value(records, *definitions[name]),
            "bootstrap_mean_ms": float(array.mean()),
            "ci95_ms": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
            "p_one_sided": float((np.sum(array <= 0.0) + 1) / (len(array) + 1)),
        }
    return {
        "repetitions": int(repetitions),
        "group_count": len(groups),
        "effects": result,
    }


def holm_adjust(effects: Mapping[str, Mapping[str, Any]], names: Sequence[str]) -> Dict[str, float]:
    ordered = sorted(names, key=lambda name: float(effects[name]["p_one_sided"]))
    result = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - rank) * float(effects[name]["p_one_sided"]))
        running = max(running, adjusted)
        result[name] = running
    return result


def exact_pair_permutation(pair_effects: Sequence[float]) -> float:
    values = np.asarray(pair_effects, dtype="float64")
    if not len(values):
        return 1.0
    observed = float(values.mean())
    at_least = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        total += 1
        statistic = float(np.mean(values * np.asarray(signs)))
        at_least += int(statistic >= observed - 1.0e-12)
    return float(at_least / total)


def aggregate_candidate_sessions(records: Sequence[TimingRecord], sessions: int) -> List[TimingRecord]:
    groups: Dict[str, List[TimingRecord]] = {}
    for record in records:
        if record.correctness_passed:
            groups.setdefault(record.candidate_id, []).append(record)
    aggregated = []
    for candidate_id, values in groups.items():
        if len({record.session for record in values}) < int(sessions):
            continue
        first = values[0]
        field_median = lambda field: median_positive(getattr(record, field) for record in values)
        prediction = {name: field_median(name + "_ms") for name in MODEL_NAMES}
        aggregated.append(
            TimingRecord(
                candidate_id=candidate_id,
                group_id=first.pair_id,
                pair_id=first.pair_id,
                session="median_of_{}_sessions".format(sessions),
                source_path=";".join(sorted(record.source_path for record in values)),
                measured_cycle_ms=field_median("measured_cycle_ms"),
                stage_devices=first.stage_devices,
                stage_threads=first.stage_threads,
                stage_run_ms=[
                    median_positive(record.stage_run_ms[index] for record in values)
                    for index in range(len(first.stage_run_ms))
                ],
                stage_communication_ms=[
                    median_positive(record.stage_communication_ms[index] for record in values)
                    for index in range(len(first.stage_communication_ms))
                ],
                stage_service_ms=[
                    median_positive(record.stage_service_ms[index] for record in values)
                    for index in range(len(first.stage_service_ms))
                ],
                queue_depth=first.queue_depth,
                b0_compute_only_ms=prediction["b0_compute_only"],
                b1_communication_ms=prediction["b1_communication"],
                b2_shared_ms=prediction["b2_shared"],
                b3_full_ms=prediction["b3_full"],
                b3_bottleneck=statistics.mode(record.b3_bottleneck for record in values),
                correctness_passed=True,
            )
        )
    return aggregated


def prospective_statistics(
    records: Sequence[TimingRecord], pairs: Mapping[str, Any], sessions: int, bootstrap: int, seed: int
) -> Dict[str, Any]:
    aggregated = aggregate_candidate_sessions(records, sessions)
    by_candidate = {record.candidate_id: record for record in aggregated}
    complete_pairs = []
    pair_effects = {name: [] for name in PRIMARY_EFFECTS}
    definitions = {
        "b1_minus_b0": ("b0_compute_only", "b1_communication"),
        "b3_minus_b2": ("b2_shared", "b3_full"),
    }
    for pair in pairs.get("pairs", []):
        values = [by_candidate.get(candidate_id) for candidate_id in pair["candidate_ids"]]
        if any(value is None for value in values):
            continue
        complete_pairs.append(pair["pair_id"])
        for name, (baseline, enhanced) in definitions.items():
            pair_effects[name].append(effect_value(values, baseline, enhanced))
    bootstrap_result = grouped_effect_bootstrap(aggregated, bootstrap, seed)
    exact_p = {name: exact_pair_permutation(values) for name, values in pair_effects.items()}
    exact_payload = {
        name: {
            "pair_count": len(pair_effects[name]),
            "mean_effect_ms": float(np.mean(pair_effects[name])) if pair_effects[name] else 0.0,
            "p_one_sided": exact_p[name],
        }
        for name in PRIMARY_EFFECTS
    }
    exact_adjusted = holm_adjust(exact_payload, PRIMARY_EFFECTS)
    bootstrap_adjusted = holm_adjust(bootstrap_result["effects"], PRIMARY_EFFECTS)
    for name in PRIMARY_EFFECTS:
        exact_payload[name]["holm_adjusted_p"] = exact_adjusted[name]
        bootstrap_result["effects"][name]["holm_adjusted_p"] = bootstrap_adjusted[name]
    metrics = {name: metric_values(aggregated, name) for name in MODEL_NAMES} if aggregated else {}
    return {
        "valid_candidate_count": len(aggregated),
        "complete_pair_count": len(complete_pairs),
        "complete_pair_ids": complete_pairs,
        "aggregated_records": [asdict(record) for record in aggregated],
        "metrics": metrics,
        "grouped_bootstrap": bootstrap_result,
        "exact_pair_permutation": exact_payload,
    }


def historical_statistics(records: Sequence[TimingRecord], bootstrap: int, seed: int) -> Dict[str, Any]:
    metrics = {name: metric_values(records, name) for name in MODEL_NAMES}
    effects = grouped_effect_bootstrap(records, bootstrap, seed)
    adjusted = holm_adjust(effects["effects"], PRIMARY_EFFECTS)
    for name in PRIMARY_EFFECTS:
        effects["effects"][name]["holm_adjusted_p"] = adjusted[name]
    return {
        "record_count": len(records),
        "outer_span_group_count": len({record.group_id for record in records}),
        "metrics": metrics,
        "grouped_bootstrap": effects,
    }


def direct_profile_statistics(
    records: Sequence[Any], bootstrap: int, seed: int
) -> Dict[str, Any]:
    rows = []
    for record in records:
        profile = (record.metadata.get("direct_communication_profile") or {}).get("pipeline")
        run_ms = record.metadata.get("measured_stage_run_ms") or []
        if not profile or len(run_ms) != len(record.stages) or record.measured_cycle_ms is None:
            continue
        cpu_run_ms = [
            as_float(value)
            for stage, value in zip(record.stages, run_ms)
            if stage.device == "cpu"
        ]
        vta_run_ms = [
            as_float(value)
            for stage, value in zip(record.stages, run_ms)
            if stage.device == "vta"
        ]
        direct_copy_ms = as_float(profile.get("direct_copy_ms")) + as_float(
            profile.get("coherence_ms")
        )
        shared_compute_ms = max(max(cpu_run_ms or [0.0]), sum(vta_run_ms))
        communication_aware_ms = max(
            max(cpu_run_ms or [0.0]), sum(vta_run_ms) + direct_copy_ms
        )
        group_parts = str(record.group_id).split("_")
        outer_span = "_".join(group_parts[-2:]) if len(group_parts) >= 2 else record.group_id
        rows.append(
            {
                "candidate_id": record.candidate_id,
                "outer_span": outer_span,
                "vta_island_count": int(record.vta_island_count),
                "measured_cycle_ms": float(record.measured_cycle_ms),
                "shared_compute_ms": shared_compute_ms,
                "direct_copy_ms": direct_copy_ms,
                "communication_aware_ms": communication_aware_ms,
            }
        )
    if not rows:
        return {"record_count": 0, "rows": []}

    target = np.asarray([row["measured_cycle_ms"] for row in rows], dtype="float64")
    base = np.asarray([row["shared_compute_ms"] for row in rows], dtype="float64")
    aware = np.asarray([row["communication_aware_ms"] for row in rows], dtype="float64")

    def array_metrics(predicted: np.ndarray) -> Dict[str, float]:
        order = np.argsort(predicted)
        best = float(target.min())
        result = {
            "mae_ms": float(np.mean(np.abs(predicted - target))),
            "rmse_ms": float(np.sqrt(np.mean((predicted - target) ** 2))),
            "spearman": float(spearmanr(target, predicted).statistic),
            "kendall": float(kendalltau(target, predicted).statistic),
        }
        for budget in (1, 3, 5, 10, 20, 50, 100):
            selected = target[order[: min(budget, len(order))]]
            result["regret_at_{}".format(budget)] = float(
                (selected.min() - best) / best
            )
        for fraction in (0.95, 0.98):
            threshold = best / fraction
            evaluations = len(order)
            for index, candidate_index in enumerate(order, 1):
                if target[candidate_index] <= threshold:
                    evaluations = index
                    break
            result["evaluations_to_oracle_{}pct".format(int(100 * fraction))] = float(
                evaluations
            )
        return result

    grouped_indices: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        grouped_indices.setdefault(str(row["outer_span"]), []).append(index)
    group_names = sorted(grouped_indices)
    rng = random.Random(seed)
    effects = []
    for _ in range(int(bootstrap)):
        indices = []
        for name in (rng.choice(group_names) for _ in group_names):
            indices.extend(grouped_indices[name])
        chosen = np.asarray(indices, dtype="int64")
        effects.append(
            float(
                np.mean(np.abs(base[chosen] - target[chosen]))
                - np.mean(np.abs(aware[chosen] - target[chosen]))
            )
        )
    effect_array = np.asarray(effects, dtype="float64")
    copy_by_islands = {}
    for island_count in sorted({int(row["vta_island_count"]) for row in rows}):
        values = [
            float(row["direct_copy_ms"])
            for row in rows
            if int(row["vta_island_count"]) == island_count
        ]
        copy_by_islands[str(island_count)] = {
            "count": len(values),
            "median_ms": float(statistics.median(values)),
            "mean_ms": float(statistics.mean(values)),
            "p95_ms": float(np.quantile(values, 0.95)),
        }
    return {
        "record_count": len(rows),
        "comparison": "shared VTA compute resource without vs with direct profiled boundary copy",
        "base": array_metrics(base),
        "communication_aware": array_metrics(aware),
        "mae_improvement_ms": float(np.mean(np.abs(base - target)) - np.mean(np.abs(aware - target))),
        "grouped_bootstrap": {
            "repetitions": int(bootstrap),
            "outer_span_group_count": len(group_names),
            "ci95_ms": [
                float(np.quantile(effect_array, 0.025)),
                float(np.quantile(effect_array, 0.975)),
            ],
            "p_one_sided": float((np.sum(effect_array <= 0.0) + 1) / (len(effect_array) + 1)),
        },
        "direct_copy_by_vta_islands": copy_by_islands,
        "rows": rows,
    }


def direct_communication_gate(profile: Mapping[str, Any]) -> Dict[str, Any]:
    bootstrap = profile.get("grouped_bootstrap") or {}
    ci95 = bootstrap.get("ci95_ms") or [float("-inf"), float("inf")]
    base = profile.get("base") or {}
    aware = profile.get("communication_aware") or {}
    checks = {
        "all_200_historical_candidates_have_direct_profiles": int(
            profile.get("record_count") or 0
        )
        == 200,
        "direct_copy_mae_improvement_ci_positive": as_float(ci95[0]) > 0.0,
        "direct_copy_one_sided_p_below_0_05": as_float(
            bootstrap.get("p_one_sided"), 1.0
        )
        < 0.05,
        "communication_aware_rank_correlation_not_worse": as_float(
            aware.get("spearman"), -1.0
        )
        >= as_float(base.get("spearman"), 1.0),
    }
    supported = all(checks.values())
    return {
        "status": "communication_mechanism_supported" if supported else "inconclusive",
        "communication_mechanism_supported": supported,
        "continue_old_set_get_board_preexperiment": False if supported else None,
        "continue_full_ramps_identification": supported,
        "full_b3_validated": False,
        "scope": (
            "Direct VTA runtime profiler boundary-copy counters added to a shared-compute "
            "model; this validates communication awareness, not the complete B3 event graph."
        ),
        "checks": checks,
    }


def candidate_metadata(roots: Sequence[str]) -> Dict[str, Any]:
    records, _ = collect_resnet_records(roots, {})
    return {record.candidate_id: record for record in records}


def safe_extract_tar(tar_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError("unsafe archive member: {}".format(member.name))
        archive.extractall(destination)


def prepare_temp_package(
    candidate_id: str, package_meta: Mapping[str, Any], temp_root: Path
) -> Path:
    package_dir = temp_root / "buildability" / candidate_id / "package"
    receipt = package_dir.parent / "extract_receipt.json"
    expected = str(package_meta["sha256"])
    if receipt.is_file() and package_dir.is_dir():
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        if payload.get("package_tar_sha256") == expected:
            return package_dir
    if package_dir.parent.exists():
        shutil.rmtree(package_dir.parent)
    package_dir.mkdir(parents=True)
    safe_extract_tar(Path(str(package_meta["path"])), package_dir)
    write_json(receipt, {"package_tar_sha256": expected, "candidate_id": candidate_id})
    return package_dir


def ssh_options(output_dir: Path) -> List[str]:
    known_hosts = output_dir / "board_known_hosts"
    return [
        "-oHostKeyAlgorithms=+ssh-rsa",
        "-oPubkeyAcceptedAlgorithms=+ssh-rsa",
        "-oBatchMode=yes",
        "-oConnectTimeout=10",
        "-oStrictHostKeyChecking=accept-new",
        "-oUserKnownHostsFile={}".format(known_hosts.resolve()),
    ]


def run_logged(command: Sequence[str], log_path: Path, timeout: int) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    proc = subprocess.run(
        [str(item) for item in command],
        cwd=str(REPO_ROOT),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout,
    )
    log_path.write_text(
        "$ {}\n\n{}\n\nreturncode={} elapsed_s={:.3f}\n".format(
            " ".join(str(item) for item in command), proc.stdout, proc.returncode, time.time() - started
        ),
        encoding="utf-8",
    )
    return proc


def preflight(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    command = ["ssh"] + ssh_options(output_dir) + [
        args.board,
        "test -d /mnt/sd/tvm_deploy && "
        "test $(cat /sys/class/fpga_manager/fpga0/state) = operating && "
        "test $(cat /sys/class/u-dma-buf/udmabuf0/size) = 201326592 && "
        "df -Pk /var/volatile && "
        "ps | grep vta_stage_pipeline_runner | grep -v grep || true",
    ]
    proc = run_logged(command, output_dir / "preflight.log", 30)
    payload = {
        "board": args.board,
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "output": proc.stdout,
    }
    write_json(output_dir / "preflight.json", payload)
    return payload


def baseline_top1(path: Path) -> Dict[int, int]:
    result = {}
    for row in read_jsonl(path):
        result[int(row["input_index"])] = int(row["top1"])
    if not result:
        raise RuntimeError("empty RPC baseline: {}".format(path))
    return result


def correctness_from_results(serial_path: Path, pipeline_path: Path, baseline_path: Path) -> bool:
    serial = {int(row["frame_id"]): row for row in read_jsonl(serial_path)}
    pipeline = {int(row["frame_id"]): row for row in read_jsonl(pipeline_path)}
    if not serial or set(serial) != set(pipeline):
        return False
    baseline = baseline_top1(baseline_path)
    for frame_id in serial:
        left, right = serial[frame_id], pipeline[frame_id]
        if any(left.get(key) != right.get(key) for key in ("input_index", "input_file", "top1")):
            return False
        expected = baseline.get(int(left["input_index"]))
        actual = int(left["top1"])
        if expected is None:
            return False
        if actual != expected and not (
            actual in CAT_EQUIVALENT_TOP1 and expected in CAT_EQUIVALENT_TOP1
        ):
            return False
    return True


def deploy_command(
    args: argparse.Namespace,
    output_dir: Path,
    package_dir: Path,
    candidate_id: str,
    threads: Sequence[int],
    remote_dir: str,
    skip_run: bool,
) -> List[str]:
    script = REPO_ROOT / "vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py"
    command = [
        sys.executable,
        str(script),
        "--board",
        args.board,
        "--remote-dir",
        remote_dir,
        "--remote-min-free-mb",
        "128",
        "--reuse-package-dir",
        str(package_dir),
        "--candidate-id",
        candidate_id,
        "--runs",
        str(args.runs),
        "--queue-depth",
        str(args.queue_depth),
        "--runtime-num-threads",
        "4",
        "--stage-runtime-num-threads",
        ",".join(str(value) for value in threads),
        "--correctness-policy",
        "cat_equivalent",
        "--ssh-command-timeout-s",
        str(args.ssh_timeout_s),
        "--scp-timeout-s",
        str(args.scp_timeout_s),
        "--serial-timeout-s",
        str(args.serial_timeout_s),
        "--pipeline-timeout-s",
        str(args.pipeline_timeout_s),
        "--fetch-timeout-s",
        str(args.fetch_timeout_s),
        "--cleanup-remote-after-run",
        "--no-ssh-control-master",
    ]
    for option in ssh_options(Path(args.output_dir)):
        command.append("--ssh-option={}".format(option))
    if skip_run:
        command.append("--skip-run")
    else:
        command.extend(
            [
                "--fetch-results-dir",
                str(output_dir),
                "--run-serial-before-pipeline",
                "--compare-serial-pipeline",
                "--rpc-baseline-result",
                str(Path(args.rpc_baseline_result).resolve()),
            ]
        )
    return command


def pair_lookup(pair_payload: Mapping[str, Any]) -> Dict[str, str]:
    result = {}
    for pair in pair_payload.get("pairs", []):
        for candidate_id in pair["candidate_ids"]:
            result[candidate_id] = pair["pair_id"]
    return result


def write_progress(
    output_dir: Path, done: int, total: int, candidate_id: str, status: str, failures: int
) -> None:
    width = 30
    filled = int(round(width * done / max(1, total)))
    payload = {
        "done": int(done),
        "total": int(total),
        "candidate_id": candidate_id,
        "status": status,
        "failure_count": int(failures),
        "updated_at_epoch_s": time.time(),
    }
    write_json(output_dir / "progress.json", payload)
    line = "[{}{}] {}/{} current={} status={} failures={}".format(
        "#" * filled,
        "-" * (width - filled),
        done,
        total,
        candidate_id or "-",
        status,
        failures,
    )
    (output_dir / "progress.txt").write_text(line + "\n", encoding="utf-8")
    print(line, flush=True)


def measure_board(
    args: argparse.Namespace,
    output_dir: Path,
    pairs: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    if not args.dry_run and not os.environ.get("SDKTARGETSYSROOT"):
        raise RuntimeError("source the AXU5EVB SDK environment before --mode measure")
    if not Path(args.rpc_baseline_result).is_file():
        raise RuntimeError("RPC baseline is missing: {}".format(args.rpc_baseline_result))
    if not args.dry_run:
        status = preflight(args, output_dir)
        if not status["passed"]:
            raise RuntimeError("board preflight failed; see {}".format(output_dir / "preflight.log"))

    package_by_candidate = {}
    candidate_ids = []
    pair_by_candidate = pair_lookup(pairs)
    for pair in pairs["pairs"]:
        for candidate_id in pair["candidate_ids"]:
            candidate_ids.append(candidate_id)
            package_by_candidate[candidate_id] = pair["packages"][candidate_id]

    temp_root = Path("/tmp/ramps_communication_preexperiment_packages")
    failures_path = output_dir / "board" / "failures.json"
    failures = (
        json.loads(failures_path.read_text(encoding="utf-8")).get("rows", [])
        if failures_path.is_file()
        else []
    )
    completed = 0
    total = len(candidate_ids) * int(args.sessions)

    order_files = []
    orders = []
    for session_index in range(1, int(args.sessions) + 1):
        order = list(candidate_ids)
        random.Random(args.seed + session_index).shuffle(order)
        if session_index == 1:
            smoke = list(pairs["pairs"][0]["candidate_ids"])
            order = smoke + [item for item in order if item not in smoke]
        order_path = output_dir / "board" / "orders" / "session{:03d}.txt".format(session_index)
        order_path.parent.mkdir(parents=True, exist_ok=True)
        order_path.write_text("".join(item + "\n" for item in order), encoding="utf-8")
        order_files.append(str(order_path))
        orders.append(order)

    smoke_candidates = set(pairs["pairs"][0]["candidate_ids"])
    for session_index, order in enumerate(orders, 1):
        for candidate_id in order:
            result_dir = (
                output_dir
                / "board"
                / "session{:03d}".format(session_index)
                / candidate_id
            )
            native_path = result_dir / "native_result.jsonl"
            serial_path = result_dir / "stage_serial_result.jsonl"
            if args.resume and native_path.is_file() and serial_path.is_file():
                try:
                    if correctness_from_results(
                        serial_path, native_path, Path(args.rpc_baseline_result)
                    ):
                        completed += 1
                        write_progress(
                            output_dir, completed, total, candidate_id, "reused", len(failures)
                        )
                        continue
                except Exception:
                    pass
            record = metadata[candidate_id]
            threads = [stage.threads for stage in record.stages]
            failures = [
                row
                for row in failures
                if not (
                    int(row.get("session", -1)) == session_index
                    and row.get("candidate_id") == candidate_id
                )
            ]
            write_json(failures_path, {"rows": failures})
            package_dir = (
                temp_root / "buildability" / candidate_id / "package"
                if args.dry_run
                else prepare_temp_package(
                    candidate_id, package_by_candidate[candidate_id], temp_root
                )
            )
            prewarm_marker = output_dir / "board" / "prewarm" / (candidate_id + ".done")
            try:
                write_progress(output_dir, completed, total, candidate_id, "prewarming", len(failures))
                if not prewarm_marker.is_file():
                    prewarm_dir = args.remote_dir.rstrip("/") + "/prewarm_" + candidate_id
                    command = deploy_command(
                        args,
                        result_dir,
                        package_dir,
                        candidate_id,
                        threads,
                        prewarm_dir,
                        skip_run=True,
                    )
                    if args.dry_run:
                        result_dir.mkdir(parents=True, exist_ok=True)
                        (result_dir / "prewarm_command.txt").write_text(
                            " ".join(command) + "\n", encoding="utf-8"
                        )
                        prewarm_marker.parent.mkdir(parents=True, exist_ok=True)
                        prewarm_marker.write_text("dry-run\n", encoding="utf-8")
                    else:
                        proc = run_logged(command, result_dir / "prewarm.log", args.scp_timeout_s + 300)
                        if proc.returncode != 0:
                            raise RuntimeError("prewarm failed with returncode {}".format(proc.returncode))
                        prewarm_marker.parent.mkdir(parents=True, exist_ok=True)
                        prewarm_marker.write_text("ok\n", encoding="utf-8")

                write_progress(output_dir, completed, total, candidate_id, "running", len(failures))
                remote_dir = args.remote_dir.rstrip("/") + "/s{:03d}_{}".format(
                    session_index, candidate_id
                )
                command = deploy_command(
                    args,
                    result_dir,
                    package_dir,
                    candidate_id,
                    threads,
                    remote_dir,
                    skip_run=False,
                )
                result_dir.mkdir(parents=True, exist_ok=True)
                (result_dir / "run_command.txt").write_text(
                    " ".join(command) + "\n", encoding="utf-8"
                )
                if args.dry_run:
                    completed += 1
                    write_progress(
                        output_dir,
                        completed,
                        total,
                        candidate_id,
                        "completed",
                        len(failures),
                    )
                    continue
                proc = run_logged(
                    command,
                    result_dir / "run.log",
                    args.scp_timeout_s + args.serial_timeout_s + args.pipeline_timeout_s + 600,
                )
                if proc.returncode != 0:
                    raise RuntimeError("measurement failed with returncode {}".format(proc.returncode))
                correct = correctness_from_results(
                    serial_path, native_path, Path(args.rpc_baseline_result)
                )
                if not correct:
                    raise RuntimeError("serial/pipeline/RPC correctness gate failed")
                completed += 1
            except Exception as err:  # pylint: disable=broad-except
                failures.append(
                    {
                        "session": session_index,
                        "candidate_id": candidate_id,
                        "pair_id": pair_by_candidate[candidate_id],
                        "error": repr(err),
                        "output_dir": str(result_dir),
                    }
                )
                write_json(failures_path, {"rows": failures})
                if session_index == 1 and candidate_id in smoke_candidates:
                    write_progress(output_dir, completed, total, candidate_id, "smoke_failed", len(failures))
                    raise RuntimeError("first matched-pair smoke failed") from err
            write_progress(output_dir, completed, total, candidate_id, "completed", len(failures))
    write_json(
        output_dir / "board" / "measurement_manifest.json",
        {
            "board": args.board,
            "sessions": args.sessions,
            "runs_per_process": args.runs,
            "warmup_frames": args.skip_first,
            "measured_frames": args.runs - args.skip_first,
            "queue_depth": args.queue_depth,
            "order_files": order_files,
            "completed_process_count": completed,
            "failure_count": len(failures),
        },
    )


def collect_board_records(
    args: argparse.Namespace,
    output_dir: Path,
    pairs: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> List[TimingRecord]:
    result = []
    lookup = pair_lookup(pairs)
    for session_index in range(1, int(args.sessions) + 1):
        session = "session{:03d}".format(session_index)
        for candidate_id, pair_id in lookup.items():
            result_dir = output_dir / "board" / session / candidate_id
            native = result_dir / "native_result.jsonl"
            serial = result_dir / "stage_serial_result.jsonl"
            if not native.is_file() or not serial.is_file():
                continue
            record = metadata[candidate_id]
            correct = correctness_from_results(serial, native, Path(args.rpc_baseline_result))
            result.append(
                timing_record_from_result(
                    candidate_id=candidate_id,
                    group_id=pair_id,
                    pair_id=pair_id,
                    session=session,
                    source_path=native,
                    devices=[stage.device for stage in record.stages],
                    threads=[stage.threads for stage in record.stages],
                    queue_depth=args.queue_depth,
                    skip_first=args.skip_first,
                    correctness_passed=correct,
                )
            )
    return result


def gate_result(
    args: argparse.Namespace,
    prospective: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not prospective:
        return {"status": "historical_only", "continue_ramps": False, "checks": {}}
    effects = prospective["grouped_bootstrap"]["effects"]
    permutation = prospective["exact_pair_permutation"]
    metrics = prospective.get("metrics", {})
    checks = {
        "minimum_valid_candidates": prospective["valid_candidate_count"]
        >= args.minimum_valid_candidates,
        "minimum_complete_pairs": prospective["complete_pair_count"]
        >= args.minimum_complete_pairs,
        "b1_minus_b0_ci_positive": effects["b1_minus_b0"]["ci95_ms"][0] > 0.0,
        "b3_minus_b2_ci_positive": effects["b3_minus_b2"]["ci95_ms"][0] > 0.0,
        "b1_minus_b0_holm_significant": permutation["b1_minus_b0"]["holm_adjusted_p"] < 0.05,
        "b3_minus_b2_holm_significant": permutation["b3_minus_b2"]["holm_adjusted_p"] < 0.05,
        "b3_minus_b0_mae_positive": effects["b3_minus_b0"]["point_estimate_ms"] > 0.0,
        "b3_top10_regret_not_worse": bool(metrics)
        and metrics["b3_full"]["regret_at_10"] <= metrics["b0_compute_only"]["regret_at_10"],
    }
    passed = all(checks.values())
    return {
        "status": "go" if passed else "no_go",
        "continue_ramps": passed,
        "checks": checks,
        "failure_policy": "no post-hoc threshold or candidate replacement",
    }


def render_report(
    historical: Optional[Mapping[str, Any]],
    direct_profile: Mapping[str, Any],
    direct_gate: Mapping[str, Any],
    prospective: Optional[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> str:
    lines = [
        "# RAMPS Communication Go/No-Go Pre-Experiment",
        "",
        "This is an oracle-service mechanism test. It does not establish zero-shot static prediction.",
        "The canonical paper-ready evidence is maintained in `../PAPER_EVIDENCE.md`.",
    ]
    if historical:
        lines.extend(
            [
                "",
                "## Deprecated Set/Get Proxy Reproduction",
                "",
                "These metrics reproduce the archived pilot only. Whole-stage `set/get` "
                "mixes copy, synchronization, allocation and runtime overhead, so these "
                "values are excluded from communication inference and model fitting.",
                "",
                "- Records: `{}`".format(historical["record_count"]),
            ]
        )
    direct_bootstrap = direct_profile["grouped_bootstrap"]
    lines.extend(
        [
            "",
            "## Direct Runtime-Profiler Communication Ablation",
            "",
            "This analysis uses the saved Host-to-VTA and VTA-to-Host copy timers at the "
            "runtime boundary. It does not infer communication from whole-stage `set/get` time.",
            "",
            "- Records with direct profiles: `{}`.".format(direct_profile["record_count"]),
            "- Shared-compute MAE: `{:.4f} ms`.".format(direct_profile["base"]["mae_ms"]),
            "- With direct boundary copy MAE: `{:.4f} ms`.".format(
                direct_profile["communication_aware"]["mae_ms"]
            ),
            "- MAE improvement: `{:.4f} ms`, grouped 95% CI `[{:.4f}, {:.4f}]`, "
            "one-sided p `{:.6f}`.".format(
                direct_profile["mae_improvement_ms"],
                direct_bootstrap["ci95_ms"][0],
                direct_bootstrap["ci95_ms"][1],
                direct_bootstrap["p_one_sided"],
            ),
            "- Spearman: `{:.4f}` without communication, `{:.4f}` with communication.".format(
                direct_profile["base"]["spearman"],
                direct_profile["communication_aware"]["spearman"],
            ),
            "- Oracle-service regret@1: `{:.4f}` without communication, `{:.4f}` "
            "with communication; evaluations to 95% oracle: `{:.0f}` versus `{:.0f}`.".format(
                direct_profile["base"]["regret_at_1"],
                direct_profile["communication_aware"]["regret_at_1"],
                direct_profile["base"]["evaluations_to_oracle_95pct"],
                direct_profile["communication_aware"]["evaluations_to_oracle_95pct"],
            ),
            "- These low-budget values consume candidate-specific stage/copy timings. "
            "They are an upper-bound mechanism result, not a zero-feedback search result.",
            "- Mechanism decision: **{}**.".format(direct_gate["status"]),
            "- This stops the old `set+get` board protocol; it does not claim that full B3 "
            "has already been validated.",
        ]
    )
    if prospective:
        lines.extend(
            [
                "",
                "## Prospective Board Replication",
                "",
                "- Valid candidates: `{}`".format(prospective["valid_candidate_count"]),
                "- Complete pairs: `{}`".format(prospective["complete_pair_count"]),
            ]
        )
        for name, row in prospective["grouped_bootstrap"]["effects"].items():
            lines.append(
                "- Prospective `{}`: effect `{:.4f} ms`, 95% CI `[{:.4f}, {:.4f}]`.".format(
                    name, row["point_estimate_ms"], row["ci95_ms"][0], row["ci95_ms"][1]
                )
            )
        for name, row in prospective["exact_pair_permutation"].items():
            lines.append(
                "- Exact `{}`: Holm p `{:.6f}` over `{}` pairs.".format(
                    name, row["holm_adjusted_p"], row["pair_count"]
                )
            )
    lines.extend(["", "## Decision", "", "- Status: **{}**".format(gate["status"]), ""])
    for name, passed in gate.get("checks", {}).items():
        lines.append("- [{}] `{}`".format("x" if passed else " ", name))
    lines.extend(
        [
            "",
            "This decision permits continued RAMPS identification with direct communication "
            "features. It does not validate the complete B3 event graph or cross-model "
            "zero-shot prediction.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    modes = {item.strip() for item in args.mode.split(",") if item.strip()}
    unknown = modes - {"historical", "plan", "measure", "analyze"}
    if unknown:
        raise RuntimeError("unknown modes: {}".format(sorted(unknown)))
    deprecated_modes = modes & {"plan", "measure", "analyze"}
    if deprecated_modes and not args.allow_deprecated_set_get_board_protocol:
        raise RuntimeError(
            "{} belongs to the deprecated set/get proxy protocol; pass "
            "--allow-deprecated-set-get-board-protocol only for archival reproduction".format(
                ",".join(sorted(deprecated_modes))
            )
        )
    if args.runs <= args.skip_first or args.skip_first < 0:
        raise RuntimeError("--runs must be greater than non-negative --skip-first")
    if args.sessions <= 0 or args.bootstrap <= 0:
        raise RuntimeError("--sessions and --bootstrap must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = args.resnet_root or list(DEFAULT_RESNET_ROOTS)
    build_cache = Path(args.build_cache_dir)

    historical: Optional[Dict[str, Any]] = None
    if args.allow_deprecated_set_get_board_protocol:
        historical_path = output_dir / "historical" / "deprecated_set_get_records.json"
        historical_metrics_path = output_dir / "historical" / "deprecated_set_get_metrics.json"
        historical_records = collect_historical_records(roots, args.skip_first)
        if len(historical_records) != 200:
            raise RuntimeError(
                "historical record count is {} rather than 200".format(len(historical_records))
            )
        write_json(historical_path, {"records": [asdict(record) for record in historical_records]})
        write_csv(
            output_dir / "historical" / "deprecated_set_get_records.csv",
            [asdict(record) for record in historical_records],
        )
        historical = historical_statistics(historical_records, args.bootstrap, args.seed)
        write_json(historical_metrics_path, historical)

    resource_records, _ = collect_resnet_records(roots, {})
    direct_profile = direct_profile_statistics(resource_records, args.bootstrap, args.seed + 2000)
    write_json(output_dir / "historical" / "direct_profile_metrics.json", direct_profile)
    write_csv(output_dir / "historical" / "direct_profile_records.csv", direct_profile.get("rows", []))

    pairs: Dict[str, Any] = {"pair_count": 0, "pairs": []}
    metadata: Dict[str, Any] = {}
    if args.allow_deprecated_set_get_board_protocol:
        pair_path = output_dir / "protocol" / "candidate_pairs.json"
        if "plan" in modes or not pair_path.is_file():
            pairs = select_candidate_pairs(roots, build_cache, args.pair_count_per_transition)
            write_json(pair_path, pairs)
            (pair_path.parent / "candidate_pairs.sha256").write_text(
                sha256_file(pair_path) + "  candidate_pairs.json\n", encoding="utf-8"
            )
            write_json(
                pair_path.parent / "protocol.json",
                {
                    "candidate_pairs_sha256": sha256_file(pair_path),
                    "board": args.board,
                    "runs": args.runs,
                    "skip_first": args.skip_first,
                    "measured_frames": args.runs - args.skip_first,
                    "sessions": args.sessions,
                    "queue_depth": args.queue_depth,
                    "bootstrap": args.bootstrap,
                    "seed": args.seed,
                    "selection_frozen_before_board_measurement": True,
                    "deprecated_set_get_proxy_protocol": True,
                    "preexperiment_script_sha256": sha256_file(Path(__file__)),
                    "runner_source_sha256": sha256_file(
                        REPO_ROOT / "vta/apps/native_deploy/vta_stage_pipeline_runner.cc"
                    ),
                },
            )
        pairs = json.loads(pair_path.read_text(encoding="utf-8"))
        metadata = candidate_metadata(roots)

    if "measure" in modes:
        measure_board(args, output_dir, pairs, metadata)

    prospective = None
    if "analyze" in modes:
        board_records = collect_board_records(args, output_dir, pairs, metadata)
        write_json(
            output_dir / "board" / "records.json",
            {"records": [asdict(record) for record in board_records]},
        )
        write_csv(output_dir / "board" / "records.csv", [asdict(record) for record in board_records])
        if board_records:
            prospective = prospective_statistics(
                board_records, pairs, args.sessions, args.bootstrap, args.seed + 1000
            )
            write_json(output_dir / "board" / "metrics.json", prospective)

    gate = gate_result(args, prospective)
    write_json(output_dir / "GO_NO_GO.json", gate)
    direct_gate = direct_communication_gate(direct_profile)
    write_json(output_dir / "DIRECT_COMMUNICATION_GO_NO_GO.json", direct_gate)
    if prospective is None and direct_gate["communication_mechanism_supported"]:
        gate = {
            "status": "communication_supported_old_protocol_stopped",
            "continue_ramps": True,
            "continue_old_set_get_board_preexperiment": False,
            "full_b3_validated": False,
            "checks": direct_gate["checks"],
            "scope": direct_gate["scope"],
        }
        write_json(output_dir / "GO_NO_GO.json", gate)
    (output_dir / "PREEXPERIMENT_REPORT.md").write_text(
        render_report(historical, direct_profile, direct_gate, prospective, gate),
        encoding="utf-8",
    )
    print("[RAMPS-COMM] direct-profile records:", direct_profile["record_count"])
    print("[RAMPS-COMM] candidate pairs:", pairs["pair_count"])
    print("[RAMPS-COMM] decision:", gate["status"])
    print("[RAMPS-COMM] output:", output_dir)


if __name__ == "__main__":
    main()
