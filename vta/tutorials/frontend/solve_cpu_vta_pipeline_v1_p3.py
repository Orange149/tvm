#!/usr/bin/env python3
"""Rank CPU-VTA Pipeline V1 partitions and verify the DP against enumeration."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

from build_cpu_vta_pipeline_v1_p1 import enumerate_reachable, schema_nbytes
from freeze_cpu_vta_pipeline_v1 import (
    PROTOCOL_ID,
    UNIT_ORDER,
    load_sealed_artifact,
    repo_root,
    seal_artifact,
    validate_artifact_sha256,
)


DEFAULT_OUTPUT = repo_root() / "vta/tutorials/frontend/report_out/resource_aware_maxplus"
CPU_THREAD_CHOICES = (1, 2, 3, 4)
PHYSICAL_CPU_CORES = 4
MAX_VTA_ISLANDS = 3
DEFAULT_TOP_K = 20
EPSILON = 1.0e-12
DMA_QUALIFICATION = (
    DEFAULT_OUTPUT
    / "stage4a_q_qualification/session1/dma_identification_summary.json"
)


@dataclass(frozen=True)
class Label:
    position: int
    vta_islands: int
    last_device: str
    max_cpu_stage_run_only_ms: float
    max_cpu_stage_ms: float
    last_cpu_stage_resource_ms: float
    vta_service_sum_ms: float
    vta_mutex_boundary_sum_ms: float
    cpu_boundary_work_sum_ms: float
    cpu_core_work_sum_ms: float
    cpu_stage_core_work_sum_ms: float
    cpu_stage_core_work_by_threads_ms: tuple[float, float, float, float]
    max_cpu_thread_parameter: int
    boundary_host_core_work_sum_ms: float
    shared_ddr_service_ms: float
    cpu_ddr_logical_bytes: int
    vta_ddr_physical_load_bytes: int
    vta_ddr_physical_store_bytes: int
    boundary_ddr_bytes: int
    path: tuple[tuple[str, int], ...]

    @property
    def state(self):
        return (self.position, self.vta_islands, self.last_device)

    @property
    def cpu_core_pool_lower_bound_ms(self):
        physical_pool_bound = self.cpu_core_work_sum_ms / PHYSICAL_CPU_CORES
        return max(physical_pool_bound, self.cpu_prefix_mask_lower_bound_ms)

    @property
    def cpu_prefix_mask_capacity_bounds_ms(self):
        """Capacity bounds for nested shared masks ``[0, threads)``."""
        confined_work = 0.0
        bounds = []
        for width, work in enumerate(self.cpu_stage_core_work_by_threads_ms, 1):
            confined_work += work
            bounds.append(confined_work / width)
        return tuple(bounds)

    @property
    def cpu_prefix_mask_lower_bound_ms(self):
        return max(self.cpu_prefix_mask_capacity_bounds_ms, default=0.0)

    @property
    def search_lower_bound(self):
        """Monotonic bound used by k-best search before the final CPU mask is known."""
        return max(
            self.max_cpu_stage_ms,
            self.vta_service_sum_ms + self.vta_mutex_boundary_sum_ms,
            self.cpu_core_pool_lower_bound_ms,
            self.shared_ddr_service_ms,
        )

    @property
    def score(self):
        return max(
            self.max_cpu_stage_ms,
            self.vta_service_sum_ms + self.vta_mutex_boundary_sum_ms,
            self.cpu_core_pool_lower_bound_ms,
            self.shared_ddr_service_ms,
        )

    @property
    def score_without_direct_boundary(self):
        return max(
            self.max_cpu_stage_run_only_ms,
            self.vta_service_sum_ms,
            (self.cpu_core_work_sum_ms - self.boundary_host_core_work_sum_ms)
            / PHYSICAL_CPU_CORES,
            self.cpu_prefix_mask_lower_bound_ms,
            self.shared_ddr_service_ms,
        )


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def segment_id(device, start, end):
    return "{}:{:02d}:{:02d}".format(device, start, end)


def boundary_id(cut, direction):
    return "boundary:{:02d}:{}".format(cut, direction)


def candidate_id(path):
    return "__".join(
        "{}_t{}".format(item[0].replace(":", "-"), item[1]) for item in path
    )


def topology_id(path):
    return "__".join(item[0].replace(":", "-") for item in path)


def positive_allocations(count):
    """Enumerate independent per-stage TVM thread parameters."""
    yield from itertools.product(CPU_THREAD_CHOICES, repeat=count)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_vta_physical_traffic(local_cost):
    """Fit physical VTA LOAD/STORE bytes per logical GOP from P2 profiler totals."""
    points = {}
    for measurement in local_cost["fit_stage_measurements"]:
        vta_stages = [stage for stage in measurement["stages"] if stage["device"] == "vta"]
        if len(vta_stages) != 1:
            raise RuntimeError("P2 VTA traffic fit requires one VTA stage per case")
        stage = vta_stages[0]
        key = tuple(stage["unit_names"])
        rows = int(measurement["serial_rows"])
        profiler = measurement["serial_vta_profiler"]
        point = {
            "group": measurement["scheme"],
            "logical_gops": float(stage["logical_compute_gops_est"]),
            "load_bytes_per_inference": float(profiler["load_bytes"]) / rows,
            "store_bytes_per_inference": float(profiler["store_bytes"]) / rows,
        }
        previous = points.get(key)
        if previous and any(
            not math.isclose(previous[name], point[name], rel_tol=0, abs_tol=EPSILON)
            for name in ("logical_gops", "load_bytes_per_inference", "store_bytes_per_inference")
        ):
            raise RuntimeError("thread sweep changed VTA physical traffic")
        points[key] = point
    unique = list(points.values())
    denominator = sum(point["logical_gops"] ** 2 for point in unique)
    if denominator <= 0:
        raise RuntimeError("VTA traffic fit is singular")
    load_rate = sum(
        point["logical_gops"] * point["load_bytes_per_inference"] for point in unique
    ) / denominator
    store_rate = sum(
        point["logical_gops"] * point["store_bytes_per_inference"] for point in unique
    ) / denominator
    holdout = local_cost["grouped_holdout_measurement"]
    holdout_stage = next(stage for stage in holdout["stages"] if stage["device"] == "vta")
    holdout_rows = int(holdout["serial_rows"])
    holdout_profiler = holdout["serial_vta_profiler"]
    holdout_observed_total = (
        float(holdout_profiler["load_bytes"]) + float(holdout_profiler["store_bytes"])
    ) / holdout_rows
    holdout_predicted_total = float(holdout_stage["logical_compute_gops_est"]) * (
        load_rate + store_rate
    )
    return {
        "fit_policy": "nonnegative_through_origin_deduplicated_segment_v1",
        "fit_point_count": len(unique),
        "load_bytes_per_logical_gop": load_rate,
        "store_bytes_per_logical_gop": store_rate,
        "fit_points": unique,
        "grouped_holdout_total_traffic_ape": abs(
            holdout_predicted_total - holdout_observed_total
        )
        / holdout_observed_total,
    }


def fit_boundary_ownership(local_cost):
    """Split direct boundary service by the runner resource that executes it."""
    grouped = {}
    for measurement in local_cost["fit_stage_measurements"]:
        stages = measurement["stages"]
        if [stage["device"] for stage in stages] != ["cpu", "vta", "cpu"]:
            raise RuntimeError("boundary ownership fit requires CPU-VTA-CPU cases")
        observations = {
            "cpu_to_vta": {
                "logical_bytes": int(stages[0]["output_bytes"]),
                "cpu_owner_ms": float(stages[0]["get_ms"]),
                "vta_mutex_ms": float(stages[1]["set_ms"]),
            },
            "vta_to_cpu": {
                "logical_bytes": int(stages[1]["output_bytes"]),
                "cpu_owner_ms": float(stages[2]["set_ms"]),
                "vta_mutex_ms": float(stages[1]["get_ms"]),
            },
        }
        for direction, observation in observations.items():
            grouped.setdefault((measurement["scheme"], direction), []).append(observation)

    points = {"cpu_to_vta": [], "vta_to_cpu": []}
    for (group, direction), observations in grouped.items():
        byte_values = {item["logical_bytes"] for item in observations}
        if len(byte_values) != 1:
            raise RuntimeError("thread sweep changed boundary bytes")
        points[direction].append(
            {
                "group": group,
                "logical_bytes": byte_values.pop(),
                "cpu_owner_ms": median_value(
                    item["cpu_owner_ms"] for item in observations
                ),
                "vta_mutex_ms": median_value(
                    item["vta_mutex_ms"] for item in observations
                ),
            }
        )

    models = {}
    for direction, direction_points in points.items():
        denominator = sum(item["logical_bytes"] ** 2 for item in direction_points)
        if denominator <= 0:
            raise RuntimeError("boundary ownership fit is singular")
        models[direction] = {
            "fit_policy": "through_origin_group_median_v1",
            "fit_point_count": len(direction_points),
            "cpu_owner_ms_per_byte": sum(
                item["logical_bytes"] * item["cpu_owner_ms"]
                for item in direction_points
            )
            / denominator,
            "vta_mutex_ms_per_byte": sum(
                item["logical_bytes"] * item["vta_mutex_ms"]
                for item in direction_points
            )
            / denominator,
            "fit_points": direction_points,
        }

    holdout = local_cost["grouped_holdout_measurement"]
    stages = holdout["stages"]
    holdout_observations = {
        "cpu_to_vta": {
            "logical_bytes": int(stages[0]["output_bytes"]),
            "total_ms": float(stages[0]["get_ms"]) + float(stages[1]["set_ms"]),
        },
        "vta_to_cpu": {
            "logical_bytes": int(stages[1]["output_bytes"]),
            "total_ms": float(stages[1]["get_ms"]) + float(stages[2]["set_ms"]),
        },
    }
    for direction, observation in holdout_observations.items():
        model = models[direction]
        predicted = observation["logical_bytes"] * (
            model["cpu_owner_ms_per_byte"] + model["vta_mutex_ms_per_byte"]
        )
        model["grouped_holdout_total_ape"] = abs(
            predicted - observation["total_ms"]
        ) / observation["total_ms"]
    return models


def median_value(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def load_dma_qualification(path=DMA_QUALIFICATION):
    payload = json.loads(Path(path).read_text())
    if not payload.get("contiguous_load_store_identified"):
        raise RuntimeError("qualified VTA LOAD/STORE model is unavailable")
    if float(payload.get("fit_r_squared", 0.0)) < 0.99:
        raise RuntimeError("qualified VTA LOAD/STORE model failed R2 gate")
    return {
        "path": str(Path(path).relative_to(repo_root())),
        "sha256": sha256_file(path),
        "load_bandwidth_GBps": float(payload["fitted_load_bandwidth_GBps"]),
        "store_bandwidth_GBps": float(payload["fitted_store_bandwidth_GBps"]),
        "fit_r_squared": float(payload["fit_r_squared"]),
        "measurement_path": payload["measurement_path"],
        "reuse_scope": "provisional V1 device-DMA component rate; P4 shortlist must validate",
    }


def build_context(profile, local_cost):
    profile_segments = {item["segment_id"]: item for item in profile["segments"]}
    segment_costs = {item["segment_id"]: item for item in local_cost["segment_costs"]}
    profile_boundaries = {item["boundary_id"]: item for item in profile["boundaries"]}
    boundary_costs = {item["boundary_id"]: item for item in local_cost["boundary_costs"]}
    if set(profile_segments) != set(segment_costs):
        raise RuntimeError("P1 segment manifest and P2 cost table differ")
    if set(profile_boundaries) != set(boundary_costs):
        raise RuntimeError("P1 boundary manifest and P2 cost table differ")

    memory_models = {
        (item["operation"], int(item["threads"])): item
        for item in local_cost["shared_ddr"]
    }
    traffic_model = fit_vta_physical_traffic(local_cost)
    boundary_ownership = fit_boundary_ownership(local_cost)
    dma_qualification = load_dma_qualification()
    return {
        "profile_segments": profile_segments,
        "segment_costs": segment_costs,
        "profile_boundaries": profile_boundaries,
        "boundary_costs": boundary_costs,
        "cpu_memory_models": memory_models,
        "vta_traffic_model": traffic_model,
        "boundary_ownership_models": boundary_ownership,
        "vta_dma_qualification": dma_qualification,
        "local_cost_table_artifact_sha256": local_cost["artifact_sha256"],
        "profile_manifest_artifact_sha256": profile["artifact_sha256"],
    }


def build_outgoing(context):
    outgoing = {}
    for item in context["profile_segments"].values():
        outgoing.setdefault((item["start_index"], item["device"]), []).append(item)
    for items in outgoing.values():
        items.sort(key=lambda item: (item["end_index"], item["segment_id"]))
    return outgoing


def stage_ddr_demand(segment, threads, context):
    if segment["device"] == "cpu":
        input_bytes = schema_nbytes(segment["input_contract"])
        output_bytes = schema_nbytes(segment["output_contract"])
        read = context["cpu_memory_models"][("read", int(threads))]
        write = context["cpu_memory_models"][("write", int(threads))]
        service = input_bytes / (float(read["bandwidth_GBps_median"]) * 1.0e6)
        service += output_bytes / (float(write["bandwidth_GBps_median"]) * 1.0e6)
        return {
            "cpu_logical_bytes": input_bytes + output_bytes,
            "vta_load_bytes": 0,
            "vta_store_bytes": 0,
            "service_ms": service,
        }
    gops = float(segment["logical_compute_gops_est"])
    traffic = context["vta_traffic_model"]
    dma = context["vta_dma_qualification"]
    load_bytes = int(round(gops * traffic["load_bytes_per_logical_gop"]))
    store_bytes = int(round(gops * traffic["store_bytes_per_logical_gop"]))
    service = load_bytes / (dma["load_bandwidth_GBps"] * 1.0e6)
    service += store_bytes / (dma["store_bandwidth_GBps"] * 1.0e6)
    return {
        "cpu_logical_bytes": 0,
        "vta_load_bytes": load_bytes,
        "vta_store_bytes": store_bytes,
        "service_ms": service,
    }


def boundary_components(boundary_cost, context):
    model = context["boundary_ownership_models"][boundary_cost["direction"]]
    logical_bytes = int(boundary_cost["logical_bytes"])
    cpu_owner_ms = logical_bytes * model["cpu_owner_ms_per_byte"]
    vta_mutex_ms = logical_bytes * model["vta_mutex_ms_per_byte"]
    return {
        "cpu_owner_ms": cpu_owner_ms,
        "vta_mutex_ms": vta_mutex_ms,
        "total_ms": cpu_owner_ms + vta_mutex_ms,
        "host_core_demand_ms": float(boundary_cost["host_core_demand_ms"]),
    }


def add_cpu_stage_core_work_by_threads(current, threads, core_work_ms):
    """Accumulate work by the width of its shared prefix affinity mask."""
    threads = int(threads)
    if threads not in CPU_THREAD_CHOICES:
        raise RuntimeError("invalid CPU stage thread parameter: {}".format(threads))
    updated = list(current)
    updated[threads - 1] += float(core_work_ms)
    return tuple(updated)


def advance_label(label, segment, threads, context):
    """Apply one already-validated segment transition without scanning alternatives."""
    next_device = "cpu" if not label.last_device else (
        "vta" if label.last_device == "cpu" else "cpu"
    )
    threads = int(threads)
    if segment["device"] != next_device or int(segment["start_index"]) != label.position:
        raise RuntimeError("segment does not continue the current alternating-device prefix")
    if threads not in [int(value) for value in segment["thread_choices"]]:
        raise RuntimeError("segment has an invalid TVM thread parameter")
    end_position = int(segment["end_index"]) + 1
    if end_position == len(UNIT_ORDER) and next_device != "cpu":
        raise RuntimeError("complete path must end on CPU")

    boundary_cpu_ms = 0.0
    boundary_vta_ms = 0.0
    boundary_host_core_ms = 0.0
    boundary_bytes = 0
    if label.last_device:
        direction = "{}_to_{}".format(label.last_device, next_device)
        boundary_cost = context["boundary_costs"][boundary_id(label.position, direction)]
        components = boundary_components(boundary_cost, context)
        boundary_cpu_ms = components["cpu_owner_ms"]
        boundary_vta_ms = components["vta_mutex_ms"]
        boundary_host_core_ms = components["host_core_demand_ms"]
        boundary_bytes = int(boundary_cost["logical_bytes"])

    service = float(
        context["segment_costs"][segment["segment_id"]]["service_ms_by_threads"][
            str(threads)
        ]
    )
    host_core_demand = float(
        context["segment_costs"][segment["segment_id"]][
            "host_core_demand_ms_by_threads"
        ][str(threads)]
    )
    ddr = stage_ddr_demand(segment, threads, context)
    if next_device == "cpu":
        return Label(
            position=end_position,
            vta_islands=label.vta_islands,
            last_device="cpu",
            max_cpu_stage_run_only_ms=max(label.max_cpu_stage_run_only_ms, service),
            max_cpu_stage_ms=max(label.max_cpu_stage_ms, service + boundary_cpu_ms),
            last_cpu_stage_resource_ms=service + boundary_cpu_ms,
            vta_service_sum_ms=label.vta_service_sum_ms,
            vta_mutex_boundary_sum_ms=label.vta_mutex_boundary_sum_ms + boundary_vta_ms,
            cpu_boundary_work_sum_ms=label.cpu_boundary_work_sum_ms + boundary_cpu_ms,
            cpu_core_work_sum_ms=label.cpu_core_work_sum_ms
            + host_core_demand
            + boundary_host_core_ms,
            cpu_stage_core_work_sum_ms=label.cpu_stage_core_work_sum_ms
            + host_core_demand,
            cpu_stage_core_work_by_threads_ms=add_cpu_stage_core_work_by_threads(
                label.cpu_stage_core_work_by_threads_ms, threads, host_core_demand
            ),
            max_cpu_thread_parameter=max(label.max_cpu_thread_parameter, threads),
            boundary_host_core_work_sum_ms=label.boundary_host_core_work_sum_ms
            + boundary_host_core_ms,
            shared_ddr_service_ms=label.shared_ddr_service_ms + ddr["service_ms"],
            cpu_ddr_logical_bytes=label.cpu_ddr_logical_bytes + ddr["cpu_logical_bytes"],
            vta_ddr_physical_load_bytes=label.vta_ddr_physical_load_bytes,
            vta_ddr_physical_store_bytes=label.vta_ddr_physical_store_bytes,
            boundary_ddr_bytes=label.boundary_ddr_bytes + boundary_bytes,
            path=label.path + ((segment["segment_id"], threads),),
        )
    if label.vta_islands >= MAX_VTA_ISLANDS:
        raise RuntimeError("VTA island limit exceeded")
    return Label(
        position=end_position,
        vta_islands=label.vta_islands + 1,
        last_device="vta",
        max_cpu_stage_run_only_ms=label.max_cpu_stage_run_only_ms,
        max_cpu_stage_ms=max(
            label.max_cpu_stage_ms,
            label.last_cpu_stage_resource_ms + boundary_cpu_ms,
        ),
        last_cpu_stage_resource_ms=0.0,
        vta_service_sum_ms=label.vta_service_sum_ms + service,
        vta_mutex_boundary_sum_ms=label.vta_mutex_boundary_sum_ms + boundary_vta_ms,
        cpu_boundary_work_sum_ms=label.cpu_boundary_work_sum_ms + boundary_cpu_ms,
        cpu_core_work_sum_ms=label.cpu_core_work_sum_ms
        + host_core_demand
        + boundary_host_core_ms,
        cpu_stage_core_work_sum_ms=label.cpu_stage_core_work_sum_ms,
        cpu_stage_core_work_by_threads_ms=label.cpu_stage_core_work_by_threads_ms,
        max_cpu_thread_parameter=label.max_cpu_thread_parameter,
        boundary_host_core_work_sum_ms=label.boundary_host_core_work_sum_ms
        + boundary_host_core_ms,
        shared_ddr_service_ms=label.shared_ddr_service_ms + ddr["service_ms"],
        cpu_ddr_logical_bytes=label.cpu_ddr_logical_bytes,
        vta_ddr_physical_load_bytes=label.vta_ddr_physical_load_bytes + ddr["vta_load_bytes"],
        vta_ddr_physical_store_bytes=label.vta_ddr_physical_store_bytes
        + ddr["vta_store_bytes"],
        boundary_ddr_bytes=label.boundary_ddr_bytes + boundary_bytes,
        path=label.path + ((segment["segment_id"], threads),),
    )


def expand_label(label, context, outgoing):
    next_device = "cpu" if not label.last_device else (
        "vta" if label.last_device == "cpu" else "cpu"
    )
    for segment in outgoing.get((label.position, next_device), []):
        if int(segment["end_index"]) + 1 == len(UNIT_ORDER) and next_device != "cpu":
            continue
        if next_device == "vta" and label.vta_islands >= MAX_VTA_ISLANDS:
            continue
        for threads in segment["thread_choices"]:
            yield advance_label(label, segment, threads, context)


def is_complete(label):
    return (
        label.position == len(UNIT_ORDER)
        and label.last_device == "cpu"
        and 1 <= label.vta_islands <= MAX_VTA_ISLANDS
    )


def dominates(left, right):
    left_metrics = (
        left.max_cpu_stage_ms,
        left.last_cpu_stage_resource_ms,
        left.vta_service_sum_ms + left.vta_mutex_boundary_sum_ms,
        left.cpu_core_pool_lower_bound_ms,
        left.shared_ddr_service_ms,
    )
    right_metrics = (
        right.max_cpu_stage_ms,
        right.last_cpu_stage_resource_ms,
        right.vta_service_sum_ms + right.vta_mutex_boundary_sum_ms,
        right.cpu_core_pool_lower_bound_ms,
        right.shared_ddr_service_ms,
    )
    return all(a <= b + EPSILON for a, b in zip(left_metrics, right_metrics)) and (
        any(a < b - EPSILON for a, b in zip(left_metrics, right_metrics))
        or left.path <= right.path
    )


def insert_pareto(frontier, candidate):
    if any(dominates(item, candidate) for item in frontier):
        return frontier, False, 1
    kept = [item for item in frontier if not dominates(candidate, item)]
    removed = len(frontier) - len(kept)
    kept.append(candidate)
    kept.sort(key=lambda item: (item.score, item.path))
    return kept, True, removed


def initial_label():
    return Label(
        position=0,
        vta_islands=0,
        last_device="",
        max_cpu_stage_run_only_ms=0.0,
        max_cpu_stage_ms=0.0,
        last_cpu_stage_resource_ms=0.0,
        vta_service_sum_ms=0.0,
        vta_mutex_boundary_sum_ms=0.0,
        cpu_boundary_work_sum_ms=0.0,
        cpu_core_work_sum_ms=0.0,
        cpu_stage_core_work_sum_ms=0.0,
        cpu_stage_core_work_by_threads_ms=(0.0, 0.0, 0.0, 0.0),
        max_cpu_thread_parameter=0,
        boundary_host_core_work_sum_ms=0.0,
        shared_ddr_service_ms=0.0,
        cpu_ddr_logical_bytes=0,
        vta_ddr_physical_load_bytes=0,
        vta_ddr_physical_store_bytes=0,
        boundary_ddr_bytes=0,
        path=tuple(),
    )


def solve_single_best_pareto(context):
    outgoing = build_outgoing(context)
    start = initial_label()
    states = {start.state: [start]}
    generated = 0
    pruned = 0
    for position in range(len(UNIT_ORDER)):
        current = [
            label
            for state, labels in list(states.items())
            if state[0] == position
            for label in labels
        ]
        for label in current:
            for successor in expand_label(label, context, outgoing):
                generated += 1
                frontier = states.get(successor.state, [])
                frontier, inserted, removed = insert_pareto(frontier, successor)
                pruned += removed + (0 if inserted else 1)
                states[successor.state] = frontier
    complete = [
        label for labels in states.values() for label in labels if is_complete(label)
    ]
    if not complete:
        raise RuntimeError("Pareto DP found no complete candidate")
    best = min(complete, key=lambda item: (item.score, candidate_id(item.path)))
    return best, {
        "state_count": len(states),
        "generated_label_count": generated,
        "pareto_pruned_label_count": pruned,
        "complete_frontier_label_count": len(complete),
    }


def solve_k_best_label_setting(context, top_k=DEFAULT_TOP_K):
    """Exact k-best search using the monotonic partial objective as a lower bound."""
    outgoing = build_outgoing(context)
    start = initial_label()
    serial = itertools.count()
    heap = [(start.search_lower_bound, next(serial), start)]
    complete = []
    threshold = None
    expanded = 0
    generated = 0
    peak_queue = 1
    while heap:
        if threshold is not None and heap[0][0] > threshold + EPSILON:
            break
        _, _, label = heapq.heappop(heap)
        if is_complete(label):
            complete.append(label)
            if len(complete) >= top_k:
                threshold = sorted(item.score for item in complete)[top_k - 1]
            continue
        if label.position >= len(UNIT_ORDER):
            continue
        expanded += 1
        for successor in expand_label(label, context, outgoing):
            generated += 1
            heapq.heappush(
                heap, (successor.search_lower_bound, next(serial), successor)
            )
        peak_queue = max(peak_queue, len(heap))
    ranked = sorted(complete, key=lambda item: (item.score, candidate_id(item.path)))
    unique = []
    seen = set()
    for item in ranked:
        cid = candidate_id(item.path)
        if cid not in seen:
            unique.append(item)
            seen.add(cid)
    return unique[:top_k], {
        "expanded_label_count": expanded,
        "generated_label_count": generated,
        "complete_labels_at_stop": len(complete),
        "peak_queue_size": peak_queue,
        "stop_threshold_ms": threshold,
        "safe_pruning": "stop only when every queued monotonic lower bound exceeds "
        "kth complete score",
        "pareto_pruning_for_top_k": False,
    }


def path_from_scheme(scheme, allocation):
    cpu_ordinal = 0
    path = []
    for stage in scheme:
        start = UNIT_ORDER.index(stage["unit_names"][0])
        end = UNIT_ORDER.index(stage["unit_names"][-1])
        threads = 1
        if stage["device"] == "cpu":
            threads = allocation[cpu_ordinal]
            cpu_ordinal += 1
        path.append((segment_id(stage["device"], start, end), threads))
    return tuple(path)


def label_from_path(path, context, outgoing=None):
    label = initial_label()
    for sid, threads in path:
        try:
            segment = context["profile_segments"][sid]
        except KeyError as err:
            raise RuntimeError("unknown segment: {}".format(sid)) from err
        label = advance_label(label, segment, threads, context)
    if not is_complete(label):
        raise RuntimeError("path is incomplete")
    return label


def _retain_top_k(items, candidate, key, top_k):
    if len(items) < top_k or key(candidate) < key(items[-1]):
        items.append(candidate)
        items.sort(key=key)
        del items[top_k:]


def enumerate_oracle(unit_schema, context, top_k=DEFAULT_TOP_K):
    """Stream every execution configuration while retaining only exact Top-K lists."""
    schemes, _, _ = enumerate_reachable(unit_schema)
    outgoing = build_outgoing(context)
    ranked = []
    no_boundary_ranked = []
    topology_ids = set()
    execution_count = 0
    for scheme in schemes:
        cpu_stages = sum(stage["device"] == "cpu" for stage in scheme)
        for allocation in positive_allocations(cpu_stages):
            path = path_from_scheme(scheme, allocation)
            label = label_from_path(path, context, outgoing=outgoing)
            execution_count += 1
            _retain_top_k(ranked, label, rank_key, top_k)
            _retain_top_k(
                no_boundary_ranked,
                label,
                lambda item: (
                    item.score_without_direct_boundary,
                    candidate_id(item.path),
                ),
                top_k,
            )
            topology_ids.add(topology_id(path))
    ids = [candidate_id(item.path) for item in ranked]
    if len(ids) != len(set(ids)):
        raise RuntimeError("streaming enumeration produced duplicate Top-K candidates")
    return ranked, no_boundary_ranked, len(topology_ids), execution_count


def candidate_record(label, rank, context):
    stages = []
    boundaries = []
    for ordinal, (sid, threads) in enumerate(label.path):
        profile = context["profile_segments"][sid]
        cost = context["segment_costs"][sid]
        service = float(cost["service_ms_by_threads"][str(threads)])
        host_core_demand = float(cost["host_core_demand_ms_by_threads"][str(threads)])
        ddr = stage_ddr_demand(profile, threads, context)
        stages.append(
            {
                "stage_index": ordinal,
                "segment_id": sid,
                "device": profile["device"],
                "threads": threads,
                "cpu_core_mask": list(range(threads)) if profile["device"] == "cpu" else [],
                "cpu_thread_semantics": "per-stage TVM runtime parameter",
                "run_service_ms": service,
                "host_core_demand_ms": host_core_demand,
                "owned_boundary_service_ms": 0.0,
                "resource_service_ms": service,
                "shared_ddr_demand": ddr,
                "logical_compute_gops_est": cost["logical_compute_gops_est"],
                "compiler_status": cost["compiler_status"],
                "ddr_accounting_ids": [
                    "{}:stage{}:{}".format(
                        candidate_id(label.path),
                        ordinal,
                        suffix,
                    )
                    for suffix in (
                        ("cpu_logical_input_read", "cpu_logical_output_write")
                        if profile["device"] == "cpu"
                        else ("lowered_total_load_est", "lowered_total_store_est")
                    )
                ],
            }
        )
        if ordinal:
            previous = stages[ordinal - 1]
            direction = "{}_to_{}".format(previous["device"], profile["device"])
            bid = boundary_id(profile["start_index"], direction)
            boundary = context["profile_boundaries"][bid]
            boundary_cost = context["boundary_costs"][bid]
            components = boundary_components(boundary_cost, context)
            cpu_stage = stages[ordinal - 1] if direction == "cpu_to_vta" else stages[ordinal]
            cpu_stage["owned_boundary_service_ms"] += components["cpu_owner_ms"]
            cpu_stage["resource_service_ms"] += components["cpu_owner_ms"]
            boundaries.append(
                {
                    "boundary_id": bid,
                    "direction": direction,
                    "logical_bytes": boundary["logical_bytes"],
                    "service_ms": components["total_ms"],
                    "cpu_owner_ms": components["cpu_owner_ms"],
                    "vta_mutex_ms": components["vta_mutex_ms"],
                    "host_core_demand_ms": components["host_core_demand_ms"],
                    "p2_aggregate_service_ms": boundary_cost["service_ms"],
                    "host_accounting_ids": [
                        slot["host_accounting_id"] for slot in boundary["slots"]
                    ],
                    "stage_dma_accounting_ids": [
                        slot["dma_accounting_id"] for slot in boundary["slots"]
                    ],
                    "stage_dma_additional_service_ms": 0.0,
                    "ownership": "service_ms is host adapter/set-get only; stage DMA is "
                    "already included in the VTA physical LOAD/STORE estimate",
                }
            )
    cid = candidate_id(label.path)
    return {
        "rank": rank,
        "candidate_id": cid,
        "topology_id": topology_id(label.path),
        "stage_count": len(stages),
        "vta_island_count": label.vta_islands,
        "cpu_stage_thread_parameters": [
            stage["threads"] for stage in stages if stage["device"] == "cpu"
        ],
        "cpu_thread_sum_is_a_constraint": False,
        "stages": stages,
        "boundaries": boundaries,
        "resource_load_ms": {
            "max_cpu_stage_run_only": label.max_cpu_stage_run_only_ms,
            "max_cpu_stage": label.max_cpu_stage_ms,
            "single_vta_service": label.vta_service_sum_ms,
            "cpu_owned_boundary_work_total": label.cpu_boundary_work_sum_ms,
            "vta_mutex_boundary_work_total": label.vta_mutex_boundary_sum_ms,
            "direct_boundary_communication": label.cpu_boundary_work_sum_ms
            + label.vta_mutex_boundary_sum_ms,
            "single_vta_plus_mutex_boundaries": label.vta_service_sum_ms
            + label.vta_mutex_boundary_sum_ms,
            "cpu_core_pool_work": label.cpu_core_work_sum_ms,
            "cpu_stage_core_work": label.cpu_stage_core_work_sum_ms,
            "cpu_prefix_mask_capacity": label.max_cpu_thread_parameter,
            "cpu_stage_core_work_by_threads": list(
                label.cpu_stage_core_work_by_threads_ms
            ),
            "cpu_physical_pool_lower_bound": label.cpu_core_work_sum_ms
            / PHYSICAL_CPU_CORES,
            "cpu_prefix_mask_capacity_bounds": list(
                label.cpu_prefix_mask_capacity_bounds_ms
            ),
            "cpu_prefix_mask_lower_bound": label.cpu_prefix_mask_lower_bound_ms,
            "cpu_prefix_union_average_legacy_lower_bound": (
                label.cpu_stage_core_work_sum_ms / label.max_cpu_thread_parameter
            ),
            "cpu_core_pool_lower_bound": label.cpu_core_pool_lower_bound_ms,
            "boundary_host_core_work_total": label.boundary_host_core_work_sum_ms,
            "shared_ddr_physical_demand": label.shared_ddr_service_ms,
        },
        "shared_ddr_traffic": {
            "cpu_logical_bytes_lower_bound": label.cpu_ddr_logical_bytes,
            "vta_physical_load_bytes_est": label.vta_ddr_physical_load_bytes,
            "vta_physical_store_bytes_est": label.vta_ddr_physical_store_bytes,
            "direct_boundary_bytes": label.boundary_ddr_bytes,
        },
        "predicted_ii_ms": label.score,
        "predicted_fps": 1000.0 / label.score,
        "ablation_without_direct_boundary_ii_ms": label.score_without_direct_boundary,
        "requires_shortlist_native_compile": any(
            stage["compiler_status"] != "native_compile_passed" for stage in stages
        ),
        "cost_provenance": {
            "profile_manifest_artifact_sha256": context[
                "profile_manifest_artifact_sha256"
            ],
            "local_cost_table_artifact_sha256": context[
                "local_cost_table_artifact_sha256"
            ],
            "candidate_measured_throughput_consumed": False,
            "single_vta_policy": "sum every VTA island and only the VTA-side set/get of "
            "each heterogeneous boundary; CPU-side boundary work belongs to its CPU stage",
            "ddr_policy": "sum CPU logical input/output floor and VTA physical LOAD/STORE "
            "estimate; direct-boundary bytes are already represented by adjacent stage traffic",
            "vta_traffic_model": context["vta_traffic_model"],
            "boundary_ownership_models": context["boundary_ownership_models"],
            "vta_dma_qualification": context["vta_dma_qualification"],
            "ddr_scope_limitation": "VTA traffic is a two-segment logical-GOP fit; CPU internal "
            "weights/activations/cache and concurrent contention remain excluded",
            "cpu_core_pool_policy": "take the maximum of total host core-ms divided by four "
            "physical cores and every nested [0,k) prefix capacity bound; workers share all "
            "cores in their stage mask, so equal per-core placement is not assumed",
            "cpu_contention_scope_limitation": "the work-conservation lower bound is modeled; "
            "contention-induced cache, scheduler, and efficiency loss remains unmodeled and must "
            "not be replaced by a sum(threads) legality rule",
            "no_double_count_rule": "VTA LOAD/STORE service remains inside VTA run_ms; DDR demand "
            "is a parallel resource constraint and direct boundary wall time is not charged to DDR",
            "boundary_overlap_rule": "CPU-side set/get belongs to its CPU stage; only VTA-side "
            "set/get is serialized by the global VTA mutex",
        },
    }


def rank_key(label):
    return (label.score, candidate_id(label.path))


def solve(output_dir, top_k=DEFAULT_TOP_K):
    unit_schema = load_sealed_artifact(
        output_dir / "v1_unit_and_boundary_schema.json",
        "cpu_vta_pipeline_v1_unit_and_boundary_schema",
    )
    profile = load_sealed_artifact(
        output_dir / "v1_profile_manifest.json", "cpu_vta_pipeline_v1_profile_manifest"
    )
    local_cost = load_sealed_artifact(
        output_dir / "v1_local_cost_table.json",
        "cpu_vta_pipeline_v1_local_cost_table",
    )
    execution_state = load_sealed_artifact(
        output_dir / "v1_execution_state.json", "cpu_vta_pipeline_v1_execution_state"
    )
    if execution_state["current_stage"] not in {"V1-P2", "V1-P3"}:
        raise RuntimeError("P3 requires a completed P2 state")
    if execution_state["current_stage"] == "V1-P2" and execution_state[
        "current_stage_status"
    ] != "completed_awaiting_user_confirmation":
        raise RuntimeError("P2 gate is not complete")

    context = build_context(profile, local_cost)
    dp_top, dp_stats = solve_k_best_label_setting(context, top_k)
    oracle_top, no_boundary_oracle, topology_count, execution_count = enumerate_oracle(
        unit_schema, context, top_k=top_k
    )

    dp_ids = [candidate_id(item.path) for item in dp_top]
    oracle_ids = [candidate_id(item.path) for item in oracle_top]
    score_deltas = [abs(left.score - right.score) for left, right in zip(dp_top, oracle_top)]
    accounting_ids = [
        slot[key]
        for boundary in profile["boundaries"]
        for slot in boundary["slots"]
        for key in ("host_accounting_id", "dma_accounting_id")
    ]
    gate = {
        "topology_count_matches_p1": topology_count
        == profile["coverage"]["legal_candidate_count"]
        == 4623,
        "execution_candidate_count_is_972528": execution_count == 972528,
        "exact_top_k_ids_match_enumeration": dp_ids == oracle_ids,
        "exact_top_k_scores_match_enumeration": max(score_deltas, default=0.0) <= EPSILON,
        "no_duplicate_dp_candidates": len(dp_ids) == len(set(dp_ids)),
        "candidate_throughput_not_consumed": True,
        "cpu_stage_threads_in_declared_domain": all(
            threads in CPU_THREAD_CHOICES
            for item in dp_top
            for sid, threads in item.path
            if sid.startswith("cpu:")
        ),
        "accounting_ids_globally_unique": len(accounting_ids) == len(set(accounting_ids)),
        "shared_ddr_demand_nonnegative": all(
            item.shared_ddr_service_ms >= 0 for item in dp_top
        ),
        "cpu_core_pool_bound_present": all(
            item.cpu_core_work_sum_ms > 0 for item in dp_top
        ),
        "physical_vta_traffic_present": all(
            item.vta_ddr_physical_load_bytes > 0
            and item.vta_ddr_physical_store_bytes > 0
            for item in dp_top
        ),
        "qualified_directional_dma_rates_loaded": context["vta_dma_qualification"][
            "fit_r_squared"
        ]
        >= 0.99,
        "vta_traffic_grouped_holdout_total_ape_le_20pct": context[
            "vta_traffic_model"
        ]["grouped_holdout_total_traffic_ape"]
        <= 0.20,
        "boundary_ownership_grouped_holdout_total_ape_le_5pct": all(
            model["grouped_holdout_total_ape"] <= 0.05
            for model in context["boundary_ownership_models"].values()
        ),
    }
    if not all(gate.values()):
        raise RuntimeError("P3 gate failed: {}".format(gate))

    ranked = seal_artifact(
        {
            "schema_version": 3,
            "kind": "cpu_vta_pipeline_v1_dp_ranked_candidates",
            "protocol_id": PROTOCOL_ID,
            "profile_manifest_artifact_sha256": profile["artifact_sha256"],
            "local_cost_table_artifact_sha256": local_cost["artifact_sha256"],
            "objective": "max(max_cpu_run+owned_boundary, "
            "single_vta_service+vta_mutex_boundary, total_host_core_ms/4, "
            "shared_ddr_physical_demand)",
            "top_k": top_k,
            "legal_topology_count": topology_count,
            "legal_execution_candidate_count": execution_count,
            "candidate_measured_throughput_consumed": False,
            "search": {
                "state": "prefix_position+vta_islands+last_device",
                "label": "max_cpu_stage_run_only+max_cpu_stage_with_owned_boundary+"
                "last_cpu_stage_with_owned_boundary+"
                "vta_service_sum+vta_mutex_boundary_sum+cpu_boundary_work_sum+"
                "cpu_core_work_sum+boundary_host_core_work_sum+"
                "shared_ddr_physical_demand+predecessor",
                "k_best": dp_stats,
            },
            "ranked_candidates": [
                candidate_record(item, rank, context)
                for rank, item in enumerate(dp_top, start=1)
            ],
        }
    )
    oracle_digest_payload = [
        {"candidate_id": candidate_id(item.path), "predicted_ii_ms": item.score}
        for item in oracle_top
    ]
    report = seal_artifact(
        {
            "schema_version": 3,
            "kind": "cpu_vta_pipeline_v1_dp_vs_enumeration_report",
            "protocol_id": PROTOCOL_ID,
            "ranked_candidates_artifact_sha256": ranked["artifact_sha256"],
            "oracle_scope": "host-only full streaming enumeration retaining exact Top-K; "
            "no compile and no board execution",
            "legal_topology_count": topology_count,
            "legal_execution_candidate_count": execution_count,
            "oracle_top_k_sha256": seal_artifact(
                {"ranking": oracle_digest_payload}
            )["artifact_sha256"],
            "dp_top_k_candidate_ids": dp_ids,
            "enumeration_top_k_candidate_ids": oracle_ids,
            "top_k_max_absolute_score_delta_ms": max(score_deltas, default=0.0),
            "single_best_candidate_id": oracle_ids[0],
            "single_best_predicted_ii_ms": oracle_top[0].score,
            "single_best_predicted_fps": 1000.0 / oracle_top[0].score,
            "direct_boundary_ablation": {
                "scope": "remove measured CPU-VTA direct boundary wall/core service; "
                "retain VTA run_ms including internal LOAD/STORE",
                "top_k_overlap_count": len(
                    {candidate_id(item.path) for item in oracle_top}
                    & {candidate_id(item.path) for item in no_boundary_oracle}
                ),
                "full_model_top1_candidate_id": oracle_ids[0],
                "without_direct_boundary_top1_candidate_id": candidate_id(
                    no_boundary_oracle[0].path
                ),
                "full_model_top1_ii_ms": oracle_top[0].score,
                "without_direct_boundary_top1_ii_ms": no_boundary_oracle[
                    0
                ].score_without_direct_boundary,
                "historical_mechanism_evidence": {
                    "candidate_count": 200,
                    "communication_aware_mae_ms": 12.8220,
                    "without_direct_boundary_mae_ms": 17.3617,
                    "relative_mae_improvement_percent": 26.15,
                    "source": "PAPER_EVIDENCE.md E1; not consumed as candidate labels",
                },
            },
            "top_k_diagnostics": {
                "vta_island_count": {
                    str(count): sum(item.vta_islands == count for item in dp_top)
                    for count in range(1, MAX_VTA_ISLANDS + 1)
                },
                "bottleneck_count": {
                    "cpu": sum(
                        item.max_cpu_stage_ms
                        >= item.vta_service_sum_ms + item.vta_mutex_boundary_sum_ms
                        and item.max_cpu_stage_ms >= item.shared_ddr_service_ms
                        for item in dp_top
                    ),
                    "single_vta_plus_mutex_boundary": sum(
                        item.vta_service_sum_ms + item.vta_mutex_boundary_sum_ms
                        > item.max_cpu_stage_ms
                        and item.vta_service_sum_ms + item.vta_mutex_boundary_sum_ms
                        >= item.shared_ddr_service_ms
                        for item in dp_top
                    ),
                    "shared_ddr_physical_demand": sum(
                        item.shared_ddr_service_ms > item.max_cpu_stage_ms
                        and item.shared_ddr_service_ms
                        > item.vta_service_sum_ms + item.vta_mutex_boundary_sum_ms
                        for item in dp_top
                    ),
                },
                "direct_boundary_service_min_ms": min(
                    item.cpu_boundary_work_sum_ms + item.vta_mutex_boundary_sum_ms
                    for item in dp_top
                ),
                "direct_boundary_service_max_ms": max(
                    item.cpu_boundary_work_sum_ms + item.vta_mutex_boundary_sum_ms
                    for item in dp_top
                ),
                "cpu_owned_boundary_work_min_ms": min(
                    item.cpu_boundary_work_sum_ms for item in dp_top
                ),
                "cpu_owned_boundary_work_max_ms": max(
                    item.cpu_boundary_work_sum_ms for item in dp_top
                ),
                "vta_mutex_boundary_work_min_ms": min(
                    item.vta_mutex_boundary_sum_ms for item in dp_top
                ),
                "vta_mutex_boundary_work_max_ms": max(
                    item.vta_mutex_boundary_sum_ms for item in dp_top
                ),
                "shared_ddr_demand_min_ms": min(
                    item.shared_ddr_service_ms for item in dp_top
                ),
                "shared_ddr_demand_max_ms": max(
                    item.shared_ddr_service_ms for item in dp_top
                ),
                "shortlist_native_compile_required_count": sum(
                    any(
                        context["segment_costs"][sid]["compiler_status"]
                        != "native_compile_passed"
                        for sid, _ in item.path
                    )
                    for item in dp_top
                ),
            },
            "gate": gate,
            "proof_scope": {
                "top_k": "partial objective is monotonic; search stops only after every queued "
                "lower bound exceeds the kth complete score; Pareto pruning is disabled for k-best",
                "pareto_shortcut": "not used for production ranking because equal-objective "
                "paths can have different deterministic candidate-id tie order",
                "not_proven": "hardware ranking accuracy, CPU internal physical traffic, "
                "contention-induced CPU efficiency loss, concurrent DDR contention, numerical "
                "package correctness, and candidate throughput",
            },
        }
    )
    state = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_execution_state",
            "protocol_id": PROTOCOL_ID,
            "current_stage": "V1-P3",
            "current_stage_status": "completed_awaiting_user_confirmation",
            "next_stage": "V1-P4",
            "next_stage_may_start": False,
            "advance_requires_user_confirmation": True,
            "blocking_evidence": [],
            "profile_manifest_artifact_sha256": profile["artifact_sha256"],
            "local_cost_table_artifact_sha256": local_cost["artifact_sha256"],
            "ranked_candidates_artifact_sha256": ranked["artifact_sha256"],
            "dp_vs_enumeration_artifact_sha256": report["artifact_sha256"],
        }
    )
    outputs = {
        "v1_dp_ranked_candidates.json": ranked,
        "v1_dp_vs_enumeration_report.json": report,
        "v1_execution_state.json": state,
    }
    for name, payload in outputs.items():
        validate_artifact_sha256(payload)
        write_json(output_dir / name, payload)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    outputs = solve(Path(args.output_dir), args.top_k)
    report = outputs["v1_dp_vs_enumeration_report.json"]
    print(
        json.dumps(
            {
                "gate": report["gate"],
                "single_best_candidate_id": report["single_best_candidate_id"],
                "single_best_predicted_ii_ms": report["single_best_predicted_ii_ms"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
