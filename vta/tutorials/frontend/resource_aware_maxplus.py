#!/usr/bin/env python3
"""Resource-aware Max-Plus cost model for native CPU/VTA pipelines.

The module is deliberately independent from TVM.  Search frontends convert
their model-specific candidates into :class:`PipelineRecord` objects, while
the solver and residual model operate only on physical resource quantities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


EPSILON = 1.0e-9
MODEL_VERSION = 3
BASE_PARAMETER_BOUNDS = {
    "cpu_compute_scale": (0.50, 2.00),
    "cpu_memory_scale": (0.50, 3.00),
    "vta_compute_scale": (0.50, 2.00),
    "vta_overlap_scale": (0.25, 2.00),
    "dma_load_scale": (0.50, 3.00),
    "dma_store_scale": (0.50, 3.00),
    "spill_scale": (0.50, 4.00),
    "boundary_scale": (0.50, 4.00),
    "bridge_scale": (0.50, 4.00),
    "launch_scale": (0.50, 4.00),
    "cpu_contention_scale": (0.50, 4.00),
}
BASE_PARAMETER_NAMES = tuple(BASE_PARAMETER_BOUNDS)
RESIDUAL_FEATURE_NAMES = (
    "fragmentation_ms",
    "spill_ratio",
    "cpu_oversubscription",
    "inverse_transaction_kib",
    "extra_segment_count",
    "small_island_count",
)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except (TypeError, ValueError):
        return float(default)


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def parse_json_value(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default
    return value if value is not None else default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_compute_ms_from_calibration(
    ops: float, measurement: Mapping[str, Any]
) -> float:
    """Convert a component-level compute calibration into service time.

    End-to-end stage GOP/s includes DMA and runtime overhead and therefore cannot
    be combined with separately calibrated DMA/submit terms in the physical
    model.  This strict entry point prevents that double counting.
    """

    if not measurement.get("eligible_for_physical_compute_model", False):
        raise ValueError("measurement is not eligible for the physical compute model")
    if measurement.get("measurement_kind") not in {
        "cpu_compute_slope", "vta_compute_slope", "component_compute_rate"
    }:
        raise ValueError("physical compute requires a component-level slope measurement")
    if "effective_stage_gops" in measurement or "effective_gops" in measurement:
        raise ValueError("effective stage GOP/s is a black-box diagnostic, not compute service")
    compute_gops = as_float(measurement.get("compute_gops"), 0.0)
    if compute_gops <= 0.0:
        raise ValueError("component compute measurement must provide positive compute_gops")
    return max(0.0, float(ops)) / compute_gops / 1.0e6


def cpu_service_ms(
    ops: float,
    memory_bytes: float,
    component_compute_gops: float,
    memory_bandwidth_gbps: float,
    launch_ms: float = 0.0,
) -> Dict[str, float]:
    """Return a thread-aware CPU Roofline service decomposition."""

    compute_ms = max(0.0, float(ops)) / max(
        float(component_compute_gops), EPSILON
    ) / 1.0e6
    memory_ms = max(0.0, float(memory_bytes)) / max(
        float(memory_bandwidth_gbps) * 1.0e9, EPSILON
    ) * 1000.0
    return {
        "compute_ms": compute_ms,
        "memory_ms": memory_ms,
        "launch_ms": max(0.0, float(launch_ms)),
        "service_ms": max(compute_ms, memory_ms) + max(0.0, float(launch_ms)),
    }


def effective_dma_bandwidth_gbps(
    peak_gbps: float,
    avg_bytes_per_call: float,
    small_call_ratio: float = 0.0,
    strided_call_ratio: float = 0.0,
    padded_call_ratio: float = 0.0,
) -> float:
    """Model transaction efficiency without architecture-specific labels."""

    granularity = min(1.0, max(0.02, float(avg_bytes_per_call) / 16384.0))
    irregularity = (
        0.55 * max(0.0, float(small_call_ratio))
        + 0.35 * max(0.0, float(strided_call_ratio))
        + 0.25 * max(0.0, float(padded_call_ratio))
    )
    efficiency = max(0.03, granularity / (1.0 + irregularity))
    return max(EPSILON, float(peak_gbps) * efficiency)


def vta_service_ms(
    compute_ops: float,
    component_compute_gops: float,
    load_bytes: float,
    store_bytes: float,
    load_bandwidth_gbps: float,
    store_bandwidth_gbps: float,
    overlap_factor: float,
    submit_ms: float = 0.0,
    sync_ms: float = 0.0,
    spill_bytes: float = 0.0,
    spill_calls: int = 0,
    dma_call_latency_ms: float = 0.0,
) -> Dict[str, float]:
    """Return load/compute/store service with partial hardware overlap."""

    compute_ms = max(0.0, float(compute_ops)) / max(
        float(component_compute_gops), EPSILON
    ) / 1.0e6
    load_ms = max(0.0, float(load_bytes)) / max(
        float(load_bandwidth_gbps) * 1.0e9, EPSILON
    ) * 1000.0
    store_ms = max(0.0, float(store_bytes)) / max(
        float(store_bandwidth_gbps) * 1.0e9, EPSILON
    ) * 1000.0
    dominant = max(compute_ms, load_ms, store_ms)
    overlap = min(1.0, max(0.0, float(overlap_factor)))
    overlapped_ms = dominant + overlap * (compute_ms + load_ms + store_ms - dominant)
    spill_ms = max(0.0, float(spill_bytes)) / max(
        min(float(load_bandwidth_gbps), float(store_bandwidth_gbps)) * 1.0e9,
        EPSILON,
    ) * 1000.0 + max(0, int(spill_calls)) * max(0.0, float(dma_call_latency_ms))
    launch_ms = max(0.0, float(submit_ms)) + max(0.0, float(sync_ms))
    return {
        "compute_ms": compute_ms,
        "load_ms": load_ms,
        "store_ms": store_ms,
        "dma_ms": load_ms + store_ms,
        "spill_ms": spill_ms,
        "launch_ms": launch_ms,
        "service_ms": overlapped_ms + spill_ms + launch_ms,
    }


def boundary_service_ms(
    boundary_bytes: float,
    peak_bandwidth_gbps: float,
    avg_bytes_per_call: float,
    call_count: int,
    call_latency_ms: float,
    pack_ms: float = 0.0,
    unpack_ms: float = 0.0,
    small_call_ratio: float = 0.0,
    strided_call_ratio: float = 0.0,
    padded_call_ratio: float = 0.0,
) -> Dict[str, float]:
    bandwidth = effective_dma_bandwidth_gbps(
        peak_bandwidth_gbps,
        avg_bytes_per_call,
        small_call_ratio,
        strided_call_ratio,
        padded_call_ratio,
    )
    transfer_ms = max(0.0, float(boundary_bytes)) / (bandwidth * 1.0e9) * 1000.0
    transaction_ms = max(0, int(call_count)) * max(0.0, float(call_latency_ms))
    return {
        "effective_bandwidth_gbps": bandwidth,
        "transfer_ms": transfer_ms,
        "transaction_ms": transaction_ms,
        "pack_ms": max(0.0, float(pack_ms)),
        "unpack_ms": max(0.0, float(unpack_ms)),
        "service_ms": transfer_ms
        + transaction_ms
        + max(0.0, float(pack_ms))
        + max(0.0, float(unpack_ms)),
    }


@dataclass
class StageService:
    name: str
    device: str
    threads: int = 1
    compute_ms: float = 0.0
    memory_ms: float = 0.0
    dma_ms: float = 0.0
    load_ms: float = 0.0
    store_ms: float = 0.0
    overlap_factor: float = 1.0
    spill_ms: float = 0.0
    launch_ms: float = 0.0
    bridge_ms: float = 0.0
    measured_ms: Optional[float] = None
    core_demand_ms: Optional[float] = None
    core_demand_source: str = ""
    sequential_unit_service_ms: Optional[float] = None
    unit_roofline_components_ms: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def unscaled_service_ms(self) -> float:
        if self.device == "cpu":
            body_ms = (
                sum(
                    max(
                        max(0.0, as_float(item.get("compute_ms"))),
                        max(0.0, as_float(item.get("memory_ms"))),
                    )
                    for item in self.unit_roofline_components_ms
                )
                if self.unit_roofline_components_ms
                else (
                    max(0.0, self.sequential_unit_service_ms)
                    if self.sequential_unit_service_ms is not None
                    else max(self.compute_ms, self.memory_ms)
                )
            )
            return body_ms + self.launch_ms + self.bridge_ms
        load_ms, store_ms = self.dma_channels_ms
        longest = max(self.compute_ms, load_ms, store_ms)
        serialized_remainder = self.compute_ms + load_ms + store_ms - longest
        return (
            longest
            + min(1.0, max(0.0, self.overlap_factor)) * serialized_remainder
            + self.spill_ms
            + self.launch_ms
            + self.bridge_ms
        )

    @property
    def dma_channels_ms(self) -> Tuple[float, float]:
        if self.load_ms > 0.0 or self.store_ms > 0.0:
            return max(0.0, self.load_ms), max(0.0, self.store_ms)
        return max(0.0, self.dma_ms), 0.0


@dataclass
class BoundaryService:
    name: str
    producer: str
    consumer: str
    direction: str
    queue_depth: int
    transfer_resource: str
    pack_ms: float = 0.0
    transfer_ms: float = 0.0
    transaction_ms: float = 0.0
    unpack_ms: float = 0.0

    @property
    def service_ms(self) -> float:
        return (
            max(0.0, self.pack_ms)
            + max(0.0, self.transfer_ms)
            + max(0.0, self.transaction_ms)
            + max(0.0, self.unpack_ms)
        )


@dataclass
class PipelineRecord:
    model: str
    candidate_id: str
    source_path: str
    group_id: str
    measured_cycle_ms: Optional[float]
    queue_depth: int
    stages: List[StageService]
    boundaries: List[BoundaryService] = field(default_factory=list)
    resource_capacities: Dict[str, int] = field(default_factory=dict)
    boundary_ms: float = 0.0
    fragmentation_ms: float = 0.0
    spill_ratio: float = 0.0
    avg_bytes_per_call: float = 0.0
    effective_segment_count: int = 0
    vta_island_count: int = 0
    small_island_count: int = 0
    legacy_score_ms: float = 0.0
    correctness_passed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PipelineRecord":
        data = dict(payload)
        data["stages"] = [
            item if isinstance(item, StageService) else StageService(**item)
            for item in data.get("stages", [])
        ]
        data["boundaries"] = [
            item if isinstance(item, BoundaryService) else BoundaryService(**item)
            for item in data.get("boundaries", [])
        ]
        return cls(**data)


@dataclass
class CycleConstraint:
    name: str
    resource: str
    service_ms: float
    tokens: int
    members: List[str]

    @property
    def cycle_ms(self) -> float:
        return float(self.service_ms) / max(1, int(self.tokens))


def default_base_parameters() -> Dict[str, float]:
    return {name: 1.0 for name in BASE_PARAMETER_NAMES}


def resource_capacity(record: PipelineRecord, name: str, legacy_default: int) -> int:
    value = int((record.resource_capacities or {}).get(name, legacy_default))
    if value <= 0:
        raise ValueError("resource capacity {} must be positive".format(name))
    return value


def scaled_stage_service(stage: StageService, parameters: Mapping[str, float]) -> float:
    cpu_compute_scale = as_float(parameters.get("cpu_compute_scale"), 1.0)
    cpu_memory_scale = as_float(parameters.get("cpu_memory_scale"), 1.0)
    vta_scale = as_float(parameters.get("vta_compute_scale"), 1.0)
    dma_load_scale = as_float(parameters.get("dma_load_scale"), 1.0)
    dma_store_scale = as_float(parameters.get("dma_store_scale"), 1.0)
    spill_scale = as_float(parameters.get("spill_scale"), 1.0)
    bridge_scale = as_float(parameters.get("bridge_scale"), 1.0)
    launch_scale = as_float(parameters.get("launch_scale"), 1.0)
    if stage.device == "cpu":
        if stage.unit_roofline_components_ms:
            return (
                sum(
                    max(
                        cpu_compute_scale * max(0.0, as_float(item.get("compute_ms"))),
                        cpu_memory_scale * max(0.0, as_float(item.get("memory_ms"))),
                    )
                    for item in stage.unit_roofline_components_ms
                )
                + launch_scale * stage.launch_ms
                + bridge_scale * stage.bridge_ms
            )
        if stage.sequential_unit_service_ms is not None:
            return (
                max(cpu_compute_scale, cpu_memory_scale)
                * max(0.0, stage.sequential_unit_service_ms)
                + launch_scale * stage.launch_ms
                + bridge_scale * stage.bridge_ms
            )
        return (
            max(
                cpu_compute_scale * stage.compute_ms,
                cpu_memory_scale * stage.memory_ms,
            )
            + launch_scale * stage.launch_ms
            + bridge_scale * stage.bridge_ms
        )
    compute_ms = vta_scale * stage.compute_ms
    raw_load_ms, raw_store_ms = stage.dma_channels_ms
    load_ms = dma_load_scale * raw_load_ms
    store_ms = dma_store_scale * raw_store_ms
    longest = max(compute_ms, load_ms, store_ms)
    serialized_remainder = compute_ms + load_ms + store_ms - longest
    overlap = min(
        1.0,
        max(
            0.0,
            stage.overlap_factor * as_float(parameters.get("vta_overlap_scale"), 1.0),
        ),
    )
    return (
        longest
        + overlap * serialized_remainder
        + spill_scale * stage.spill_ms
        + launch_scale * stage.launch_ms
        + bridge_scale * stage.bridge_ms
    )


def cpu_core_demand_ms(
    stage: StageService,
    stage_service_ms: float,
    parameters: Mapping[str, float],
    require_measured: bool = False,
) -> Tuple[float, str]:
    """Return CPU core-time demand, separate from stage wall time.

    Old records do not contain process/thread CPU time.  They remain readable
    for historical analysis, but publication-mode predictions must reject the
    legacy wall-time-times-requested-threads proxy.
    """

    if stage.device != "cpu":
        return 0.0, "not_cpu"
    scale = as_float(parameters.get("cpu_contention_scale"), 1.0)
    if stage.core_demand_ms is not None:
        source = stage.core_demand_source or "measured_or_calibrated"
        return scale * max(0.0, as_float(stage.core_demand_ms)), source
    if require_measured:
        raise RuntimeError(
            "CPU stage {} lacks core_demand_ms required by publication mode".format(
                stage.name
            )
        )
    return (
        scale * max(0.0, stage_service_ms) * max(1, int(stage.threads)),
        "legacy_wall_time_x_requested_threads",
    )


def build_cycle_constraints(
    record: PipelineRecord,
    parameters: Optional[Mapping[str, float]] = None,
    require_measured_core_demand: bool = False,
) -> List[CycleConstraint]:
    """Build marked-graph cycle constraints for one periodic pipeline.

    A stage has a one-token worker cycle.  Shared hardware resources add
    resource cycles.  FIFO backpressure is represented by adjacent-stage and
    full-path cycles whose token counts grow with queue depth.
    """

    params = dict(default_base_parameters())
    params.update(parameters or {})
    service = [scaled_stage_service(stage, params) for stage in record.stages]
    service_by_name = {
        stage.name: stage_ms for stage, stage_ms in zip(record.stages, service)
    }
    boundary_services = [
        BoundaryService(
            **{
                **asdict(boundary),
                "pack_ms": params["bridge_scale"] * boundary.pack_ms,
                "transfer_ms": params["boundary_scale"] * boundary.transfer_ms,
                "transaction_ms": params["boundary_scale"] * boundary.transaction_ms,
                "unpack_ms": params["bridge_scale"] * boundary.unpack_ms,
            }
        )
        for boundary in record.boundaries
    ]
    constraints: List[CycleConstraint] = []
    for stage, stage_ms in zip(record.stages, service):
        constraints.append(
            CycleConstraint(
                name="stage:{}".format(stage.name),
                resource="stage_worker",
                service_ms=stage_ms,
                tokens=1,
                members=[stage.name],
            )
        )
    for boundary in boundary_services:
        constraints.append(
            CycleConstraint(
                name="boundary:{}".format(boundary.name),
                resource="boundary_worker",
                service_ms=boundary.service_ms,
                tokens=1,
                members=[boundary.producer, boundary.consumer],
            )
        )

    cpu_demands = [
        cpu_core_demand_ms(
            stage,
            stage_ms,
            params,
            require_measured=require_measured_core_demand,
        )[0]
        for stage, stage_ms in zip(record.stages, service)
        if stage.device == "cpu"
    ]
    cpu_core_ms = sum(cpu_demands)
    if cpu_core_ms > 0.0:
        constraints.append(
            CycleConstraint(
                "resource:cpu_core_pool",
                "cpu_core_pool",
                cpu_core_ms,
                resource_capacity(record, "cpu_core_pool", 4),
                [],
            )
        )

    cpu_memory_ms = sum(
        params["cpu_memory_scale"] * stage.memory_ms
        for stage in record.stages
        if stage.device == "cpu"
    )
    cpu_memory_ms += sum(
        boundary.transfer_ms + boundary.transaction_ms
        for boundary in boundary_services
        if boundary.transfer_resource == "cpu_memory"
    )
    if cpu_memory_ms > 0.0:
        constraints.append(
            CycleConstraint(
                "resource:cpu_memory",
                "cpu_memory",
                cpu_memory_ms,
                resource_capacity(record, "cpu_memory", 1),
                [],
            )
        )

    vta_ms = sum(
        stage_ms
        for stage, stage_ms in zip(record.stages, service)
        if stage.device == "vta"
    )
    if vta_ms > 0.0:
        constraints.append(
            CycleConstraint(
                "resource:vta_mutex",
                "vta_mutex",
                vta_ms,
                resource_capacity(record, "vta_mutex", 1),
                [],
            )
        )

    ps_pl_load_ms = sum(
        params["dma_load_scale"] * stage.dma_channels_ms[0]
        + 0.5 * params["spill_scale"] * stage.spill_ms
        for stage in record.stages
        if stage.device == "vta"
    )
    ps_pl_load_ms += sum(
        boundary.transfer_ms + boundary.transaction_ms
        for boundary in boundary_services
        if boundary.transfer_resource == "ps_pl" and boundary.direction.endswith("_to_vta")
    )
    ps_pl_store_ms = sum(
        params["dma_store_scale"] * stage.dma_channels_ms[1]
        + 0.5 * params["spill_scale"] * stage.spill_ms
        for stage in record.stages
        if stage.device == "vta"
    )
    ps_pl_store_ms += sum(
        boundary.transfer_ms + boundary.transaction_ms
        for boundary in boundary_services
        if boundary.transfer_resource == "ps_pl" and boundary.direction.startswith("vta_to_")
    )
    if ps_pl_load_ms > 0.0:
        constraints.append(
            CycleConstraint(
                "resource:ps_pl_load",
                "ps_pl_load",
                ps_pl_load_ms,
                resource_capacity(record, "ps_pl_load", 1),
                [],
            )
        )
    if ps_pl_store_ms > 0.0:
        constraints.append(
            CycleConstraint(
                "resource:ps_pl_store",
                "ps_pl_store",
                ps_pl_store_ms,
                resource_capacity(record, "ps_pl_store", 1),
                [],
            )
        )

    boundary_transfer_ms = params["boundary_scale"] * max(0.0, record.boundary_ms)
    boundary_resource_ms = params["bridge_scale"] * sum(
        max(0.0, stage.bridge_ms) for stage in record.stages
    )
    boundary_resource_ms += sum(
        boundary.pack_ms + boundary.unpack_ms for boundary in boundary_services
    )
    if not record.boundaries:
        boundary_resource_ms += boundary_transfer_ms
    if boundary_resource_ms > 0.0:
        constraints.append(
            CycleConstraint(
                "resource:boundary",
                "bridge_pack_unpack",
                boundary_resource_ms,
                resource_capacity(record, "bridge_pack_unpack", 1),
                [],
            )
        )

    queue_depth = max(1, int(record.queue_depth))
    if record.boundaries:
        for boundary in boundary_services:
            constraints.append(
                CycleConstraint(
                    name="fifo:{}".format(boundary.name),
                    resource="fifo_backpressure",
                    service_ms=(
                        service_by_name[boundary.producer]
                        + boundary.service_ms
                        + service_by_name[boundary.consumer]
                    ),
                    tokens=max(1, int(boundary.queue_depth)) + 1,
                    members=[boundary.producer, boundary.name, boundary.consumer],
                )
            )
    else:
        for idx in range(max(0, len(service) - 1)):
            constraints.append(
                CycleConstraint(
                    name="fifo:{}:{}".format(record.stages[idx].name, record.stages[idx + 1].name),
                    resource="fifo_backpressure",
                    service_ms=service[idx] + service[idx + 1],
                    tokens=queue_depth + 1,
                    members=[record.stages[idx].name, record.stages[idx + 1].name],
                )
            )
    if service and not record.boundaries:
        inflight_tokens = 1 + queue_depth * max(0, len(service) - 1)
        constraints.append(
            CycleConstraint(
                "pipeline:finite_buffers",
                "finite_buffers",
                sum(service) + boundary_transfer_ms,
                inflight_tokens,
                [stage.name for stage in record.stages],
            )
        )
    return constraints


def maximum_cycle_mean(constraints: Sequence[CycleConstraint]) -> Tuple[float, Optional[CycleConstraint]]:
    if not constraints:
        return 0.0, None
    bottleneck = max(constraints, key=lambda item: item.cycle_ms)
    return bottleneck.cycle_ms, bottleneck


def lightweight_cycle_scores(
    record: PipelineRecord,
    parameters: Optional[Mapping[str, float]] = None,
    require_measured_core_demand: bool = False,
    require_zero_feedback: bool = False,
) -> Dict[str, Any]:
    """Return the staged models used by budgeted candidate search.

    ``compute_balance_ms`` is the cheap compute-only baseline.  It already
    serializes all VTA islands because the native executor owns one physical
    VTA.  ``communication_aware_ms`` adds lowered VTA DMA/runtime service and
    heterogeneous boundary adapters to that serialized path.  The final
    ``executor_resource_ms`` is the existing resource lower bound without the
    optional FIFO/event-graph constraints.

    These scores are deliberately separate so experiments can establish that
    each extra mechanism improves low-budget candidate selection before the
    next one is enabled.
    """

    provenance = lightweight_score_provenance(
        record,
        require_measured_core_demand=require_measured_core_demand,
    )
    if require_zero_feedback and not provenance["zero_feedback_eligible"]:
        raise RuntimeError(
            "candidate {} is not zero-feedback eligible: {}".format(
                record.candidate_id,
                ", ".join(provenance["blockers"]),
            )
        )

    params = dict(default_base_parameters())
    params.update(parameters or {})
    stage_service = [scaled_stage_service(stage, params) for stage in record.stages]

    cpu_compute = max(
        [
            max(0.0, stage.compute_ms, stage.memory_ms)
            for stage in record.stages
            if stage.device == "cpu"
        ]
        or [0.0]
    )
    vta_compute = sum(
        max(0.0, stage.compute_ms)
        for stage in record.stages
        if stage.device == "vta"
    )
    compute_balance = max(cpu_compute, vta_compute, EPSILON)

    cpu_stage = max(
        [
            service_ms
            for stage, service_ms in zip(record.stages, stage_service)
            if stage.device == "cpu"
        ]
        or [0.0]
    )
    vta_serial = sum(
        service_ms
        for stage, service_ms in zip(record.stages, stage_service)
        if stage.device == "vta"
    )
    heterogeneous_adapter = sum(
        boundary.service_ms
        for boundary in record.boundaries
        if "vta" in boundary.direction
    )
    bridge_demand = sum(boundary.service_ms for boundary in record.boundaries)
    communication_aware = max(
        cpu_stage,
        vta_serial + heterogeneous_adapter,
        bridge_demand,
        EPSILON,
    )

    constraints = build_cycle_constraints(
        record,
        params,
        require_measured_core_demand=require_measured_core_demand,
    )
    resource_constraints = [
        item
        for item in constraints
        if item.resource not in ("fifo_backpressure", "finite_buffers")
    ]
    executor_resource, bottleneck = maximum_cycle_mean(resource_constraints)
    return {
        "compute_balance_ms": compute_balance,
        "communication_aware_ms": communication_aware,
        "executor_resource_ms": max(executor_resource, EPSILON),
        "executor_bottleneck": asdict(bottleneck) if bottleneck is not None else None,
        "components": {
            "cpu_compute_stage_ms": cpu_compute,
            "vta_compute_mutex_ms": vta_compute,
            "cpu_full_stage_ms": cpu_stage,
            "vta_full_mutex_ms": vta_serial,
            "heterogeneous_adapter_ms": heterogeneous_adapter,
            "bridge_demand_ms": bridge_demand,
        },
        "provenance": provenance,
        "scope": "lightweight_candidate_ranking_no_fifo_or_fitted_residual",
    }


def lightweight_score_provenance(
    record: PipelineRecord,
    require_measured_core_demand: bool = False,
) -> Dict[str, Any]:
    """Describe whether lightweight ranking can run before candidate measurement.

    Historical records intentionally retain measured outcomes for retrospective
    evaluation.  Their presence is not leakage by itself: the lightweight
    scorer never reads ``measured_cycle_ms``, ``StageService.measured_ms``, or
    direct runtime-profiler payloads.  This gate instead checks the declared
    origin of the service parameters that the scorer does consume.
    """

    declared = record.metadata.get("score_input_provenance") or {}
    service = declared.get("service_parameters") or {}
    workload = declared.get("workload_features") or {}
    service_kind = str(service.get("kind") or "unspecified")
    workload_kind = str(workload.get("kind") or "unspecified")
    allowed_service_kinds = {
        "compiler_static",
        "independent_hardware_profile",
        "legacy_default_static",
    }
    allowed_workload_kinds = {"compiler_static", "lowering_static"}
    blockers = []
    if service_kind not in allowed_service_kinds:
        blockers.append("service_parameters:{}".format(service_kind))
    if workload_kind not in allowed_workload_kinds:
        blockers.append("workload_features:{}".format(workload_kind))
    missing_core_demand = [
        stage.name
        for stage in record.stages
        if stage.device == "cpu" and stage.core_demand_ms is None
    ]
    if require_measured_core_demand and missing_core_demand:
        blockers.append("missing_cpu_core_demand")

    direct_profile = record.metadata.get("direct_communication_profile") or {}
    candidate_measurement_fields = []
    if record.measured_cycle_ms is not None:
        candidate_measurement_fields.append("measured_cycle_ms")
    if any(stage.measured_ms is not None for stage in record.stages):
        candidate_measurement_fields.append("stage.measured_ms")
    if record.metadata.get("measured_stage_run_ms"):
        candidate_measurement_fields.append("metadata.measured_stage_run_ms")
    if direct_profile.get("available"):
        candidate_measurement_fields.append("metadata.direct_communication_profile")

    return {
        "zero_feedback_eligible": not blockers,
        "blockers": blockers,
        "service_parameter_kind": service_kind,
        "service_parameter_source": str(service.get("source") or "unspecified"),
        "workload_feature_kind": workload_kind,
        "workload_feature_source": str(workload.get("source") or "unspecified"),
        "candidate_measurement_fields_present_for_evaluation": candidate_measurement_fields,
        "candidate_measurement_fields_used_by_score": [],
        "missing_cpu_core_demand_stages": missing_core_demand,
    }


def critical_cycle_constraints(
    constraints: Sequence[CycleConstraint],
    rel_tol: float = 1.0e-9,
    abs_tol: float = 1.0e-9,
) -> List[CycleConstraint]:
    """Return every cycle tied at the maximum mean, not just one tie-break winner."""

    cycle_ms, _ = maximum_cycle_mean(constraints)
    return [
        item
        for item in constraints
        if math.isclose(item.cycle_ms, cycle_ms, rel_tol=rel_tol, abs_tol=abs_tol)
    ]


def resource_demand_matrix(
    record: PipelineRecord,
    parameters: Optional[Mapping[str, float]] = None,
    require_measured_core_demand: bool = False,
) -> Dict[str, Any]:
    """Return D[stage, resource] in milliseconds per pipeline item."""

    params = dict(default_base_parameters())
    params.update(parameters or {})
    resources = [
        "cpu_core_pool",
        "cpu_memory",
        "vta_mutex",
        "ps_pl_load",
        "ps_pl_store",
        "bridge_pack_unpack",
    ]
    rows = []
    for stage in record.stages:
        service_ms = scaled_stage_service(stage, params)
        demand = {name: 0.0 for name in resources}
        demand["bridge_pack_unpack"] = params["bridge_scale"] * max(
            0.0, stage.bridge_ms
        )
        if stage.device == "cpu":
            core_demand, core_source = cpu_core_demand_ms(
                stage,
                service_ms,
                params,
                require_measured=require_measured_core_demand,
            )
            demand["cpu_core_pool"] = core_demand
            demand["cpu_memory"] = params["cpu_memory_scale"] * max(
                0.0, stage.memory_ms
            )
            demand["core_demand_source"] = core_source
        else:
            demand["vta_mutex"] = service_ms
            demand["ps_pl_load"] = max(
                0.0,
                params["dma_load_scale"] * stage.dma_channels_ms[0]
                + 0.5 * params["spill_scale"] * stage.spill_ms,
            )
            demand["ps_pl_store"] = max(
                0.0,
                params["dma_store_scale"] * stage.dma_channels_ms[1]
                + 0.5 * params["spill_scale"] * stage.spill_ms,
            )
        rows.append({"stage": stage.name, "device": stage.device, **demand})
    for boundary in record.boundaries:
        demand = {name: 0.0 for name in resources}
        adapter_ms = params["bridge_scale"] * (
            max(0.0, boundary.pack_ms) + max(0.0, boundary.unpack_ms)
        )
        transfer_ms = params["boundary_scale"] * (
            max(0.0, boundary.transfer_ms) + max(0.0, boundary.transaction_ms)
        )
        demand["bridge_pack_unpack"] = adapter_ms
        if boundary.transfer_resource == "cpu_memory":
            demand["cpu_memory"] = transfer_ms
        elif boundary.transfer_resource == "ps_pl" and boundary.direction.endswith("_to_vta"):
            demand["ps_pl_load"] = transfer_ms
        elif boundary.transfer_resource == "ps_pl" and boundary.direction.startswith("vta_to_"):
            demand["ps_pl_store"] = transfer_ms
        rows.append({"stage": boundary.name, "device": "boundary", **demand})
    return {
        "resources": resources,
        "rows": rows,
        "resource_capacities": {
            name: resource_capacity(
                record,
                name,
                4 if name == "cpu_core_pool" else 1,
            )
            for name in resources
        },
        "cpu_core_tokens": resource_capacity(record, "cpu_core_pool", 4),
        "vta_mutex_tokens": resource_capacity(record, "vta_mutex", 1),
        "queue_depth": int(record.queue_depth),
        "boundary_bridge_ms": (
            sum(
                params["boundary_scale"]
                * (max(0.0, boundary.transfer_ms) + max(0.0, boundary.transaction_ms))
                + params["bridge_scale"]
                * (max(0.0, boundary.pack_ms) + max(0.0, boundary.unpack_ms))
                for boundary in record.boundaries
            )
            if record.boundaries
            else params["boundary_scale"] * max(0.0, record.boundary_ms)
            + params["bridge_scale"]
            * sum(max(0.0, stage.bridge_ms) for stage in record.stages)
        ),
    }


def marked_event_graph_payload(
    record: PipelineRecord,
    parameters: Optional[Mapping[str, float]] = None,
    require_measured_core_demand: bool = False,
) -> Dict[str, Any]:
    """Describe the actor, dependency, resource, and bounded-buffer graph.

    This representation makes the modelling assumptions inspectable.  The
    steady-state solver uses the derived resource cycles below; data channels
    have zero initial tokens and reverse free-slot channels carry q tokens,
    which is the standard bounded-buffer backpressure construction.
    """

    params = dict(default_base_parameters())
    params.update(parameters or {})
    services = [scaled_stage_service(stage, params) for stage in record.stages]
    scaled_boundaries = [
        BoundaryService(
            **{
                **asdict(boundary),
                "pack_ms": params["bridge_scale"] * boundary.pack_ms,
                "transfer_ms": params["boundary_scale"] * boundary.transfer_ms,
                "transaction_ms": params["boundary_scale"] * boundary.transaction_ms,
                "unpack_ms": params["bridge_scale"] * boundary.unpack_ms,
            }
        )
        for boundary in record.boundaries
    ]
    matrix = resource_demand_matrix(
        record,
        params,
        require_measured_core_demand=require_measured_core_demand,
    )
    actors = [
        {
            "id": stage.name,
            "device": stage.device,
            "service_ms": service_ms,
            "threads": int(stage.threads),
        }
        for stage, service_ms in zip(record.stages, services)
    ]
    actors.extend(
        {
            "id": boundary.name,
            "device": "boundary",
            "service_ms": boundary.service_ms,
            "threads": 1,
        }
        for boundary in scaled_boundaries
    )
    channels: List[Dict[str, Any]] = []
    queue_depth = max(1, int(record.queue_depth))
    for stage in record.stages:
        channels.append(
            {
                "name": "worker:{}".format(stage.name),
                "kind": "worker_reuse",
                "producer": stage.name,
                "consumer": stage.name,
                "initial_tokens": 1,
                "capacity_tokens": 1,
            }
        )
    graph_boundaries = scaled_boundaries or [
        BoundaryService(
            name="legacy_{}_{}".format(record.stages[index].name, record.stages[index + 1].name),
            producer=record.stages[index].name,
            consumer=record.stages[index + 1].name,
            direction="legacy",
            queue_depth=queue_depth,
            transfer_resource="cpu_memory",
        )
        for index in range(max(0, len(record.stages) - 1))
    ]
    for boundary in graph_boundaries:
        producer = boundary.producer
        consumer = boundary.consumer
        data_consumer = boundary.name if scaled_boundaries else consumer
        channels.extend(
            [
                {
                    "name": "data:{}".format(boundary.name),
                    "kind": "data_dependency",
                    "producer": producer,
                    "consumer": data_consumer,
                    "initial_tokens": 0,
                    "capacity_tokens": max(1, int(boundary.queue_depth)),
                },
                {
                    "name": "slots:{}".format(boundary.name),
                    "kind": "fifo_free_slots",
                    "producer": consumer,
                    "consumer": producer,
                    "initial_tokens": max(1, int(boundary.queue_depth)),
                    "capacity_tokens": max(1, int(boundary.queue_depth)),
                },
            ]
        )
        if scaled_boundaries:
            channels.append(
                {
                    "name": "adapted:{}:{}".format(boundary.name, consumer),
                    "kind": "adapted_data_dependency",
                    "producer": boundary.name,
                    "consumer": consumer,
                    "initial_tokens": 0,
                    "capacity_tokens": max(1, int(boundary.queue_depth)),
                }
            )
    capacities = matrix["resource_capacities"]
    resources = []
    for resource in matrix["resources"]:
        users = [
            {"stage": row["stage"], "demand_ms": as_float(row.get(resource))}
            for row in matrix["rows"]
            if as_float(row.get(resource)) > 0.0
        ]
        resources.append(
            {
                "name": resource,
                "capacity_tokens": capacities[resource],
                "users": users,
            }
        )
    return {
        "actors": actors,
        "channels": channels,
        "resources": resources,
        "queue_depth": queue_depth,
        "semantics": (
            "data channels enforce producer-consumer precedence; reverse slot "
            "channels enforce bounded FIFO backpressure; shared resources are "
            "represented by finite-capacity tokens"
        ),
    }


def expanded_timed_event_graph(
    constraints: Sequence[CycleConstraint],
) -> Dict[str, Any]:
    """Expand analytical resource bounds into a first-order max-plus graph.

    A constraint with service ``s`` and ``q`` tokens becomes a q-state ring.
    Its cycle mean is therefore s/q.  The block-diagonal union has spectral
    radius equal to the maximum of all resource-cycle constraints.  This is an
    algebraic encoding check, not an independently reconstructed hardware event
    graph; physical ordering and contention require Stage 5 validation.
    """

    nodes = []
    edges = []
    for constraint_index, constraint in enumerate(constraints):
        token_count = max(1, int(constraint.tokens))
        block = []
        for token_index in range(token_count):
            node_id = len(nodes)
            block.append(node_id)
            nodes.append(
                {
                    "id": node_id,
                    "name": "{}:delay{}".format(constraint.name, token_index),
                    "constraint": constraint.name,
                    "resource": constraint.resource,
                }
            )
        for token_index, source in enumerate(block):
            target = block[(token_index + 1) % token_count]
            weight = float(constraint.service_ms) if token_index == token_count - 1 else 0.0
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "weight_ms": weight,
                    "constraint_index": constraint_index,
                }
            )
    return {"nodes": nodes, "edges": edges}


def karp_maximum_cycle_mean(node_count: int, edges: Sequence[Mapping[str, Any]]) -> float:
    """Compute the max-plus spectral radius of a finite weighted graph."""

    if node_count <= 0:
        return 0.0
    incoming: List[List[Tuple[int, float]]] = [[] for _ in range(node_count)]
    for edge in edges:
        incoming[int(edge["target"])].append(
            (int(edge["source"]), as_float(edge.get("weight_ms")))
        )
    negative_infinity = float("-inf")
    dynamic = np.full((node_count + 1, node_count), negative_infinity, dtype="float64")
    dynamic[0, :] = 0.0
    for length in range(1, node_count + 1):
        for target in range(node_count):
            if incoming[target]:
                dynamic[length, target] = max(
                    dynamic[length - 1, source] + weight
                    for source, weight in incoming[target]
                )
    candidates = []
    for node in range(node_count):
        if not math.isfinite(float(dynamic[node_count, node])):
            continue
        slopes = []
        for length in range(node_count):
            if math.isfinite(float(dynamic[length, node])):
                slopes.append(
                    (dynamic[node_count, node] - dynamic[length, node])
                    / float(node_count - length)
                )
        if slopes:
            candidates.append(min(slopes))
    return float(max(candidates)) if candidates else 0.0


def timed_event_graph_payload(
    record: PipelineRecord,
    parameters: Optional[Mapping[str, float]] = None,
    require_measured_core_demand: bool = False,
) -> Dict[str, Any]:
    constraints = build_cycle_constraints(
        record,
        parameters,
        require_measured_core_demand=require_measured_core_demand,
    )
    graph = expanded_timed_event_graph(constraints)
    analytical, bottleneck = maximum_cycle_mean(constraints)
    spectral = karp_maximum_cycle_mean(len(graph["nodes"]), graph["edges"])
    critical = critical_cycle_constraints(constraints)
    return {
        "candidate_id": record.candidate_id,
        "model": record.model,
        "maxplus_equation": "x(k+1) = A(theta) tensor x(k)",
        "solver_scope": "analytical_resource_cycle_lower_bounds",
        "karp_validation_kind": "algebraic_encoding_consistency_only",
        "physical_event_graph_validation": "pending_stage5_controlled_pipeline",
        "marked_event_graph": marked_event_graph_payload(
            record,
            parameters,
            require_measured_core_demand=require_measured_core_demand,
        ),
        "resource_demand_matrix": resource_demand_matrix(
            record,
            parameters,
            require_measured_core_demand=require_measured_core_demand,
        ),
        "constraints": [asdict(item) | {"cycle_ms": item.cycle_ms} for item in constraints],
        "state_graph": graph,
        "analytical_cycle_ms": analytical,
        "karp_spectral_radius_ms": spectral,
        "spectral_validation_abs_error_ms": abs(analytical - spectral),
        "bottleneck_cycle": asdict(bottleneck) if bottleneck is not None else None,
        "critical_cycles": [asdict(item) | {"cycle_ms": item.cycle_ms} for item in critical],
    }


def residual_feature_values(record: PipelineRecord) -> Dict[str, float]:
    cpu_threads = sum(max(1, stage.threads) for stage in record.stages if stage.device == "cpu")
    transaction_kib = max(record.avg_bytes_per_call / 1024.0, 1.0 / 1024.0)
    return {
        "fragmentation_ms": max(0.0, record.fragmentation_ms),
        "spill_ratio": max(0.0, record.spill_ratio),
        "cpu_oversubscription": float(max(0, cpu_threads - 4)),
        "inverse_transaction_kib": 1.0 / transaction_kib,
        "extra_segment_count": float(max(0, record.effective_segment_count - 3)),
        "small_island_count": float(max(0, record.small_island_count)),
    }


def monotonic_residual(record: PipelineRecord, residual_model: Mapping[str, Any]) -> float:
    values = residual_feature_values(record)
    result = as_float(residual_model.get("intercept_ms"), 0.0)
    for feature in residual_model.get("features", []) or []:
        value = values.get(str(feature.get("name")), 0.0)
        for knot, coefficient in zip(
            feature.get("knots", []) or [], feature.get("coefficients", []) or []
        ):
            result += max(0.0, value - as_float(knot)) * max(0.0, as_float(coefficient))
    return result


def monotonic_scale(record: PipelineRecord, scale_model: Mapping[str, Any]) -> float:
    return max(EPSILON, monotonic_residual(record, scale_model))


def predict_record(
    record: PipelineRecord,
    model: Optional[Mapping[str, Any]] = None,
    include_residual: Optional[bool] = None,
) -> Dict[str, Any]:
    payload = model or {}
    options = payload.get("prediction_options", {}) or {}
    require_core_demand = bool(options.get("require_measured_core_demand", False))
    if include_residual is None:
        include_residual = bool(options.get("include_residual", False))
    base_parameters = dict(default_base_parameters())
    base_parameters.update(payload.get("base_parameters", {}) or {})
    constraints = build_cycle_constraints(
        record,
        base_parameters,
        require_measured_core_demand=require_core_demand,
    )
    maxplus_ms, bottleneck = maximum_cycle_mean(constraints)
    critical = critical_cycle_constraints(constraints)
    available_residual_ms = monotonic_residual(
        record, payload.get("residual_model", {}) or {}
    )
    residual_ms = available_residual_ms if include_residual else 0.0
    mean_ms = max(EPSILON, maxplus_ms + residual_ms)
    uncertainty = payload.get("uncertainty", {}) or {}
    global_std = max(0.0, as_float(uncertainty.get("residual_std_ms"), 0.0))
    scale_model = uncertainty.get("scale_model", {}) or {}
    std_ms = monotonic_scale(record, scale_model) if scale_model else global_std
    std_ms = max(as_float(uncertainty.get("minimum_scale_ms"), 0.0), std_ms)
    if "normalized_residual_q90" in uncertainty:
        q90 = std_ms * max(0.0, as_float(uncertainty.get("normalized_residual_q90")))
        q95 = std_ms * max(0.0, as_float(uncertainty.get("normalized_residual_q95")))
    else:
        q90 = max(0.0, as_float(uncertainty.get("abs_residual_q90_ms"), 0.0))
        q95 = max(0.0, as_float(uncertainty.get("abs_residual_q95_ms"), 0.0))
    return {
        "prediction_mode": "physical_plus_residual" if include_residual else "physical",
        "maxplus_cycle_ms": maxplus_ms,
        "residual_ms": residual_ms,
        "available_residual_ms": available_residual_ms,
        "predicted_cycle_ms": mean_ms,
        "predicted_fps": 1000.0 / mean_ms,
        "predicted_std_ms": std_ms,
        "risk_score_ms": mean_ms + 1.645 * std_ms,
        "prediction_interval_90_ms": [max(0.0, mean_ms - q90), mean_ms + q90],
        "prediction_interval_95_ms": [max(0.0, mean_ms - q95), mean_ms + q95],
        "bottleneck_cycle": asdict(bottleneck) if bottleneck is not None else None,
        "critical_cycles": [asdict(item) | {"cycle_ms": item.cycle_ms} for item in critical],
        "cycle_constraints": [asdict(item) | {"cycle_ms": item.cycle_ms} for item in constraints],
        "core_demand_sources": {
            stage.name: cpu_core_demand_ms(
                stage,
                scaled_stage_service(stage, base_parameters),
                base_parameters,
                require_measured=require_core_demand,
            )[1]
            for stage in record.stages
            if stage.device == "cpu"
        },
    }


def _targets(records: Sequence[PipelineRecord]) -> np.ndarray:
    return np.asarray([as_float(item.measured_cycle_ms) for item in records], dtype="float64")


def fit_base_parameters(records: Sequence[PipelineRecord]) -> Dict[str, float]:
    from scipy.optimize import least_squares

    if not records:
        return default_base_parameters()
    target = _targets(records)
    scale = max(1.0, float(np.median(target)) * 0.05)
    stage_observations = [
        (record, stage, float(stage.measured_ms))
        for record in records
        for stage in record.stages
        if stage.measured_ms is not None and float(stage.measured_ms) > 0.0
    ]
    stage_scale = max(
        1.0,
        float(np.median([value for _, _, value in stage_observations])) * 0.05
        if stage_observations
        else scale,
    )

    def residual(vector: np.ndarray) -> np.ndarray:
        params = dict(zip(BASE_PARAMETER_NAMES, vector.tolist()))
        predicted = np.asarray(
            [maximum_cycle_mean(build_cycle_constraints(item, params))[0] for item in records]
        )
        data_residual = (predicted - target) / scale
        stage_residual = np.asarray(
            [
                (scaled_stage_service(stage, params) - measured) / stage_scale
                for _, stage, measured in stage_observations
            ],
            dtype="float64",
        )
        # Calibration supplies a physical prior of one.  Weak shrinkage prevents
        # collinear controlled configurations from producing nonphysical scales.
        prior_residual = math.sqrt(0.02) * (vector - 1.0)
        return np.concatenate((data_residual, 0.5 * stage_residual, prior_residual))

    lower = np.asarray([BASE_PARAMETER_BOUNDS[name][0] for name in BASE_PARAMETER_NAMES])
    upper = np.asarray([BASE_PARAMETER_BOUNDS[name][1] for name in BASE_PARAMETER_NAMES])

    result = least_squares(
        residual,
        np.ones(len(BASE_PARAMETER_NAMES), dtype="float64"),
        bounds=(lower, upper),
        loss="huber",
        f_scale=1.0,
        max_nfev=2000,
    )
    return dict(zip(BASE_PARAMETER_NAMES, [float(item) for item in result.x]))


def _residual_basis(
    records: Sequence[PipelineRecord], feature_specs: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    rows = []
    for record in records:
        values = residual_feature_values(record)
        row = [1.0]
        for spec in feature_specs:
            value = values.get(str(spec["name"]), 0.0)
            row.extend(max(0.0, value - float(knot)) for knot in spec["knots"])
        rows.append(row)
    return np.asarray(rows, dtype="float64")


def _feature_specs(records: Sequence[PipelineRecord]) -> List[Dict[str, Any]]:
    specs = []
    for name in RESIDUAL_FEATURE_NAMES:
        values = np.asarray([residual_feature_values(item)[name] for item in records], dtype="float64")
        if values.size == 0 or float(values.max() - values.min()) <= EPSILON:
            continue
        knots = sorted(set(float(item) for item in np.quantile(values, [0.25, 0.5, 0.75])))
        specs.append({"name": name, "knots": knots})
    return specs


def fit_monotonic_residual(
    records: Sequence[PipelineRecord],
    base_parameters: Mapping[str, float],
    rank_weight: float = 0.20,
    seed: int = 20260713,
) -> Dict[str, Any]:
    from scipy.optimize import minimize

    specs = _feature_specs(records)
    matrix = _residual_basis(records, specs)
    target = _targets(records)
    base = np.asarray(
        [maximum_cycle_mean(build_cycle_constraints(item, base_parameters))[0] for item in records]
    )
    target_residual = target - base
    scale = max(1.0, float(np.median(target)) * 0.05)
    rng = random.Random(seed)
    pairs = [(i, j) for i in range(len(records)) for j in range(i + 1, len(records))]
    if len(pairs) > 20000:
        pairs = rng.sample(pairs, 20000)

    initial = np.zeros(matrix.shape[1], dtype="float64")
    initial[0] = float(np.median(target_residual))
    bounds = [(None, None)] + [(0.0, None)] * (matrix.shape[1] - 1)

    def objective(coefficients: np.ndarray) -> float:
        prediction = base + matrix.dot(coefficients)
        error = (prediction - target) / scale
        abs_error = np.abs(error)
        huber = np.where(abs_error <= 1.0, 0.5 * error * error, abs_error - 0.5).mean()
        rank_loss = 0.0
        if pairs and rank_weight > 0.0:
            values = []
            tau = max(1.0, float(np.std(target)) * 0.10)
            for left, right in pairs:
                delta = target[right] - target[left]
                if abs(delta) < 1.0:
                    continue
                sign = 1.0 if delta > 0.0 else -1.0
                margin = sign * (prediction[right] - prediction[left]) / tau
                values.append(float(np.logaddexp(0.0, -margin)))
            rank_loss = statistics.mean(values) if values else 0.0
        regularization = 1.0e-5 * float(np.dot(coefficients[1:], coefficients[1:]))
        return float(huber + rank_weight * rank_loss + regularization)

    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
    coefficients = result.x
    offset = 1
    features = []
    for spec in specs:
        count = len(spec["knots"])
        features.append(
            {
                "name": spec["name"],
                "knots": list(spec["knots"]),
                "coefficients": [float(item) for item in coefficients[offset : offset + count]],
            }
        )
        offset += count
    return {
        "kind": "monotonic_hinge_gam",
        "intercept_ms": float(coefficients[0]),
        "features": features,
        "rank_weight": float(rank_weight),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def fit_monotonic_scale(
    records: Sequence[PipelineRecord], absolute_errors: Sequence[float]
) -> Dict[str, Any]:
    from scipy.optimize import lsq_linear

    if len(records) != len(absolute_errors) or not records:
        raise ValueError("uncertainty scale fit requires equal non-empty inputs")
    specs = _feature_specs(records)
    matrix = _residual_basis(records, specs)
    target = np.maximum(EPSILON, np.asarray(absolute_errors, dtype="float64"))
    result = lsq_linear(matrix, target, bounds=(0.0, np.inf), lsmr_tol="auto")
    coefficients = result.x
    offset = 1
    features = []
    for spec in specs:
        count = len(spec["knots"])
        features.append(
            {
                "name": spec["name"],
                "knots": list(spec["knots"]),
                "coefficients": [
                    float(item) for item in coefficients[offset : offset + count]
                ],
            }
        )
        offset += count
    return {
        "kind": "monotonic_nonnegative_absolute_error_gam",
        "intercept_ms": float(coefficients[0]),
        "features": features,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def fit_ramps_model(
    records: Sequence[PipelineRecord],
    rank_weight: float = 0.20,
    seed: int = 20260713,
    identification_records: Optional[Sequence[PipelineRecord]] = None,
    conformal_fraction: float = 0.20,
) -> Dict[str, Any]:
    usable = [item for item in records if item.measured_cycle_ms and item.correctness_passed]
    if len(usable) < 10:
        raise RuntimeError("RAMPS fit requires at least 10 measured records")
    identified = [
        item
        for item in (identification_records or [])
        if item.measured_cycle_ms and item.correctness_passed
    ]
    base_parameters = fit_base_parameters(identified or usable)

    groups = sorted(set(item.group_id for item in usable))
    ordered_groups = sorted(
        groups,
        key=lambda name: hashlib.sha256("{}:{}".format(seed, name).encode()).hexdigest(),
    )
    calibration_group_count = max(1, int(round(len(ordered_groups) * conformal_fraction)))
    calibration_groups = set(ordered_groups[:calibration_group_count])
    residual_fit = [item for item in usable if item.group_id not in calibration_groups]
    conformal = [item for item in usable if item.group_id in calibration_groups]
    if len(residual_fit) < 10 or len(conformal) < 5:
        residual_fit = usable
        conformal = usable
        calibration_groups = set(groups)
    residual_model = fit_monotonic_residual(
        residual_fit, base_parameters, rank_weight, seed
    )
    partial = {
        "version": MODEL_VERSION,
        "kind": "resource_aware_maxplus_monotonic_residual",
        "base_parameters": base_parameters,
        "residual_model": residual_model,
        "prediction_options": {
            "include_residual": False,
            "require_measured_core_demand": False,
        },
    }
    fit_errors = np.asarray(
        [
            predict_record(item, partial, include_residual=True)["predicted_cycle_ms"]
            - item.measured_cycle_ms
            for item in residual_fit
        ]
    )
    scale_model = fit_monotonic_scale(residual_fit, np.abs(fit_errors))
    minimum_scale_ms = max(1.0e-3, float(np.quantile(np.abs(fit_errors), 0.10)))
    errors = np.asarray(
        [
            predict_record(item, partial, include_residual=True)["predicted_cycle_ms"]
            - item.measured_cycle_ms
            for item in conformal
        ]
    )
    conformal_scale = np.asarray(
        [max(minimum_scale_ms, monotonic_scale(item, scale_model)) for item in conformal]
    )

    def conformal_quantile(values: np.ndarray, alpha: float) -> float:
        level = min(1.0, math.ceil((len(values) + 1) * (1.0 - alpha)) / len(values))
        return float(np.quantile(values, level, method="higher"))

    normalized_errors = np.abs(errors) / conformal_scale

    partial["uncertainty"] = {
        "method": "monotonic_heteroscedastic_grouped_split_conformal",
        "residual_std_ms": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
        "minimum_scale_ms": minimum_scale_ms,
        "scale_model": scale_model,
        "normalized_residual_q90": conformal_quantile(normalized_errors, 0.10),
        "normalized_residual_q95": conformal_quantile(normalized_errors, 0.05),
        "calibration_sample_count": len(conformal),
        "calibration_group_count": len(calibration_groups),
        "calibration_groups": sorted(calibration_groups),
    }
    partial["training"] = {
        "sample_count": len(usable),
        "identification_sample_count": len(identified),
        "residual_fit_sample_count": len(residual_fit),
        "conformal_sample_count": len(conformal),
        "models": sorted(set(item.model for item in usable)),
        "seed": int(seed),
        "rank_weight": float(rank_weight),
        "feature_policy": "physical_only_no_model_or_candidate_identity",
    }
    return partial


def ranking_metrics(records: Sequence[PipelineRecord], predictions: Sequence[float]) -> Dict[str, float]:
    from scipy.stats import kendalltau, spearmanr

    target = _targets(records)
    predicted = np.asarray(predictions, dtype="float64")
    if len(target) != len(predicted) or not len(target):
        raise ValueError("metrics require non-empty equal-length target and prediction")
    actual_order = np.argsort(target)
    predicted_order = np.argsort(predicted)
    pair_correct = 0
    pair_total = 0
    for left in range(len(target)):
        for right in range(left + 1, len(target)):
            if abs(target[left] - target[right]) <= EPSILON:
                continue
            pair_total += 1
            pair_correct += int(
                (target[left] < target[right]) == (predicted[left] < predicted[right])
            )

    def recall(actual_k: int, predicted_k: int) -> float:
        actual = set(actual_order[: min(actual_k, len(target))].tolist())
        predicted_set = set(predicted_order[: min(predicted_k, len(target))].tolist())
        return len(actual & predicted_set) / max(1, len(actual))

    best = float(target.min())
    result: Dict[str, float] = {
        "sample_count": float(len(target)),
        "mae_ms": float(np.mean(np.abs(predicted - target))),
        "rmse_ms": float(math.sqrt(np.mean((predicted - target) ** 2))),
        "mape": float(np.mean(np.abs(predicted - target) / np.maximum(target, EPSILON))),
        "spearman": float(spearmanr(target, predicted).statistic),
        "kendall": float(kendalltau(target, predicted).statistic),
        "pairwise_accuracy": pair_correct / max(1, pair_total),
        "top10_recall_at_20": recall(10, 20),
        "top20_recall_at_50": recall(20, 50),
    }
    for budget in (1, 3, 5, 10, 20, 50, 100):
        selected = target[predicted_order[: min(budget, len(target))]]
        result["regret_at_{}".format(budget)] = float((selected.min() - best) / best)
    for fraction in (0.95, 0.98):
        threshold = best / fraction
        evaluations = len(target)
        for index, candidate_index in enumerate(predicted_order, 1):
            if target[candidate_index] <= threshold:
                evaluations = index
                break
        result["evaluations_to_oracle_{}pct".format(int(100 * fraction))] = float(
            evaluations
        )
    return result


def grouped_bootstrap_metrics(
    records: Sequence[PipelineRecord],
    predictions: Sequence[float],
    repetitions: int = 500,
    seed: int = 20260713,
) -> Dict[str, Any]:
    groups: Dict[str, List[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.group_id, []).append(index)
    names = sorted(groups)
    rng = random.Random(seed)
    samples: Dict[str, List[float]] = {}
    for _ in range(max(1, int(repetitions))):
        selected: List[int] = []
        for name in (rng.choice(names) for _ in names):
            selected.extend(groups[name])
        subset_records = [records[index] for index in selected]
        subset_predictions = [predictions[index] for index in selected]
        metrics = ranking_metrics(subset_records, subset_predictions)
        for key, value in metrics.items():
            if math.isfinite(float(value)):
                samples.setdefault(key, []).append(float(value))
    intervals = {}
    for key, values in samples.items():
        intervals[key] = {
            "mean": float(np.mean(values)),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        }
    return {"repetitions": int(repetitions), "group_count": len(names), "metrics": intervals}


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2, sort_keys=True)
        out.write("\n")
