#!/usr/bin/env python3
"""Model-independent static workload schema for RAMPS.

This module is the DNN-facing layer of RAMPS.  A frontend lowers a partition
with the frozen compiler configuration and emits only physical work and tensor
properties.  Model names and candidate measurements are identity/evaluation
metadata; they never become service-model features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SCHEMA_VERSION = 3
SUPPORTED_DEVICES = {"cpu", "vta"}
SUPPORTED_PRIMITIVES = {
    "conv2d",
    "dense",
    "elementwise",
    "pool_reduction",
    "concat_copy",
    "resize",
    "layout_transform",
    "other_cpu",
}
SUPPORTED_ACCESS_KINDS = {
    "large_contiguous",
    "small_contiguous",
    "strided",
    "padded",
}
SUPPORTED_TRANSFER_RESOURCES = {"cpu_memory", "stage_dma", "zero_copy"}
SUPPORTED_ADAPTER_KINDS = {"identity", "pack", "unpack", "pack_unpack", "requantize"}
SRAM_BANKS = ("inp", "wgt", "acc", "out")
ACCOUNTING_RESOURCES = (
    "cpu_memory",
    "ps_pl_load",
    "ps_pl_store",
    "bridge_pack_unpack",
)
SUPPORTED_DMA_TRAFFIC_SCOPES = {"total_physical_including_spill"}
FORBIDDEN_FEATURE_KEYS = {
    "model",
    "model_name",
    "layer_name",
    "candidate_id",
    "measured_ms",
    "measured_cycle_ms",
    "pipeline_throughput_fps",
    "throughput_fps",
}
DTYPE_BYTES = {
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "uint32": 4,
    "float32": 4,
    "int64": 8,
    "uint64": 8,
    "float64": 8,
}


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError("{} must be finite and non-negative".format(name))


def _positive_shape(name: str, shape: Sequence[int]) -> None:
    if not shape or any(int(value) <= 0 for value in shape):
        raise ValueError("{} must contain positive dimensions".format(name))


def _sha256(name: str, value: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", str(value)):
        raise ValueError("{} must be a lowercase SHA256".format(name))


def _tensor_bytes(shape: Sequence[int], dtype: str, name: str) -> int:
    if dtype not in DTYPE_BYTES:
        raise ValueError("{} uses unsupported fixed-width dtype {}".format(name, dtype))
    elements = 1
    for dimension in shape:
        elements *= int(dimension)
    return elements * DTYPE_BYTES[dtype]


def _reject_forbidden_features(value: Any, path: str = "features") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in FORBIDDEN_FEATURE_KEYS:
                raise ValueError("{} contains forbidden prediction key {}".format(path, key))
            _reject_forbidden_features(item, "{}.{}".format(path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_features(item, "{}[{}]".format(path, index))


@dataclass(frozen=True)
class TensorSignature:
    logical_shape: List[int]
    physical_shape: List[int]
    producer_dtype: str
    producer_layout: str
    transport_dtype: str
    transport_layout: str
    consumer_dtype: str
    consumer_layout: str
    logical_bytes: int
    physical_bytes: int
    quantization: Dict[str, Any] = field(default_factory=dict)
    channel_padding: Dict[str, Any] = field(default_factory=dict)
    slice_policy: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _positive_shape("logical_shape", self.logical_shape)
        _positive_shape("physical_shape", self.physical_shape)
        if not all(
            (
                self.producer_dtype,
                self.producer_layout,
                self.transport_dtype,
                self.transport_layout,
                self.consumer_dtype,
                self.consumer_layout,
            )
        ):
            raise ValueError("producer, transport and consumer dtype/layout are required")
        if int(self.logical_bytes) <= 0 or int(self.physical_bytes) <= 0:
            raise ValueError("tensor byte counts must be positive")
        expected_logical = _tensor_bytes(
            self.logical_shape, self.producer_dtype, "logical tensor"
        )
        expected_physical = _tensor_bytes(
            self.physical_shape, self.transport_dtype, "physical tensor"
        )
        if int(self.logical_bytes) != expected_logical:
            raise ValueError("logical_bytes is inconsistent with logical shape/dtype")
        if int(self.physical_bytes) != expected_physical:
            raise ValueError("physical_bytes is inconsistent with physical shape/dtype")
        _reject_forbidden_features(self.quantization, "tensor.quantization")
        _reject_forbidden_features(self.channel_padding, "tensor.channel_padding")
        _reject_forbidden_features(self.slice_policy, "tensor.slice_policy")
        if (
            self.producer_dtype != self.transport_dtype
            or self.transport_dtype != self.consumer_dtype
        ) and not self.quantization:
            raise ValueError("dtype-changing boundary requires explicit quantization metadata")
        padded_channels = int(self.channel_padding.get("physical_channels", 0) or 0)
        logical_channels = int(self.channel_padding.get("logical_channels", 0) or 0)
        if padded_channels and padded_channels < logical_channels:
            raise ValueError("physical channel padding cannot reduce channel count")
        if padded_channels > logical_channels and not self.slice_policy:
            raise ValueError("padded channels require an explicit consumer slice policy")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TensorSignature":
        item = cls(**dict(payload))
        item.validate()
        return item


@dataclass(frozen=True)
class OperatorWorkload:
    op_id: str
    primitive: str
    logical_ops: float
    physical_ops: float
    input_bytes: int
    weight_bytes: int
    output_bytes: int
    memory_bytes: int
    load_bytes: int = 0
    store_bytes: int = 0
    load_calls: int = 0
    store_calls: int = 0
    avg_bytes_per_call: float = 0.0
    access_kind: str = "large_contiguous"
    tile_count: int = 1
    spill_bytes: int = 0
    spill_calls: int = 0
    dma_traffic_scope: str = "total_physical_including_spill"
    schedule_id: str = ""
    fusion_group_id: str = ""
    unit_kind: str = "backend_fused_unit"
    sram_peak_bytes: Dict[str, int] = field(default_factory=dict)
    resource_accounting_ids: Dict[str, List[str]] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.op_id:
            raise ValueError("op_id is required for graph reconstruction")
        if self.primitive not in SUPPORTED_PRIMITIVES:
            raise ValueError("unsupported primitive {}".format(self.primitive))
        if self.access_kind not in SUPPORTED_ACCESS_KINDS:
            raise ValueError("unsupported access kind {}".format(self.access_kind))
        if self.dma_traffic_scope not in SUPPORTED_DMA_TRAFFIC_SCOPES:
            raise ValueError(
                "unsupported DMA traffic scope {}; fixed-schedule RAMPS requires total "
                "physical DMA including spill".format(self.dma_traffic_scope)
            )
        for name in (
            "logical_ops",
            "physical_ops",
            "input_bytes",
            "weight_bytes",
            "output_bytes",
            "memory_bytes",
            "load_bytes",
            "store_bytes",
            "load_calls",
            "store_calls",
            "avg_bytes_per_call",
            "spill_bytes",
            "spill_calls",
        ):
            _finite_nonnegative(name, getattr(self, name))
        if self.physical_ops + 1.0e-9 < self.logical_ops:
            raise ValueError("physical_ops cannot be lower than logical_ops")
        if int(self.tile_count) <= 0:
            raise ValueError("tile_count must be positive")
        if not self.schedule_id:
            raise ValueError("fixed schedule_id is required")
        if not self.fusion_group_id:
            raise ValueError("fusion_group_id is required")
        if self.unit_kind != "backend_fused_unit":
            raise ValueError("RAMPS models backend_fused_unit workloads, not raw Relay ops")
        unknown_banks = set(self.sram_peak_bytes) - set(SRAM_BANKS)
        if unknown_banks:
            raise ValueError("unknown SRAM banks: {}".format(sorted(unknown_banks)))
        for bank, byte_count in self.sram_peak_bytes.items():
            _finite_nonnegative("sram_peak_bytes.{}".format(bank), byte_count)
        unknown_resources = set(self.resource_accounting_ids) - set(ACCOUNTING_RESOURCES)
        if unknown_resources:
            raise ValueError(
                "unknown accounting resources: {}".format(sorted(unknown_resources))
            )
        for resource, identifiers in self.resource_accounting_ids.items():
            if not identifiers or any(not str(item) for item in identifiers):
                raise ValueError("{} accounting ids must be non-empty".format(resource))
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("{} accounting ids must be unique".format(resource))
        _reject_forbidden_features(self.attributes, "operator.attributes")

    def prediction_features(self) -> Dict[str, Any]:
        """Return the only fields a hardware service model may consume."""

        self.validate()
        return {
            "primitive": self.primitive,
            "logical_ops": float(self.logical_ops),
            "physical_ops": float(self.physical_ops),
            "input_bytes": int(self.input_bytes),
            "weight_bytes": int(self.weight_bytes),
            "output_bytes": int(self.output_bytes),
            "memory_bytes": int(self.memory_bytes),
            "load_bytes": int(self.load_bytes),
            "store_bytes": int(self.store_bytes),
            "load_calls": int(self.load_calls),
            "store_calls": int(self.store_calls),
            "avg_bytes_per_call": float(self.avg_bytes_per_call),
            "access_kind": self.access_kind,
            "tile_count": int(self.tile_count),
            "spill_bytes": int(self.spill_bytes),
            "spill_calls": int(self.spill_calls),
            "dma_traffic_scope": self.dma_traffic_scope,
            "schedule_id": self.schedule_id,
            "unit_kind": self.unit_kind,
            "sram_peak_bytes": dict(self.sram_peak_bytes),
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperatorWorkload":
        item = cls(**dict(payload))
        item.validate()
        return item


@dataclass(frozen=True)
class BoundaryWorkload:
    boundary_id: str
    producer_stage: str
    consumer_stage: str
    tensor: TensorSignature
    adapter_kind: str
    transfer_bytes: int
    call_count: int
    avg_bytes_per_call: float
    access_kind: str = "large_contiguous"
    transfer_resource: str = "cpu_memory"
    queue_depth: int = 1
    resource_accounting_ids: Dict[str, List[str]] = field(default_factory=dict)
    pack_elements: int = 0
    unpack_elements: int = 0
    adapter_threads: int = 1

    def validate(self) -> None:
        self.tensor.validate()
        if not self.boundary_id or not self.producer_stage or not self.consumer_stage:
            raise ValueError("boundary identity and endpoints are required")
        if self.adapter_kind not in SUPPORTED_ADAPTER_KINDS:
            raise ValueError("unsupported adapter_kind {}".format(self.adapter_kind))
        if self.access_kind not in SUPPORTED_ACCESS_KINDS:
            raise ValueError("unsupported boundary access kind {}".format(self.access_kind))
        if self.transfer_resource not in SUPPORTED_TRANSFER_RESOURCES:
            raise ValueError("unsupported transfer resource {}".format(self.transfer_resource))
        if int(self.queue_depth) <= 0:
            raise ValueError("boundary queue_depth must be positive")
        if int(self.adapter_threads) <= 0:
            raise ValueError("boundary adapter_threads must be positive")
        for name in (
            "transfer_bytes",
            "call_count",
            "avg_bytes_per_call",
            "pack_elements",
            "unpack_elements",
        ):
            _finite_nonnegative(name, getattr(self, name))
        if self.transfer_resource == "zero_copy":
            if int(self.transfer_bytes) != 0 or int(self.call_count) != 0:
                raise ValueError("zero-copy boundary cannot report transfer bytes or calls")
        elif self.transfer_resource == "stage_dma":
            if int(self.transfer_bytes) < int(self.tensor.physical_bytes):
                raise ValueError(
                    "stage-DMA boundary must retain the physical tensor byte count"
                )
            if int(self.call_count) != 0 or float(self.avg_bytes_per_call) != 0.0:
                raise ValueError(
                    "stage-DMA calls belong to producer/consumer lowered operators, "
                    "not the boundary"
                )
            forbidden = set(self.resource_accounting_ids).intersection(
                {"ps_pl_load", "ps_pl_store"}
            )
            if forbidden:
                raise ValueError(
                    "stage-DMA boundary cannot duplicate PS-PL accounting ids: {}".format(
                        sorted(forbidden)
                    )
                )
        elif int(self.transfer_bytes) <= 0:
            raise ValueError("materialized boundary must report transfer bytes")
        else:
            if int(self.call_count) <= 0 or float(self.avg_bytes_per_call) <= 0.0:
                raise ValueError("materialized boundary requires calls and average transaction size")
            if int(self.transfer_bytes) < int(self.tensor.physical_bytes):
                raise ValueError("boundary transfer bytes cannot omit the physical tensor")
            expected_average = float(self.transfer_bytes) / int(self.call_count)
            if not math.isclose(
                float(self.avg_bytes_per_call), expected_average, rel_tol=1.0e-6, abs_tol=1.0e-6
            ):
                raise ValueError("avg_bytes_per_call is inconsistent with bytes/calls")
        if self.adapter_kind == "identity":
            schemas = {
                (self.tensor.producer_dtype, self.tensor.producer_layout),
                (self.tensor.transport_dtype, self.tensor.transport_layout),
                (self.tensor.consumer_dtype, self.tensor.consumer_layout),
            }
            if len(schemas) != 1 or self.pack_elements or self.unpack_elements:
                raise ValueError("identity adapter cannot change dtype/layout or process elements")
        unknown_resources = set(self.resource_accounting_ids) - set(ACCOUNTING_RESOURCES)
        if unknown_resources:
            raise ValueError(
                "unknown boundary accounting resources: {}".format(
                    sorted(unknown_resources)
                )
            )
        for resource, identifiers in self.resource_accounting_ids.items():
            if not identifiers or any(not str(item) for item in identifiers):
                raise ValueError("{} boundary accounting ids must be non-empty".format(resource))
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("{} boundary accounting ids must be unique".format(resource))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BoundaryWorkload":
        data = dict(payload)
        if not isinstance(data.get("tensor"), TensorSignature):
            data["tensor"] = TensorSignature.from_dict(data["tensor"])
        item = cls(**data)
        item.validate()
        return item


@dataclass(frozen=True)
class StageWorkload:
    stage_id: str
    device: str
    threads: int
    operators: List[OperatorWorkload]

    def validate(self) -> None:
        if not self.stage_id:
            raise ValueError("stage_id is required")
        if self.device not in SUPPORTED_DEVICES:
            raise ValueError("unsupported device {}".format(self.device))
        if int(self.threads) <= 0:
            raise ValueError("threads must be positive")
        if self.device == "vta" and int(self.threads) != 1:
            raise ValueError("the current VTA runtime requires one host stage thread")
        if not self.operators:
            raise ValueError("a stage must contain at least one lowered operator")
        for operator in self.operators:
            operator.validate()
            required_accounting = []
            if self.device == "cpu" and operator.memory_bytes > 0:
                required_accounting.append("cpu_memory")
            if self.device == "vta" and operator.load_bytes > 0:
                required_accounting.append("ps_pl_load")
            if self.device == "vta" and operator.store_bytes > 0:
                required_accounting.append("ps_pl_store")
            missing = [
                name
                for name in required_accounting
                if not operator.resource_accounting_ids.get(name)
            ]
            if missing:
                raise ValueError(
                    "operator {} lacks accounting ids for {}".format(
                        operator.op_id, sorted(missing)
                    )
                )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageWorkload":
        data = dict(payload)
        data["operators"] = [
            item if isinstance(item, OperatorWorkload) else OperatorWorkload.from_dict(item)
            for item in data.get("operators", [])
        ]
        item = cls(**data)
        item.validate()
        return item


@dataclass(frozen=True)
class PartitionWorkload:
    graph_id: str
    candidate_id: str
    compiler_config_sha256: str
    hardware_fingerprint_sha256: str
    queue_depth: int
    stages: List[StageWorkload]
    boundaries: List[BoundaryWorkload]
    static_contract_validated: bool
    static_validation_sha256: str
    identity_metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.graph_id or not self.candidate_id:
            raise ValueError("graph_id and candidate_id are required as non-feature identity")
        _sha256("compiler_config_sha256", self.compiler_config_sha256)
        _sha256("hardware_fingerprint_sha256", self.hardware_fingerprint_sha256)
        if self.static_contract_validated is not True:
            raise ValueError("partition requires a hashed static boundary validation")
        _sha256("static_validation_sha256", self.static_validation_sha256)
        if int(self.queue_depth) <= 0:
            raise ValueError("queue_depth must be positive")
        if not self.stages:
            raise ValueError("partition must contain stages")
        for stage in self.stages:
            stage.validate()
        stage_ids = {stage.stage_id for stage in self.stages}
        if len(stage_ids) != len(self.stages):
            raise ValueError("stage ids must be unique")
        stage_order = {stage.stage_id: index for index, stage in enumerate(self.stages)}
        boundary_ids = set()
        accounting_ids = {resource: set() for resource in ACCOUNTING_RESOURCES}
        for stage in self.stages:
            for operator in stage.operators:
                for resource, identifiers in operator.resource_accounting_ids.items():
                    overlap = accounting_ids[resource].intersection(identifiers)
                    if overlap:
                        raise ValueError(
                            "duplicate {} accounting ids: {}".format(
                                resource, sorted(overlap)
                            )
                        )
                    accounting_ids[resource].update(identifiers)
        incoming = {stage.stage_id: 0 for stage in self.stages}
        for boundary in self.boundaries:
            boundary.validate()
            if boundary.boundary_id in boundary_ids:
                raise ValueError("boundary ids must be unique")
            boundary_ids.add(boundary.boundary_id)
            if boundary.producer_stage not in stage_ids or boundary.consumer_stage not in stage_ids:
                raise ValueError("boundary endpoint is not a stage in this partition")
            if stage_order[boundary.producer_stage] >= stage_order[boundary.consumer_stage]:
                raise ValueError("stage boundary graph must be acyclic and topologically ordered")
            producer_device = self.stages[stage_order[boundary.producer_stage]].device
            consumer_device = self.stages[stage_order[boundary.consumer_stage]].device
            direction = "{}_to_{}".format(producer_device, consumer_device)
            allowed_transfer_resources = {
                "cpu_to_cpu": {"cpu_memory", "zero_copy"},
                "cpu_to_vta": {"stage_dma", "zero_copy"},
                "vta_to_cpu": {"stage_dma", "zero_copy"},
                "vta_to_vta": {"stage_dma", "zero_copy"},
            }[direction]
            if boundary.transfer_resource not in allowed_transfer_resources:
                raise ValueError(
                    "boundary {} uses {} for {}; PS-PL ownership belongs to lowered "
                    "stage DMA".format(
                        boundary.boundary_id, boundary.transfer_resource, direction
                    )
                )
            required_boundary_resources = []
            if boundary.transfer_resource == "cpu_memory":
                required_boundary_resources = ["cpu_memory"]
            elif boundary.transfer_resource == "stage_dma":
                required_boundary_resources = []
            if boundary.adapter_kind != "identity":
                required_boundary_resources.append("bridge_pack_unpack")
            missing = [
                name
                for name in required_boundary_resources
                if not boundary.resource_accounting_ids.get(name)
            ]
            if missing:
                raise ValueError(
                    "boundary {} lacks accounting ids for {}".format(
                        boundary.boundary_id, sorted(missing)
                    )
                )
            for resource, identifiers in boundary.resource_accounting_ids.items():
                overlap = accounting_ids[resource].intersection(identifiers)
                if overlap:
                    raise ValueError(
                        "duplicate {} accounting ids: {}".format(
                            resource, sorted(overlap)
                        )
                    )
                accounting_ids[resource].update(identifiers)
            incoming[boundary.consumer_stage] += 1
        for stage in self.stages[1:]:
            if incoming[stage.stage_id] == 0:
                raise ValueError("non-source stage {} has no input boundary".format(stage.stage_id))

    def prediction_payload(self) -> Dict[str, Any]:
        """Strip identity/evaluation metadata before service prediction."""

        self.validate()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "compiler_config_sha256": self.compiler_config_sha256,
            "hardware_fingerprint_sha256": self.hardware_fingerprint_sha256,
            "queue_depth": int(self.queue_depth),
            "static_validation_sha256": self.static_validation_sha256,
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "device": stage.device,
                    "threads": int(stage.threads),
                    "operators": [operator.prediction_features() for operator in stage.operators],
                }
                for stage in self.stages
            ],
            "boundaries": [
                {
                    key: value
                    for key, value in asdict(boundary).items()
                    if key != "resource_accounting_ids"
                }
                for boundary in self.boundaries
            ],
        }
        _reject_forbidden_features(payload)
        return payload

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PartitionWorkload":
        data = dict(payload)
        data.pop("schema_version", None)
        data["stages"] = [
            item if isinstance(item, StageWorkload) else StageWorkload.from_dict(item)
            for item in data.get("stages", [])
        ]
        data["boundaries"] = [
            item if isinstance(item, BoundaryWorkload) else BoundaryWorkload.from_dict(item)
            for item in data.get("boundaries", [])
        ]
        item = cls(**data)
        item.validate()
        return item


def workload_primitives(workloads: Iterable[PartitionWorkload]) -> List[str]:
    return sorted(
        {
            operator.primitive
            for workload in workloads
            for stage in workload.stages
            for operator in stage.operators
        }
    )
