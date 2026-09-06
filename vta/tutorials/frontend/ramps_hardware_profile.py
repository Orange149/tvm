#!/usr/bin/env python3
"""Hardware-facing service layer for portable RAMPS predictions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ramps_workload_schema import (
    FORBIDDEN_FEATURE_KEYS,
    BoundaryWorkload,
    PartitionWorkload,
    StageWorkload,
)
from resource_aware_maxplus import BoundaryService, PipelineRecord, StageService


PROFILE_SCHEMA_VERSION = 3
REQUIRED_GATES = (
    "correctness_gate_passed",
    "determinism_gate_passed",
    "precision_gate_passed",
    "service_fit_validated",
)
REQUIRED_EVIDENCE_HASHES = (
    "protocol_sha256",
    "measurement_plan_sha256",
    "calibration_cost_ledger_sha256",
    "raw_data_sha256",
    "fit_report_sha256",
    "reference_correctness_report_sha256",
)
REQUIRED_RESOURCE_CAPACITIES = (
    "cpu_core_pool",
    "cpu_memory",
    "vta_mutex",
    "ps_pl_load",
    "ps_pl_store",
    "bridge_pack_unpack",
)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def with_profile_sha256(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(payload)
    result.pop("profile_sha256", None)
    result["profile_sha256"] = canonical_sha256(result)
    return result


def _number(value: Any, name: str, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as err:
        raise ValueError("{} must be numeric".format(name)) from err
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError("{} must be finite and {}".format(name, qualifier))
    return result


def _require_sha256(name: str, value: Any) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", str(value or "")):
        raise ValueError("{} must be a lowercase SHA256".format(name))


def _reject_model_specific_keys(value: Any, path: str = "hardware_profile") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in FORBIDDEN_FEATURE_KEYS:
                raise ValueError("{} contains model-specific key {}".format(path, key))
            _reject_model_specific_keys(item, "{}.{}".format(path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_model_specific_keys(item, "{}[{}]".format(path, index))


def _validate_surface_schema(
    surface: Mapping[str, Any],
    label: str,
    output_names: Sequence[str],
    require_holdout: bool,
) -> None:
    if surface.get("interpolation") != "normalized_inverse_distance_v1":
        raise ValueError("{} uses an unsupported interpolation method".format(label))
    feature_names = list(surface.get("feature_names") or [])
    points = list(surface.get("support_points") or [])
    if not feature_names or len(points) < 2:
        raise ValueError("{} requires features and at least two support points".format(label))
    for index, point in enumerate(points):
        features = point.get("features") or {}
        for name in feature_names:
            _number(features.get(name), "{}.support[{}].{}".format(label, index, name))
        for name in output_names:
            _number(point.get(name), "{}.support[{}].{}".format(label, index, name))
    if require_holdout:
        validation = surface.get("validation") or {}
        if validation.get("holdout_passed") is not True:
            raise ValueError("{} lacks a passing grouped holdout".format(label))
        _number(validation.get("holdout_mape"), "{}.holdout_mape".format(label))


def boundary_conversion_signature(boundary: BoundaryWorkload) -> str:
    """Return the calibrated adapter domain key for a physical boundary."""

    tensor = boundary.tensor
    quant_scheme = str((tensor.quantization or {}).get("scheme") or "none")
    padded = int(
        int((tensor.channel_padding or {}).get("physical_channels", 0) or 0)
        > int((tensor.channel_padding or {}).get("logical_channels", 0) or 0)
    )
    return (
        "{}|{}:{}>{}:{}>{}:{}|quant={}|padded={}|access={}".format(
            boundary.adapter_kind,
            tensor.producer_dtype,
            tensor.producer_layout,
            tensor.transport_dtype,
            tensor.transport_layout,
            tensor.consumer_dtype,
            tensor.consumer_layout,
            quant_scheme,
            padded,
            boundary.access_kind,
        )
    )


@dataclass(frozen=True)
class HardwareProfile:
    payload: Dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "HardwareProfile":
        with Path(path).open(encoding="utf-8") as inp:
            return cls.from_dict(json.load(inp))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HardwareProfile":
        profile = cls(dict(payload))
        profile.validate(require_publication=False)
        return profile

    @property
    def fingerprint_sha256(self) -> str:
        return str((self.payload.get("hardware_fingerprint") or {}).get("sha256") or "")

    def validate(self, require_publication: bool = True) -> None:
        if int(self.payload.get("schema_version", 0)) != PROFILE_SCHEMA_VERSION:
            raise ValueError("hardware profile schema_version must be {}".format(PROFILE_SCHEMA_VERSION))
        if self.payload.get("profile_kind") != "portable_hardware_service_profile":
            raise ValueError("profile_kind must be portable_hardware_service_profile")
        _require_sha256("hardware fingerprint", self.fingerprint_sha256)
        unhashed = dict(self.payload)
        actual_profile_sha256 = str(unhashed.pop("profile_sha256", ""))
        if not actual_profile_sha256 or actual_profile_sha256 != canonical_sha256(unhashed):
            raise ValueError("hardware profile SHA256 is missing or invalid")
        _require_sha256("compiler configuration", self.payload.get("compiler_config_sha256"))
        if not self.payload.get("allowed_schedule_ids"):
            raise ValueError("hardware profile must bind at least one fixed schedule")
        if self.payload.get("fallback_used") or self.payload.get("estimated_from_used"):
            raise ValueError("portable hardware profile cannot contain fallback estimates")
        _reject_model_specific_keys(self.payload)
        for section in ("cpu", "vta", "dma", "boundary"):
            if not isinstance(self.payload.get(section), Mapping):
                raise ValueError("hardware profile is missing {} section".format(section))
        accounting = self.payload.get("component_accounting_policy") or {}
        required_accounting = {
            "vta_dma_scope": "total_physical_including_spill",
            "boundary_ps_pl_owner": "lowered_stage_dma",
            "spill_time_policy": "informational_not_additive",
        }
        for name, expected in required_accounting.items():
            if accounting.get(name) != expected:
                raise ValueError(
                    "hardware profile accounting policy {} must be {}".format(
                        name, expected
                    )
                )
        capacities = self.payload.get("resource_capacities") or {}
        for name in REQUIRED_RESOURCE_CAPACITIES:
            value = int(capacities.get(name, 0) or 0)
            if value <= 0:
                raise ValueError("hardware profile lacks positive {} capacity".format(name))
        if require_publication:
            if self.payload.get("optimization_objective") != "top_k_candidate_triage":
                raise ValueError(
                    "publication profile must optimize top_k_candidate_triage"
                )
            budget = self.payload.get("calibration_budget") or {}
            support_count = int(budget.get("support_pool_case_count", 0) or 0)
            measured_count = int(budget.get("measured_case_count", 0) or 0)
            holdout_count = int(budget.get("holdout_case_count", 0) or 0)
            if support_count <= 0 or measured_count <= 0:
                raise ValueError("publication profile lacks calibration cost counts")
            if measured_count >= support_count or budget.get("full_pool_measured") is not False:
                raise ValueError(
                    "publication profile must use a bounded subset, not the full support pool"
                )
            if measured_count > 80 or holdout_count < 16 or holdout_count >= measured_count:
                raise ValueError(
                    "publication profile violates the frozen 80-case/16-holdout budget"
                )
            legacy_fields = (
                ("cpu.compute_gops", (self.payload["cpu"].get("compute_gops"))),
                (
                    "cpu.memory_bandwidth_GBps",
                    self.payload["cpu"].get("memory_bandwidth_GBps"),
                ),
                ("vta.compute_gops", self.payload["vta"].get("compute_gops")),
                (
                    "vta.non_overlap_fraction",
                    self.payload["vta"].get("non_overlap_fraction"),
                ),
            )
            present = [name for name, value in legacy_fields if value is not None]
            if present:
                raise ValueError(
                    "publication profile contains shape-insensitive legacy fields: {}".format(
                        ", ".join(present)
                    )
                )
            if not self.payload["cpu"].get("unit_models") or not self.payload["cpu"].get(
                "memory_models"
            ):
                raise ValueError("publication profile requires CPU unit and memory surfaces")
            if not self.payload["vta"].get("unit_models"):
                raise ValueError("publication profile requires VTA unit surfaces")
            for primitive, by_threads in self.payload["cpu"]["unit_models"].items():
                for threads, surface in by_threads.items():
                    _validate_surface_schema(
                        surface,
                        "CPU {} t{}".format(primitive, threads),
                        ("compute_gops", "compute_core_scale", "memory_core_scale"),
                        require_holdout=True,
                    )
            for threads, surface in self.payload["cpu"]["memory_models"].items():
                _validate_surface_schema(
                    surface,
                    "CPU memory t{}".format(threads),
                    ("bandwidth_GBps",),
                    require_holdout=True,
                )
            overlap_models = self.payload["vta"].get("system_overlap_models") or {}
            for primitive, surface in self.payload["vta"]["unit_models"].items():
                _validate_surface_schema(
                    surface,
                    "VTA {}".format(primitive),
                    ("compute_gops",),
                    require_holdout=True,
                )
                overlap_surface = overlap_models.get(primitive)
                if not isinstance(overlap_surface, Mapping) or overlap_surface.get(
                    "validation_source"
                ) != "stage5_controlled_pipeline":
                    raise ValueError(
                        "VTA {} lacks Stage 5 overlap validation".format(primitive)
                    )
                _validate_surface_schema(
                    overlap_surface,
                    "VTA overlap {}".format(primitive),
                    ("non_overlap_fraction",),
                    require_holdout=True,
                )
            gates = self.payload.get("validation_gates") or {}
            failed = [name for name in REQUIRED_GATES if gates.get(name) is not True]
            if failed:
                raise ValueError("hardware profile has not passed gates: {}".format(", ".join(failed)))
            evidence = self.payload.get("validation_evidence") or {}
            missing = [name for name in REQUIRED_EVIDENCE_HASHES if not evidence.get(name)]
            if missing:
                raise ValueError("hardware profile lacks validation evidence: {}".format(", ".join(missing)))
            for name in REQUIRED_EVIDENCE_HASHES:
                _require_sha256(name, evidence[name])

    def _validate_operator_domain(self, operator: Any, device: str) -> None:
        if operator.schedule_id not in set(self.payload.get("allowed_schedule_ids") or []):
            raise ValueError("operator schedule {} was not calibrated".format(operator.schedule_id))
        domain = self.payload.get("supported_workload_domain") or {}
        primitives = set(domain.get("primitives") or [])
        if primitives and operator.primitive not in primitives:
            raise ValueError("primitive {} is outside calibrated domain".format(operator.primitive))
        for key, limits in (domain.get("attribute_ranges") or {}).items():
            if key not in operator.attributes:
                continue
            value = _number(operator.attributes[key], "operator attribute {}".format(key))
            low, high = [float(item) for item in limits]
            if value < low or value > high:
                raise ValueError(
                    "operator attribute {}={} is outside calibrated [{}, {}]".format(
                        key, value, low, high
                    )
                )
        if device == "vta":
            capacities = self.payload["vta"].get("sram_capacity_bytes") or {}
            missing_banks = set(("inp", "wgt", "acc", "out")) - set(
                operator.sram_peak_bytes
            )
            if missing_banks:
                raise ValueError(
                    "VTA workload lacks fixed-schedule SRAM peaks for {}".format(
                        sorted(missing_banks)
                    )
                )
            for bank, peak in operator.sram_peak_bytes.items():
                if bank not in capacities:
                    raise ValueError("hardware profile lacks {} SRAM capacity".format(bank))
                if int(peak) > int(capacities[bank]):
                    raise ValueError("operator exceeds {} SRAM capacity".format(bank))

    @staticmethod
    def _operator_feature(operator: Any, name: str) -> float:
        direct = {
            "logical_ops": operator.logical_ops,
            "physical_ops": operator.physical_ops,
            "input_bytes": operator.input_bytes,
            "weight_bytes": operator.weight_bytes,
            "output_bytes": operator.output_bytes,
            "memory_bytes": operator.memory_bytes,
            "load_bytes": operator.load_bytes,
            "store_bytes": operator.store_bytes,
            "load_calls": operator.load_calls,
            "store_calls": operator.store_calls,
            "avg_bytes_per_call": operator.avg_bytes_per_call,
            "tile_count": operator.tile_count,
        }
        if name in direct:
            return _number(direct[name], "operator feature {}".format(name))
        if name not in operator.attributes:
            raise ValueError("operator lacks calibrated feature {}".format(name))
        return _number(operator.attributes[name], "operator feature {}".format(name))

    @staticmethod
    def _interpolate_surface(
        surface: Mapping[str, Any],
        feature_values: Mapping[str, float],
        output_names: Sequence[str],
        label: str,
    ) -> Dict[str, float]:
        """Interpolate measured support points without silent extrapolation.

        Normalized inverse-distance interpolation is deliberately simple and
        inspectable.  Formal profiles must report its grouped holdout error.
        """

        if surface.get("interpolation") != "normalized_inverse_distance_v1":
            raise ValueError("{} uses an unsupported interpolation method".format(label))
        feature_names = list(surface.get("feature_names") or [])
        points = list(surface.get("support_points") or [])
        if not feature_names or not points:
            raise ValueError("{} lacks feature names or support points".format(label))
        ranges: Dict[str, Tuple[float, float]] = {}
        for name in feature_names:
            values = [
                _number((point.get("features") or {}).get(name), "{}.{}".format(label, name))
                for point in points
            ]
            low, high = min(values), max(values)
            query = _number(feature_values.get(name), "query feature {}".format(name))
            if query < low or query > high:
                raise ValueError(
                    "{} feature {}={} is outside calibrated [{}, {}]".format(
                        label, name, query, low, high
                    )
                )
            ranges[name] = (low, high)
        distances: List[Tuple[float, Mapping[str, Any]]] = []
        for point in points:
            squared = 0.0
            features = point.get("features") or {}
            for name in feature_names:
                low, high = ranges[name]
                scale = max(high - low, 1.0)
                squared += ((float(feature_values[name]) - float(features[name])) / scale) ** 2
            distances.append((math.sqrt(squared), point))
        exact = [point for distance, point in distances if distance <= 1.0e-12]
        selected = exact or [point for _, point in sorted(distances, key=lambda item: item[0])[:4]]
        if exact:
            weights = [1.0] * len(selected)
        else:
            lookup = {id(point): distance for distance, point in distances}
            weights = [1.0 / max(lookup[id(point)] ** 2, 1.0e-12) for point in selected]
        total_weight = sum(weights)
        result = {}
        for output in output_names:
            result[output] = sum(
                weight * _number(point.get(output), "{}.{}".format(label, output))
                for point, weight in zip(selected, weights)
            ) / total_weight
        return result

    def _cpu_unit_parameters(self, operator: Any, threads: int) -> Dict[str, float]:
        primitive = (self.payload["cpu"].get("unit_models") or {}).get(operator.primitive)
        surface = (primitive or {}).get(str(int(threads))) if isinstance(primitive, Mapping) else None
        if not isinstance(surface, Mapping):
            raise ValueError(
                "hardware profile lacks CPU unit model {} at {} threads".format(
                    operator.primitive, threads
                )
            )
        if operator.schedule_id not in set(surface.get("schedule_ids") or []):
            raise ValueError("CPU unit surface does not cover schedule {}".format(operator.schedule_id))
        values = {
            name: self._operator_feature(operator, name)
            for name in surface.get("feature_names") or []
        }
        return self._interpolate_surface(
            surface,
            values,
            ("compute_gops", "compute_core_scale", "memory_core_scale"),
            "CPU {} t{}".format(operator.primitive, threads),
        )

    def _cpu_memory_bandwidth(self, threads: int, working_set_bytes: float) -> float:
        surface = (self.payload["cpu"].get("memory_models") or {}).get(str(int(threads)))
        if not isinstance(surface, Mapping):
            raise ValueError("hardware profile lacks CPU memory model at {} threads".format(threads))
        bandwidth = self._interpolate_surface(
            surface,
            {"working_set_bytes": float(working_set_bytes)},
            ("bandwidth_GBps",),
            "CPU memory t{}".format(threads),
        )["bandwidth_GBps"]
        if bandwidth <= 0.0:
            raise ValueError("CPU memory bandwidth must be positive")
        return bandwidth

    def _vta_unit_parameters(self, operator: Any) -> Dict[str, float]:
        surface = (self.payload["vta"].get("unit_models") or {}).get(operator.primitive)
        if not isinstance(surface, Mapping):
            raise ValueError("hardware profile lacks VTA unit model {}".format(operator.primitive))
        if operator.schedule_id not in set(surface.get("schedule_ids") or []):
            raise ValueError("VTA unit surface does not cover schedule {}".format(operator.schedule_id))
        values = {
            name: self._operator_feature(operator, name)
            for name in surface.get("feature_names") or []
        }
        result = self._interpolate_surface(
            surface,
            values,
            ("compute_gops",),
            "VTA {}".format(operator.primitive),
        )
        if result["compute_gops"] <= 0.0:
            raise ValueError("VTA compute rate must be positive")
        overlap_surface = (self.payload["vta"].get("system_overlap_models") or {}).get(
            operator.primitive
        )
        if not isinstance(overlap_surface, Mapping) or overlap_surface.get(
            "validation_source"
        ) != "stage5_controlled_pipeline":
            raise ValueError(
                "VTA overlap for {} requires Stage 5 controlled-pipeline validation".format(
                    operator.primitive
                )
            )
        overlap_values = {
            name: self._operator_feature(operator, name)
            for name in overlap_surface.get("feature_names") or []
        }
        result.update(
            self._interpolate_surface(
                overlap_surface,
                overlap_values,
                ("non_overlap_fraction",),
                "VTA overlap {}".format(operator.primitive),
            )
        )
        if result["non_overlap_fraction"] > 1.0:
            raise ValueError("VTA non-overlap fraction must be in [0, 1]")
        return result

    def _dma_parameters(self, access_kind: str) -> Tuple[float, float, float, float]:
        entry = (self.payload["dma"].get("access_models") or {}).get(access_kind)
        if not isinstance(entry, Mapping):
            raise ValueError("hardware profile lacks DMA access kind {}".format(access_kind))
        load = entry.get("load") or {}
        store = entry.get("store") or {}
        return (
            _number(load.get("bandwidth_GBps"), "DMA load bandwidth", positive=True),
            _number(store.get("bandwidth_GBps"), "DMA store bandwidth", positive=True),
            _number(load.get("call_latency_ms"), "DMA load call latency"),
            _number(store.get("call_latency_ms"), "DMA store call latency"),
        )

    def estimate_stage(self, stage: StageWorkload) -> StageService:
        stage.validate()
        for operator in stage.operators:
            self._validate_operator_domain(operator, stage.device)
        if stage.device == "cpu":
            launch_ms = _number(self.payload["cpu"].get("launch_ms", 0.0), "CPU launch")
            compute_ms = 0.0
            memory_ms = 0.0
            sequential_service_ms = 0.0
            unit_roofline_components_ms = []
            core_demand_ms = _number(
                self.payload["cpu"].get("launch_core_ms", 0.0), "CPU launch core time"
            )
            for operator in stage.operators:
                unit = self._cpu_unit_parameters(operator, stage.threads)
                if unit["compute_gops"] <= 0.0:
                    raise ValueError("CPU compute rate must be positive")
                memory_bandwidth = self._cpu_memory_bandwidth(
                    stage.threads, operator.memory_bytes
                )
                unit_compute_ms = (
                    operator.physical_ops
                    / unit["compute_gops"]
                    / 1.0e6
                )
                unit_memory_ms = operator.memory_bytes / memory_bandwidth / 1.0e6
                compute_ms += unit_compute_ms
                memory_ms += unit_memory_ms
                sequential_service_ms += max(unit_compute_ms, unit_memory_ms)
                unit_roofline_components_ms.append(
                    {
                        "op_id": operator.op_id,
                        "compute_ms": unit_compute_ms,
                        "memory_ms": unit_memory_ms,
                    }
                )
                core_demand_ms += unit_compute_ms * _number(
                    unit.get("compute_core_scale"), "compute core-demand scale"
                )
                core_demand_ms += unit_memory_ms * _number(
                    unit.get("memory_core_scale"), "memory core-demand scale"
                )
            return StageService(
                name=stage.stage_id,
                device="cpu",
                threads=stage.threads,
                compute_ms=compute_ms,
                memory_ms=memory_ms,
                launch_ms=launch_ms,
                core_demand_ms=core_demand_ms,
                core_demand_source="portable_hardware_profile",
                sequential_unit_service_ms=sequential_service_ms,
                unit_roofline_components_ms=unit_roofline_components_ms,
            )

        compute_ms = 0.0
        load_ms = 0.0
        store_ms = 0.0
        overlap_weighted_ms = 0.0
        overlap_weight = 0.0
        for operator in stage.operators:
            if operator.dma_traffic_scope != "total_physical_including_spill":
                raise ValueError("VTA operator DMA must include fixed-schedule spill traffic")
            unit = self._vta_unit_parameters(operator)
            unit_compute_ms = operator.physical_ops / unit["compute_gops"] / 1.0e6
            load_bw, store_bw, load_latency, store_latency = self._dma_parameters(
                operator.access_kind
            )
            unit_load_ms = operator.load_bytes / load_bw / 1.0e6
            unit_load_ms += operator.load_calls * load_latency
            unit_store_ms = operator.store_bytes / store_bw / 1.0e6
            unit_store_ms += operator.store_calls * store_latency
            compute_ms += unit_compute_ms
            load_ms += unit_load_ms
            store_ms += unit_store_ms
            unit_weight = unit_compute_ms + unit_load_ms + unit_store_ms
            overlap_weighted_ms += unit["non_overlap_fraction"] * unit_weight
            overlap_weight += unit_weight
        non_overlap = overlap_weighted_ms / overlap_weight if overlap_weight else 1.0
        return StageService(
            name=stage.stage_id,
            device="vta",
            threads=1,
            compute_ms=compute_ms,
            load_ms=load_ms,
            store_ms=store_ms,
            overlap_factor=non_overlap,
            spill_ms=0.0,
            launch_ms=_number(self.payload["vta"].get("submit_ms", 0.0), "VTA submit")
            + _number(self.payload["vta"].get("sync_ms", 0.0), "VTA sync"),
        )

    def estimate_boundary(self, boundary: BoundaryWorkload, direction: str) -> Dict[str, float]:
        boundary.validate()
        if boundary.transfer_resource in {"zero_copy", "stage_dma"}:
            transfer_ms = 0.0
            transaction_ms = 0.0
        else:
            direction_entry = (self.payload["boundary"].get("directions") or {}).get(direction)
            if not isinstance(direction_entry, Mapping):
                raise ValueError("hardware profile lacks boundary direction {}".format(direction))
            dma_entry = (direction_entry.get("access_models") or {}).get(boundary.access_kind)
            if not isinstance(dma_entry, Mapping):
                raise ValueError(
                    "hardware profile lacks boundary access {} for {}".format(
                        boundary.access_kind, direction
                    )
                )
            bandwidth = _number(dma_entry.get("bandwidth_GBps"), "boundary bandwidth", positive=True)
            call_latency = _number(dma_entry.get("call_latency_ms", 0.0), "boundary call latency")
            transfer_ms = boundary.transfer_bytes / bandwidth / 1.0e6
            transaction_ms = boundary.call_count * call_latency
        direction_entry = (self.payload["boundary"].get("directions") or {}).get(direction) or {}
        signature = boundary_conversion_signature(boundary)
        signature_entry = (direction_entry.get("adapter_models") or {}).get(signature)
        adapter_entry = (
            signature_entry.get(str(int(boundary.adapter_threads)))
            if isinstance(signature_entry, Mapping)
            else None
        )
        if not isinstance(adapter_entry, Mapping):
            raise ValueError(
                "hardware profile lacks adapter signature {} at {} threads for {}".format(
                    signature, boundary.adapter_threads, direction
                )
            )
        adapter_intercept_ms = _number(
            adapter_entry.get("intercept_ms", 0.0), "adapter intercept"
        )
        pack_rate = _number(adapter_entry.get("pack_elements_per_ms", 0.0), "pack rate")
        unpack_rate = _number(adapter_entry.get("unpack_elements_per_ms", 0.0), "unpack rate")
        if boundary.pack_elements and pack_rate <= 0.0:
            raise ValueError("non-empty pack adapter requires a positive measured pack rate")
        if boundary.unpack_elements and unpack_rate <= 0.0:
            raise ValueError("non-empty unpack adapter requires a positive measured unpack rate")
        pack_ms = boundary.pack_elements / pack_rate if boundary.pack_elements else 0.0
        unpack_ms = (
            boundary.unpack_elements / unpack_rate
            if boundary.unpack_elements
            else 0.0
        )
        if boundary.adapter_kind != "identity":
            # Attribute the fixed adapter cost to the phase that actually runs.
            # This does not change total service, but keeps the phase breakdown
            # usable by the resource event graph.
            if boundary.adapter_kind == "unpack":
                unpack_ms += adapter_intercept_ms
            else:
                pack_ms += adapter_intercept_ms
        return {
            "transfer_ms": transfer_ms,
            "transaction_ms": transaction_ms,
            "pack_ms": pack_ms,
            "unpack_ms": unpack_ms,
            "service_ms": transfer_ms + transaction_ms + pack_ms + unpack_ms,
            "conversion_signature": signature,
            "transfer_accounting": (
                "owned_by_lowered_stage_dma"
                if boundary.transfer_resource == "stage_dma"
                else "owned_by_boundary"
            ),
        }

    def build_pipeline_record(
        self,
        workload: PartitionWorkload,
        require_publication: bool = True,
    ) -> PipelineRecord:
        self.validate(require_publication=require_publication)
        workload.validate()
        if workload.hardware_fingerprint_sha256 != self.fingerprint_sha256:
            raise ValueError("workload and hardware profile fingerprints do not match")
        if workload.compiler_config_sha256 != self.payload.get("compiler_config_sha256"):
            raise ValueError("workload and hardware profile compiler configurations do not match")
        stages = [self.estimate_stage(stage) for stage in workload.stages]
        devices = {stage.stage_id: stage.device for stage in workload.stages}
        boundary_services = []
        boundary_breakdown = []
        for boundary in workload.boundaries:
            direction = "{}_to_{}".format(
                devices[boundary.producer_stage], devices[boundary.consumer_stage]
            )
            allowed_resources = {
                "cpu_to_cpu": {"cpu_memory", "zero_copy"},
                "cpu_to_vta": {"stage_dma", "zero_copy"},
                "vta_to_cpu": {"stage_dma", "zero_copy"},
                "vta_to_vta": {"stage_dma", "zero_copy"},
            }[direction]
            if boundary.transfer_resource not in allowed_resources:
                raise ValueError(
                    "boundary {} uses {} for {}".format(
                        boundary.boundary_id, boundary.transfer_resource, direction
                    )
                )
            estimate = self.estimate_boundary(boundary, direction)
            boundary_services.append(
                BoundaryService(
                    name=boundary.boundary_id,
                    producer=boundary.producer_stage,
                    consumer=boundary.consumer_stage,
                    direction=direction,
                    queue_depth=boundary.queue_depth,
                    transfer_resource=boundary.transfer_resource,
                    pack_ms=estimate["pack_ms"],
                    transfer_ms=estimate["transfer_ms"],
                    transaction_ms=estimate["transaction_ms"],
                    unpack_ms=estimate["unpack_ms"],
                )
            )
            boundary_breakdown.append(
                {"boundary_id": boundary.boundary_id, "direction": direction, **estimate}
            )
        vta_islands = sum(1 for stage in stages if stage.device == "vta")
        return PipelineRecord(
            model=workload.graph_id,
            candidate_id=workload.candidate_id,
            source_path="portable_static_workload",
            group_id="{}:portable".format(workload.graph_id),
            measured_cycle_ms=None,
            queue_depth=workload.queue_depth,
            stages=stages,
            boundaries=boundary_services,
            resource_capacities={
                name: int(self.payload["resource_capacities"][name])
                for name in REQUIRED_RESOURCE_CAPACITIES
            },
            boundary_ms=0.0,
            effective_segment_count=len(stages),
            vta_island_count=vta_islands,
            correctness_passed=False,
            metadata={
                "prediction_input": "hardware_profile_plus_static_workload",
                "score_input_provenance": {
                    "workload_features": {
                        "kind": "lowering_static",
                        "source": "validated_partition_workload",
                    },
                    "service_parameters": {
                        "kind": "independent_hardware_profile",
                        "source": str(self.payload.get("profile_id") or "hardware_profile"),
                    },
                    "candidate_outcome": {
                        "kind": "not_available",
                        "use": "prospective_prediction",
                    },
                },
                "static_contract_validated": True,
                "static_validation_sha256": workload.static_validation_sha256,
                "runtime_correctness_status": "not_measured",
                "compiler_config_sha256": workload.compiler_config_sha256,
                "hardware_fingerprint_sha256": workload.hardware_fingerprint_sha256,
                "boundary_breakdown": boundary_breakdown,
            },
        )
