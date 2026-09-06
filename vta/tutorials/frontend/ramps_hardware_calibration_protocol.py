#!/usr/bin/env python3
"""Budgeted, model-independent calibration protocol for RAMPS.

The protocol describes a semantic case pool.  It does not require every case
to be compiled or measured.  H2 audits one representative per builder
equivalence class; H3 freezes a bounded fit/holdout measurement plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PROTOCOL_VERSION = 4
BOARD_CASE_BUDGET = 80
MIN_GROUPED_HOLDOUT_CASES = 16
CPU_THREADS = (1, 2, 3, 4)
TRANSFER_SIZES = (4 * 1024, 64 * 1024, 1024 * 1024)
DMA_CALL_COUNTS = (1, 8, 64)
DMA_ACCESS_KINDS = ("large_contiguous", "strided", "padded")
CPU_MEMORY_POINTS = (
    (64 * 1024, "cache_hot"),
    (4 * 1024 * 1024, "streaming_prefaulted"),
    (64 * 1024 * 1024, "streaming_prefaulted"),
)
CONV_POINTS = (
    (3, 32, 224, 7, 2),
    (16, 32, 128, 3, 1),
    (32, 64, 64, 3, 2),
    (64, 128, 32, 3, 1),
    (128, 256, 16, 3, 2),
    (255, 256, 8, 1, 1),
)
TENSOR_POINTS = ((16, 128), (64, 32), (256, 8))
DENSE_POINTS = ((128, 16), (256, 255), (1024, 256))
BOUNDARY_SHAPES = ((1, 16, 8, 8), (1, 64, 32, 32), (1, 256, 64, 64))


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _builder(template: str, assertions: Iterable[str]) -> Dict[str, Any]:
    return {
        "template": template,
        "compile_mode": "native_static_package",
        "required_lowering_assertions": list(assertions),
        "fallback_allowed": False,
    }


def _reference(kind: str) -> Dict[str, Any]:
    return {
        "kind": kind,
        "implementation": "independent_host_reference",
        "runtime_correctness_status": "unmeasured_until_board_execution",
        "determinism_required": True,
    }


def _case(
    case_id: str,
    family: str,
    template: str,
    includes: Sequence[str],
    excludes: Sequence[str],
    semantics: Mapping[str, Any],
    assertions: Sequence[str],
) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "family": family,
        "case_semantics_version": 1,
        "measurement_kind": "isolated_native_service",
        "includes": list(includes),
        "excludes": list(excludes),
        "semantics": dict(semantics),
        "builder_contract": _builder(template, assertions),
        "reference_contract": _reference(template),
    }


def operator_cases() -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for device in ("cpu", "vta"):
        threads_values = CPU_THREADS if device == "cpu" else (1,)
        for index, (ci, co, spatial, kernel, stride) in enumerate(CONV_POINTS):
            for threads in threads_values:
                result.append(
                    _case(
                        "op_{}_conv_p{}_t{}".format(device, index, threads),
                        "operator_cases",
                        "{}_conv2d_fixed_schedule".format(device),
                        ["{}_compute".format(device), "{}_memory".format(device)],
                        ["boundary_adapter", "candidate_throughput"],
                        {
                            "device": device,
                            "primitive": "conv2d",
                            "input_shape": [1, ci, spatial, spatial],
                            "channels_out": co,
                            "kernel": kernel,
                            "stride": stride,
                            "padding": kernel // 2,
                            "dtype": "float32" if device == "cpu" else "int8",
                            "threads": threads,
                            "arithmetic_repeats": 1,
                            "schedule_id": "fixed_schedule",
                        },
                        ["exact_shape", "exact_dtype", "fixed_schedule", "no_fallback"],
                    )
                )
        for index, (channels, spatial) in enumerate(TENSOR_POINTS):
            for threads in threads_values:
                result.append(
                    _case(
                        "op_{}_elementwise_p{}_t{}".format(device, index, threads),
                        "operator_cases",
                        "{}_elementwise_repeat".format(device),
                        ["{}_compute".format(device), "{}_memory".format(device)],
                        ["boundary_adapter", "candidate_throughput"],
                        {
                            "device": device,
                            "primitive": "elementwise",
                            "input_shape": [1, channels, spatial, spatial],
                            "opcode": "add",
                            "dtype": "float32" if device == "cpu" else "int8",
                            "threads": threads,
                            "arithmetic_repeats": 8,
                            "schedule_id": "fixed_schedule",
                        },
                        ["fixed_input_output_traffic", "exact_opcode", "no_fallback"],
                    )
                )
        for index, (features_in, features_out) in enumerate(DENSE_POINTS):
            for threads in threads_values:
                result.append(
                    _case(
                        "op_{}_dense_p{}_t{}".format(device, index, threads),
                        "operator_cases",
                        "{}_dense_fixed_schedule".format(device),
                        ["{}_compute".format(device), "{}_memory".format(device)],
                        ["boundary_adapter", "candidate_throughput"],
                        {
                            "device": device,
                            "primitive": "dense",
                            "input_shape": [1, features_in],
                            "features_out": features_out,
                            "dtype": "float32" if device == "cpu" else "int8",
                            "threads": threads,
                            "schedule_id": "fixed_schedule",
                        },
                        ["exact_shape", "fixed_schedule", "no_fallback"],
                    )
                )
    return result


def seed_operator_cases() -> List[Dict[str, Any]]:
    """Compatibility alias for callers written against protocol v3."""

    return operator_cases()


def cpu_memory_cases() -> List[Dict[str, Any]]:
    result = []
    for operation in ("read", "write", "copy"):
        for threads in CPU_THREADS:
            for byte_count, policy in CPU_MEMORY_POINTS:
                result.append(
                    _case(
                        "cpu_memory_{}_t{}_b{}".format(operation, threads, byte_count),
                        "cpu_memory_cases",
                        "cpu_memory_{}".format(operation),
                        ["cpu_memory_{}".format(operation)],
                        ["operator_compute", "page_fault", "candidate_throughput"],
                        {
                            "operation": operation,
                            "threads": threads,
                            "bytes": byte_count,
                            "working_set_policy": policy,
                            "page_policy": "prefaulted",
                            "checksum_required": True,
                        },
                        ["memory_access_not_eliminated", "cpu_affinity_recorded"],
                    )
                )
    return result


def _dma_geometry(access: str, byte_count: int, calls: int) -> Dict[str, Any]:
    if access == "large_contiguous":
        rows, pitch, padding = 1, byte_count, [0, 0, 0, 0]
    elif access == "strided":
        rows = 8
        pitch = max(2, byte_count // rows + 16)
        padding = [0, 0, 0, 0]
    else:
        rows = 8
        pitch = max(1, byte_count // rows)
        padding = [1, 1, 1, 1]
    return {
        "bytes": byte_count,
        "calls": calls,
        "rows": rows,
        "row_bytes": max(1, byte_count // rows),
        "pitch_bytes": pitch,
        "padding": padding,
    }


def dma_cases() -> List[Dict[str, Any]]:
    result = []
    for direction in ("load", "store"):
        for access in DMA_ACCESS_KINDS:
            for byte_count in TRANSFER_SIZES:
                for calls in DMA_CALL_COUNTS:
                    control = "large_contiguous" if access != "large_contiguous" else access
                    result.append(
                        _case(
                            "dma_{}_{}_b{}_n{}".format(direction, access, byte_count, calls),
                            "dma_cases",
                            "vta_dma_{}_{}".format(direction, access),
                            ["ps_pl_{}_service".format(direction)],
                            ["vta_compute", "opposite_direction_dma", "candidate_throughput"],
                            {
                                "direction": direction,
                                "access_kind": access,
                                "matched_control_access": control,
                                "physical_geometry": _dma_geometry(access, byte_count, calls),
                                "alu_opcode": "add",
                                "alu_calls": 1,
                                "physical_output_shape": [1, 16, 8, 8],
                                "opposite_direction_calls": 1,
                            },
                            [
                                "actual_dma_instruction_count",
                                "matched_alu_and_output",
                                "matched_opposite_direction_dma",
                            ],
                        )
                    )
    return result


def boundary_cases() -> List[Dict[str, Any]]:
    definitions = (
        ("cpu_to_cpu", "identity", "float32", "NCHW", "float32", "NCHW", "none"),
        ("cpu_to_vta", "pack", "float32", "NCHW", "int8", "NCHW16c", "int8_symmetric"),
        ("vta_to_cpu", "unpack", "int8", "NCHW16c", "float32", "NCHW", "int8_symmetric"),
    )
    result = []
    for direction, adapter, src_dtype, src_layout, dst_dtype, dst_layout, quant in definitions:
        for shape_index, shape in enumerate(BOUNDARY_SHAPES):
            logical_channels = shape[1]
            physical_channels = ((logical_channels + 15) // 16) * 16
            for threads in CPU_THREADS:
                result.append(
                    _case(
                        "boundary_{}_p{}_t{}".format(direction, shape_index, threads),
                        "boundary_cases",
                        "boundary_{}".format(adapter),
                        ["host_adapter", "cpu_memory_materialization"],
                        ["stage_dma", "vta_compute", "candidate_throughput"],
                        {
                            "direction": direction,
                            "adapter_kind": adapter,
                            "logical_shape": list(shape),
                            "physical_shape": [shape[0], physical_channels, shape[2], shape[3]],
                            "source_dtype": src_dtype,
                            "source_layout": src_layout,
                            "destination_dtype": dst_dtype,
                            "destination_layout": dst_layout,
                            "quantization": quant,
                            "logical_channels": logical_channels,
                            "physical_channels": physical_channels,
                            "slice_required": physical_channels != logical_channels,
                            "threads": threads,
                            "transfer_owner": (
                                "cpu_memory" if direction == "cpu_to_cpu" else "lowered_stage_dma"
                            ),
                        },
                        ["independent_value_reference", "adapter_only_timing", "no_dma_double_count"],
                    )
                )
    return result


def vta_phase_cases() -> List[Dict[str, Any]]:
    result = []
    for point_index, point in enumerate(CONV_POINTS[:4]):
        for repeats in (1, 2, 4, 8, 16):
            ci, co, spatial, kernel, stride = point
            result.append(
                _case(
                    "vta_phase_p{}_r{}".format(point_index, repeats),
                    "vta_phase_cases",
                    "vta_compute_repeat",
                    ["vta_compute"],
                    ["variable_dma", "submit", "sync", "candidate_throughput"],
                    {
                        "input_shape": [1, ci, spatial, spatial],
                        "channels_out": co,
                        "kernel": kernel,
                        "stride": stride,
                        "compute_repeats": repeats,
                        "dma_repeats": 1,
                        "schedule_id": "fixed_schedule",
                    },
                    ["fixed_dma_instructions", "increasing_compute_instructions", "no_fallback"],
                )
            )
    return result


def submit_sync_cases() -> List[Dict[str, Any]]:
    result = []
    for policy, sleep_ns in (("busy_poll", 0), ("runtime_default", 1000)):
        for kernel, repeats in (("empty", 0), ("short", 1), ("long", 64)):
            result.append(
                _case(
                    "submit_sync_{}_{}".format(policy, kernel),
                    "submit_sync_cases",
                    "host_submit_sync",
                    ["driver_submit", "host_sync_wait"],
                    ["set_input", "get_output", "candidate_throughput"],
                    {
                        "poll_policy": policy,
                        "poll_sleep_ns": sleep_ns,
                        "kernel_class": kernel,
                        "device_compute_repeats": repeats,
                    },
                    ["submit_and_sync_timestamped_separately", "fixed_runtime_policy"],
                )
            )
    return result


def _template_smoke_ids(cases: Sequence[Mapping[str, Any]]) -> List[str]:
    selected: Dict[str, str] = {}
    for case in cases:
        template = str(case["builder_contract"]["template"])
        selected.setdefault(template, str(case["case_id"]))
    return sorted(selected.values())


def build_protocol() -> Dict[str, Any]:
    families = {
        "operator_cases": operator_cases(),
        "cpu_memory_cases": cpu_memory_cases(),
        "dma_cases": dma_cases(),
        "boundary_cases": boundary_cases(),
        "vta_phase_cases": vta_phase_cases(),
        "submit_sync_cases": submit_sync_cases(),
    }
    all_cases = [case for values in families.values() for case in values]
    protocol: Dict[str, Any] = {
        "schema_version": PROTOCOL_VERSION,
        "stage": "hardware_minimum_viable_calibration",
        "optimization_objective": "top_k_candidate_triage_not_exact_cycle_regression",
        "candidate_throughput_labels_allowed": False,
        "full_support_pool_compile_required": False,
        "full_support_pool_measurement_allowed": False,
        "support_domain_kind": "semantic_case_pool",
        "support_domain_case_count": len(all_cases),
        "board_budget": {
            "maximum_total_cases": BOARD_CASE_BUDGET,
            "minimum_grouped_holdout_cases": MIN_GROUPED_HOLDOUT_CASES,
            "default_fit_cases": BOARD_CASE_BUDGET - MIN_GROUPED_HOLDOUT_CASES,
            "adaptive_extension": "requires_preregistered_holdout_failure_and_user_confirmation",
        },
        "selection": {
            "status": "deferred_until_h2_observed_lowering_features",
            "method": "grouped_d_optimal_with_signature_coverage",
            "forbidden_inputs": ["model_name", "candidate_id", "candidate_throughput"],
        },
        "h2_compile_audit": {
            "scope": "one_case_per_builder_equivalence_class",
            "case_ids": _template_smoke_ids(all_cases),
            "runtime_correctness_claimed": False,
        },
        "fit_models": {
            "cpu": "minimum_viable_compute_memory_and_core_demand",
            "vta": "minimum_viable_compute_plus_total_dma",
            "dma": "direction_access_bytes_calls",
            "boundary": "adapter_signature_and_size",
            "vta_overlap": "deferred_to_stage5_only_if_top_k_ablation_justifies_it",
        },
        "measurement_contract": {
            "performance_path": "native_static_package_only",
            "rpc_performance_allowed": False,
            "fallback_allowed": False,
            "required_common_fields": [
                "case_id",
                "protocol_sha256",
                "measurement_plan_sha256",
                "reference_correctness_passed",
                "determinism_passed",
                "wall_ms",
                "process_cpu_ms",
                "failure_reason",
            ],
        },
        **families,
        "family_case_counts": {name: len(values) for name, values in families.items()},
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    return protocol


def _all_cases(protocol: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return [
        case
        for family in protocol.get("family_case_counts", {})
        for case in protocol.get(family, [])
    ]


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if int(protocol.get("schema_version", 0)) != PROTOCOL_VERSION:
        raise ValueError("calibration protocol must use schema version {}".format(PROTOCOL_VERSION))
    if protocol.get("optimization_objective") != "top_k_candidate_triage_not_exact_cycle_regression":
        raise ValueError("calibration objective must be candidate triage")
    if protocol.get("candidate_throughput_labels_allowed") is not False:
        raise ValueError("candidate throughput labels must not select calibration cases")
    if protocol.get("full_support_pool_compile_required") is not False:
        raise ValueError("full support-pool compilation is outside the lean protocol")
    if protocol.get("full_support_pool_measurement_allowed") is not False:
        raise ValueError("full support-pool measurement is outside the lean protocol")
    cases = _all_cases(protocol)
    if len(cases) != int(protocol.get("support_domain_case_count", -1)):
        raise ValueError("support-domain case count mismatch")
    ids = [str(case.get("case_id") or "") for case in cases]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("case ids must be non-empty and globally unique")
    for case in cases:
        if not case.get("case_semantics_version") or not case.get("semantics"):
            raise ValueError("{} lacks executable semantics".format(case.get("case_id")))
        if not case.get("builder_contract") or not case.get("reference_contract"):
            raise ValueError("{} lacks builder/reference contracts".format(case.get("case_id")))
        if not case.get("includes") or "excludes" not in case:
            raise ValueError("{} lacks component ownership".format(case.get("case_id")))
    budget = protocol.get("board_budget") or {}
    if int(budget.get("maximum_total_cases", 0)) != BOARD_CASE_BUDGET:
        raise ValueError("board case budget must be frozen at {}".format(BOARD_CASE_BUDGET))
    if int(budget.get("minimum_grouped_holdout_cases", 0)) < MIN_GROUPED_HOLDOUT_CASES:
        raise ValueError("grouped holdout budget is too small")
    dma = list(protocol.get("dma_cases") or [])
    for direction in ("load", "store"):
        for access in DMA_ACCESS_KINDS:
            selected = [
                row["semantics"]["physical_geometry"]
                for row in dma
                if row["semantics"]["direction"] == direction
                and row["semantics"]["access_kind"] == access
            ]
            if {row["bytes"] for row in selected} != set(TRANSFER_SIZES):
                raise ValueError("DMA bytes do not vary independently")
            if {row["calls"] for row in selected} != set(DMA_CALL_COUNTS):
                raise ValueError("DMA calls do not vary independently")
    if max(row["semantics"]["bytes"] for row in protocol["cpu_memory_cases"]) < 64 * 1024 * 1024:
        raise ValueError("CPU memory cases do not reach a DRAM-sized working set")
    known = set(ids)
    smokes = list((protocol.get("h2_compile_audit") or {}).get("case_ids") or [])
    if not smokes or not set(smokes) <= known or len(smokes) >= len(cases):
        raise ValueError("H2 must audit a strict representative subset")
    text = json.dumps(protocol, sort_keys=True).lower()
    if "resnet" in text or "yolo" in text:
        raise ValueError("hardware protocol contains a model-specific name")
    unhashed = dict(protocol)
    actual = unhashed.pop("protocol_sha256", "")
    if actual != canonical_sha256(unhashed):
        raise ValueError("calibration protocol SHA256 mismatch")


def freeze_measurement_plan(
    protocol: Mapping[str, Any],
    fit_case_ids: Sequence[str],
    holdout_case_ids: Sequence[str],
    observed_feature_manifest_sha256: str,
) -> Dict[str, Any]:
    """Freeze a label-free measurement plan under the protocol board budget."""

    validate_protocol(protocol)
    if not re.fullmatch(r"[0-9a-f]{64}", observed_feature_manifest_sha256 or ""):
        raise ValueError("observed feature manifest SHA256 is required")
    fit = list(fit_case_ids)
    holdout = list(holdout_case_ids)
    if len(fit) != len(set(fit)) or len(holdout) != len(set(holdout)):
        raise ValueError("fit and holdout ids must be unique")
    if set(fit) & set(holdout):
        raise ValueError("fit and holdout cases must be disjoint")
    known = {str(case["case_id"]) for case in _all_cases(protocol)}
    unknown = (set(fit) | set(holdout)) - known
    if unknown:
        raise ValueError("unknown calibration case ids: {}".format(sorted(unknown)))
    if len(fit) + len(holdout) > BOARD_CASE_BUDGET:
        raise ValueError("measurement plan exceeds the {}-case budget".format(BOARD_CASE_BUDGET))
    if len(holdout) < MIN_GROUPED_HOLDOUT_CASES:
        raise ValueError("measurement plan needs at least {} holdout cases".format(MIN_GROUPED_HOLDOUT_CASES))
    plan: Dict[str, Any] = {
        "schema_version": 1,
        "protocol_sha256": protocol["protocol_sha256"],
        "optimization_objective": "top_k_candidate_triage",
        "selection_method": "grouped_d_optimal_with_signature_coverage",
        "candidate_throughput_labels_used": False,
        "observed_feature_manifest_sha256": observed_feature_manifest_sha256,
        "fit_case_ids": fit,
        "holdout_case_ids": holdout,
        "measured_case_count": len(fit) + len(holdout),
        "maximum_case_budget": BOARD_CASE_BUDGET,
    }
    plan["measurement_plan_sha256"] = canonical_sha256(plan)
    return plan


def build_calibration_cost_ledger(
    measurement_plan: Mapping[str, Any],
    compile_seconds: float,
    board_seconds: float,
    failed_case_count: int,
) -> Dict[str, Any]:
    """Create the auditable cost record used by the candidate-triage claim."""

    unhashed_plan = dict(measurement_plan)
    actual_plan_hash = str(unhashed_plan.pop("measurement_plan_sha256", ""))
    if actual_plan_hash != canonical_sha256(unhashed_plan):
        raise ValueError("measurement plan SHA256 is missing or invalid")
    for name, value in (("compile_seconds", compile_seconds), ("board_seconds", board_seconds)):
        if float(value) < 0.0:
            raise ValueError("{} must be non-negative".format(name))
    failed = int(failed_case_count)
    measured = int(measurement_plan.get("measured_case_count", 0) or 0)
    if failed < 0 or failed > measured:
        raise ValueError("failed_case_count is outside the measurement plan")
    ledger: Dict[str, Any] = {
        "schema_version": 1,
        "measurement_plan_sha256": actual_plan_hash,
        "planned_case_count": measured,
        "completed_case_count": measured - failed,
        "failed_case_count": failed,
        "compile_seconds": float(compile_seconds),
        "board_seconds": float(board_seconds),
        "total_profile_seconds": float(compile_seconds) + float(board_seconds),
    }
    ledger["calibration_cost_ledger_sha256"] = canonical_sha256(ledger)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    protocol = build_protocol()
    validate_protocol(protocol)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary or not args.output:
        print(
            "RAMPS protocol v{}: pool={} h2_smokes={} board_budget={} holdout={} sha256={}".format(
                PROTOCOL_VERSION,
                protocol["support_domain_case_count"],
                len(protocol["h2_compile_audit"]["case_ids"]),
                BOARD_CASE_BUDGET,
                MIN_GROUPED_HOLDOUT_CASES,
                protocol["protocol_sha256"],
            )
        )


if __name__ == "__main__":
    main()
