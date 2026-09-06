#!/usr/bin/env python3
"""Tests for the portable two-layer RAMPS contract and solver boundary."""

import pytest

import ramps_hardware_calibration_protocol as calibration_protocol
from ramps_hardware_profile import HardwareProfile, with_profile_sha256
from ramps_workload_schema import (
    BoundaryWorkload,
    OperatorWorkload,
    PartitionWorkload,
    StageWorkload,
    TensorSignature,
)
import resource_aware_dataset as dataset
import resource_aware_maxplus as ramps


HARDWARE_SHA = "1" * 64
COMPILER_SHA = "2" * 64
STATIC_SHA = "3" * 64
EVIDENCE_SHA = "4" * 64


def profile_payload():
    adapter_signatures = {
        "cpu_to_cpu": "identity|float32:NCHW>float32:NCHW>float32:NCHW|quant=none|padded=0|access=large_contiguous",
        "cpu_to_vta": "pack|float32:NCHW>int8:NCHW16c>int8:NCHW16c|quant=int8_symmetric|padded=1|access=large_contiguous",
        "vta_to_cpu": "unpack|int8:NCHW16c>int8:NCHW16c>float32:NCHW|quant=int8_symmetric|padded=1|access=large_contiguous",
    }
    directions = {}
    for direction in ("cpu_to_cpu", "cpu_to_vta", "vta_to_cpu", "vta_to_vta"):
        directions[direction] = {
            "access_models": {
                access: {"bandwidth_GBps": 1.0, "call_latency_ms": 0.01}
                for access in ("large_contiguous", "small_contiguous", "strided", "padded")
            },
            "adapter_models": {},
        }
        if direction in adapter_signatures:
            directions[direction]["adapter_models"][adapter_signatures[direction]] = {
                str(threads): {
                    "intercept_ms": 0.01 if direction != "cpu_to_cpu" else 0.0,
                    "pack_elements_per_ms": 1000.0,
                    "unpack_elements_per_ms": 1000.0,
                }
                for threads in (1, 2, 3, 4)
            }
    unit_models = {}
    for primitive, base_rate in (("conv2d", 2.0), ("elementwise", 1.0)):
        unit_models[primitive] = {}
        for threads in (1, 2, 3, 4):
            unit_models[primitive][str(threads)] = {
                "interpolation": "normalized_inverse_distance_v1",
                "feature_names": ["physical_ops"],
                "schedule_ids": ["fixed-schedule"],
                "validation": {"holdout_passed": True, "holdout_mape": 0.05},
                "support_points": [
                    {
                        "features": {"physical_ops": ops},
                        "compute_gops": base_rate * threads,
                        "compute_core_scale": float(threads),
                        "memory_core_scale": 0.5 * float(threads),
                    }
                    for ops in (1.0e6, 1.0e9)
                ],
            }
    memory_models = {
        str(threads): {
            "interpolation": "normalized_inverse_distance_v1",
            "feature_names": ["working_set_bytes"],
            "validation": {"holdout_passed": True, "holdout_mape": 0.05},
            "support_points": [
                {
                    "features": {"working_set_bytes": byte_count},
                    "bandwidth_GBps": 0.5 + 0.5 * threads,
                }
                for byte_count in (1000, 6000, 500_000_000)
            ],
        }
        for threads in (1, 2, 3, 4)
    }
    vta_surface = {
        "interpolation": "normalized_inverse_distance_v1",
        "feature_names": ["physical_ops"],
        "schedule_ids": ["fixed-schedule"],
        "validation": {"holdout_passed": True, "holdout_mape": 0.05},
        "support_points": [
            {"features": {"physical_ops": ops}, "compute_gops": 20.0}
            for ops in (1.0e6, 1.0e9)
        ],
    }
    overlap_surface = {
        "validation_source": "stage5_controlled_pipeline",
        "interpolation": "normalized_inverse_distance_v1",
        "feature_names": ["physical_ops"],
        "validation": {"holdout_passed": True, "holdout_mape": 0.05},
        "support_points": [
            {"features": {"physical_ops": ops}, "non_overlap_fraction": 0.25}
            for ops in (1.0e6, 1.0e9)
        ],
    }
    payload = {
        "schema_version": 3,
        "profile_kind": "portable_hardware_service_profile",
        "hardware_fingerprint": {"sha256": HARDWARE_SHA},
        "compiler_config_sha256": COMPILER_SHA,
        "allowed_schedule_ids": ["fixed-schedule"],
        "resource_capacities": {
            "cpu_core_pool": 4,
            "cpu_memory": 1,
            "vta_mutex": 1,
            "ps_pl_load": 1,
            "ps_pl_store": 1,
            "bridge_pack_unpack": 1,
        },
        "fallback_used": False,
        "estimated_from_used": False,
        "component_accounting_policy": {
            "vta_dma_scope": "total_physical_including_spill",
            "boundary_ps_pl_owner": "lowered_stage_dma",
            "spill_time_policy": "informational_not_additive",
        },
        "supported_workload_domain": {
            "primitives": ["conv2d", "elementwise"],
            "attribute_ranges": {"kernel": [1, 7], "stride": [1, 2]},
        },
        "validation_gates": {
            "correctness_gate_passed": True,
            "determinism_gate_passed": True,
            "precision_gate_passed": True,
            "service_fit_validated": True,
        },
        "validation_evidence": {
            "protocol_sha256": EVIDENCE_SHA,
            "measurement_plan_sha256": EVIDENCE_SHA,
            "calibration_cost_ledger_sha256": EVIDENCE_SHA,
            "raw_data_sha256": EVIDENCE_SHA,
            "fit_report_sha256": EVIDENCE_SHA,
            "reference_correctness_report_sha256": EVIDENCE_SHA,
        },
        "optimization_objective": "top_k_candidate_triage",
        "calibration_budget": {
            "support_pool_case_count": 212,
            "measured_case_count": 80,
            "holdout_case_count": 16,
            "full_pool_measured": False,
        },
        "cpu": {
            "unit_models": unit_models,
            "memory_models": memory_models,
            "launch_ms": 0.1,
            "launch_core_ms": 0.05,
        },
        "vta": {
            "unit_models": {"conv2d": vta_surface},
            "system_overlap_models": {"conv2d": overlap_surface},
            "submit_ms": 0.05,
            "sync_ms": 0.10,
            "sram_capacity_bytes": {
                "inp": 1 << 20,
                "wgt": 1 << 20,
                "acc": 1 << 20,
                "out": 1 << 20,
            },
        },
        "dma": {
            "access_models": {
                "large_contiguous": {
                    "load": {"bandwidth_GBps": 1.0, "call_latency_ms": 0.01},
                    "store": {"bandwidth_GBps": 0.8, "call_latency_ms": 0.015},
                },
                "small_contiguous": {
                    "load": {"bandwidth_GBps": 0.5, "call_latency_ms": 0.02},
                    "store": {"bandwidth_GBps": 0.4, "call_latency_ms": 0.025},
                },
                "strided": {
                    "load": {"bandwidth_GBps": 0.4, "call_latency_ms": 0.03},
                    "store": {"bandwidth_GBps": 0.4, "call_latency_ms": 0.035},
                },
                "padded": {
                    "load": {"bandwidth_GBps": 0.3, "call_latency_ms": 0.04},
                    "store": {"bandwidth_GBps": 0.3, "call_latency_ms": 0.045},
                },
            }
        },
        "boundary": {"directions": directions},
    }
    return with_profile_sha256(payload)


def operator(
    op_id,
    primitive="conv2d",
    ops=1.0e9,
    memory_bytes=6000,
    access="large_contiguous",
    sram_peak=4096,
):
    return OperatorWorkload(
        op_id=op_id,
        primitive=primitive,
        logical_ops=ops,
        physical_ops=ops,
        input_bytes=1000,
        weight_bytes=2000,
        output_bytes=3000,
        memory_bytes=memory_bytes,
        load_bytes=3000,
        store_bytes=3000,
        load_calls=2,
        store_calls=2,
        avg_bytes_per_call=1500,
        access_kind=access,
        schedule_id="fixed-schedule",
        fusion_group_id="fusion-{}".format(op_id),
        sram_peak_bytes={bank: sram_peak for bank in ("inp", "wgt", "acc", "out")},
        resource_accounting_ids={
            resource: ["{}:{}".format(op_id, resource)]
            for resource in ("cpu_memory", "ps_pl_load", "ps_pl_store")
        },
        attributes={"kernel": 3, "stride": 1},
    )


def tensor_for(adapter):
    if adapter == "identity":
        return TensorSignature(
            logical_shape=[1, 15, 8, 8],
            physical_shape=[1, 15, 8, 8],
            producer_dtype="float32",
            producer_layout="NCHW",
            transport_dtype="float32",
            transport_layout="NCHW",
            consumer_dtype="float32",
            consumer_layout="NCHW",
            logical_bytes=3840,
            physical_bytes=3840,
        )
    if adapter == "pack":
        producer_dtype, producer_layout = "float32", "NCHW"
        consumer_dtype, consumer_layout = "int8", "NCHW16c"
    else:
        producer_dtype, producer_layout = "int8", "NCHW16c"
        consumer_dtype, consumer_layout = "float32", "NCHW"
    return TensorSignature(
        logical_shape=[1, 15, 8, 8],
        physical_shape=[1, 1, 8, 8, 1, 16],
        producer_dtype=producer_dtype,
        producer_layout=producer_layout,
        transport_dtype="int8",
        transport_layout="NCHW16c",
        consumer_dtype=consumer_dtype,
        consumer_layout=consumer_layout,
        logical_bytes=3840 if producer_dtype == "float32" else 960,
        physical_bytes=1024,
        quantization={"scheme": "int8_symmetric", "scale_source": "lowered_graph"},
        channel_padding={"logical_channels": 15, "physical_channels": 16, "axis": 1},
        slice_policy={"axis": 1, "begin": 0, "end": 15},
    )


def boundary(boundary_id, producer, consumer, adapter, queue_depth=2):
    tensor = tensor_for(adapter)
    transfer_bytes = tensor.physical_bytes
    return BoundaryWorkload(
        boundary_id=boundary_id,
        producer_stage=producer,
        consumer_stage=consumer,
        tensor=tensor,
        adapter_kind=adapter,
        transfer_bytes=transfer_bytes,
        call_count=1 if adapter == "identity" else 0,
        avg_bytes_per_call=transfer_bytes if adapter == "identity" else 0.0,
        transfer_resource="cpu_memory" if adapter == "identity" else "stage_dma",
        queue_depth=queue_depth,
        resource_accounting_ids=(
            {"cpu_memory": ["{}:cpu_memory".format(boundary_id)]}
            if adapter == "identity"
            else {"bridge_pack_unpack": ["{}:adapter".format(boundary_id)]}
        ),
        pack_elements=960 if adapter == "pack" else 0,
        unpack_elements=960 if adapter == "unpack" else 0,
    )


def workload():
    return PartitionWorkload(
        graph_id="unseen_graph",
        candidate_id="candidate-1",
        compiler_config_sha256=COMPILER_SHA,
        hardware_fingerprint_sha256=HARDWARE_SHA,
        queue_depth=2,
        stages=[
            StageWorkload("stage0", "cpu", 3, [operator("op0")]),
            StageWorkload("stage1", "vta", 1, [operator("op1")]),
            StageWorkload("stage2", "cpu", 2, [operator("op2", "elementwise", 1.0e6)]),
        ],
        boundaries=[
            boundary("b0", "stage0", "stage1", "pack"),
            boundary("b1", "stage1", "stage2", "unpack"),
        ],
        static_contract_validated=True,
        static_validation_sha256=STATIC_SHA,
        identity_metadata={"model": "kept_for_reporting_only", "throughput_fps": 999.0},
    )


def test_hardware_protocol_contains_no_model_specific_names():
    protocol = calibration_protocol.build_protocol()
    calibration_protocol.validate_protocol(protocol)
    text = str(protocol).lower()
    assert "resnet" not in text
    assert "yolo" not in text
    assert protocol["candidate_throughput_labels_allowed"] is False
    assert protocol["schema_version"] == 4
    assert protocol["full_support_pool_compile_required"] is False
    assert protocol["full_support_pool_measurement_allowed"] is False
    assert protocol["board_budget"]["maximum_total_cases"] == 80
    assert protocol["board_budget"]["minimum_grouped_holdout_cases"] == 16
    assert len(protocol["h2_compile_audit"]["case_ids"]) < protocol["support_domain_case_count"]
    assert protocol["fit_models"]["vta_overlap"].startswith("deferred_to_stage5")
    required = protocol["measurement_contract"]["required_common_fields"]
    assert "reference_correctness_passed" in required
    assert "determinism_passed" in required


def test_hardware_protocol_separates_dma_bytes_calls_and_reaches_dram_sizes():
    protocol = calibration_protocol.build_protocol()
    load_contiguous = [
        row
        for row in protocol["dma_cases"]
        if row["semantics"]["direction"] == "load"
        and row["semantics"]["access_kind"] == "large_contiguous"
    ]
    combinations = {
        (
            row["semantics"]["physical_geometry"]["bytes"],
            row["semantics"]["physical_geometry"]["calls"],
        )
        for row in load_contiguous
    }
    assert combinations == {
        (byte_count, calls)
        for byte_count in calibration_protocol.TRANSFER_SIZES
        for calls in calibration_protocol.DMA_CALL_COUNTS
    }
    assert max(row["semantics"]["bytes"] for row in protocol["cpu_memory_cases"]) >= 64 * 1024 * 1024
    assert all(
        row["semantics"]["transfer_owner"] == "lowered_stage_dma"
        for row in protocol["boundary_cases"]
        if row["semantics"]["direction"] != "cpu_to_cpu"
    )


def test_hardware_measurement_plan_is_label_free_and_budgeted():
    protocol = calibration_protocol.build_protocol()
    case_ids = [
        case["case_id"]
        for family in protocol["family_case_counts"]
        for case in protocol[family]
    ]
    plan = calibration_protocol.freeze_measurement_plan(
        protocol,
        case_ids[:64],
        case_ids[64:80],
        "a" * 64,
    )
    assert plan["measured_case_count"] == 80
    assert plan["candidate_throughput_labels_used"] is False
    assert len(plan["measurement_plan_sha256"]) == 64
    ledger = calibration_protocol.build_calibration_cost_ledger(plan, 12.5, 87.5, 2)
    assert ledger["total_profile_seconds"] == pytest.approx(100.0)
    assert ledger["completed_case_count"] == 78
    assert len(ledger["calibration_cost_ledger_sha256"]) == 64
    with pytest.raises(ValueError, match="exceeds"):
        calibration_protocol.freeze_measurement_plan(
            protocol, case_ids[:65], case_ids[65:81], "a" * 64
        )


def test_hardware_protocol_cases_have_executable_semantics():
    protocol = calibration_protocol.build_protocol()
    for family in protocol["family_case_counts"]:
        for case in protocol[family]:
            assert case["semantics"]
            assert case["builder_contract"]["required_lowering_assertions"]
            assert case["reference_contract"]["implementation"] == "independent_host_reference"


def test_prediction_payload_strips_identity_and_measurement_metadata():
    payload = workload().prediction_payload()
    assert "identity_metadata" not in payload
    assert "candidate_id" not in str(payload)
    assert "throughput_fps" not in str(payload)
    assert "resource_accounting_ids" not in str(payload)
    assert "fusion-op0" not in str(payload)


def test_portable_layers_build_explicit_boundary_services_and_unmeasured_status():
    record = dataset.portable_record_from_workload(
        workload(), HardwareProfile.from_dict(profile_payload())
    )
    assert record.metadata["prediction_input"] == "hardware_profile_plus_static_workload"
    assert ramps.lightweight_cycle_scores(
        record, require_zero_feedback=True
    )["provenance"]["zero_feedback_eligible"]
    assert record.metadata["runtime_correctness_status"] == "not_measured"
    assert record.correctness_passed is False
    assert record.stages[0].core_demand_source == "portable_hardware_profile"
    assert record.stages[1].load_ms > 0.0
    assert record.boundary_ms == 0.0
    assert len(record.boundaries) == 2
    assert sum(item.service_ms for item in record.boundaries) > 0.0
    prediction = ramps.predict_record(
        record,
        {"prediction_options": {"require_measured_core_demand": True}},
    )
    assert prediction["maxplus_cycle_ms"] > 0.0


def test_cpu_roofline_is_applied_per_sequential_fused_unit():
    compute_heavy = operator("compute", ops=1.0e9, memory_bytes=1000)
    memory_heavy = operator("memory", ops=1.0e6, memory_bytes=500_000_000)
    stage = StageWorkload("cpu", "cpu", 4, [compute_heavy, memory_heavy])
    service = HardwareProfile.from_dict(profile_payload()).estimate_stage(stage)
    aggregate_roofline = max(service.compute_ms, service.memory_ms)
    assert service.sequential_unit_service_ms > aggregate_roofline * 1.5
    assert len(service.unit_roofline_components_ms) == 2
    assert service.unscaled_service_ms == pytest.approx(
        service.sequential_unit_service_ms + service.launch_ms
    )
    scaled = ramps.scaled_stage_service(
        service, {"cpu_compute_scale": 2.0, "cpu_memory_scale": 0.5}
    )
    expected = sum(
        max(2.0 * item["compute_ms"], 0.5 * item["memory_ms"])
        for item in service.unit_roofline_components_ms
    ) + service.launch_ms
    assert scaled == pytest.approx(expected)


def test_explicit_branch_dag_is_preserved_without_adjacent_edge_fabrication():
    item = workload()
    branch = PartitionWorkload(
        **{
            **item.__dict__,
            "boundaries": [
                boundary("route", "stage0", "stage1", "pack", queue_depth=1),
                boundary("skip", "stage0", "stage2", "identity", queue_depth=4),
                boundary("body", "stage1", "stage2", "unpack", queue_depth=2),
            ],
        }
    )
    record = HardwareProfile.from_dict(profile_payload()).build_pipeline_record(branch)
    graph = ramps.marked_event_graph_payload(record)
    data_edges = {
        (edge["producer"], edge["consumer"])
        for edge in graph["channels"]
        if edge["kind"] == "data_dependency"
    }
    assert data_edges == {("stage0", "route"), ("stage0", "skip"), ("stage1", "body")}
    assert len({edge["name"] for edge in graph["channels"]}) == len(graph["channels"])
    fifo = {
        item.name: item
        for item in ramps.build_cycle_constraints(record)
        if item.resource == "fifo_backpressure"
    }
    assert fifo["fifo:route"].tokens == 2
    assert fifo["fifo:skip"].tokens == 5
    assert fifo["fifo:body"].tokens == 3


def test_quantized_physical_transport_can_be_smaller_than_logical_float_tensor():
    tensor = tensor_for("pack")
    tensor.validate()
    assert tensor.physical_bytes < tensor.logical_bytes


def test_dtype_conversion_requires_quantization_contract():
    payload = {**tensor_for("pack").__dict__, "quantization": {}}
    with pytest.raises(ValueError, match="quantization metadata"):
        TensorSignature(**payload).validate()


def test_identity_adapter_cannot_hide_layout_conversion():
    tensor = TensorSignature(**{**tensor_for("identity").__dict__, "consumer_layout": "NHWC"})
    broken = BoundaryWorkload(
        **{**boundary("x", "stage0", "stage2", "identity").__dict__, "tensor": tensor}
    )
    with pytest.raises(ValueError, match="identity adapter"):
        broken.validate()


def test_missing_primitive_is_rejected_instead_of_falling_back():
    item = workload()
    missing = OperatorWorkload(**{**operator("missing").__dict__, "primitive": "resize"})
    altered = PartitionWorkload(
        **{
            **item.__dict__,
            "stages": [StageWorkload("stage0", "cpu", 3, [missing]), *item.stages[1:]],
        }
    )
    with pytest.raises(ValueError, match="outside calibrated domain"):
        HardwareProfile.from_dict(profile_payload()).build_pipeline_record(altered)


def test_schedule_compiler_and_hardware_mismatches_are_rejected():
    profile = HardwareProfile.from_dict(profile_payload())
    item = workload()
    bad_schedule = OperatorWorkload(**{**operator("bad").__dict__, "schedule_id": "other"})
    with pytest.raises(ValueError, match="was not calibrated"):
        profile.estimate_stage(StageWorkload("cpu", "cpu", 1, [bad_schedule]))
    with pytest.raises(ValueError, match="compiler configurations"):
        profile.build_pipeline_record(
            PartitionWorkload(**{**item.__dict__, "compiler_config_sha256": "5" * 64})
        )
    with pytest.raises(ValueError, match="fingerprints do not match"):
        profile.build_pipeline_record(
            PartitionWorkload(**{**item.__dict__, "hardware_fingerprint_sha256": "6" * 64})
        )


def test_resource_capacity_comes_from_hardware_profile_not_board_constant():
    payload = profile_payload()
    payload["resource_capacities"]["cpu_core_pool"] = 2
    profile = HardwareProfile.from_dict(with_profile_sha256(payload))
    record = profile.build_pipeline_record(workload())
    matrix = ramps.resource_demand_matrix(record, require_measured_core_demand=True)
    assert matrix["cpu_core_tokens"] == 2
    cpu_cycle = next(
        item
        for item in ramps.build_cycle_constraints(
            record, require_measured_core_demand=True
        )
        if item.resource == "cpu_core_pool"
    )
    assert cpu_cycle.tokens == 2


def test_out_of_domain_shape_and_vta_sram_overflow_are_rejected():
    profile = HardwareProfile.from_dict(profile_payload())
    bad_shape = OperatorWorkload(
        **{**operator("shape").__dict__, "attributes": {"kernel": 9, "stride": 1}}
    )
    with pytest.raises(ValueError, match="outside calibrated"):
        profile.estimate_stage(StageWorkload("cpu", "cpu", 1, [bad_shape]))
    overflow = operator("overflow", sram_peak=(1 << 20) + 1)
    with pytest.raises(ValueError, match="SRAM capacity"):
        profile.estimate_stage(StageWorkload("vta", "vta", 1, [overflow]))
    profile.estimate_stage(StageWorkload("cpu", "cpu", 1, [overflow]))


def test_missing_measured_pack_rate_is_rejected():
    payload = profile_payload()
    adapter_models = payload["boundary"]["directions"]["cpu_to_vta"][
        "adapter_models"
    ]
    signature = next(iter(adapter_models))
    adapter_models[signature]["1"]["pack_elements_per_ms"] = 0.0
    profile = HardwareProfile.from_dict(with_profile_sha256(payload))
    with pytest.raises(ValueError, match="positive measured pack rate"):
        profile.build_pipeline_record(workload())


def test_unpack_adapter_intercept_is_charged_to_unpack_phase():
    profile = HardwareProfile.from_dict(profile_payload())
    edge = boundary("unpack", "stage1", "stage2", "unpack")
    estimate = profile.estimate_boundary(edge, "vta_to_cpu")
    assert estimate["pack_ms"] == 0.0
    assert estimate["unpack_ms"] == pytest.approx(960.0 / 1000.0 + 0.01)


def test_adapter_lookup_distinguishes_access_geometry():
    profile = HardwareProfile.from_dict(profile_payload())
    edge = boundary("pack", "stage0", "stage1", "pack")
    strided = BoundaryWorkload(**{**edge.__dict__, "access_kind": "strided"})
    with pytest.raises(ValueError, match=r"access=strided"):
        profile.estimate_boundary(strided, "cpu_to_vta")


def test_partition_rejects_double_counted_stage_and_boundary_transaction():
    item = workload()
    first = item.boundaries[0]
    duplicated = BoundaryWorkload(
        **{
            **first.__dict__,
            "resource_accounting_ids": {"ps_pl_load": ["op1:ps_pl_load"]},
        }
    )
    with pytest.raises(ValueError, match="cannot duplicate PS-PL accounting ids"):
        duplicated.validate()


def test_shape_aware_service_surface_changes_rate_and_rejects_extrapolation():
    payload = profile_payload()
    points = payload["cpu"]["unit_models"]["conv2d"]["1"]["support_points"]
    points[0]["compute_gops"] = 1.0
    points[1]["compute_gops"] = 4.0
    profile = HardwareProfile.from_dict(with_profile_sha256(payload))
    small = profile.estimate_stage(
        StageWorkload("small", "cpu", 1, [operator("small", ops=1.0e6)])
    )
    large = profile.estimate_stage(
        StageWorkload("large", "cpu", 1, [operator("large", ops=1.0e9)])
    )
    assert small.compute_ms == pytest.approx(1.0)
    assert large.compute_ms == pytest.approx(250.0)
    with pytest.raises(ValueError, match="outside calibrated"):
        profile.estimate_stage(
            StageWorkload("outside", "cpu", 1, [operator("outside", ops=2.0e9)])
        )


def test_fixed_schedule_spill_is_diagnostic_not_additive():
    profile = HardwareProfile.from_dict(profile_payload())
    base = operator("vta-base")
    spilled = OperatorWorkload(
        **{**base.__dict__, "op_id": "vta-spill", "spill_bytes": 4096, "spill_calls": 4}
    )
    base_service = profile.estimate_stage(StageWorkload("base", "vta", 1, [base]))
    spill_service = profile.estimate_stage(StageWorkload("spill", "vta", 1, [spilled]))
    assert spill_service.spill_ms == 0.0
    assert spill_service.unscaled_service_ms == pytest.approx(base_service.unscaled_service_ms)


def test_profile_hash_and_publication_evidence_are_enforced():
    payload = profile_payload()
    payload["cpu"]["launch_ms"] = 99.0
    with pytest.raises(ValueError, match="SHA256"):
        HardwareProfile.from_dict(payload)
    payload = profile_payload()
    payload["validation_gates"]["service_fit_validated"] = False
    profile = HardwareProfile.from_dict(with_profile_sha256(payload))
    with pytest.raises(ValueError, match="service_fit_validated"):
        profile.build_pipeline_record(workload())


def test_publication_profile_requires_per_surface_holdout_evidence():
    payload = profile_payload()
    del payload["cpu"]["unit_models"]["conv2d"]["1"]["validation"]
    profile = HardwareProfile.from_dict(with_profile_sha256(payload))
    with pytest.raises(ValueError, match="lacks a passing grouped holdout"):
        profile.build_pipeline_record(workload())


def test_new_search_cannot_silently_use_legacy_yolo_estimator():
    with pytest.raises(RuntimeError, match="legacy aggregate estimator"):
        dataset.yolo_record_from_candidate({"candidate_id": "x"})


def test_model_specific_operator_features_are_rejected():
    with pytest.raises(ValueError, match="forbidden prediction key layer_name"):
        OperatorWorkload(
            **{
                **operator("op").__dict__,
                "attributes": {"layer_name": "layer4_block0"},
            }
        ).validate()
