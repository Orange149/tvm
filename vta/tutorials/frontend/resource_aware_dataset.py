#!/usr/bin/env python3
"""Historical artifact adapters for RAMPS evaluation.

The ResNet/YOLO reconstruction below preserves old experiments and is not the
portable prediction path.  New DNNs must emit ``PartitionWorkload`` objects
and combine them with a validated ``HardwareProfile``; see
``portable_record_from_workload``.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from resource_aware_maxplus import (
    PipelineRecord,
    StageService,
    as_float,
    as_int,
    parse_json_value,
)
from ramps_hardware_profile import HardwareProfile
from ramps_workload_schema import PartitionWorkload


LEGACY_MODEL_SPECIFIC_ADAPTERS = True


DEFAULT_RESNET_ROOTS = (
    "vta/tutorials/frontend/report_out/native_stage_pipeline_searches/20260507_v23_warm100",
    "vta/tutorials/frontend/report_out/native_stage_pipeline_searches/20260512_theory_build100_board",
)
DEFAULT_YOLO_SUMMARIES = (
    "vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/"
    "20260514_resnet_style_cpu_vta_cpu_100/summary.json",
    "vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/"
    "20260514_yolo_multisplit100/summary.json",
)
DEFAULT_CONTROLLED_RESNET_SUMMARY = (
    "vta/tutorials/frontend/report_out/resource_aware_maxplus/20260713_v1/"
    "identification/resnet72/controlled72_summary.json"
)
REPO_ROOT = Path(__file__).resolve().parents[3]


def portable_record_from_workload(
    workload: PartitionWorkload,
    hardware_profile: HardwareProfile,
    require_publication: bool = True,
) -> PipelineRecord:
    """Build a prediction record without model-name or layer-name features."""

    return hardware_profile.build_pipeline_record(
        workload,
        require_publication=require_publication,
    )

# idx, name, input channels, output channels, output height/width, kernel
YOLO_CONVS = (
    (0, "conv0", 3, 16, 416, 416, 3),
    (1, "conv2", 16, 32, 208, 208, 3),
    (2, "conv4", 32, 64, 104, 104, 3),
    (3, "conv6", 64, 128, 52, 52, 3),
    (4, "conv8", 128, 256, 26, 26, 3),
    (5, "conv10", 256, 512, 13, 13, 3),
    (6, "conv12", 512, 1024, 13, 13, 3),
    (7, "conv13", 1024, 256, 13, 13, 1),
    (8, "conv18", 256, 128, 13, 13, 1),
    (9, "conv21", 384, 256, 26, 26, 3),
    (10, "conv22", 256, 255, 26, 26, 1),
    (11, "conv14", 256, 512, 13, 13, 3),
    (12, "conv15", 512, 255, 13, 13, 1),
)
RESNET_BLOCK_SHAPES = {
    "layer1_block0": (64, 64, 56, False),
    "layer1_block1": (64, 64, 56, False),
    "layer2_block0": (64, 128, 28, True),
    "layer2_block1": (128, 128, 28, False),
    "layer3_block0": (128, 256, 14, True),
    "layer3_block1": (256, 256, 14, False),
    "layer4_block0": (256, 512, 7, True),
    "layer4_block1": (512, 512, 7, False),
}


def load_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as inp:
        return json.load(inp)


def calibration_is_measured(calibration: Optional[Mapping[str, Any]]) -> bool:
    if not calibration:
        return False
    source = str(calibration.get("source", ""))
    if not source or "default_static" in source:
        return False
    sections = []
    sections.extend((calibration.get("cpu_eff_gops") or {}).values())
    sections.extend((calibration.get("vta_eff_gops") or {}).values())
    sections.extend((calibration.get("dma") or {}).values())
    return bool(sections) and all(
        "default_static" not in str(item.get("estimated_from", "")) for item in sections
    )


def _cpu_gops(calibration: Mapping[str, Any], bucket: str, threads: int, fallback: float) -> float:
    entry = (calibration.get("cpu_eff_gops") or {}).get(bucket, {})
    return max(0.05, as_float(entry.get(str(max(1, int(threads)))), fallback))


def _vta_gops(calibration: Mapping[str, Any], bucket: str, fallback: float) -> float:
    entry = (calibration.get("vta_eff_gops") or {}).get(bucket, {})
    return max(0.05, as_float(entry.get("gops"), fallback))


def _dma_ms_from_parts(parts: Sequence[Mapping[str, Any]], calibration: Mapping[str, Any]) -> float:
    total = 0.0
    for part in parts:
        entry = (calibration.get("dma") or {}).get(str(part.get("bucket")), {})
        load_bw = as_float(entry.get("ps_pl_load_bw_GBps"), 1.5)
        store_bw = as_float(entry.get("ps_pl_store_bw_GBps"), 1.5)
        bandwidth = max(0.05, min(load_bw, store_bw))
        total += as_float(part.get("bytes")) / (bandwidth * 1.0e9) * 1000.0
    return total


def _dma_channels_from_parts(
    parts: Sequence[Mapping[str, Any]], calibration: Mapping[str, Any]
) -> Tuple[float, float]:
    load_ms = 0.0
    store_ms = 0.0
    for part in parts:
        entry = (calibration.get("dma") or {}).get(str(part.get("bucket")), {})
        total_profile_bytes = as_float(entry.get("load_bytes")) + as_float(
            entry.get("store_bytes")
        )
        if total_profile_bytes <= 0.0:
            raise RuntimeError(
                "measured DMA bucket {} lacks load/store byte composition".format(
                    part.get("bucket")
                )
            )
        load_fraction = as_float(entry.get("load_bytes")) / total_profile_bytes
        byte_count = as_float(part.get("bytes"))
        load_bytes = byte_count * load_fraction
        store_bytes = byte_count - load_bytes
        load_ms += load_bytes / max(0.05, as_float(entry.get("ps_pl_load_bw_GBps"))) / 1.0e6
        store_ms += store_bytes / max(0.05, as_float(entry.get("ps_pl_store_bw_GBps"))) / 1.0e6
    return load_ms, store_ms


def _stage_measured_lookup(row: Mapping[str, Any]) -> Dict[str, float]:
    summaries = parse_json_value(row.get("stage_ms_summary"), []) or []
    return {str(item.get("name")): as_float(item.get("ms")) for item in summaries}


def _resolve_output_dir(row: Mapping[str, Any], source_path: str) -> Optional[Path]:
    raw = str(row.get("output_dir") or "")
    if not raw:
        return None
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([REPO_ROOT / path, Path(source_path).parent / path.name])
    return next((item for item in candidates if item.is_dir()), None)


def _profile_per_inference(status: Mapping[str, Any], runs: int) -> Dict[str, float]:
    divisor = float(max(1, int(runs)))
    h2d_us = as_float(status.get("mem_copy_from_host_us"))
    d2h_us = as_float(status.get("mem_copy_to_host_us"))
    flush_us = as_float(status.get("flush_cache_us"))
    invalidate_us = as_float(status.get("invalidate_cache_us"))
    return {
        "host_to_vta_copy_ms": h2d_us / 1000.0 / divisor,
        "vta_to_host_copy_ms": d2h_us / 1000.0 / divisor,
        "direct_copy_ms": (h2d_us + d2h_us) / 1000.0 / divisor,
        "direct_copy_bytes": (
            as_float(status.get("mem_copy_from_host_bytes"))
            + as_float(status.get("mem_copy_to_host_bytes"))
        )
        / divisor,
        "direct_copy_calls": (
            as_float(status.get("mem_copy_from_host_calls"))
            + as_float(status.get("mem_copy_to_host_calls"))
        )
        / divisor,
        "coherence_ms": (flush_us + invalidate_us) / 1000.0 / divisor,
        "flush_cache_ms": flush_us / 1000.0 / divisor,
        "invalidate_cache_ms": invalidate_us / 1000.0 / divisor,
        "dma_load_bytes": as_float(status.get("load_buffer_2d_bytes")) / divisor,
        "dma_store_bytes": as_float(status.get("store_buffer_2d_bytes")) / divisor,
        "dma_load_calls": as_float(status.get("load_buffer_2d_calls")) / divisor,
        "dma_store_calls": as_float(status.get("store_buffer_2d_calls")) / divisor,
        "device_run_wait_ms": as_float(status.get("device_run_wait_us")) / 1000.0 / divisor,
    }


def _resnet_direct_communication_profile(
    row: Mapping[str, Any], source_path: str
) -> Dict[str, Any]:
    output_dir = _resolve_output_dir(row, source_path)
    if output_dir is None:
        return {"available": False}
    result: Dict[str, Any] = {
        "available": False,
        "source": "vta_runtime_profiler",
        "output_dir": str(output_dir),
    }
    modes = {
        "serial": ("single_run_status.json", 1),
        "pipeline": ("benchmark_totals_status.json", max(1, as_int(row.get("runs"), 1))),
    }
    for mode, (filename, runs) in modes.items():
        path = output_dir / "profile" / mode / filename
        if not path.is_file():
            continue
        result[mode] = _profile_per_inference(load_json(path), runs)
        result[mode]["status_path"] = str(path)
        result["available"] = True
    return result


def _round_up(value: int, multiple: int = 16) -> int:
    return ((int(value) + multiple - 1) // multiple) * multiple


def _resnet_unit_memory_bytes(name: str) -> float:
    if name == "stem":
        return float(
            4
            * (
                3 * 224 * 224
                + 64 * 3 * 7 * 7
                + 64 * 112 * 112
                + 64 * 56 * 56
            )
        )
    if name == "head":
        return float(4 * (512 * 7 * 7 + 512 + 512 * 1000 + 1000 + 1000))
    match = re.match(r"(layer\d_block\d)_(main_preadd|skip_proj|add_relu_tail)$", name)
    if not match:
        return 0.0
    block, kind = match.groups()
    channels_in, channels_out, out_hw, downsample = RESNET_BLOCK_SHAPES[block]
    input_hw = out_hw * 2 if downsample else out_hw
    input_elements = channels_in * input_hw * input_hw
    output_elements = channels_out * out_hw * out_hw
    if kind == "main_preadd":
        weights = channels_in * channels_out * 3 * 3 + channels_out * channels_out * 3 * 3
        intermediate = output_elements
        residual_route = input_elements
        return float(
            4
            * (
                input_elements
                + weights
                + 2 * intermediate
                + output_elements
                + residual_route
            )
        )
    if kind == "skip_proj":
        return float(4 * (input_elements + channels_in * channels_out + output_elements))
    return float(4 * 3 * output_elements)


def _yolo_conv_ops(index: int, padded: bool = False) -> float:
    _, _, channels_in, channels_out, height, width, kernel = YOLO_CONVS[int(index)]
    if padded:
        channels_in = _round_up(channels_in)
        channels_out = _round_up(channels_out)
    return float(2 * height * width * channels_in * channels_out * kernel * kernel)


def _yolo_conv_memory_bytes(index: int, element_bytes: int, padded: bool) -> float:
    _, _, channels_in, channels_out, height, width, kernel = YOLO_CONVS[int(index)]
    if padded:
        channels_in = _round_up(channels_in)
        channels_out = _round_up(channels_out)
    input_bytes = channels_in * height * width * element_bytes
    weight_bytes = channels_in * channels_out * kernel * kernel * element_bytes
    output_bytes = channels_out * height * width * element_bytes
    return float(input_bytes + weight_bytes + output_bytes)


def _yolo_cpu_bucket(index: int) -> str:
    _, _, channels_in, channels_out, height, _, kernel = YOLO_CONVS[int(index)]
    if int(index) == 0 or height >= 104:
        return "stem_large_input_conv"
    if kernel == 1:
        return "skip_proj_1x1_conv"
    if channels_in == channels_out:
        return "residual_3x3_conv"
    return "residual_3x3_conv"


def _yolo_vta_bucket(index: int) -> str:
    _, _, channels_in, channels_out, _, _, kernel = YOLO_CONVS[int(index)]
    if kernel == 1:
        return "conv1x1"
    if max(channels_in, channels_out) <= 128:
        return "conv3x3_c_small"
    return "conv3x3_c_large"


def _yolo_dma_bucket(index: int) -> str:
    _, _, channels_in, channels_out, height, width, _ = YOLO_CONVS[int(index)]
    physical_bytes = _round_up(channels_out) * height * width
    if channels_in % 16 or channels_out % 16:
        return "padded_load"
    if physical_bytes < 64 * 1024:
        return "small_tensor_load_store"
    return "large_contiguous_load_store"


def _yolo_stage_conv_lists(
    row: Mapping[str, Any], devices: Sequence[str]
) -> List[List[int]]:
    plan = row.get("stage_plan") or []
    if len(plan) == len(devices):
        return [[as_int(item) for item in stage.get("convs", [])] for stage in plan]
    cpu_lists = row.get("cpu_stage_convs") or []
    if not cpu_lists:
        cpu_lists = [row.get("cpu_prefix_convs") or [], row.get("cpu_tail_convs") or []]
    vta_lists = row.get("vta_stage_convs") or []
    if not vta_lists:
        vta_lists = [row.get("vta_convs") or []]
    cpu_index = 0
    vta_index = 0
    result = []
    for device in devices:
        if device == "cpu":
            values = cpu_lists[cpu_index] if cpu_index < len(cpu_lists) else []
            cpu_index += 1
        else:
            values = vta_lists[vta_index] if vta_index < len(vta_lists) else []
            vta_index += 1
        result.append([as_int(item) for item in values])
    return result


def _yolo_calibrated_stages(
    row: Mapping[str, Any], devices: Sequence[str], calibration: Mapping[str, Any]
) -> List[StageService]:
    conv_lists = _yolo_stage_conv_lists(row, devices)
    measured_ms = [as_float(item) for item in row.get("stage_times_ms", []) or []]
    runtime = row.get("runtime_config") or {}
    stage_overheads = calibration.get("stage_overheads") or {}
    stages = []
    for index, (device, convs) in enumerate(zip(devices, conv_lists)):
        if device == "cpu":
            threads = max(
                1,
                as_int(
                    runtime.get("stage0_threads" if index == 0 else "stage2_threads"),
                    3 if index == 0 else 4,
                ),
            )
            compute_ms = 0.0
            memory_bytes = 0.0
            for conv_index in convs:
                bucket = _yolo_cpu_bucket(conv_index)
                gops = as_float(calibration["cpu_eff_gops"][bucket].get(str(threads)))
                if gops <= 0.0:
                    raise RuntimeError(
                        "missing YOLO CPU calibration bucket {} thread {}".format(
                            bucket, threads
                        )
                    )
                compute_ms += _yolo_conv_ops(conv_index) / gops / 1.0e6
                memory_bytes += _yolo_conv_memory_bytes(conv_index, 4, False)
            bandwidth = as_float(
                calibration["cpu_memory_bandwidth_GBps"].get(str(threads))
            )
            if bandwidth <= 0.0:
                raise RuntimeError("missing CPU memory bandwidth thread {}".format(threads))
            overhead = stage_overheads.get("cpu") or {}
            stages.append(
                StageService(
                    name="stage{}_cpu".format(index),
                    device="cpu",
                    threads=threads,
                    compute_ms=compute_ms,
                    memory_ms=memory_bytes / bandwidth / 1.0e6,
                    launch_ms=as_float(overhead.get("set_input_ms"))
                    + as_float(overhead.get("get_output_ms")),
                    measured_ms=measured_ms[index] if index < len(measured_ms) else None,
                )
            )
            continue

        compute_ms = 0.0
        load_ms = 0.0
        store_ms = 0.0
        overlap_values = []
        for conv_index in convs:
            bucket = _yolo_vta_bucket(conv_index)
            entry = calibration["vta_eff_gops"][bucket]
            gops = as_float(entry.get("gops"))
            if gops <= 0.0:
                raise RuntimeError("missing YOLO VTA calibration bucket {}".format(bucket))
            compute_ms += _yolo_conv_ops(conv_index, padded=True) / gops / 1.0e6
            dma_bucket = _yolo_dma_bucket(conv_index)
            dma_entry = calibration["dma"][dma_bucket]
            _, _, channels_in, channels_out, height, width, kernel = YOLO_CONVS[conv_index]
            channels_in = _round_up(channels_in)
            channels_out = _round_up(channels_out)
            load_bytes = channels_in * height * width + channels_in * channels_out * kernel * kernel
            store_bytes = channels_out * height * width
            load_ms += load_bytes / as_float(dma_entry.get("ps_pl_load_bw_GBps")) / 1.0e6
            store_ms += store_bytes / as_float(dma_entry.get("ps_pl_store_bw_GBps")) / 1.0e6
            overlap_values.append(as_float(entry.get("overlap_factor"), 0.0))
        overhead = stage_overheads.get("vta") or {}
        stages.append(
            StageService(
                name="stage{}_vta".format(index),
                device="vta",
                threads=1,
                compute_ms=compute_ms,
                load_ms=load_ms,
                store_ms=store_ms,
                overlap_factor=(sum(overlap_values) / len(overlap_values) if overlap_values else 0.0),
                launch_ms=as_float(overhead.get("runner_submit_ms"))
                + as_float(overhead.get("sync_wait_ms")),
                bridge_ms=as_float(overhead.get("bridge_pack_ms")),
                measured_ms=measured_ms[index] if index < len(measured_ms) else None,
            )
        )
    return stages


def _resnet_group(row: Mapping[str, Any]) -> str:
    islands = parse_json_value(row.get("vta_islands"), []) or []
    if not islands:
        return "resnet:no_vta"
    first = min(as_int(item.get("start_idx")) for item in islands)
    last = max(as_int(item.get("end_idx")) for item in islands)
    return "resnet:i{}_{}_{}".format(len(islands), first, last)


def _physical_spill_ms(payload: Mapping[str, Any]) -> float:
    """Read only lowering/profiler spill time with physical units.

    Historical static_tile_spill_penalty and SRAM risk fields are heuristic
    scores, not milliseconds, and must not enter the fixed-tile model.
    """

    for key in ("physical_spill_ms", "lowered_spill_ms", "profiled_spill_ms"):
        if key in payload:
            return max(0.0, as_float(payload.get(key)))
    return 0.0


def _physical_spill_ratio(payload: Mapping[str, Any]) -> float:
    spill_bytes = max(0.0, as_float(payload.get("physical_spill_bytes")))
    compulsory_bytes = max(0.0, as_float(payload.get("compulsory_external_bytes")))
    return spill_bytes / compulsory_bytes if compulsory_bytes > 0.0 else 0.0


def _resnet_score_input_provenance(
    source_path: str,
    calibration: Mapping[str, Any],
    measured_calibration: bool,
) -> Dict[str, Any]:
    """Record the origin of fields used by pre-measurement candidate scores."""

    if measured_calibration:
        service_parameters = {
            "kind": "independent_hardware_profile",
            "source": str(calibration.get("source") or "measured_calibration"),
        }
    else:
        cost_model_path = Path(source_path).parent / "cost_model.json"
        cost_model = load_json(cost_model_path) if cost_model_path.exists() else {}
        cost_source = str(cost_model.get("source") or "legacy_embedded_static_estimate")
        service_parameters = {
            "kind": (
                "legacy_default_static"
                if "default_static" in cost_source
                else "compiler_static"
            ),
            "source": cost_source,
            "source_path": str(cost_model_path) if cost_model_path.exists() else "",
        }
    return {
        "workload_features": {
            "kind": "compiler_static",
            "source": "score_components_json_before_candidate_execution",
        },
        "service_parameters": service_parameters,
        "candidate_outcome": {
            "kind": "candidate_measurement",
            "use": "retrospective_evaluation_only",
        },
    }


def resnet_record_from_row(
    row: Mapping[str, Any],
    source_path: str,
    calibration: Optional[Mapping[str, Any]] = None,
) -> PipelineRecord:
    components = parse_json_value(row.get("score_components_json"), {}) or {}
    measured_lookup = _stage_measured_lookup(row)
    stages = []
    stage_static_parts = []
    measured_calibration = calibration_is_measured(calibration)
    calibration = calibration or {}
    for raw in components.get("stages", []) or []:
        device = str(raw.get("device"))
        threads = max(1, as_int(raw.get("threads"), 1))
        compute_ms = as_float(raw.get("static_run_ms_est"))
        dma_ms = as_float(raw.get("static_vta_dma_ms_est"))
        load_ms = 0.0
        store_ms = 0.0
        overlap_factor = 1.0
        memory_ms = 0.0
        launch_ms = 0.0
        bridge_ms = 0.0
        if device == "cpu" and measured_calibration:
            compute_ms = 0.0
            for part in raw.get("static_compute_parts", []) or []:
                bucket = str(part.get("bucket"))
                fallback = as_float(part.get("gops"), 1.0)
                gops = _cpu_gops(calibration, bucket, threads, fallback)
                compute_ms += as_float(part.get("ops")) / gops / 1.0e6
            bandwidth = as_float(
                (calibration.get("cpu_memory_bandwidth_GBps") or {}).get(str(threads))
            )
            if bandwidth <= 0.0:
                raise RuntimeError("missing ResNet CPU memory bandwidth thread {}".format(threads))
            memory_bytes = sum(
                _resnet_unit_memory_bytes(str(name)) for name in raw.get("unit_names", []) or []
            )
            memory_ms = memory_bytes / bandwidth / 1.0e6
            overhead = (calibration.get("stage_overheads") or {}).get("cpu") or {}
            launch_ms = as_float(overhead.get("set_input_ms")) + as_float(
                overhead.get("get_output_ms")
            )
        elif device == "vta":
            compute_ms = as_float(raw.get("static_vta_compute_ms_est"), compute_ms)
            if measured_calibration:
                compute_ms = 0.0
                for part in raw.get("static_compute_parts", []) or []:
                    bucket = str(part.get("bucket"))
                    fallback = as_float(part.get("gops"), 1.0)
                    compute_ms += as_float(part.get("ops")) / _vta_gops(
                        calibration, bucket, fallback
                    ) / 1.0e6
                load_ms, store_ms = _dma_channels_from_parts(
                    raw.get("static_dma_parts", []) or [], calibration
                )
                dma_ms = load_ms + store_ms
                overlap_values = [
                    as_float(
                        (calibration.get("vta_eff_gops") or {})
                        .get(str(part.get("bucket")), {})
                        .get("overlap_factor"),
                        0.0,
                    )
                    for part in raw.get("static_compute_parts", []) or []
                ]
                overlap_factor = (
                    sum(overlap_values) / len(overlap_values) if overlap_values else 0.0
                )
                overhead = (calibration.get("stage_overheads") or {}).get("vta") or {}
                launch_ms = as_float(overhead.get("runner_submit_ms")) + as_float(
                    overhead.get("sync_wait_ms")
                )
                bridge_ms = as_float(overhead.get("bridge_pack_ms"))
            else:
                launch_ms = as_float(raw.get("static_runner_submit_ms_est")) + as_float(
                    raw.get("static_sync_wait_ms_est")
                )
                bridge_ms = as_float(raw.get("static_bridge_pack_ms_est")) + as_float(
                    raw.get("static_bridge_unpack_ms_est")
                )
        elif device == "cpu":
            launch_ms = as_float(raw.get("static_set_input_ms_est")) + as_float(
                raw.get("static_get_output_ms_est")
            )
        stages.append(
            StageService(
                name=str(raw.get("name", "stage{}".format(len(stages)))),
                device=device,
                threads=threads,
                compute_ms=max(0.0, compute_ms),
                memory_ms=max(0.0, memory_ms),
                dma_ms=max(0.0, dma_ms),
                load_ms=max(0.0, load_ms),
                store_ms=max(0.0, store_ms),
                overlap_factor=min(1.0, max(0.0, overlap_factor)),
                spill_ms=_physical_spill_ms(raw),
                launch_ms=max(0.0, launch_ms),
                bridge_ms=max(0.0, bridge_ms),
                measured_ms=measured_lookup.get(str(raw.get("name"))),
                core_demand_ms=(
                    max(0.0, as_float(raw.get("core_demand_ms")))
                    if raw.get("core_demand_ms") not in (None, "")
                    else None
                ),
                core_demand_source=str(raw.get("core_demand_source") or ""),
            )
        )
        stage_static_parts.append(
            {
                "name": str(raw.get("name", "stage{}".format(len(stages) - 1))),
                "device": device,
                "unit_names": list(raw.get("unit_names", []) or []),
                "compute_parts": list(raw.get("static_compute_parts", []) or []),
                "dma_parts": list(raw.get("static_dma_parts", []) or []),
            }
        )
    fps = as_float(row.get("pipeline_throughput_fps"))
    spill_ratio = _physical_spill_ratio(row)
    direct_profile = _resnet_direct_communication_profile(row, source_path)
    measured_stage_run_ms = [
        as_float(row.get("stage{}_run_ms".format(index))) for index in range(len(stages))
    ]
    output_dir = _resolve_output_dir(row, source_path)
    runtime_manifest = {}
    if output_dir is not None and (output_dir / "manifest.json").exists():
        runtime_manifest = load_json(output_dir / "manifest.json")
    affinity = runtime_manifest.get("stage_cpu_affinity") or {}
    runtime_provenance = {
        "queue_depth": max(1, as_int(row.get("queue_depth"), 2)),
        "poll_sleep_ns": as_int(row.get("poll_sleep_ns"), -1),
        "cpu_affinity_policy": affinity.get(
            "policy", "legacy_default_threadpool_affinity_unrecorded"
        ),
        "stage_cpu_affinity": affinity,
        "manifest_path": (
            str(output_dir / "manifest.json") if runtime_manifest else ""
        ),
        "current_source_fingerprint_available": bool(
            runtime_manifest.get("hardware_fingerprint_artifact_sha256")
        ),
    }
    return PipelineRecord(
        model="resnet18",
        candidate_id=str(row.get("candidate_id") or row.get("scheme_name")),
        source_path=str(source_path),
        group_id=_resnet_group(row),
        measured_cycle_ms=1000.0 / fps if fps > 0.0 else None,
        queue_depth=max(1, as_int(row.get("queue_depth"), 2)),
        stages=stages,
        boundary_ms=(
            as_float(row.get("boundary_bytes"))
            / max(as_float(calibration.get("boundary_bw_GBps")), 1.0e-9)
            / 1.0e6
            if measured_calibration
            else as_float(
                components.get("boundary_penalty_ms"),
                row.get("boundary_penalty_ms_est", 0.0),
            )
        ),
        fragmentation_ms=as_float(
            components.get("dma_fragmentation_penalty_ms"),
            row.get("dma_fragmentation_penalty_ms_est", 0.0),
        ),
        spill_ratio=spill_ratio,
        avg_bytes_per_call=as_float(row.get("dma_avg_bytes_per_call")),
        effective_segment_count=len(stages),
        vta_island_count=as_int(row.get("vta_island_count")),
        small_island_count=sum(
            1
            for item in parse_json_value(row.get("vta_islands"), []) or []
            if as_int(item.get("end_idx")) - as_int(item.get("start_idx")) + 1 <= 2
        ),
        legacy_score_ms=as_float(row.get("static_score_ms")),
        correctness_passed=bool(row.get("passes_correctness_gate", True)),
        metadata={
            "thread_config": row.get("thread_config", row.get("stage_runtime_threads", "")),
            "stage_devices": row.get("stage_devices", ""),
            "boundary_bytes": as_int(row.get("boundary_bytes")),
            "selection_bucket": row.get("selection_bucket", ""),
            "calibration_source": calibration.get("source", "historical_static"),
            "score_input_provenance": _resnet_score_input_provenance(
                source_path,
                calibration,
                measured_calibration,
            ),
            "stage_static_parts": stage_static_parts,
            "measured_stage_run_ms": measured_stage_run_ms,
            "direct_communication_profile": direct_profile,
            "runtime_provenance": runtime_provenance,
        },
    )


def _parse_devices(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [item for item in str(value or "").split("/") if item]


def _yolo_group(row: Mapping[str, Any]) -> str:
    islands = row.get("islands") or []
    if islands:
        first = islands[0].get("start_name", islands[0].get("start_point_idx", "x"))
        last = islands[-1].get("end_name", islands[-1].get("end_point_idx", "x"))
        return "yolo:i{}_{}_{}".format(len(islands), first, last)
    return "yolo:i1_{}_{}".format(row.get("start_name", "x"), row.get("end_name", "x"))


def _yolo_static_stage_ms(row: Mapping[str, Any], devices: Sequence[str]) -> List[float]:
    cpu_values = list(row.get("stage_cpu_ms_est") or [])
    vta_values = list(row.get("stage_vta_ms_est") or [])
    if not cpu_values and not vta_values:
        cpu_values = [as_float(row.get("stage0_ms_est")), as_float(row.get("stage2_ms_est"))]
        vta_values = [as_float(row.get("stage1_vta_ms_est"))]
    cpu_index = 0
    vta_index = 0
    values = []
    for device in devices:
        if device == "cpu":
            values.append(as_float(cpu_values[cpu_index]) if cpu_index < len(cpu_values) else 0.0)
            cpu_index += 1
        else:
            values.append(as_float(vta_values[vta_index]) if vta_index < len(vta_values) else 0.0)
            vta_index += 1
    return values


def yolo_record_from_row(
    row: Mapping[str, Any],
    source_path: str,
    calibration: Optional[Mapping[str, Any]] = None,
) -> PipelineRecord:
    devices = _parse_devices(row.get("stage_devices"))
    runtime = row.get("runtime_config") or {}
    measured_calibration = calibration_is_measured(calibration)
    if measured_calibration:
        stages = _yolo_calibrated_stages(row, devices, calibration or {})
    else:
        static_ms = _yolo_static_stage_ms(row, devices)
        measured_ms = [as_float(item) for item in row.get("stage_times_ms", []) or []]
        vta_count = max(1, sum(1 for item in devices if item == "vta"))
        total_vta_dma = as_float(row.get("vta_dma_ms"))
        stages = []
        for index, (device, service_ms) in enumerate(zip(devices, static_ms)):
            if device == "cpu":
                threads = max(
                    1,
                    as_int(
                        runtime.get("stage0_threads" if index == 0 else "stage2_threads"),
                        3 if index == 0 else 4,
                    ),
                )
                compute_ms = service_ms
                dma_ms = 0.0
            else:
                threads = 1
                dma_ms = total_vta_dma / vta_count
                compute_ms = max(0.0, service_ms - dma_ms)
            stages.append(
                StageService(
                    name="stage{}_{}".format(index, device),
                    device=device,
                    threads=threads,
                    compute_ms=compute_ms,
                    dma_ms=dma_ms,
                    measured_ms=measured_ms[index] if index < len(measured_ms) else None,
                )
            )
    fps = as_float(row.get("pipeline_throughput_fps"), row.get("throughput_fps", 0.0))
    island_count = as_int(row.get("island_count"), sum(1 for item in devices if item == "vta"))
    return PipelineRecord(
        model="yolov3_tiny",
        candidate_id=str(row.get("candidate_id")),
        source_path=str(source_path),
        group_id=_yolo_group(row),
        measured_cycle_ms=as_float(row.get("measured_cycle_ms"), 1000.0 / fps if fps else 0.0)
        or None,
        queue_depth=max(1, as_int(runtime.get("queue_depth"), 2)),
        stages=stages,
        boundary_ms=(
            as_float(row.get("boundary_bytes_est"))
            / max(as_float((calibration or {}).get("boundary_bw_GBps")), 1.0e-9)
            / 1.0e6
            if measured_calibration
            else as_float(row.get("boundary_ms"))
        ),
        fragmentation_ms=as_float(row.get("dma_fragmentation_ms")),
        spill_ratio=_physical_spill_ratio(row),
        avg_bytes_per_call=(
            as_float(row.get("boundary_bytes_est"))
            / max(1, 2 * max(1, as_int(row.get("effective_segment_count"), len(stages)) - 1))
        ),
        effective_segment_count=as_int(row.get("effective_segment_count"), len(stages)),
        vta_island_count=island_count,
        small_island_count=as_int(row.get("small_island_count")),
        legacy_score_ms=as_float(row.get("predicted_cycle_ms")),
        correctness_passed=bool(row.get("passes_correctness_gate", False)) and bool(
            row.get("raw_sanity_passed", True)
        ),
        metadata={
            "stage_devices": "/".join(devices),
            "boundary_bytes": as_int(row.get("boundary_bytes_est")),
            "runtime_config": runtime,
            "detection_gate_passed": bool(row.get("detection_gate_passed", False)),
            "hardware_service_source": (
                "measured_calibration" if measured_calibration else "legacy_static_fallback"
            ),
            "stage_conv_indices": _yolo_stage_conv_lists(row, devices),
        },
    )


def resnet_summary_paths(roots: Sequence[str] = DEFAULT_RESNET_ROOTS) -> List[Path]:
    paths = []
    for root in roots:
        paths.extend(Path(item) for item in glob.glob(str(Path(root) / "batch[0-9][0-9][0-9]" / "summary.json")))
    return sorted(set(paths))


def collect_resnet_records(
    roots: Sequence[str] = DEFAULT_RESNET_ROOTS,
    calibration: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[PipelineRecord], List[Path]]:
    paths = resnet_summary_paths(roots)
    records = []
    seen = set()
    for path in paths:
        for row in load_json(path).get("rows", []) or []:
            fps = as_float(row.get("pipeline_throughput_fps"))
            candidate_id = str(row.get("candidate_id") or row.get("scheme_name"))
            if (
                row.get("status") != "ok"
                or row.get("run_kind") != "default"
                or fps <= 0.0
                or candidate_id in seen
            ):
                continue
            record = resnet_record_from_row(row, str(path), calibration)
            if record.correctness_passed:
                records.append(record)
                seen.add(candidate_id)
    return records, paths


def collect_controlled_resnet_records(
    summary_path: str = DEFAULT_CONTROLLED_RESNET_SUMMARY,
    historical_records: Optional[Sequence[PipelineRecord]] = None,
    calibration: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[PipelineRecord], List[Path]]:
    path = Path(summary_path)
    if not path.exists():
        return [], []
    historical = list(historical_records or collect_resnet_records(calibration=calibration)[0])
    lookup = {item.candidate_id: item for item in historical}
    records = []
    for row in load_json(path).get("rows", []) or []:
        if not bool(row.get("publication_valid")):
            continue
        candidate_id = str(row.get("candidate_id") or "")
        base = lookup.get(candidate_id)
        cycle_ms = as_float(row.get("cycle_median_ms"))
        if base is None or cycle_ms <= 0.0:
            continue
        record = PipelineRecord.from_dict(base.to_dict())
        thread_values = [
            int(item)
            for item in str(row.get("stage_threads") or "").split(",")
            if item
        ]
        if len(thread_values) != len(record.stages):
            raise RuntimeError(
                "controlled thread vector does not match stages for {}".format(
                    row.get("config_id")
                )
            )
        static_parts = record.metadata.get("stage_static_parts", []) or []
        for stage_index, (stage, threads) in enumerate(zip(record.stages, thread_values)):
            stage.threads = max(1, int(threads))
            if stage.device != "cpu" or not calibration_is_measured(calibration):
                continue
            raw = static_parts[stage_index] if stage_index < len(static_parts) else {}
            stage.compute_ms = sum(
                as_float(part.get("ops"))
                / _cpu_gops(
                    calibration or {},
                    str(part.get("bucket")),
                    stage.threads,
                    as_float(part.get("gops"), 1.0),
                )
                / 1.0e6
                for part in raw.get("compute_parts", []) or []
            )
            bandwidth = as_float(
                ((calibration or {}).get("cpu_memory_bandwidth_GBps") or {}).get(
                    str(stage.threads)
                )
            )
            if bandwidth <= 0.0:
                raise RuntimeError(
                    "controlled ResNet CPU memory bandwidth thread {} is missing".format(
                        stage.threads
                    )
                )
            memory_bytes = sum(
                _resnet_unit_memory_bytes(str(name))
                for name in raw.get("unit_names", []) or []
            )
            stage.memory_ms = memory_bytes / bandwidth / 1.0e6
        measured_stages = parse_json_value(row.get("stage_median_ms_json"), []) or []
        if measured_stages:
            if len(measured_stages) != len(record.stages):
                raise RuntimeError(
                    "controlled measured stage vector does not match {}".format(
                        row.get("config_id")
                    )
                )
            for stage, measured_ms in zip(record.stages, measured_stages):
                stage.measured_ms = as_float(measured_ms)
        core_demands = parse_json_value(row.get("stage_core_demand_ms_json"), []) or []
        if core_demands:
            if len(core_demands) != len(record.stages):
                raise RuntimeError(
                    "controlled core-demand vector does not match {}".format(
                        row.get("config_id")
                    )
                )
            for stage, demand_ms in zip(record.stages, core_demands):
                if stage.device == "cpu":
                    stage.core_demand_ms = max(0.0, as_float(demand_ms))
                    stage.core_demand_source = "controlled_process_cpu_time"
        record.candidate_id = str(row.get("config_id"))
        record.source_path = str(path)
        record.group_id = "controlled:{}".format(candidate_id)
        record.measured_cycle_ms = cycle_ms
        record.queue_depth = max(1, as_int(row.get("queue_depth"), 2))
        record.correctness_passed = True
        record.metadata = {
            **record.metadata,
            "identification_only": True,
            "base_candidate_id": candidate_id,
            "cpu_thread_policy": row.get("cpu_thread_policy"),
            "successful_sessions": as_int(row.get("successful_sessions")),
            "cycle_mad_ms": as_float(row.get("cycle_mad_ms")),
        }
        records.append(record)
    return records, [path]


def collect_yolo_records(
    summaries: Sequence[str] = DEFAULT_YOLO_SUMMARIES,
    calibration: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[PipelineRecord], List[Path]]:
    paths = [Path(item) for item in summaries]
    records = []
    seen = set()
    for path in paths:
        for row in load_json(path).get("rows", []) or []:
            candidate_id = str(row.get("candidate_id"))
            fps = as_float(row.get("pipeline_throughput_fps"), row.get("throughput_fps", 0.0))
            if (
                row.get("mode") != "pipeline"
                or row.get("status") != "ok"
                or fps <= 0.0
                or candidate_id in seen
            ):
                continue
            record = yolo_record_from_row(row, str(path), calibration)
            if record.correctness_passed:
                records.append(record)
                seen.add(candidate_id)
    return records, paths


def yolo_record_from_candidate(
    candidate: Mapping[str, Any],
    source_path: str = "enumerated",
    calibration: Optional[Mapping[str, Any]] = None,
    *,
    legacy_analysis: bool = False,
) -> PipelineRecord:
    if not legacy_analysis:
        raise RuntimeError(
            "yolo_record_from_candidate is a legacy aggregate estimator; "
            "new ranking must lower a PartitionWorkload and use portable_record_from_workload"
        )
    devices = _parse_devices(candidate.get("stage_devices"))
    if calibration_is_measured(calibration):
        row = dict(candidate)
        row.setdefault("runtime_config", candidate.get("runtime_config") or {})
        record = yolo_record_from_row(row, source_path, calibration)
        record.measured_cycle_ms = None
        record.metadata["prediction_mode"] = "legacy_analysis"
        return record
    static_ms = _yolo_static_stage_ms(candidate, devices)
    vta_count = max(1, sum(1 for device in devices if device == "vta"))
    total_dma = as_float(candidate.get("vta_dma_ms"))
    stages = []
    for index, (device, service_ms) in enumerate(zip(devices, static_ms)):
        dma_ms = total_dma / vta_count if device == "vta" else 0.0
        stages.append(
            StageService(
                name="stage{}_{}".format(index, device),
                device=device,
                threads=3 if index == 0 and device == "cpu" else (4 if device == "cpu" else 1),
                compute_ms=max(0.0, service_ms - dma_ms),
                dma_ms=dma_ms,
            )
        )
    return PipelineRecord(
        model="yolov3_tiny",
        candidate_id=str(candidate.get("candidate_id")),
        source_path=source_path,
        group_id=_yolo_group(candidate),
        measured_cycle_ms=None,
        queue_depth=2,
        stages=stages,
        boundary_ms=as_float(candidate.get("boundary_ms")),
        fragmentation_ms=as_float(candidate.get("dma_fragmentation_ms")),
        spill_ratio=_physical_spill_ratio(candidate),
        avg_bytes_per_call=as_float(candidate.get("boundary_bytes_est"))
        / max(1, 2 * max(1, len(stages) - 1)),
        effective_segment_count=as_int(candidate.get("effective_segment_count"), len(stages)),
        vta_island_count=as_int(candidate.get("island_count"), vta_count),
        small_island_count=as_int(candidate.get("small_island_count")),
        legacy_score_ms=as_float(candidate.get("predicted_cycle_ms")),
        correctness_passed=True,
        metadata={
            "stage_devices": "/".join(devices),
            "boundary_bytes": as_int(candidate.get("boundary_bytes_est")),
            "prediction_mode": "legacy_analysis",
        },
    )
