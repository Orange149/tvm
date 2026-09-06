#!/usr/bin/env python3
"""Run and aggregate the minimal native V1-P2 profile session."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

from freeze_cpu_vta_pipeline_v1 import PROTOCOL_ID, repo_root, seal_artifact
from deploy_classification_stage_pipeline_native import CAT_EQUIVALENT_TOP1
from split_resnet18_stages import build_resnet18_unit_compute_metadata


OUTPUT_ROOT = repo_root() / "vta/tutorials/frontend/report_out/resource_aware_maxplus"
SESSION_ROOT = OUTPUT_ROOT / "v1_p2_session1"
REMOTE_ROOT = "/var/volatile/ramps_v1_p2"
UDMABUF_BYTES = 201326592
SSH_OPTIONS = [
    "HostKeyAlgorithms=+ssh-rsa",
    "PubkeyAcceptedAlgorithms=+ssh-rsa",
]
STAGE_CASES = [
    *[
        {
            "case_id": "three_stage_e_t{}".format(threads),
            "scheme": "three_stage_e",
            "threads": threads,
            "run_serial_before_pipeline": False,
            "role": "fit_thread_sweep",
        }
        for threads in (1, 2, 3, 4)
    ],
    {
        "case_id": "three_stage_a_t2",
        "scheme": "three_stage_a",
        "threads": 2,
        "run_serial_before_pipeline": False,
        "role": "fit_shape_anchor",
    },
    {
        "case_id": "three_stage_b_t2_matched",
        "scheme": "three_stage_b",
        "threads": 2,
        "run_serial_before_pipeline": True,
        "role": "grouped_holdout_matched_control",
    },
]
MEMORY_CASES = [
    {
        "case_id": "ddr_{}_t{}".format(operation, threads),
        "operation": operation,
        "threads": threads,
    }
    for operation in ("read", "write", "copy")
    for threads in (1, 2, 3, 4)
]


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_logged(command, ledger, case_id, kind):
    started = time.time()
    started_monotonic = time.monotonic()
    record = {
        "case_id": case_id,
        "kind": kind,
        "command": [str(item) for item in command],
        "started_unix_s": started,
        "status": "running",
    }
    ledger.append(record)
    write_json(SESSION_ROOT / "progress.json", {"cases": ledger})
    try:
        subprocess.run([str(item) for item in command], check=True)
    except Exception as error:
        record["status"] = "failed"
        record["error"] = repr(error)
        raise
    finally:
        record["elapsed_s"] = time.monotonic() - started_monotonic
        if record["status"] == "running":
            record["status"] = "completed"
        write_json(SESSION_ROOT / "progress.json", {"cases": ledger})


def stage_case_complete(case):
    case_dir = SESSION_ROOT / "stage" / case["case_id"]
    serial = case_dir / "stage_serial_result.jsonl"
    if not serial.is_file() or len(read_jsonl(serial)) != 12:
        return False
    if case["run_serial_before_pipeline"]:
        pipeline = case_dir / "native_result.jsonl"
        return pipeline.is_file() and len(read_jsonl(pipeline)) == 12
    return True


def memory_case_complete(case):
    output = SESSION_ROOT / "memory" / (case["case_id"] + ".jsonl")
    return output.is_file() and len(read_jsonl(output)) == 8


def independent_mxnet_reference():
    import mxnet as mx
    import numpy as np
    from PIL import Image
    from mxnet.gluon.model_zoo import vision

    image_path = repo_root() / "build_axu_aarch64/cat.png"
    image = Image.open(image_path).resize((224, 224)).convert("RGB")
    array = np.asarray(image).astype("float32")
    array -= np.asarray([123.0, 117.0, 104.0], dtype="float32")
    array /= np.asarray([58.395, 57.12, 57.375], dtype="float32")
    array = array.transpose((2, 0, 1))[np.newaxis, :]
    model = vision.get_model("resnet18_v1", pretrained=True)
    top1 = int(np.argmax(model(mx.nd.array(array)).asnumpy()[0]))
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p2_independent_reference",
            "model": "mxnet.gluon.model_zoo.resnet18_v1_pretrained",
            "image": str(image_path),
            "image_sha256": file_sha256(image_path),
            "preprocess": "resize224_rgb_mean_std_chw",
            "top1": top1,
        }
    )


def frozen_plan(memory_binary):
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p2_frozen_plan",
            "protocol_id": PROTOCOL_ID,
            "session_count": 1,
            "native_stage_case_count": 7,
            "native_stage_invocation_count": len(STAGE_CASES),
            "memory_case_count": len(MEMORY_CASES),
            "total_case_count": 19,
            "stage_runs_per_mode": 12,
            "stage_scored_runs_per_mode": 10,
            "memory_runs_per_case": 8,
            "memory_warmup_per_case": 2,
            "memory_inner_repeats": 2,
            "memory_working_set_bytes": 64 * 1024 * 1024,
            "stage_cases": STAGE_CASES,
            "memory_cases": MEMORY_CASES,
            "memory_binary_sha256": file_sha256(memory_binary),
            "performance_path": "native_static_package_only",
            "rpc_performance_allowed": False,
            "reference": "independent MXNet pretrained model with identical input preprocessing",
            "stop_rule": "stop on correctness, affinity, or ownership failure",
        }
    )


def measure(args):
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    plan = frozen_plan(args.memory_binary)
    write_json(OUTPUT_ROOT / "v1_p2_frozen_plan.json", plan)
    progress_path = SESSION_ROOT / "progress.json"
    ledger = json.loads(progress_path.read_text()).get("cases", []) if progress_path.is_file() else []
    ssh_base = ["ssh"]
    scp_base = ["scp"]
    for option in SSH_OPTIONS:
        ssh_base.extend(["-o", option])
        scp_base.extend(["-o", option])
    run_logged(
        ssh_base
        + [
            args.board,
            (
                "test $(cat /sys/class/u-dma-buf/udmabuf0/size) = {} && "
                "test -c /dev/udmabuf0 && "
                "test $(cat /sys/class/fpga_manager/fpga0/state) = operating"
            ).format(UDMABUF_BYTES),
        ],
        ledger,
        "native_vta_preflight",
        "setup",
    )
    deploy = repo_root() / "vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py"
    for case in STAGE_CASES:
        if stage_case_complete(case):
            print("[RESUME] stage case already complete:", case["case_id"])
            continue
        case_dir = SESSION_ROOT / "stage" / case["case_id"]
        package = (
            OUTPUT_ROOT / "v1_p1_packages" / case["scheme"] / "package"
        )
        command = [
            sys.executable,
            deploy,
            "--board",
            args.board,
            "--remote-dir",
            "{}/{}".format(REMOTE_ROOT, case["case_id"]),
            "--remote-min-free-mb",
            "128",
            "--deploy-mode",
            "sync",
            "--candidate-id",
            case["case_id"],
            "--runs",
            "12",
            "--queue-depth",
            "2",
            "--stage-runtime-num-threads",
            "{0},1,{0}".format(case["threads"]),
            "--runner-output-mode",
            "raw_all_stages",
            "--fetch-results-dir",
            case_dir,
            "--vta-runtime-profile-dir",
            "profile",
            "--vta-runtime-profile-events-limit",
            "256",
            "--serial-timeout-s",
            "300",
            "--pipeline-timeout-s",
            "300",
            "--fetch-timeout-s",
            "180",
            "--cleanup-remote-after-run",
            "--reuse-package-dir",
            package,
        ]
        if case["run_serial_before_pipeline"]:
            command.append("--run-serial-before-pipeline")
        else:
            command.append("--serial")
        for option in SSH_OPTIONS:
            command.extend(["--ssh-option", option])
        run_logged(command, ledger, case["case_id"], "native_stage")

    remote_root = REMOTE_ROOT + "/memory"
    run_logged(
        ssh_base + [args.board, "mkdir -p {}".format(remote_root)],
        ledger,
        "memory_prepare_dir",
        "setup",
    )
    run_logged(
        scp_base
        + [args.memory_binary, "{}:{}/ramps_cpu_memory_microbench".format(args.board, remote_root)],
        ledger,
        "memory_upload_binary",
        "setup",
    )
    memory_dir = SESSION_ROOT / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    for case in MEMORY_CASES:
        if memory_case_complete(case):
            print("[RESUME] memory case already complete:", case["case_id"])
            continue
        remote_output = "{}/{}.jsonl".format(remote_root, case["case_id"])
        remote_command = " ".join(
            shlex.quote(str(item))
            for item in [
                "{}/ramps_cpu_memory_microbench".format(remote_root),
                "--operation",
                case["operation"],
                "--case-id",
                case["case_id"],
                "--output",
                remote_output,
                "--threads",
                case["threads"],
                "--streams",
                1,
                "--warmup",
                2,
                "--runs",
                8,
                "--inner-repeats",
                2,
                "--bytes",
                64 * 1024 * 1024,
            ]
        )
        run_logged(
            ssh_base + [args.board, remote_command],
            ledger,
            case["case_id"],
            "native_memory",
        )
        run_logged(
            scp_base + ["{}:{}".format(args.board, remote_output), memory_dir],
            ledger,
            case["case_id"] + "_fetch",
            "fetch",
        )
    write_json(SESSION_ROOT / "measurement_ledger.json", {"cases": ledger})
    run_logged(
        ssh_base + [args.board, "rm -rf -- {}".format(REMOTE_ROOT)],
        ledger,
        "remote_cleanup",
        "cleanup",
    )
    write_json(SESSION_ROOT / "measurement_ledger.json", {"cases": ledger})


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def median(rows, key):
    return float(statistics.median(float(row[key]) for row in rows))


def schema_nbytes(schema):
    dtype_bytes = {"int8": 1, "uint8": 1, "int32": 4, "float32": 4}
    total = 0
    for slot in schema.get("slots", []):
        elements = 1
        for extent in slot["shape"]:
            elements *= int(extent)
        total += elements * dtype_bytes[slot["dtype"]]
    return int(total)


def fit_through_origin(points, feature_key, value_key, coefficient_name):
    denominator = sum(float(item[feature_key]) ** 2 for item in points)
    if denominator <= 0:
        raise RuntimeError("cannot fit {} from zero features".format(coefficient_name))
    coefficient = sum(
        float(item[feature_key]) * float(item[value_key]) for item in points
    ) / denominator
    errors = []
    for item in points:
        actual = float(item[value_key])
        predicted = coefficient * float(item[feature_key])
        errors.append(abs(predicted - actual) / actual if actual > 0 else 0.0)
    return {
        coefficient_name: coefficient,
        "fit_point_count": len(points),
        "fit_mape": sum(errors) / len(errors),
        "fit_max_ape": max(errors),
        "intercept_ms": 0.0,
        "fit_policy": "nonnegative_through_origin_v1",
    }


def median_grouped_points(points, group_keys, feature_key, value_key):
    grouped = {}
    for item in points:
        key = tuple(item[name] for name in group_keys)
        grouped.setdefault(key, []).append(item)
    result = []
    for key, rows in sorted(grouped.items()):
        result.append(
            {
                **{name: value for name, value in zip(group_keys, key)},
                feature_key: float(rows[0][feature_key]),
                value_key: float(statistics.median(row[value_key] for row in rows)),
                "replicate_count": len(rows),
            }
        )
    return result


def normalize_cost_commands(cases):
    """Make legacy timing anomalies explicit without rewriting the raw ledger."""
    normalized = []
    for item in cases:
        record = dict(item)
        elapsed = float(record["elapsed_s"])
        if elapsed < 0:
            record["elapsed_s_raw"] = elapsed
            record["elapsed_s"] = 0.0
            record["timing_anomaly"] = "negative_legacy_elapsed_clamped"
        normalized.append(record)
    return normalized


def raw_signatures(rows):
    return {
        tuple(item["fnv1a64"] for item in row.get("raw_outputs", [])) for row in rows
    }


def summarize_vta_profiler(case_dir, mode):
    path = case_dir / "profile" / mode / "benchmark_totals_status.json"
    status = json.loads(path.read_text())
    return {
        "driver_run_calls": int(status["driver_run_calls"]),
        "driver_timeout_calls": int(status["driver_timeout_calls"]),
        "synchronize_calls": int(status["synchronize_calls"]),
        "load_bytes": int(status["synchronize_load_bytes"]),
        "store_bytes": int(status["synchronize_store_bytes"]),
        "mem_copy_from_host_bytes": int(status["mem_copy_from_host_bytes"]),
        "mem_copy_to_host_bytes": int(status["mem_copy_to_host_bytes"]),
    }


def summarize_stage_case(case):
    case_dir = SESSION_ROOT / "stage" / case["case_id"]
    manifest = json.loads((case_dir / "manifest.json").read_text())
    compute = build_resnet18_unit_compute_metadata(1, 224)
    serial = read_jsonl(case_dir / "stage_serial_result.jsonl")
    serial_scored = serial[2:]
    stage_count = int(serial_scored[0]["stage_count"])
    stages = []
    for index in range(stage_count):
        stage_manifest = manifest["stages"][index]
        logical_gops = sum(
            float(compute[name]["compute_gops_est"])
            for name in stage_manifest["unit_names"]
        )
        stages.append(
            {
                "index": index,
                "device": stage_manifest["device"],
                "unit_names": list(stage_manifest["unit_names"]),
                "logical_compute_gops_est": logical_gops,
                "input_bytes": schema_nbytes(stage_manifest["input_schema"]),
                "output_bytes": schema_nbytes(stage_manifest["output_schema"]),
                "threads": int(case["threads"])
                if stage_manifest["device"] == "cpu"
                else 1,
                "wall_ms": median(serial_scored, "stage{}_ms".format(index)),
                "set_ms": median(serial_scored, "stage{}_set_ms".format(index)),
                "run_ms": median(serial_scored, "stage{}_run_ms".format(index)),
                "get_ms": median(serial_scored, "stage{}_get_ms".format(index)),
                "process_cpu_ms": median(
                    serial_scored, "stage{}_process_cpu_ms".format(index)
                ),
                "set_process_cpu_ms": median(
                    serial_scored, "stage{}_set_process_cpu_ms".format(index)
                ),
                "run_process_cpu_ms": median(
                    serial_scored, "stage{}_run_process_cpu_ms".format(index)
                ),
                "get_process_cpu_ms": median(
                    serial_scored, "stage{}_get_process_cpu_ms".format(index)
                ),
            }
        )
    summary = {
        **case,
        "serial_rows": len(serial),
        "serial_top1_values": sorted({int(row["top1"]) for row in serial}),
        "serial_raw_signature_count": len(raw_signatures(serial)),
        "serial_total_latency_ms": median(serial_scored, "total_latency_ms"),
        "serial_vta_profiler": summarize_vta_profiler(case_dir, "serial"),
        "stage_cpu_affinity": manifest.get("stage_cpu_affinity", {}),
        "stages": stages,
    }
    if case["run_serial_before_pipeline"]:
        pipeline = read_jsonl(case_dir / "native_result.jsonl")
        pipeline_scored = sorted(pipeline[2:], key=lambda row: row["completion_index"])
        end_key = "stage{}_end_ms".format(stage_count - 1)
        completion_span = float(pipeline_scored[-1][end_key]) - float(
            pipeline_scored[0][end_key]
        )
        measured_cycle = completion_span / max(1, len(pipeline_scored) - 1)
        measured_serial_max_stage = max(stage["wall_ms"] for stage in stages)
        summary["pipeline"] = {
            "rows": len(pipeline),
            "top1_values": sorted({int(row["top1"]) for row in pipeline}),
            "raw_signature_count": len(raw_signatures(pipeline)),
            "measured_cycle_ms": measured_cycle,
            "measured_serial_max_stage_ms": measured_serial_max_stage,
            "serial_to_pipeline_residual_diagnostic": (
                measured_cycle - measured_serial_max_stage
            )
            / measured_serial_max_stage,
            "serial_pipeline_raw_signatures_equal": raw_signatures(serial)
            == raw_signatures(pipeline),
            "vta_profiler": summarize_vta_profiler(case_dir, "pipeline"),
        }
    return summary


def aggregate(args):
    plan_path = OUTPUT_ROOT / "v1_p2_frozen_plan.json"
    plan = json.loads(plan_path.read_text())
    stage_summaries = [summarize_stage_case(case) for case in STAGE_CASES]
    reference = independent_mxnet_reference()
    write_json(OUTPUT_ROOT / "v1_p2_reference.json", reference)
    memory = []
    for case in MEMORY_CASES:
        rows = read_jsonl(SESSION_ROOT / "memory" / (case["case_id"] + ".jsonl"))
        memory.append(
            {
                **case,
                "sample_count": len(rows),
                "cpu_affinity_values": sorted(
                    {tuple(row["cpu_affinity"]) for row in rows}
                ),
                "bandwidth_GBps_median": median(rows, "bandwidth_GBps"),
                "wall_ms_median": median(rows, "wall_ms"),
                "process_cpu_ms_median": median(rows, "process_cpu_ms"),
                "correct": all(row["correct"] for row in rows),
            }
        )

    fit = [item for item in stage_summaries if item["role"].startswith("fit")]
    holdout = next(item for item in stage_summaries if item["role"].startswith("grouped"))

    cpu_fit_points = []
    vta_fit_points = []
    boundary_fit_points = []
    for item in fit:
        for stage in item["stages"]:
            point = {
                "group": item["scheme"],
                "segment": tuple(stage["unit_names"]),
                "threads": stage["threads"],
                "logical_compute_gops_est": stage["logical_compute_gops_est"],
                "run_ms": stage["run_ms"],
                "run_process_cpu_ms": stage["run_process_cpu_ms"],
            }
            (cpu_fit_points if stage["device"] == "cpu" else vta_fit_points).append(point)
        boundary_fit_points.extend(
            [
                {
                    "group": item["scheme"],
                    "direction": "cpu_to_vta",
                    "bytes_mib": item["stages"][0]["output_bytes"] / (1024.0 * 1024.0),
                    "service_ms": item["stages"][0]["get_ms"]
                    + item["stages"][1]["set_ms"],
                    "host_core_demand_ms": item["stages"][0]["get_process_cpu_ms"]
                    + item["stages"][1]["set_process_cpu_ms"],
                },
                {
                    "group": item["scheme"],
                    "direction": "vta_to_cpu",
                    "bytes_mib": item["stages"][1]["output_bytes"] / (1024.0 * 1024.0),
                    "service_ms": item["stages"][1]["get_ms"]
                    + item["stages"][2]["set_ms"],
                    "host_core_demand_ms": item["stages"][1]["get_process_cpu_ms"]
                    + item["stages"][2]["set_process_cpu_ms"],
                },
            ]
        )

    cpu_models = {}
    cpu_core_models = {}
    for threads in (1, 2, 3, 4):
        points = [item for item in cpu_fit_points if item["threads"] == threads]
        cpu_models[str(threads)] = fit_through_origin(
            points, "logical_compute_gops_est", "run_ms", "ms_per_logical_gop"
        )
        cpu_core_models[str(threads)] = fit_through_origin(
            points,
            "logical_compute_gops_est",
            "run_process_cpu_ms",
            "core_ms_per_logical_gop",
        )
    distinct_vta_points = median_grouped_points(
        vta_fit_points,
        ("group", "segment"),
        "logical_compute_gops_est",
        "run_ms",
    )
    vta_model = fit_through_origin(
        distinct_vta_points,
        "logical_compute_gops_est",
        "run_ms",
        "ms_per_logical_gop",
    )
    distinct_vta_core_points = median_grouped_points(
        vta_fit_points,
        ("group", "segment"),
        "logical_compute_gops_est",
        "run_process_cpu_ms",
    )
    vta_host_core_model = fit_through_origin(
        distinct_vta_core_points,
        "logical_compute_gops_est",
        "run_process_cpu_ms",
        "core_ms_per_logical_gop",
    )
    boundary_models = {}
    boundary_core_models = {}
    for direction in ("cpu_to_vta", "vta_to_cpu"):
        points = median_grouped_points(
            [item for item in boundary_fit_points if item["direction"] == direction],
            ("group", "direction", "bytes_mib"),
            "bytes_mib",
            "service_ms",
        )
        boundary_models[direction] = fit_through_origin(
            points, "bytes_mib", "service_ms", "ms_per_mib"
        )
        core_points = median_grouped_points(
            [item for item in boundary_fit_points if item["direction"] == direction],
            ("group", "direction", "bytes_mib"),
            "bytes_mib",
            "host_core_demand_ms",
        )
        boundary_core_models[direction] = fit_through_origin(
            core_points,
            "bytes_mib",
            "host_core_demand_ms",
            "core_ms_per_mib",
        )

    profile_manifest = json.loads((OUTPUT_ROOT / "v1_profile_manifest.json").read_text())
    segment_costs = []
    for segment in profile_manifest["segments"]:
        gops = float(segment["logical_compute_gops_est"])
        if segment["device"] == "cpu":
            estimates = {
                threads: gops * model["ms_per_logical_gop"]
                for threads, model in cpu_models.items()
            }
            core_estimates = {
                threads: gops * model["core_ms_per_logical_gop"]
                for threads, model in cpu_core_models.items()
            }
        else:
            estimates = {"1": gops * vta_model["ms_per_logical_gop"]}
            core_estimates = {
                "1": gops * vta_host_core_model["core_ms_per_logical_gop"]
            }
        segment_costs.append(
            {
                "segment_id": segment["segment_id"],
                "device": segment["device"],
                "logical_compute_gops_est": gops,
                "service_ms_by_threads": estimates,
                "host_core_demand_ms_by_threads": core_estimates,
                "compiler_status": segment["compiler_status"],
            }
        )
    boundary_costs = []
    for boundary in profile_manifest["boundaries"]:
        model = boundary_models[boundary["direction"]]
        boundary_costs.append(
            {
                "boundary_id": boundary["boundary_id"],
                "direction": boundary["direction"],
                "logical_bytes": boundary["logical_bytes"],
                "service_ms": boundary["logical_bytes"]
                / (1024.0 * 1024.0)
                * model["ms_per_mib"],
                "host_core_demand_ms": boundary["logical_bytes"]
                / (1024.0 * 1024.0)
                * boundary_core_models[boundary["direction"]]["core_ms_per_mib"],
            }
        )

    holdout_predictions = []
    for stage in holdout["stages"]:
        model = (
            cpu_models[str(stage["threads"])]
            if stage["device"] == "cpu"
            else vta_model
        )
        predicted = stage["logical_compute_gops_est"] * model["ms_per_logical_gop"]
        holdout_predictions.append(
            {
                "stage_index": stage["index"],
                "device": stage["device"],
                "predicted_run_ms": predicted,
                "measured_run_ms": stage["run_ms"],
                "run_ape": abs(predicted - stage["run_ms"]) / stage["run_ms"],
            }
        )
    holdout_c2v = (
        holdout["stages"][0]["output_bytes"]
        / (1024.0 * 1024.0)
        * boundary_models["cpu_to_vta"]["ms_per_mib"]
    )
    holdout_v2c = (
        holdout["stages"][1]["output_bytes"]
        / (1024.0 * 1024.0)
        * boundary_models["vta_to_cpu"]["ms_per_mib"]
    )
    predicted_resource_services = {
        "cpu_stage0_ms": holdout_predictions[0]["predicted_run_ms"],
        "single_vta_path_ms": holdout_predictions[1]["predicted_run_ms"]
        + holdout_c2v
        + holdout_v2c,
        "cpu_stage2_ms": holdout_predictions[2]["predicted_run_ms"],
    }
    predicted_bottleneck = max(predicted_resource_services, key=predicted_resource_services.get)
    predicted_cycle = predicted_resource_services[predicted_bottleneck]
    exact_top1_match_count = sum(
        item["serial_top1_values"] == [reference["top1"]] for item in stage_summaries
    )
    all_top1_class_equivalent = all(
        item["serial_top1_values"] == [reference["top1"]]
        or (
            reference["top1"] in CAT_EQUIVALENT_TOP1
            and set(item["serial_top1_values"]).issubset(CAT_EQUIVALENT_TOP1)
        )
        for item in stage_summaries
    )
    all_hashes_stable = all(item["serial_raw_signature_count"] == 1 for item in stage_summaries)
    thread_sweep = {
        int(item["threads"]): item for item in fit if item["role"] == "fit_thread_sweep"
    }
    memory_affinity_observed = all(
        item["cpu_affinity_values"] == [tuple(range(item["threads"]))] for item in memory
    )
    stage_thread_sweep_effective = set(thread_sweep) == {1, 2, 3, 4} and all(
        thread_sweep[4]["stages"][index]["run_ms"]
        < thread_sweep[1]["stages"][index]["run_ms"]
        for index in (0, 2)
    )
    profiler_records = [item["serial_vta_profiler"] for item in stage_summaries]
    profiler_records.append(holdout["pipeline"]["vta_profiler"])
    vta_profiler_valid = all(
        item["driver_run_calls"] > 0
        and item["synchronize_calls"] > 0
        and item["load_bytes"] > 0
        and item["store_bytes"] > 0
        and item["driver_timeout_calls"] == 0
        for item in profiler_records
    )
    holdout_pipeline = holdout["pipeline"]
    serial_affinity_matches_current = all(
        all(
            list(
                item["stage_cpu_affinity"]
                .get("serial", {})
                .get("stage{}".format(stage["index"]), [])
            )
            == (list(range(stage["threads"])) if stage["device"] == "cpu" else [])
            for stage in item["stages"]
        )
        for item in stage_summaries
    )
    holdout_pipeline_affinity_matches_current = all(
        list(
            holdout["stage_cpu_affinity"]
            .get("pipeline", {})
            .get("stage{}".format(stage["index"]), [])
        )
        == (list(range(stage["threads"])) if stage["device"] == "cpu" else [])
        for stage in holdout["stages"]
    )
    local_cost = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_local_cost_table",
            "protocol_id": PROTOCOL_ID,
            "p2_plan_artifact_sha256": plan["artifact_sha256"],
            "measurement_path": "native_static_package",
            "rpc_performance_used": False,
            "fit_stage_measurements": fit,
            "grouped_holdout_measurement": holdout,
            "service_models": {
                "cpu": cpu_models,
                "cpu_core_demand": cpu_core_models,
                "vta": vta_model,
                "vta_host_core_demand": vta_host_core_model,
                "boundary": boundary_models,
                "boundary_host_core_demand": boundary_core_models,
            },
            "segment_costs": segment_costs,
            "boundary_costs": boundary_costs,
            "profile_manifest_artifact_sha256": profile_manifest["artifact_sha256"],
            "shared_ddr": memory,
            "runtime_policy_compatibility": {
                "current_affinity_policy": "independent_overlapping_prefix_masks_v1",
                "serial_fit_masks_match_current": serial_affinity_matches_current,
                "grouped_pipeline_holdout_masks_match_current": (
                    holdout_pipeline_affinity_matches_current
                ),
                "grouped_pipeline_holdout_scope": (
                    "historical_disjoint_affinity_diagnostic_only"
                    if not holdout_pipeline_affinity_matches_current
                    else "current_runtime_policy_validation"
                ),
            },
            "ownership": {
                "cpu_service": "stage run_ms; run process CPU time prices shared core work",
                "vta_service": "VTA stage run_ms including submit/sync/LOAD/STORE",
                "vta_host_core_demand": "VTA run process CPU time, excluding set/get",
                "cpu_to_vta_boundary": "CPU get_ms + VTA set_ms",
                "vta_to_cpu_boundary": "VTA get_ms + CPU set_ms",
                "boundary_host_core_demand": "process CPU time of both set/get owners",
                "ddr": "64 MiB sustained read/write/copy diagnostic lower bound",
                "additive_spill_penalty": False,
            },
            "scope": "all 172 reachable segments priced by frozen logical-op service rates; "
            "native shortlist compile remains mandatory",
            "correctness_scope": "semantic class equivalence, deterministic raw outputs, and "
            "serial/pipeline equivalence; numerical reference remains mandatory for P3 shortlist",
        }
    )
    progress = json.loads((SESSION_ROOT / "measurement_ledger.json").read_text())
    cost_commands = normalize_cost_commands(progress["cases"])
    cost_ledger = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_profile_cost_ledger",
            "protocol_id": PROTOCOL_ID,
            "p2_plan_artifact_sha256": plan["artifact_sha256"],
            "completed_command_count": sum(
                item["status"] == "completed" for item in cost_commands
            ),
            "failed_command_count": sum(item["status"] == "failed" for item in cost_commands),
            "timing_anomaly_count": sum("timing_anomaly" in item for item in cost_commands),
            "wall_clock_s": sum(float(item["elapsed_s"]) for item in cost_commands),
            "native_stage_case_count": 7,
            "memory_case_count": 12,
            "candidate_throughput_evaluations": 0,
            "commands": cost_commands,
        }
    )
    write_json(OUTPUT_ROOT / "v1_local_cost_table.json", local_cost)
    write_json(OUTPUT_ROOT / "v1_profile_cost_ledger.json", cost_ledger)

    measured_cycle = float(holdout_pipeline["measured_cycle_ms"])
    residual = (measured_cycle - predicted_cycle) / predicted_cycle
    holdout_validation = {
        "stage_run_predictions": holdout_predictions,
        "predicted_resource_services": predicted_resource_services,
        "predicted_bottleneck": predicted_bottleneck,
        "measured_serial_bottleneck": "cpu_stage{}".format(
            max(range(3), key=lambda index: holdout["stages"][index]["wall_ms"])
        ),
        "predicted_cycle_ms": predicted_cycle,
        "measured_pipeline_cycle_ms": measured_cycle,
        "pipeline_cycle_relative_residual": residual,
        "uses_holdout_timing_for_fit": False,
        "affinity_policy_at_measurement": holdout["stage_cpu_affinity"].get("policy"),
        "matches_current_pipeline_affinity": holdout_pipeline_affinity_matches_current,
        "evidence_scope": (
            "historical_affinity_diagnostic_only"
            if not holdout_pipeline_affinity_matches_current
            else "current_runtime_policy"
        ),
    }
    gate = {
        "independent_reference_class_equivalent": all_top1_class_equivalent,
        "deterministic_output_hash": all_hashes_stable,
        "serial_pipeline_equivalence": holdout_pipeline[
            "serial_pipeline_raw_signatures_equal"
        ],
        "native_vta_profiler_valid": vta_profiler_valid,
        "cpu_thread_control_effective": memory_affinity_observed
        and stage_thread_sweep_effective,
        "serial_profile_masks_match_current_policy": serial_affinity_matches_current,
        "memory_correctness": all(item["correct"] for item in memory),
        "historical_affinity_holdout_residual_within_20_percent": abs(residual) <= 0.20,
        "historical_affinity_holdout_bottleneck_preserved": (
            predicted_bottleneck == "cpu_stage0_ms"
        ),
    }
    lines = [
        "# V1-P2 Holdout Report",
        "",
        "- Native performance only; RPC performance used: `false`.",
        "- Stage cases: `7`; DDR cases: `12`; candidate throughput evaluations: `0`.",
        "- Independent MXNet reference top1: `{}`.".format(reference["top1"]),
        "- Exact top1 matches: `{}/{}`; remaining outputs use the preregistered ImageNet "
        "cat-equivalent policy `{}`.".format(
            exact_top1_match_count,
            len(stage_summaries),
            sorted(CAT_EQUIVALENT_TOP1),
        ),
        "- This is a semantic/determinism gate, not numerical tensor equivalence; P3 shortlist "
        "must pass the native numerical reference.",
        "",
        "## Gate",
        "",
    ]
    for name, passed in gate.items():
        lines.append("- `{}`: `{}`".format(name, str(bool(passed)).lower()))
    lines.extend(
        [
            "",
            "## Grouped Holdout",
            "",
            "`three_stage_b` fit-only max-load prediction: `{:.3f} ms`; measured pipeline "
            "cycle: `{:.3f} ms`; residual: `{:+.2%}`.".format(
                predicted_cycle,
                measured_cycle,
                residual,
            ),
            "",
            (
                "This holdout used `{}` and matches the current overlapping-mask policy: `{}`. "
                "Its residual is a historical-policy diagnostic and is not evidence for "
                "current-policy CPU contention. The frozen additive logical-op/core-work model "
                "prices all 172 reachable segments; P3 must still compile every shortlisted "
                "segment before board execution."
            ).format(
                holdout["stage_cpu_affinity"].get("policy", "unknown"),
                str(bool(holdout_pipeline_affinity_matches_current)).lower(),
            ),
        ]
    )
    (OUTPUT_ROOT / "v1_holdout_report.md").write_text("\n".join(lines) + "\n")
    write_json(
        SESSION_ROOT / "aggregate_summary.json",
        {"gate": gate, "stages": stage_summaries, "holdout_validation": holdout_validation},
    )
    failed_gates = sorted(name for name, passed in gate.items() if not passed)
    execution_state = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_execution_state",
            "protocol_id": PROTOCOL_ID,
            "current_stage": "V1-P2",
            "current_stage_status": "completed_awaiting_user_confirmation"
            if not failed_gates
            else "blocked_by_p2_gate",
            "next_stage": "V1-P3",
            "next_stage_may_start": False,
            "advance_requires_user_confirmation": True,
            "blocking_evidence": failed_gates,
            "profile_manifest_artifact_sha256": profile_manifest["artifact_sha256"],
            "p2_plan_artifact_sha256": plan["artifact_sha256"],
            "local_cost_table_artifact_sha256": local_cost["artifact_sha256"],
            "profile_cost_ledger_artifact_sha256": cost_ledger["artifact_sha256"],
            "independent_reference_artifact_sha256": reference["artifact_sha256"],
        }
    )
    write_json(OUTPUT_ROOT / "v1_execution_state.json", execution_state)
    print(json.dumps({"gate": gate, "holdout_residual": residual}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["measure", "aggregate", "all"], default="all")
    parser.add_argument("--board", default="root@192.168.1.234")
    parser.add_argument("--memory-binary", default="/tmp/ramps_cpu_memory_microbench_v1")
    args = parser.parse_args()
    if args.mode in {"measure", "all"}:
        measure(args)
    if args.mode in {"aggregate", "all"}:
        aggregate(args)


if __name__ == "__main__":
    main()
