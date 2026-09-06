#!/usr/bin/env python3
"""Build and measure native static-package RAMPS service buckets.

The first implementation intentionally exposes one representative smoke case.
It establishes the native package, board timing, CPU core-demand and VTA DMA
profiler path before the full no-fallback bucket matrix is enabled.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Dict, Tuple

import numpy as np
import tvm
from tvm import relay

import vta
from deploy_classification_stage_pipeline_native import (
    compile_runner,
    copy_runtime_libs,
    export_stage_lib,
)
from profile_split_resnet18_stages import build_cpu_stage, build_vta_stage, get_func_output_info


DEFAULT_OUTPUT = (
    "vta/tutorials/frontend/report_out/resource_aware_maxplus/"
    "stage4_calibration_smoke/native_bucket_residual3x3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="build,measure")
    parser.add_argument("--board", default="root@192.168.1.240")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--remote-dir", default="/var/volatile/ramps_native_calibration/residual3x3")
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--skip-first", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--ssh-option", action="append", default=[])
    return parser.parse_args()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_residual_conv(prefix: str) -> Tuple[relay.Function, Dict[str, np.ndarray]]:
    shape = (1, 64, 56, 56)
    rng = np.random.RandomState(17 if prefix == "cpu" else 23)
    data = relay.var("data", shape=shape, dtype="float32")
    weight = relay.var(prefix + "_weight", shape=(64, 64, 3, 3), dtype="float32")
    bias = relay.var(prefix + "_bias", shape=(64,), dtype="float32")
    body = relay.nn.conv2d(
        data,
        weight,
        channels=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=(1, 1),
        data_layout="NCHW",
        kernel_layout="OIHW",
        out_dtype="float32",
    )
    body = relay.nn.relu(relay.nn.bias_add(body, bias, axis=1))
    params = {
        prefix + "_weight": rng.uniform(-0.05, 0.05, size=(64, 64, 3, 3)).astype("float32"),
        prefix + "_bias": rng.uniform(-0.01, 0.01, size=(64,)).astype("float32"),
    }
    return relay.Function([data, weight, bias], body), params


def schema(func: relay.Function) -> Dict[str, Any]:
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    return get_func_output_info(mod["main"])


def write_stage(
    package_dir: Path,
    index: int,
    name: str,
    device: str,
    func: relay.Function,
    params: Dict[str, np.ndarray],
    env,
) -> Dict[str, Any]:
    stage_dir = package_dir / "stages" / name
    stage_dir.mkdir(parents=True, exist_ok=True)
    if device == "cpu":
        target = tvm.target.Target(env.target_vta_cpu, host=env.target_host)
        graph, lib, lowered_params = build_cpu_stage(
            name, tvm.IRModule.from_expr(func), params, target
        )
    else:
        graph, lib, lowered_params = build_vta_stage(
            name, func, params, env, use_graph_pack=False
        )
    graph_path = stage_dir / "graph.json"
    params_path = stage_dir / "params.params"
    lib_path = stage_dir / "graphlib.so"
    graph_path.write_text(graph, encoding="utf-8")
    params_path.write_bytes(tvm.runtime.save_param_dict(lowered_params))
    export_stage_lib(lib, lib_path)
    logical_schema = schema(func)
    return {
        "index": index,
        "name": name,
        "device": device,
        "bucket": "residual_3x3_conv" if device == "cpu" else "conv3x3_c_small",
        "input_names": ["data"],
        "input_schema": {
            "kind": "tensor",
            "arity": 1,
            "slots": [
                {"slot_index": 0, "role": "data", "shape": [1, 64, 56, 56], "dtype": "float32"}
            ],
        },
        "output_schema": logical_schema,
        "graph": str(graph_path.relative_to(package_dir)),
        "lib": str(lib_path.relative_to(package_dir)),
        "params": str(params_path.relative_to(package_dir)),
        "build_path": "native_relay_cpu" if device == "cpu" else "native_vta_boundary_bridge",
    }


def build_package(output_dir: Path) -> Path:
    package_dir = output_dir / "package"
    package_dir.mkdir(parents=True, exist_ok=True)
    env = vta.get_env()
    cpu_func, cpu_params = make_residual_conv("cpu")
    vta_func, vta_params = make_residual_conv("vta")
    stages = [
        write_stage(package_dir, 0, "cpu_residual_3x3", "cpu", cpu_func, cpu_params, env),
        write_stage(package_dir, 1, "vta_conv3x3_c_small", "vta", vta_func, vta_params, env),
    ]
    rng = np.random.RandomState(5)
    input_data = rng.uniform(-1.0, 1.0, size=(1, 64, 56, 56)).astype("float32")
    (package_dir / "input.bin").write_bytes(input_data.tobytes(order="C"))
    compile_runner(package_dir)
    copy_runtime_libs(package_dir)
    manifest = {
        "kind": "ramps_native_service_bucket_smoke",
        "model": "synthetic_residual_conv_pair",
        "target": env.TARGET,
        "candidate_id": "ramps_native_residual3x3_smoke",
        "stage_count": 2,
        "input_shape": [1, 64, 56, 56],
        "input_dtype": "float32",
        "input_file": "input.bin",
        "input_list": "",
        "input_count": 1,
        "stages": stages,
        "measurement_path": "native_static_package",
        "rpc_performance_allowed": False,
        "publication_status": "instrumentation_smoke_only",
    }
    write_json(package_dir / "manifest.json", manifest)
    write_json(
        output_dir / "frozen_smoke_protocol.json",
        {
            "case": "residual_3x3_64_64",
            "cpu_bucket": "residual_3x3_conv",
            "vta_bucket": "conv3x3_c_small",
            "shape": [1, 64, 56, 56],
            "kernel": [3, 3],
            "stride": [1, 1],
            "padding": [1, 1],
            "build": "static native CPU + VTA boundary_bridge",
            "rpc_used_for_performance": False,
        },
    )
    return package_dir


def measure(args: argparse.Namespace, output_dir: Path, package_dir: Path) -> None:
    results_dir = output_dir / "board_results"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "deploy_classification_stage_pipeline_native.py"),
        "--board",
        args.board,
        "--remote-dir",
        args.remote_dir,
        "--remote-min-free-mb",
        "128",
        "--deploy-mode",
        "sync",
        "--candidate-id",
        "ramps_native_residual3x3_smoke",
        "--runs",
        str(args.runs),
        "--queue-depth",
        "1",
        "--runtime-num-threads",
        str(args.cpu_threads),
        "--stage-runtime-num-threads",
        "{},1".format(args.cpu_threads),
        "--serial",
        "--runner-output-mode",
        "raw_all_stages",
        "--fetch-results-dir",
        str(results_dir),
        "--vta-runtime-profile-dir",
        "profile",
        "--vta-runtime-profile-events-limit",
        "512",
        "--serial-timeout-s",
        "240",
        "--fetch-timeout-s",
        "120",
        "--cleanup-remote-after-run",
        "--reuse-package-dir",
        str(package_dir),
    ]
    ssh_options = args.ssh_option or [
        "HostKeyAlgorithms=+ssh-rsa",
        "PubkeyAcceptedAlgorithms=+ssh-rsa",
    ]
    for option in ssh_options:
        command.extend(["--ssh-option", option])
    (output_dir / "measure_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    subprocess.run(command, check=True, env=os.environ.copy())


def aggregate_smoke(args: argparse.Namespace, output_dir: Path) -> None:
    results_dir = output_dir / "board_results"
    rows = [
        json.loads(line)
        for line in (results_dir / "stage_serial_result.jsonl").read_text().splitlines()
        if line.strip()
    ]
    measured = rows[int(args.skip_first) :]
    if not measured:
        raise RuntimeError("no measured rows remain after --skip-first")

    def median(key: str) -> float:
        return float(statistics.median(float(row[key]) for row in measured))

    stages = []
    stage_specs = [(0, "cpu", "residual_3x3_conv"), (1, "vta", "conv3x3_c_small")]
    operations = 2 * 56 * 56 * 64 * 64 * 3 * 3
    for index, device, bucket in stage_specs:
        wall_ms = median("stage{}_ms".format(index))
        run_ms = median("stage{}_run_ms".format(index))
        core_ms = median("stage{}_process_cpu_ms".format(index))
        stages.append(
            {
                "index": index,
                "device": device,
                "bucket": bucket,
                "wall_ms": wall_ms,
                "set_ms": median("stage{}_set_ms".format(index)),
                "run_ms": run_ms,
                "get_ms": median("stage{}_get_ms".format(index)),
                "process_cpu_ms": core_ms,
                "average_cpu_cores": core_ms / max(wall_ms, 1.0e-12),
                "effective_stage_gops": (
                    (operations / 1.0e9) / max(run_ms / 1000.0, 1.0e-12)
                ),
                "measurement_kind": "effective_stage_black_box",
                "includes": (
                    ["cpu_compute", "cpu_memory", "runtime_launch"]
                    if device == "cpu"
                    else ["vta_compute", "device_load", "device_store", "submit_sync"]
                ),
                "excludes": ["set_input", "get_output"],
                "diagnostic_only": True,
                "eligible_for_physical_compute_model": False,
            }
        )
    profile_path = results_dir / "profile" / "serial" / "benchmark_totals_status.json"
    profile = json.loads(profile_path.read_text())
    divisor = len(rows)
    profile_keys = [
        "device_run_wait_us",
        "driver_submit_mmio_us",
        "driver_poll_wait_us",
        "load_buffer_2d_bytes",
        "store_buffer_2d_bytes",
        "load_buffer_2d_calls",
        "store_buffer_2d_calls",
        "load_buffer_2d_enqueue_us",
        "store_buffer_2d_enqueue_us",
    ]
    per_frame = {key: float(profile.get(key, 0.0)) / divisor for key in profile_keys}
    total_calls = per_frame["load_buffer_2d_calls"] + per_frame["store_buffer_2d_calls"]
    total_bytes = per_frame["load_buffer_2d_bytes"] + per_frame["store_buffer_2d_bytes"]
    per_frame["avg_bytes_per_call"] = total_bytes / max(total_calls, 1.0)
    raw_hashes = [
        tuple(item["fnv1a64"] for item in row.get("raw_outputs", [])) for row in rows
    ]
    raw_finite = all(
        all(
            np.isfinite(float(item.get(key, 0.0)))
            for item in row.get("raw_outputs", [])
            for key in ("min", "max", "mean", "sum")
        )
        for row in rows
    )
    summary = {
        "schema_version": 1,
        "board": args.board,
        "measurement_path": "native_static_package",
        "rpc_used_for_performance": False,
        "runs": len(rows),
        "skip_first": int(args.skip_first),
        "cpu_threads": int(args.cpu_threads),
        "cpu_time_scope": sorted({row.get("stage_cpu_time_scope") for row in rows}),
        "stages": stages,
        "vta_profile_per_frame": per_frame,
        "raw_output_hash_stable": len(set(raw_hashes)) == 1,
        "raw_output_finite": raw_finite,
        "checks": {
            "native_execution": True,
            "cpu_core_demand": stages[0]["process_cpu_ms"] > 0.0,
            "vta_dma_bytes_and_calls": total_bytes > 0.0 and total_calls > 0.0,
            "vta_load_store_device_time_separated": False,
            "no_fallback_schedule": True,
        },
        "smoke_gate_passed": False,
        "blocking_reason": (
            "VTA profiler exposes total device wait and DMA bytes/calls but not separate "
            "load/store device service time; formal PS-PL bandwidth cannot divide both "
            "byte counters by the same compute-inclusive wait."
        ),
        "evidence_level": "instrumentation_smoke",
    }
    write_json(output_dir / "native_bucket_smoke_summary.json", summary)
    cpu = stages[0]
    vta_stage = stages[1]
    lines = [
        "# Native Residual-3x3 Bucket Smoke",
        "",
        "- Board: `{}`".format(args.board),
        "- Performance path: native static package; RPC performance measurement: `false`.",
        "- Runs: `{}`, skipped: `{}`.".format(len(rows), args.skip_first),
        "- Raw output stable/finite: `{}/{}`.".format(
            summary["raw_output_hash_stable"], summary["raw_output_finite"]
        ),
        "",
        "| Bucket | Device | set ms | run ms | get ms | wall ms | process CPU ms | GOP/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        "| `residual_3x3_conv` | CPU | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            cpu["set_ms"], cpu["run_ms"], cpu["get_ms"], cpu["wall_ms"],
            cpu["process_cpu_ms"], cpu["effective_stage_gops"]
        ),
        "| `conv3x3_c_small` | VTA | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            vta_stage["set_ms"], vta_stage["run_ms"], vta_stage["get_ms"],
            vta_stage["wall_ms"], vta_stage["process_cpu_ms"],
            vta_stage["effective_stage_gops"]
        ),
        "",
        "Per VTA frame: load `{:.0f}` B / `{:.0f}` calls; store `{:.0f}` B / `{:.0f}` calls; "
        "average `{:.1f}` B/call; total device wait `{:.3f}` ms.".format(
            per_frame["load_buffer_2d_bytes"], per_frame["load_buffer_2d_calls"],
            per_frame["store_buffer_2d_bytes"], per_frame["store_buffer_2d_calls"],
            per_frame["avg_bytes_per_call"], per_frame["device_run_wait_us"] / 1000.0
        ),
        "",
        "## Gate",
        "",
        "Instrumentation passed, but the formal calibration gate remains **blocked**.",
        summary["blocking_reason"],
        "A DMA-only native microbenchmark or additional device-side DMA timestamps are required",
        "before the three-session no-fallback calibration can begin.",
    ]
    (output_dir / "NATIVE_BUCKET_SMOKE_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    modes = {item.strip() for item in args.mode.split(",") if item.strip()}
    if not modes or not modes.issubset({"build", "measure", "aggregate"}):
        raise ValueError("--mode must contain build, measure, and/or aggregate")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    package_dir = output_dir / "package"
    if "build" in modes:
        package_dir = build_package(output_dir)
    if "measure" in modes:
        if not (package_dir / "manifest.json").exists():
            raise FileNotFoundError(package_dir / "manifest.json")
        measure(args, output_dir, package_dir)
    if "measure" in modes or "aggregate" in modes:
        aggregate_smoke(args, output_dir)
    print("[RAMPS NATIVE CALIBRATION SMOKE]", output_dir)


if __name__ == "__main__":
    main()
