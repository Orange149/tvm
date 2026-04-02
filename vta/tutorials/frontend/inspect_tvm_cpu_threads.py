#!/usr/bin/env python3
"""Inspect TVM internal CPU threading on a representative staged CPU block."""

from __future__ import absolute_import, print_function

import argparse
import os
import time

import mxnet as mx
import numpy as np
import tvm
from mxnet.gluon.model_zoo import vision

import vta

import profile_split_resnet18_stages as ps  # pylint: disable=import-error
import split_resnet18_stages as ss  # pylint: disable=import-error


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("VTA_RPC_HOST", "192.168.1.247"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VTA_RPC_PORT", "9090")))
    parser.add_argument("--scheme", default="three_stage_b", choices=sorted(list(ss.SCHEMES.keys()) + [ss.AUTO_RESOURCE_AWARE_SCHEME]))
    parser.add_argument("--stage-name", default="stage0_cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runtime-num-threads", type=int, default=0)
    parser.add_argument("--timer-number", type=int, default=10)
    parser.add_argument("--timer-repeat", type=int, default=10)
    parser.add_argument("--run-time-evaluator", action="store_true")
    parser.add_argument("--print-llvm-parallel-markers", action="store_true")
    parser.add_argument(
        "--resource-aware-candidate-name",
        default="",
        help="When --scheme auto_resource_aware, inspect this specific enumerated candidate",
    )
    return parser.parse_args()


def build_host_inputs(input_schema):
    values = []
    for slot in input_schema["slots"]:
        values.append(np.zeros(tuple(slot["shape"]), dtype=slot["dtype"]))
    return values[0] if len(values) == 1 else tuple(values)


def collect_parallel_markers(lib):
    texts = []
    for obj in [lib] + list(getattr(lib, "imported_modules", [])):
        for fmt in ["ll", "asm", "c"]:
            try:
                texts.append(obj.get_source(fmt))
                break
            except Exception:  # pylint: disable=broad-except
                continue
    text = "\n".join(texts)
    return {
        "__TVMBackendParallelLaunch": text.count("__TVMBackendParallelLaunch"),
        "TVMBackendParallelLaunch": text.count("TVMBackendParallelLaunch"),
        "omp": text.count("omp"),
    }


def main():
    args = parse_args()
    env = vta.get_env()
    full_model = vision.get_model("resnet18_v1", pretrained=False)
    full_model.initialize(ctx=mx.cpu())
    _ = full_model(mx.nd.array(np.zeros((args.batch, 3, args.image_size, args.image_size), dtype="float32")))
    feature_blocks = list(full_model.features)
    output_block = full_model.output
    unit_blocks = ss.build_resnet18_unit_blocks(feature_blocks, output_block)
    unit_metadata = ss.build_resnet18_unit_metadata(unit_blocks, args.batch, args.image_size)
    scheme_cfg, selected, _ = ss.resolve_scheme_config(
        args.scheme,
        feature_blocks,
        output_block,
        args.batch,
        args.image_size,
        selected_candidate_name=args.resource_aware_candidate_name or None,
    )
    stage_cfg = next((item for item in scheme_cfg if item["name"] == args.stage_name), None)
    if stage_cfg is None:
        raise ValueError("stage {} not found in scheme {}".format(args.stage_name, selected["scheme_name"] if selected else args.scheme))
    stage_block = ss.make_stage_block(feature_blocks, output_block, stage_cfg, unit_blocks=unit_blocks)
    stage_inputs = ss.relay_inputs_for_stage(stage_cfg, unit_metadata)
    mod, params = ss.lower_stage_to_relay(stage_block, stage_inputs)
    cpu_target = tvm.target.Target(env.target_vta_cpu, host=env.target_host)
    if stage_cfg["device"] == "cpu":
        graph, lib, lowered_params = ps.build_cpu_stage(args.stage_name, mod, params, cpu_target)
    else:
        graph, lib, lowered_params = ps.build_vta_stage(
            args.stage_name,
            mod["main"],
            params,
            env,
            use_graph_pack=(selected["scheme_name"] == "all_vta" if selected else args.scheme == "all_vta"),
        )
    markers = collect_parallel_markers(lib)

    print("[INSPECT] scheme={}".format(selected["scheme_name"] if selected else args.scheme))
    print("[INSPECT] stage={}".format(args.stage_name))
    print("[INSPECT] launcher env_TVM_NUM_THREADS={}".format(os.environ.get("TVM_NUM_THREADS", "<unset>")))
    print("[INSPECT] launcher env_OMP_NUM_THREADS={}".format(os.environ.get("OMP_NUM_THREADS", "<unset>")))
    if args.print_llvm_parallel_markers:
        print("[INSPECT] llvm_markers __TVMBackendParallelLaunch={} TVMBackendParallelLaunch={} omp={}".format(
            markers["__TVMBackendParallelLaunch"],
            markers["TVMBackendParallelLaunch"],
            markers["omp"],
        ))

    remote = ps.connect_remote(env, args.host, args.port)
    threadpool_info = ps.configure_remote_threadpool(remote, num_threads=args.runtime_num_threads)
    print(
        "[INSPECT] runtime thread pool requested={} before={} after={} config_available={} num_threads_available={}".format(
            threadpool_info["requested_num_threads"] if threadpool_info["requested_num_threads"] is not None else "-",
            threadpool_info["runtime_num_threads_before"] if threadpool_info["runtime_num_threads_before"] is not None else "-",
            threadpool_info["runtime_num_threads_after"] if threadpool_info["runtime_num_threads_after"] is not None else "-",
            "yes" if threadpool_info["config_available"] else "no",
            "yes" if threadpool_info["num_threads_available"] else "no",
        )
    )

    remote_lib = ps.export_and_upload(lib, remote, env, args.stage_name + "_inspect")
    stage_module, device = ps.create_stage_module(args.stage_name, stage_cfg["device"], graph, remote_lib, remote)
    timer_device = device[0] if isinstance(device, (list, tuple)) else device
    stage_module.set_input(**lowered_params)
    current = build_host_inputs(ps.stage_input_schema_for_stage(stage_cfg, unit_metadata))
    output_schema = ps.get_func_output_info(mod["main"])

    for warmup_idx in range(args.warmup):
        if warmup_idx == 0:
            print("[INSPECT] warmup from {}".format(ps.describe_value_location(current)))
        ps.set_stage_inputs(stage_module, [name for name, _ in stage_inputs], current)
        stage_module.run()
        _ = ps.fetch_stage_output(stage_module, output_schema, remote)

    run_ms = []
    service_ms = []
    for repeat_idx in range(args.repeat):
        t_set0 = time.perf_counter()
        ps.set_stage_inputs(stage_module, [name for name, _ in stage_inputs], current)
        t0 = time.perf_counter()
        stage_module.run()
        t1 = time.perf_counter()
        _ = ps.fetch_stage_output(stage_module, output_schema, remote)
        t2 = time.perf_counter()
        run_ms.append((t1 - t0) * 1000.0)
        service_ms.append((t2 - t_set0) * 1000.0)
        print(
            "[INSPECT] repeat {}/{} run_ms={:.3f} service_ms={:.3f}".format(
                repeat_idx + 1,
                args.repeat,
                run_ms[-1],
                service_ms[-1],
            )
        )

    service_avg = float(np.mean(service_ms)) if service_ms else 0.0
    service_std = float(np.std(service_ms)) if service_ms else 0.0
    run_avg = float(np.mean(run_ms)) if run_ms else 0.0
    run_std = float(np.std(run_ms)) if run_ms else 0.0
    print(
        "[INSPECT] summary runtime_num_threads={} stage_service_avg_ms={:.3f} stage_service_std_ms={:.3f} stage_run_avg_ms={:.3f} stage_run_std_ms={:.3f}".format(
            threadpool_info["runtime_num_threads_after"] if threadpool_info["runtime_num_threads_after"] is not None else -1,
            service_avg,
            service_std,
            run_avg,
            run_std,
        )
    )
    if args.run_time_evaluator:
        timer = stage_module.module.time_evaluator(
            "run",
            timer_device,
            number=args.timer_number,
            repeat=args.timer_repeat,
        )
        timer_result = timer()
        results_ms = [float(item) * 1000.0 for item in timer_result.results]
        print(
            "[INSPECT] time_evaluator number={} repeat={} mean_ms={:.3f} std_ms={:.3f} min_ms={:.3f} max_ms={:.3f}".format(
                args.timer_number,
                args.timer_repeat,
                float(np.mean(results_ms)),
                float(np.std(results_ms)),
                float(np.min(results_ms)),
                float(np.max(results_ms)),
            )
        )
    print(
        "[INSPECT] markers __TVMBackendParallelLaunch={} TVMBackendParallelLaunch={} omp={}".format(
            markers["__TVMBackendParallelLaunch"],
            markers["TVMBackendParallelLaunch"],
            markers["omp"],
        )
    )


if __name__ == "__main__":
    main()
