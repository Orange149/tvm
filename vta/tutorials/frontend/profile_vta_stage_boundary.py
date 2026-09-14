#!/usr/bin/env python3
"""Measure the memory cost of inserting a VTA GraphExecutor boundary.

The experiment keeps the layer4 operator sequence and TopHub dispatch fixed,
then compares one Executor against two Executors with either a materialized
DDR handoff or a shared-output/input NDArray binding.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

from mxnet.gluon.model_zoo import vision
import numpy as np
import tvm
from tvm import relay, rpc
from tvm.contrib import graph_executor
import vta

from hp_hpc_quant.vta_runtime_profile_utils import (
    clear_runtime_profiler,
    fetch_runtime_profiler_hooks,
    read_runtime_events,
    read_runtime_status,
)
from profile_split_resnet18_stages import build_vta_stage, create_stage_module, export_and_upload
from split_resnet18_stages import (
    BLOCK_UNIT_GROUPS,
    build_resnet18_unit_blocks,
    build_resnet18_unit_metadata,
    lower_stage_to_relay,
    make_stage_block,
    relay_inputs_for_stage,
)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype="float64"), q))


def timing_summary(costs_s):
    return {
        "samples": len(costs_s),
        "median_ms": float(np.median(costs_s)) * 1000.0,
        "p25_ms": percentile(costs_s, 25) * 1000.0,
        "p75_ms": percentile(costs_s, 75) * 1000.0,
        "min_ms": min(costs_s) * 1000.0,
        "max_ms": max(costs_s) * 1000.0,
        "costs_s": list(costs_s),
    }


def paired_difference_summary(left_s, right_s):
    delta_ms = (np.asarray(left_s) - np.asarray(right_s)) * 1000.0
    return {
        "samples": int(delta_ms.size),
        "median_ms": float(np.median(delta_ms)),
        "mean_ms": float(np.mean(delta_ms)),
        "p25_ms": percentile(delta_ms, 25),
        "p75_ms": percentile(delta_ms, 75),
        "left_slower_count": int(np.count_nonzero(delta_ms > 0)),
        "delta_ms": delta_ms.tolist(),
    }


def build_stage(name, units, features, head, blocks, metadata, env):
    stage = {"name": name, "device": "vta", "unit_names": list(units)}
    relay_inputs = relay_inputs_for_stage(stage, metadata)
    block = make_stage_block(features, head, stage, unit_blocks=blocks)
    mod, params = lower_stage_to_relay(block, relay_inputs)
    audit = {}
    graph, lib, lowered_params = build_vta_stage(
        name, mod["main"], params, env, tuning_audit=audit
    )
    return {
        "name": name,
        "units": list(units),
        "relay_inputs": relay_inputs,
        "relay_mod": mod,
        "params": params,
        "graph": graph,
        "lib": lib,
        "lowered_params": lowered_params,
        "tuning": audit,
    }


def instantiate(stage, remote, env, suffix=""):
    remote_lib = export_and_upload(
        stage["lib"], remote, env, stage["name"] + suffix
    )
    executor, ctx = create_stage_module(
        stage["name"] + suffix, "vta", stage["graph"], remote_lib, remote
    )
    executor.set_input(**stage["lowered_params"])
    return executor, ctx


def build_cpu_reference(stage):
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[]):
            quantized = relay.quantize.quantize(stage["relay_mod"], params=stage["params"])
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        factory = relay.build(quantized, target="llvm", params=stage["params"])
    return graph_executor.GraphModule(factory["default"](tvm.cpu()))


def host_outputs(executor, remote):
    outputs = []
    for index in range(executor.get_num_outputs()):
        template = executor.get_output(index)
        host = tvm.nd.empty(tuple(template.shape), str(template.dtype), remote.cpu(0))
        outputs.append(executor.get_output(index, host).numpy())
    return outputs


def compare_outputs(actual, expected):
    if len(actual) != len(expected):
        return {"correct": False, "reason": "output count differs"}
    rows = []
    for got, want in zip(actual, expected):
        delta = got.astype("float64") - want.astype("float64")
        rows.append({
            "shape": list(got.shape),
            "dtype": str(got.dtype),
            "mismatched": int(np.count_nonzero(got != want)),
            "max_abs_error": float(np.max(np.abs(delta))),
            "sha256": hashlib.sha256(got.tobytes()).hexdigest(),
        })
    return {"correct": all(row["mismatched"] == 0 for row in rows), "outputs": rows}


def capture_profile(hooks, run, events_limit):
    clear_runtime_profiler(hooks)
    run()
    return {
        "status": read_runtime_status(hooks),
        "events": read_runtime_events(hooks, max_events=events_limit),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--events-limit", type=int, default=256)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    env = vta.get_env()
    model = vision.get_model("resnet18_v1", pretrained=True)
    features, head = list(model.features), model.output
    blocks = build_resnet18_unit_blocks(features, head)
    metadata = build_resnet18_unit_metadata(blocks, env.BATCH, 224)
    block0 = BLOCK_UNIT_GROUPS["layer4_block0"]
    block1 = BLOCK_UNIT_GROUPS["layer4_block1"]

    stages = {
        "monolithic": build_stage(
            "boundary_c_monolithic", block0 + block1, features, head, blocks, metadata, env
        ),
        "producer": build_stage(
            "boundary_c_producer", block0, features, head, blocks, metadata, env
        ),
        "consumer": build_stage(
            "boundary_c_consumer", block1, features, head, blocks, metadata, env
        ),
    }
    cpu_references = {name: build_cpu_reference(stage) for name, stage in stages.items()}
    payload = {
        "status": "built",
        "host": args.host,
        "port": args.port,
        "fixed_operator_sequence": block0 + block1,
        "split_after": block0[-1],
        "tuning": {name: stage["tuning"] for name, stage in stages.items()},
    }
    write_json(args.output, payload)

    remote = rpc.connect(args.host, args.port, session_timeout=300)
    remote.get_function("runtime.config_threadpool")(1, 1)
    hooks = fetch_runtime_profiler_hooks(remote)
    mono, _ = instantiate(stages["monolithic"], remote, env)
    ordinary_prod, _ = instantiate(stages["producer"], remote, env, "_ordinary")
    ordinary_cons, _ = instantiate(stages["consumer"], remote, env, "_ordinary")
    shared_prod, _ = instantiate(stages["producer"], remote, env, "_shared")
    shared_cons, _ = instantiate(stages["consumer"], remote, env, "_shared")

    rng = np.random.default_rng(20260907)
    input_name, input_shape = stages["monolithic"]["relay_inputs"][0]
    boundary_name = stages["consumer"]["relay_inputs"][0][0]
    sample = rng.uniform(-1.0, 1.0, input_shape).astype("float32")
    mono.set_input(input_name, sample)
    ordinary_prod.set_input(input_name, sample)
    shared_prod.set_input(input_name, sample)
    cpu_references["monolithic"].set_input(input_name, sample)
    cpu_references["producer"].set_input(input_name, sample)
    cpu_references["monolithic"].run()
    cpu_references["producer"].run()
    cpu_boundary = cpu_references["producer"].get_output(0).numpy()
    cpu_references["consumer"].set_input(boundary_name, cpu_boundary)
    cpu_references["consumer"].run()
    cpu_mono_output = [
        cpu_references["monolithic"].get_output(i).numpy()
        for i in range(cpu_references["monolithic"].get_num_outputs())
    ]
    cpu_split_output = [
        cpu_references["consumer"].get_output(i).numpy()
        for i in range(cpu_references["consumer"].get_num_outputs())
    ]

    output_template = ordinary_prod.get_output(0)
    input_template = ordinary_cons.get_input(boundary_name)
    boundary_shape = tuple(int(x) for x in output_template.shape)
    boundary_dtype = str(output_template.dtype)
    if boundary_shape != tuple(int(x) for x in input_template.shape):
        raise AssertionError("producer output and consumer input shapes differ")
    if boundary_dtype != str(input_template.dtype):
        raise AssertionError("producer output and consumer input dtypes differ")

    # Controlled materialized path: copy the producer output once into a
    # distinct ext_dev DDR allocation, then bind it to the next Executor.
    # Keeping both views on ext_dev avoids mixing RPC CPU-device conversion
    # into the boundary-copy measurement.
    ordinary_bridge = tvm.nd.empty(boundary_shape, boundary_dtype, output_template.device)
    ordinary_cons.set_input_zero_copy(boundary_name, ordinary_bridge)
    # Proposed direct handoff: producer output and consumer input are the same
    # physical DDR allocation. GraphExecutor performs no boundary copy.
    shared_bridge = tvm.nd.empty(boundary_shape, boundary_dtype, output_template.device)
    shared_prod.set_output_zero_copy(0, shared_bridge)
    shared_cons.set_input_zero_copy(boundary_name, shared_bridge)

    def run_mono():
        mono.run()

    def run_ordinary():
        ordinary_prod.run()
        ordinary_prod.get_output(0, ordinary_bridge)
        ordinary_cons.run()

    def run_shared():
        shared_prod.run()
        shared_cons.run()

    for _ in range(args.warmup):
        run_mono()
        run_ordinary()
        run_shared()

    # Use a rotating order so thermal/time drift does not favor one variant.
    costs = {"monolithic": [], "materialized": [], "shared_zero_copy": []}
    variants = [("monolithic", run_mono), ("materialized", run_ordinary),
                ("shared_zero_copy", run_shared)]
    for repeat in range(args.repeat):
        order = variants[repeat % 3:] + variants[:repeat % 3]
        for name, run in order:
            start = time.perf_counter()
            run()
            costs[name].append(time.perf_counter() - start)

    run_mono()
    expected = host_outputs(mono, remote)
    run_ordinary()
    ordinary_output = host_outputs(ordinary_cons, remote)
    run_shared()
    shared_output = host_outputs(shared_cons, remote)
    comparisons = {
        "monolithic_vta_vs_llvm": compare_outputs(expected, cpu_mono_output),
        "materialized_vta_vs_split_llvm": compare_outputs(ordinary_output, cpu_split_output),
        "shared_zero_copy_vta_vs_split_llvm": compare_outputs(shared_output, cpu_split_output),
        "shared_zero_copy_vs_materialized": compare_outputs(shared_output, ordinary_output),
        "materialized_vs_monolithic": compare_outputs(ordinary_output, expected),
        "shared_zero_copy_vs_monolithic": compare_outputs(shared_output, expected),
    }
    required_correct = [
        "monolithic_vta_vs_llvm",
        "materialized_vta_vs_split_llvm",
        "shared_zero_copy_vta_vs_split_llvm",
        "shared_zero_copy_vs_materialized",
    ]
    if not all(comparisons[name]["correct"] for name in required_correct):
        raise AssertionError("VTA execution differs from its reference: {}".format(comparisons))

    profiles = {
        "monolithic": capture_profile(hooks, run_mono, args.events_limit),
        "materialized": capture_profile(hooks, run_ordinary, args.events_limit),
        "shared_zero_copy": capture_profile(hooks, run_shared, args.events_limit),
    }
    summaries = {name: timing_summary(values) for name, values in costs.items()}
    mono_ms = summaries["monolithic"]["median_ms"]
    materialized_ms = summaries["materialized"]["median_ms"]
    shared_ms = summaries["shared_zero_copy"]["median_ms"]
    payload.update({
        "status": "completed",
        "bitstream_expected_sha256":
            "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6",
        "boundary": {
            "name": boundary_name,
            "shape": list(boundary_shape),
            "dtype": boundary_dtype,
            "bytes": int(np.prod(boundary_shape) * np.dtype(boundary_dtype).itemsize),
            "producer_internal_device": str(output_template.device),
            "consumer_internal_device": str(input_template.device),
            "materialized_bridge_device": str(ordinary_bridge.device),
            "shared_bridge_device": str(shared_bridge.device),
            "materialized_framework_copies_per_frame": 1,
            "shared_framework_copies_per_frame": 0,
        },
        "correctness": comparisons,
        "timing": summaries,
        "paired_timing_differences": {
            "materialized_minus_shared": paired_difference_summary(
                costs["materialized"], costs["shared_zero_copy"]
            ),
            "shared_minus_monolithic": paired_difference_summary(
                costs["shared_zero_copy"], costs["monolithic"]
            ),
            "materialized_minus_monolithic": paired_difference_summary(
                costs["materialized"], costs["monolithic"]
            ),
        },
        "derived": {
            "split_materialized_overhead_ms": materialized_ms - mono_ms,
            "split_shared_overhead_ms": shared_ms - mono_ms,
            "zero_copy_saved_ms": materialized_ms - shared_ms,
            "zero_copy_saved_percent_of_materialized":
                (materialized_ms - shared_ms) / materialized_ms * 100.0,
        },
        "runtime_profiles": profiles,
    })
    write_json(args.output, payload)
    print("[BOUNDARY] shape={} dtype={} bytes={}".format(
        boundary_shape, boundary_dtype, payload["boundary"]["bytes"]), flush=True)
    print("[TIMING] mono={:.3f} ms materialized={:.3f} ms shared={:.3f} ms".format(
        mono_ms, materialized_ms, shared_ms), flush=True)
    print("[CORRECT] materialized=True shared_zero_copy=True", flush=True)


if __name__ == "__main__":
    main()
