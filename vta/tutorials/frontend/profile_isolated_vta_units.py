#!/usr/bin/env python3
"""Profile isolated ResNet18 VTA units with full Relay lowering and TopHub."""

import argparse
import hashlib
import json
from pathlib import Path

from mxnet.gluon.model_zoo import vision
import numpy as np
import tvm
from tvm import relay, rpc
from tvm.contrib import graph_executor
import vta

from profile_split_resnet18_stages import build_vta_stage, create_stage_module, export_and_upload
from split_resnet18_stages import (build_resnet18_unit_blocks, build_resnet18_unit_metadata,
                                  lower_stage_to_relay, make_stage_block, relay_inputs_for_stage)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--units", default="layer2_block0_skip_proj,layer3_block0_skip_proj")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    env = vta.get_env()
    model = vision.get_model("resnet18_v1", pretrained=True)
    features, head = list(model.features), model.output
    blocks = build_resnet18_unit_blocks(features, head)
    metadata = build_resnet18_unit_metadata(blocks, env.BATCH, 224)
    remote = rpc.connect(args.host, args.port, session_timeout=180)
    remote.get_function("runtime.config_threadpool")(1, 1)
    results = []

    for unit_name in args.units.split(","):
        stage = {"name": "isolated_" + unit_name, "device": "vta", "unit_names": [unit_name]}
        relay_inputs = relay_inputs_for_stage(stage, metadata)
        block = make_stage_block(features, head, stage, unit_blocks=blocks)
        mod, params = lower_stage_to_relay(block, relay_inputs)
        rng = np.random.default_rng(0)
        inputs = {name: rng.uniform(-1, 1, shape).astype("float32")
                  for name, shape in relay_inputs}

        with tvm.transform.PassContext(opt_level=3):
            with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[]):
                quantized = relay.quantize.quantize(mod, params=params)
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            cpu_factory = relay.build(quantized, target="llvm", params=params)
        cpu = graph_executor.GraphModule(cpu_factory["default"](tvm.cpu()))
        cpu.set_input(**inputs)
        cpu.run()
        expected = [cpu.get_output(i).numpy() for i in range(cpu.get_num_outputs())]

        audit = {}
        graph, lib, lowered_params = build_vta_stage(
            stage["name"], mod["main"], params, env, tuning_audit=audit)
        remote_lib = export_and_upload(lib, remote, env, stage["name"])
        executor, ctx = create_stage_module(stage["name"], "vta", graph, remote_lib, remote)
        executor.set_input(**lowered_params)
        executor.set_input(**inputs)
        executor.run()
        actual = [executor.get_output(i).numpy() for i in range(executor.get_num_outputs())]
        comparisons = [{
            "elements": int(a.size), "mismatched": int(np.count_nonzero(a != e)),
            "max_abs_error": float(np.max(np.abs(a.astype("float64") - e.astype("float64")))),
        } for a, e in zip(actual, expected)]
        if len(actual) != len(expected) or any(row["mismatched"] for row in comparisons):
            raise AssertionError("{} differs from quantized LLVM reference: {}".format(
                unit_name, comparisons))
        timing = executor.module.time_evaluator(
            "run", ctx[0], number=1, repeat=args.repeat)().results
        remote.get_function("vta.runtime.profiler_clear")()
        executor.run()
        profile = json.loads(remote.get_function("vta.runtime.profiler_status")())
        results.append({
            "unit_name": unit_name, "relay_inputs": relay_inputs,
            "reference_correct": True, "comparisons": comparisons,
            "median_ms": float(np.median(timing)) * 1000, "costs_s": list(timing),
            "runtime_profile": profile, "tuning": audit,
            "output_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in actual],
        })
        write_json(args.output, {"status": "running", "host": args.host,
                                 "port": args.port, "results": results})
        print("[ISOLATED] {} correct median_ms={:.3f} load_calls={} load_bytes={}".format(
            unit_name, results[-1]["median_ms"], profile["load_buffer_2d_calls"],
            profile["load_buffer_2d_bytes"]), flush=True)

    payload = {"status": "completed", "host": args.host, "port": args.port,
               "bitstream_expected_sha256": "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6",
               "results": results}
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
