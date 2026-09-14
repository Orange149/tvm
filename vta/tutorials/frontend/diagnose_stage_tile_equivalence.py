"""Compare one real stage's fallback and tuned VTA outputs with quantized LLVM.

This diagnostic does not accept a numerical mismatch as a performance result.
The reference shares Relay quantization, but not VTA packing/scheduling/runtime.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tvm
from tvm import relay, rpc
from tvm.contrib import graph_executor
from mxnet.gluon.model_zoo import vision
import vta

from profile_split_resnet18_stages import build_vta_stage, create_stage_module, export_and_upload
from split_resnet18_stages import (build_resnet18_unit_blocks, build_resnet18_unit_metadata,
                                  relay_inputs_for_stage, make_stage_block, lower_stage_to_relay)
from run_stage_tile_iteration import write_json


def compare(actual, expected):
    delta = actual.astype("float64") - expected.astype("float64")
    return {"shape": list(actual.shape), "elements": actual.size,
            "mismatched": int(np.count_nonzero(actual != expected)),
            "max_abs_error": float(np.max(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--tune-log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ranking", default="vta/tutorials/frontend/report_out/resource_aware_maxplus/v1_p5b_iteration2_ranked_candidates.json")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    row = json.loads(Path(args.ranking).read_text())["rows"][0]
    stage = next(s for s in row["scheme_cfg"] if s["device"] == "vta")
    env = vta.get_env()
    model = vision.get_model("resnet18_v1", pretrained=True)
    features, head = list(model.features), model.output
    units = build_resnet18_unit_blocks(features, head)
    metadata = build_resnet18_unit_metadata(units, env.BATCH, 224)
    names = relay_inputs_for_stage(stage, metadata)
    block = make_stage_block(features, head, stage, unit_blocks=units)
    mod, params = lower_stage_to_relay(block, names)
    rng = np.random.default_rng(0)
    inputs = {name: rng.uniform(-1, 1, shape).astype("float32") for name, shape in names}
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[]):
            quantized = relay.quantize.quantize(mod, params=params)
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        factory = relay.build(quantized, target="llvm", params=params)
    cpu = graph_executor.GraphModule(factory["default"](tvm.cpu()))
    cpu.set_input(**inputs)
    cpu.run()
    expected = [cpu.get_output(i).numpy() for i in range(cpu.get_num_outputs())]
    np.savez(output / "reference.npz", *expected)
    print("[DIAG] Quantized LLVM reference ready", flush=True)
    report = {"stage": stage, "reference": "same_quantized_Relay_on_unpacked_host_LLVM",
              "reference_limit": "shares_quantizer; not independent full-network reference", "phases": {}}
    for phase, log in [("baseline", ""), ("tuned", args.tune_log)]:
        audit = {}
        graph, lib, lowered = build_vta_stage(stage["name"], mod["main"], params, env,
            tune_log=log, tuning_audit=audit, require_tuned=bool(log))
        remote = rpc.connect(args.host, args.port, session_timeout=180)
        remote.get_function("runtime.config_threadpool")(1, 1)
        library = export_and_upload(lib, remote, env, "equivalence_" + phase)
        executor, _ = create_stage_module(stage["name"], "vta", graph, library, remote)
        executor.set_input(**lowered)
        executor.set_input(**inputs)
        runs = []
        for repeat in range(2):
            executor.run()
            actual = [executor.get_output(i).numpy() for i in range(executor.get_num_outputs())]
            if len(actual) != len(expected):
                raise AssertionError("Reference output arity mismatch")
            np.savez(output / (phase + "_" + str(repeat) + ".npz"), *actual)
            runs.append([compare(a, e) for a, e in zip(actual, expected)])
        report["phases"][phase] = {"tuning": audit, "runs": runs,
            "repeat_identical": [r["actual_sha256"] for r in runs[0]] == [r["actual_sha256"] for r in runs[1]]}
        write_json(output / "comparison.json", report)
        print("[DIAG]", phase, runs, flush=True)
        # This C++ RPC server serializes client sessions. Release every remote
        # handle, including the device tuple, before connecting the next phase.
        del executor, library, remote, _


if __name__ == "__main__":
    main()
