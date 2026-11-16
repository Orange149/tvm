#!/usr/bin/env python3
"""Build one shared-quantization YOLO224 program, without candidate search."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HOLD = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
CFG = HOLD / "20260926_c3_yolo224_static_prepare_run01/resolution_224/yolov3-tiny-224.cfg"
WEIGHTS = ROOT / ".tvm_test_data/darknet/yolov3-tiny.weights"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def capture_workspace(out):
    out.mkdir(parents=True, exist_ok=False)
    for name, argv in (("git_status.txt", ["git", "status", "--short"]),
                       ("existing_changes.patch", ["git", "diff", "--binary"])):
        (out / name).write_bytes(subprocess.check_output(argv, cwd=ROOT))
    files = [ROOT / p for p in (
        "vta/apps/native_deploy/vta_stage_pipeline_runner.cc",
        "vta/apps/native_deploy/vta_shared_buffer_plan.h", "vta/runtime/runtime.cc",
        "vta/runtime/queue_capacity.h", "src/runtime/graph_executor/graph_executor.cc",
        "3rdparty/vta-hw/src/axu5evb/axu5evb_driver.cc")]
    for path in files:
        dest = out / "source_before" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    write(out / "initial_sources.json", {str(p.relative_to(ROOT)): sha(p) for p in files})


def logits_after_quantization(func, relay):
    """Follow each final convolution's scalar epilogue, before YOLO reshape/sigmoid."""
    calls = []
    class Collect(relay.ExprVisitor):
        def visit_call(self, call):
            super().visit_call(call)
            calls.append(call)
    Collect().visit(func.body)
    convs = [c for c in calls if getattr(c.op, "name", "") == "nn.conv2d"]
    if len(convs) != 13:
        raise ValueError("expected exactly 13 YOLO tiny convolutions")
    selected = []
    allowed = {"add", "nn.bias_add", "right_shift", "left_shift", "clip", "cast",
               "multiply", "divide", "round", "annotation.stop_fusion"}
    for conv in (convs[10], convs[12]):
        current = conv
        while True:
            users = [c for c in calls if any(a.same_as(current) for a in c.args)]
            print("LOGIT_CHAIN", getattr(current.op, "name", ""), str(current.checked_type),
                  [(getattr(c.op, "name", ""), str(c.checked_type)) for c in users], flush=True)
            if len(users) != 1 or getattr(users[0].op, "name", "") not in allowed:
                break
            current = users[0]
        selected.append(current)
    shapes = [[int(d) for d in x.checked_type.shape] for x in selected]
    if shapes != [[1, 255, 14, 14], [1, 255, 7, 7]]:
        raise ValueError("unexpected final detection-head shapes: " + str(shapes))
    if any(str(x.checked_type.dtype) != "int8" for x in selected):
        raise ValueError("expected common-graph int8 logits before head reshape/dequantization")
    return selected


def build(out):
    import numpy as np
    import tvm
    from tvm import relay, autotvm
    import vta
    from tvm.relay.testing import darknet
    from search_yolov3_tiny_stage_splits import (
        yolo_relay, make_multisplit_stage_funcs, build_cpu_stage, build_vta_stage,
        stage_output_schema, export_shared_lib)
    if sha(WEIGHTS) != "dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c":
        raise ValueError("weights identity mismatch")
    assets = Path("/home/orange/.tvm_test_data")
    args = SimpleNamespace(cfg_path=str(CFG), weights_path=str(WEIGHTS),
        darknet_lib_path=str(assets / "darknet/libdarknet2.0.so"),
        coco_path=str(assets / "data/coco.names"), font_path=str(assets / "data/arial.ttf"),
        image_path=str(assets / "data/person.jpg"), split_quantization_mode="full_graph")
    env, imported_assets, func, params, named, _, _ = yolo_relay(args)
    shape = [int(x) for x in func.params[0].checked_type.shape]
    if shape != [1, 3, 224, 224]:
        raise ValueError("224 deployment required")
    (out / "common_quantized_relay.json").write_text(tvm.ir.save_json(func))
    (out / "common_quantized_relay.txt").write_text(func.astext(show_meta_data=False))
    heads = logits_after_quantization(func, relay)
    named.update(small_logits51=heads[0], big_logits66=heads[1])
    candidate = {"islands": [{"start_point_idx": 3, "end_point_idx": 11}]}
    stages = make_multisplit_stage_funcs(func, named, candidate, "cpu")
    if [(s["input_point"], s["output_point"], s["device"]) for s in stages] != [
        ("data", "pool2", "cpu"), ("pool2", "logits", "vta"), ("logits", "output", "cpu")]:
        raise ValueError("fixed stage plan changed")
    package = out / "package"
    package.mkdir()
    records = []
    dispatch_records=[]
    cached=[Path('/home/orange/.tvm/tophub')/name for name in ('arm_cpu_v0.08.log','vta_v0.10.log')]
    if not all(p.is_file() for p in cached):raise FileNotFoundError('offline stock TopHub records required')
    class RecordDispatch(autotvm.task.DispatchContext):
        def _query_inside(self,target,workload):
            config=self._old_ctx.query(target,workload)
            dispatch_records.append({'target':str(target),'workload':str(workload),
                'config':config.to_json_dict() if hasattr(config,'to_json_dict') else str(config),
                'is_fallback':config.is_fallback})
            return config
    for i, stage in enumerate(stages):
        print("BUILD", i, stage["name"], flush=True)
        path = package / "stages" / ("stage" + str(i))
        path.mkdir(parents=True)
        (path / "relay.json").write_text(tvm.ir.save_json(stage["func"]))
        (path / "relay.txt").write_text(stage["func"].astext(show_meta_data=False))
        started = time.monotonic()
        with autotvm.apply_history_best([str(p) for p in cached]), RecordDispatch():
            if stage["device"] == "cpu":
                graph, lib, lowered = build_cpu_stage(stage["name"], stage["func"], params, env)
            else:
                graph, lib, lowered = build_vta_stage(stage["name"], stage["func"], params, env,
                                                     already_quantized=True)
        (path / "graph.json").write_text(graph)
        (path / "params.params").write_bytes(tvm.runtime.save_param_dict(lowered))
        # Keep generated code for the last-reader and hidden-effect audit.
        (path / "module.ll").write_text(lib.get_source("ll"))
        export_shared_lib(lib, path / "graphlib.so")
        compiled = json.loads(graph)
        schema = [{"shape": compiled["attrs"]["shape"][1][compiled["node_row_ptr"][h[0]] + h[1]],
                   "dtype": compiled["attrs"]["dltype"][1][compiled["node_row_ptr"][h[0]] + h[1]]}
                  for h in compiled["heads"]]
        record = {"index": i, "name": stage["name"], "device": stage["device"],
            "input_names": stage["input_names"],
            "input_sources": ["input:0"] if i == 0 else [f"stage{i-1}:{j}" for j in range(len(stage["input_names"]))],
            "runtime_num_threads": (3, 1, 4)[i], "output_schema": {"slots": schema},
            "build_seconds": time.monotonic() - started}
        for key, filename in (("graph", "graph.json"), ("lib", "graphlib.so"), ("params", "params.params")):
            record[key] = str((path / filename).relative_to(package))
        record["artifact_sha256"] = {key + "_sha256": sha(package / record[key]) for key in ("graph", "lib", "params")}
        records.append(record)
        print("DONE", i, schema, flush=True)
    if [len(s["output_schema"]["slots"]) for s in records] != [1, 2, 8]:
        raise ValueError("new program output arity requires review")
    write(package / "manifest.json", {"candidate_id": "yolo224_pool2_logits_common_quant_c2",
        "input_shape": shape, "stages": records, "split_quantization_mode": "full_graph",
        "quantization": {"global_scale": 23, "skip_conv_layers": [0],
                         "store_lowbit_output": True, "round_for_shift": True},
        "common_relay_sha256": sha(out / "common_quantized_relay.json"),
        "stock_dispatch_logs":{str(p):sha(p) for p in cached},
        "source_assets": {key: {"path": p, "sha256": sha(p)} for key,p in imported_assets.items()
                          if key.endswith("path")}, "C1_search": False, "C3_optimization": False})
    write(out/'stock_dispatch_records.json',dispatch_records)
    for p in cached:
        target=out/'dispatch_logs'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
    images = [assets / "data/person.jpg", ROOT.parent / "Vitis-AI/external/onnxruntime/images/dog.jpg"]
    images += sorted((ROOT / "n02123597").glob("*.JPEG"))
    seen, inputs = set(), []
    directory = package / "inputs"
    directory.mkdir()
    for image in images:
        digest = sha(image)
        if digest in seen:
            continue
        seen.add(digest)
        array = np.ascontiguousarray(darknet.load_image(str(image), 224, 224)[None], dtype="float32")
        if array.shape != (1, 3, 224, 224):
            raise ValueError("YOLO preprocessing shape mismatch")
        path = directory / (f"{len(inputs):03d}.bin")
        path.write_bytes(array.tobytes())
        inputs.append({"path": str(path.relative_to(package)), "sha256": sha(path),
                       "image": str(image), "image_sha256": digest})
        if len(inputs) == 64:
            break
    if len(inputs) != 64:
        raise ValueError("64 distinct real images required")
    write(package / "inputs.json", inputs)
    write(out / "build_complete.json", {"stages": records, "input_count": len(inputs),
        "numeric_acceptance": "all final outputs byte identical to same-program M0; no tolerance"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--init", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.init:
        capture_workspace(out)
    else:
        build(out)


if __name__ == "__main__":
    main()
