#!/usr/bin/env python3
"""Standalone YOLOv3-tiny baseline harness for future native pipeline work.

This script intentionally does not modify or depend on deploy_detection.py.  The
first supported path is an RPC all-VTA graph-executor baseline that produces a
stable JSON reference for later native serial/pipeline checks.  Native YOLO
stage-pipeline execution is reported as an explicit unsupported mode until a
YOLO stage splitter exists.
"""

from __future__ import absolute_import, print_function

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np
import tvm
import vta
from tvm import autotvm, relay, rpc
from tvm.contrib import cc, graph_executor, utils
from tvm.contrib.download import download_testdata
from tvm.relay.testing import darknet, yolo_detection
from tvm.relay.testing.darknet import __darknetffi__
from vta.top import graph_pack


MODEL_NAME = "yolov3-tiny"
REPO_URL = "https://github.com/dmlc/web-data/blob/main/darknet/"
PACK_DICT = {
    "yolov3-tiny": ["nn.max_pool2d", "cast", 4, 186],
}
RAW_OUTPUT_MAE_TOL = 1e-3
RAW_OUTPUT_MAX_ABS_TOL = 1e-2
RAW_OUTPUT_SUM_REL_TOL = 1e-6


def parse_modes(text):
    modes = []
    valid = {"rpc_baseline", "native_serial", "native_pipeline"}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in valid:
            raise argparse.ArgumentTypeError(
                "unknown mode {!r}; expected one of {}".format(item, sorted(valid))
            )
        modes.append(item)
    if not modes:
        raise argparse.ArgumentTypeError("--mode must contain at least one mode")
    return modes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone YOLOv3-tiny RPC/native-pipeline baseline harness."
    )
    parser.add_argument("--board", default="", help="SSH/RPC target such as root@192.168.1.185")
    parser.add_argument("--host", default="", help="RPC host; defaults to host parsed from --board")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--device", default="vta", choices=["vta", "arm_cpu"])
    parser.add_argument("--mode", type=parse_modes, default=parse_modes("rpc_baseline"))
    parser.add_argument(
        "--output-dir",
        default="vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/smoke",
    )
    parser.add_argument("--runs", type=int, default=20, help="Benchmark number per repeat.")
    parser.add_argument("--repeat", type=int, default=3, help="Benchmark repeat count.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--skip-reconfig-runtime", action="store_true")
    parser.add_argument("--skip-fpga-program", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.560)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--topk-raw", type=int, default=8)
    parser.add_argument("--topk-detections", type=int, default=20)
    parser.add_argument("--allow-unsupported-native", action="store_true")
    parser.add_argument(
        "--rpc-baseline-json",
        default="",
        help="Existing rpc_all_vta_baseline.json for native-only correctness comparison.",
    )
    parser.add_argument("--remote-dir", default="/var/volatile/yolov3_tiny_pipeline_baseline")
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--runtime-num-threads", type=int, default=4)
    parser.add_argument("--serial-timeout-s", type=int, default=180)
    parser.add_argument("--pipeline-timeout-s", type=int, default=180)
    parser.add_argument("--fetch-timeout-s", type=int, default=120)
    parser.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="Extra ssh/scp option, for example HostKeyAlgorithms=+ssh-rsa.",
    )
    parser.add_argument("--cfg-path", default="", help="Optional local yolov3-tiny.cfg path.")
    parser.add_argument("--weights-path", default="", help="Optional local yolov3-tiny.weights path.")
    parser.add_argument("--darknet-lib-path", default="", help="Optional local libdarknet path.")
    parser.add_argument("--coco-path", default="", help="Optional local coco.names path.")
    parser.add_argument("--font-path", default="", help="Optional local arial.ttf path.")
    parser.add_argument("--image-path", default="", help="Optional local test image path.")
    return parser.parse_args()


def board_host(args):
    if args.host:
        return args.host
    if args.board and "@" in args.board:
        return args.board.rsplit("@", 1)[1].split(":", 1)[0]
    return args.board


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2, sort_keys=True)
        out.write("\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_array(arr):
    contiguous = np.ascontiguousarray(arr)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def json_float(value):
    return float(np.asarray(value).item())


def raw_tensor_signature(arr, topk):
    flat = np.asarray(arr).reshape(-1)
    if flat.size:
        top_idx = np.argsort(np.abs(flat))[-int(topk) :][::-1]
        top_values = [
            {"flat_index": int(idx), "value": float(flat[idx])}
            for idx in top_idx
        ]
    else:
        top_values = []
    return {
        "shape": [int(item) for item in arr.shape],
        "dtype": str(arr.dtype),
        "sha256": sha256_array(arr),
        "size": int(arr.size),
        "min": float(np.min(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "std": float(np.std(arr)) if arr.size else 0.0,
        "sum": float(np.sum(arr)) if arr.size else 0.0,
        "top_abs_values": top_values,
    }


def compare_raw_outputs(reference, candidate):
    ref_outputs = reference.get("raw_outputs", [])
    cand_outputs = candidate.get("raw_outputs", [])
    rows = []
    passes = len(ref_outputs) == len(cand_outputs)
    max_mae = 0.0
    max_abs = 0.0
    for idx, (ref, cand) in enumerate(zip(ref_outputs, cand_outputs)):
        same_shape = ref.get("shape") == cand.get("shape")
        same_dtype = ref.get("dtype") == cand.get("dtype")
        same_hash = ref.get("sha256") == cand.get("sha256")
        mae = abs(float(ref.get("mean", 0.0)) - float(cand.get("mean", 0.0)))
        sum_abs = abs(float(ref.get("sum", 0.0)) - float(cand.get("sum", 0.0)))
        ref_sum_abs = abs(float(ref.get("sum", 0.0)))
        sum_rel = sum_abs / max(ref_sum_abs, 1.0)
        max_mae = max(max_mae, mae)
        max_abs = max(max_abs, sum_abs)
        row_passes = same_shape and same_dtype and (
            same_hash
            or (
                mae <= RAW_OUTPUT_MAE_TOL
                and (sum_abs <= RAW_OUTPUT_MAX_ABS_TOL or sum_rel <= RAW_OUTPUT_SUM_REL_TOL)
            )
        )
        passes = passes and row_passes
        rows.append(
            {
                "index": idx,
                "same_shape": same_shape,
                "same_dtype": same_dtype,
                "same_hash": same_hash,
                "mean_abs_delta": mae,
                "sum_abs_delta": sum_abs,
                "sum_rel_delta": sum_rel,
                "passes": row_passes,
            }
        )
    return {
        "passes_raw_output_gate": bool(passes),
        "output_count_match": len(ref_outputs) == len(cand_outputs),
        "max_mean_abs_delta": max_mae,
        "max_sum_abs_delta": max_abs,
        "rows": rows,
    }


def first_existing_or_download(path, url, filename, module):
    if path:
        path = Path(path).expanduser()
        if not path.exists():
            raise RuntimeError("asset path does not exist: {}".format(path))
        return str(path)
    return download_testdata(url, filename, module=module)


def download_yolo_assets(args):
    cfg_path = first_existing_or_download(
        args.cfg_path,
        "https://github.com/pjreddie/darknet/blob/master/cfg/"
        + MODEL_NAME
        + ".cfg?raw=true",
        MODEL_NAME + ".cfg",
        "darknet",
    )
    weights_path = first_existing_or_download(
        args.weights_path,
        "https://pjreddie.com/media/files/" + MODEL_NAME + ".weights",
        MODEL_NAME + ".weights",
        "darknet",
    )
    if sys.platform in ["linux", "linux2"]:
        darknet_lib_path = first_existing_or_download(
            args.darknet_lib_path,
            REPO_URL + "lib/libdarknet2.0.so?raw=true",
            "libdarknet2.0.so",
            "darknet",
        )
    elif sys.platform == "darwin":
        darknet_lib_path = first_existing_or_download(
            args.darknet_lib_path,
            REPO_URL + "lib_osx/libdarknet_mac2.0.so?raw=true",
            "libdarknet_mac2.0.so",
            "darknet",
        )
    else:
        raise RuntimeError("Darknet lib is not supported on {}".format(sys.platform))
    coco_path = first_existing_or_download(
        args.coco_path,
        REPO_URL + "data/coco.names?raw=true",
        "coco.names",
        "data",
    )
    font_path = first_existing_or_download(
        args.font_path,
        REPO_URL + "data/arial.ttf?raw=true",
        "arial.ttf",
        "data",
    )
    image_path = first_existing_or_download(
        args.image_path,
        REPO_URL + "data/person.jpg?raw=true",
        "person.jpg",
        "data",
    )
    with open(coco_path, encoding="utf-8") as inp:
        names = [line.strip() for line in inp if line.strip()]
    return {
        "cfg_path": cfg_path,
        "weights_path": weights_path,
        "darknet_lib_path": darknet_lib_path,
        "coco_path": coco_path,
        "font_path": font_path,
        "image_path": image_path,
        "names": names,
    }


def load_darknet_net(assets):
    return __darknetffi__.dlopen(assets["darknet_lib_path"]).load_network(
        assets["cfg_path"].encode("utf-8"),
        assets["weights_path"].encode("utf-8"),
        0,
    )


def prepare_input_data(env, net, image_path):
    neth, netw = net.h, net.w
    data_hwc = darknet.load_image(image_path, neth, netw).transpose(1, 2, 0)
    data = data_hwc.transpose((2, 0, 1))
    data = data[np.newaxis, :]
    data = np.repeat(data, env.BATCH, axis=0)
    return data.astype("float32"), data_hwc


def build_yolo_module(env, args, assets, target):
    net = load_darknet_net(assets)
    dshape = (env.BATCH, net.c, net.h, net.w)
    dtype = "float32"
    build_start = time.time()
    mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=dshape)
    if target.device_name == "vta":
        with tvm.transform.PassContext(opt_level=3):
            with relay.quantize.qconfig(
                global_scale=23.0,
                skip_conv_layers=[0],
                store_lowbit_output=True,
                round_for_shift=True,
            ):
                mod = relay.quantize.quantize(mod, params=params)
            pack_spec = PACK_DICT[MODEL_NAME]
            mod = graph_pack(
                mod["main"],
                env.BATCH,
                env.BLOCK_OUT,
                env.WGT_WIDTH,
                start_name=pack_spec[0],
                stop_name=pack_spec[1],
                start_name_idx=pack_spec[2],
                stop_name_idx=pack_spec[3],
            )
    else:
        mod = mod["main"]

    with autotvm.tophub.context(target):
        with vta.build_config(disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}):
            lib = relay.build(
                mod,
                target=tvm.target.Target(target, host=env.target_host),
                params=params,
            )
    build_s = time.time() - build_start
    data, data_hwc = prepare_input_data(env, net, assets["image_path"])
    return {
        "net": net,
        "dshape": dshape,
        "lib": lib,
        "build_s": build_s,
        "data": data,
        "data_hwc": data_hwc,
    }


def connect_remote(env, args):
    if env.TARGET in ["sim", "tsim"]:
        return rpc.LocalSession()
    host = board_host(args)
    if not host:
        raise RuntimeError("--host or --board is required for non-simulator RPC")
    remote = rpc.connect(host, int(args.port))
    if not args.skip_reconfig_runtime:
        vta.reconfig_runtime(remote)
    if args.device == "vta" and not args.skip_fpga_program:
        vta.program_fpga(remote, bitstream=None)
    return remote


def extract_raw_outputs(module, topk):
    output_count = int(module.get_num_outputs())
    outputs = []
    arrays = []
    for idx in range(output_count):
        arr = module.get_output(idx).numpy()
        arrays.append(arr)
        sig = raw_tensor_signature(arr, topk)
        sig["index"] = idx
        outputs.append(sig)
    return arrays, outputs


def yolo_layers_from_outputs(arrays):
    tvm_out = []
    for idx in range(2):
        layer_attr = arrays[idx * 4 + 3]
        layer_out = {
            "type": "Yolo",
            "biases": arrays[idx * 4 + 2],
            "mask": arrays[idx * 4 + 1],
        }
        out_shape = (
            int(layer_attr[0]),
            int(layer_attr[1] // layer_attr[0]),
            int(layer_attr[2]),
            int(layer_attr[3]),
        )
        layer_out["output"] = arrays[idx * 4].reshape(out_shape)
        layer_out["classes"] = int(layer_attr[4])
        tvm_out.append(layer_out)
    return tvm_out


def detection_summary(args, net, assets, data_hwc, arrays):
    neth, netw = net.h, net.w
    img = darknet.load_image_color(assets["image_path"])
    _, im_h, im_w = img.shape
    tvm_out = yolo_layers_from_outputs(arrays)
    dets = yolo_detection.fill_network_boxes(
        (netw, neth),
        (im_w, im_h),
        float(args.threshold),
        1,
        tvm_out,
    )
    last_layer = net.layers[net.n - 1]
    yolo_detection.do_nms_sort(dets, last_layer.classes, float(args.nms_threshold))
    detections = []
    for det in dets:
        valid, item = yolo_detection.get_detections(
            img,
            det,
            float(args.threshold),
            assets["names"],
            last_layer.classes,
        )
        if not valid:
            continue
        probs = np.asarray(det["prob"])
        score = float(np.max(probs)) if probs.size else 0.0
        class_id = int(np.argmax(probs)) if probs.size else int(item["category"])
        detections.append(
            {
                "class_id": class_id,
                "class_name": assets["names"][class_id] if class_id < len(assets["names"]) else "",
                "score": score,
                "objectness": float(det.get("objectness", 0.0)),
                "left": int(item["left"]),
                "top": int(item["top"]),
                "right": int(item["right"]),
                "bottom": int(item["bot"]),
                "labels": list(item["labelstr"]),
            }
        )
    detections.sort(key=lambda row: row["score"], reverse=True)
    return {
        "threshold": float(args.threshold),
        "nms_threshold": float(args.nms_threshold),
        "image_shape_hwc": [int(item) for item in data_hwc.shape],
        "net_shape_hw": [int(neth), int(netw)],
        "detection_count": len(detections),
        "top_detections": detections[: int(args.topk_detections)],
    }


def run_rpc_baseline(args, output_dir):
    env = vta.get_env()
    target = env.target if args.device == "vta" else env.target_vta_cpu
    assets = download_yolo_assets(args)
    build = build_yolo_module(env, args, assets, target)
    result = {
        "mode": "rpc_baseline",
        "status": "compile_only" if args.compile_only else "ok",
        "model": MODEL_NAME,
        "device": args.device,
        "target": str(target),
        "host": board_host(args),
        "port": int(args.port),
        "build_s": float(build["build_s"]),
        "input_shape": [int(item) for item in build["data"].shape],
        "env_target": env.TARGET,
        "compile_only": bool(args.compile_only),
    }
    if args.compile_only:
        write_json(Path(output_dir) / "rpc_all_vta_baseline.json", result)
        return result

    remote = connect_remote(env, args)
    ctx = remote.ext_dev(0) if args.device == "vta" else remote.cpu(0)
    temp = utils.tempdir()
    lib_path = temp.relpath("graphlib.so")
    if env.TARGET in ["sim", "tsim", "intelfocl"]:
        build["lib"].export_library(lib_path, fcompile=cc.create_shared)
        loaded = remote.load_module(lib_path)
    else:
        sysroot = os.environ.get("SDKTARGETSYSROOT", "")
        if not sysroot:
            raise RuntimeError("SDKTARGETSYSROOT is not set; source the AXU5EVB SDK first")
        fcompile = cc.cross_compiler("aarch64-xilinx-linux-g++")
        build["lib"].export_library(
            lib_path,
            fcompile=fcompile,
            options=[
                "--sysroot={}".format(sysroot),
                "-Wl,-rpath-link,{}/lib".format(sysroot),
                "-Wl,-rpath-link,{}/usr/lib".format(sysroot),
                "-L{}/lib".format(sysroot),
                "-L{}/usr/lib".format(sysroot),
            ],
        )
        remote.upload(lib_path)
        loaded = remote.load_module("graphlib.so")
    module = graph_executor.GraphModule(loaded["default"](ctx))
    module.set_input("data", build["data"])
    for _ in range(int(args.warmup)):
        module.run()
    timer = module.module.time_evaluator(
        "run",
        ctx,
        number=int(args.runs),
        repeat=int(args.repeat),
    )
    bench = timer()
    mean_ms = float(bench.mean * 1000.0)
    std_ms = float(np.std(bench.results) * 1000.0)
    module.run()
    arrays, raw_outputs = extract_raw_outputs(module, args.topk_raw)
    result.update(
        {
            "status": "ok",
            "benchmark_number": int(args.runs),
            "benchmark_repeat": int(args.repeat),
            "warmup": int(args.warmup),
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
            "per_sample_mean_ms": mean_ms / float(env.BATCH),
            "raw_outputs": raw_outputs,
            "detection_summary": detection_summary(
                args,
                build["net"],
                assets,
                build["data_hwc"],
                arrays,
            ),
        }
    )
    write_json(Path(output_dir) / "rpc_all_vta_baseline.json", result)
    return result


def repo_root():
    return Path(__file__).resolve().parents[3]


def run_cmd(cmd, timeout=None, cwd=None, stdout_path=None):
    text = " ".join(shlex.quote(str(item)) for item in cmd)
    print("[CMD]", text)
    stdout = subprocess.PIPE
    stderr = subprocess.STDOUT
    out_file = None
    try:
        if stdout_path:
            Path(stdout_path).parent.mkdir(parents=True, exist_ok=True)
            out_file = open(stdout_path, "w", encoding="utf-8")
            stdout = out_file
            stderr = subprocess.STDOUT
        proc = subprocess.run(
            [str(item) for item in cmd],
            cwd=str(cwd) if cwd else None,
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=timeout,
            check=False,
        )
    finally:
        if out_file is not None:
            out_file.close()
    if proc.returncode != 0:
        output = "" if stdout_path else (proc.stdout or "")
        raise RuntimeError("command failed rc={} cmd={}\n{}".format(proc.returncode, text, output))
    return "" if stdout_path else (proc.stdout or "")


def export_shared_lib(lib, out_path):
    sysroot = os.environ.get("SDKTARGETSYSROOT", "")
    if not sysroot:
        raise RuntimeError("SDKTARGETSYSROOT is not set; source the AXU5EVB SDK first")
    fcompile = cc.cross_compiler("aarch64-xilinx-linux-g++")
    lib.export_library(
        str(out_path),
        fcompile=fcompile,
        options=[
            "--sysroot={}".format(sysroot),
            "-Wl,-rpath-link,{}/lib".format(sysroot),
            "-Wl,-rpath-link,{}/usr/lib".format(sysroot),
            "-L{}/lib".format(sysroot),
            "-L{}/usr/lib".format(sysroot),
        ],
    )


def graph_factory_parts(factory):
    if isinstance(factory, tuple):
        return factory
    return factory.get_graph_json(), factory.get_lib(), factory.get_params()


def compile_runner(package_dir):
    root = repo_root()
    sysroot = os.environ.get("SDKTARGETSYSROOT", "")
    if not sysroot:
        raise RuntimeError("SDKTARGETSYSROOT is not set; source the AXU5EVB SDK first")
    cxx = shutil.which("aarch64-xilinx-linux-g++")
    if not cxx:
        raise RuntimeError("aarch64-xilinx-linux-g++ not found in PATH")
    source = root / "vta" / "apps" / "native_deploy" / "vta_stage_pipeline_runner.cc"
    output = package_dir / "vta_stage_pipeline_runner"
    cmd = [
        cxx,
        "-std=c++17",
        "-O2",
        "--sysroot={}".format(sysroot),
        "-I",
        root / "include",
        "-I",
        root / "3rdparty" / "dlpack" / "include",
        "-I",
        root / "3rdparty" / "dmlc-core" / "include",
        "-I",
        root / "3rdparty" / "vta-hw" / "include",
        "-I",
        root / "vta" / "include",
        source,
        "-o",
        output,
        "-L",
        root / "build_axu_aarch64",
        "-Wl,-rpath-link,{}/lib".format(sysroot),
        "-Wl,-rpath-link,{}/usr/lib".format(sysroot),
        "-ltvm_runtime",
        "-lvta",
        "-ldl",
        "-pthread",
    ]
    run_cmd(cmd)
    return output


def copy_runtime_libs(package_dir):
    root = repo_root()
    for name in ["libtvm_runtime.so", "libvta.so"]:
        src = root / "build_axu_aarch64" / name
        if not src.exists():
            raise RuntimeError("missing runtime library {}".format(src))
        shutil.copy2(str(src), str(package_dir / name))


def build_native_package(args, output_dir):
    env = vta.get_env()
    target = env.target if args.device == "vta" else env.target_vta_cpu
    assets = download_yolo_assets(args)
    build = build_yolo_module(env, args, assets, target)
    package_dir = Path(output_dir) / "native_package"
    if package_dir.exists():
        shutil.rmtree(str(package_dir))
    stage_dir = package_dir / "stages" / "stage0_all_vta"
    stage_dir.mkdir(parents=True, exist_ok=True)
    graph, lib, lowered_params = graph_factory_parts(build["lib"])
    (stage_dir / "graph.json").write_text(graph, encoding="utf-8")
    (stage_dir / "params.params").write_bytes(tvm.runtime.save_param_dict(lowered_params))
    export_shared_lib(lib, stage_dir / "graphlib.so")
    (package_dir / "input.bin").write_bytes(build["data"].tobytes(order="C"))
    compile_runner(package_dir)
    copy_runtime_libs(package_dir)

    manifest = {
        "kind": "yolov3_tiny_native_runner_package",
        "stage_split": "single_stage_all_vta",
        "note": (
            "This validates native deployment and runner raw-output handling. "
            "A true cpu/vta/cpu YOLO Relay split is the next implementation step."
        ),
        "model": MODEL_NAME,
        "device": args.device,
        "input_shape": [int(item) for item in build["data"].shape],
        "input_dtype": str(build["data"].dtype),
        "runs": int(args.runs),
        "queue_depth": int(args.queue_depth),
        "stage_count": 1,
        "stages": [
            {
                "index": 0,
                "name": "stage0_all_vta",
                "device": args.device,
                "input_names": ["data"],
                "graph": "stages/stage0_all_vta/graph.json",
                "lib": "stages/stage0_all_vta/graphlib.so",
                "params": "stages/stage0_all_vta/params.params",
            }
        ],
    }
    write_json(package_dir / "manifest.json", manifest)
    write_native_run_script(args, package_dir, serial=True)
    write_native_run_script(args, package_dir, serial=False)
    return package_dir, manifest


def write_native_run_script(args, package_dir, serial):
    output_jsonl = "native_serial_result.jsonl" if serial else "native_pipeline_result.jsonl"
    script_name = "run_native_serial.sh" if serial else "run_native_pipeline.sh"
    serial_arg = " \\\n  --serial" if serial else ""
    script = """#!/bin/sh
set -eu
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
export LD_PRELOAD="$PWD/libtvm_runtime.so:$PWD/libvta.so"
export TVM_NUM_THREADS={runtime_num_threads}
: "${{TVM_THREAD_POOL_SPIN_COUNT:=0}}"
: "${{AXU5EVB_DRIVER_POST_START_SLEEP_NS:=1000}}"
: "${{AXU5EVB_DRIVER_POLL_SLEEP_NS:=1000}}"
export TVM_THREAD_POOL_SPIN_COUNT
export AXU5EVB_DRIVER_POST_START_SLEEP_NS
export AXU5EVB_DRIVER_POLL_SLEEP_NS

exec ./vta_stage_pipeline_runner \\
  --stage0-graph stages/stage0_all_vta/graph.json \\
  --stage0-lib stages/stage0_all_vta/graphlib.so \\
  --stage0-params stages/stage0_all_vta/params.params \\
  --stage0-input-names data \\
  --stage0-name stage0_all_vta \\
  --stage0-device {device} \\
  --stage0-runtime-num-threads 1 \\
  --input input.bin \\
  --runs {runs} \\
  --queue-depth {queue_depth} \\
  --runtime-num-threads {runtime_num_threads} \\
  --output-mode raw \\
  --output-jsonl {output_jsonl}{serial_arg}
""".format(
        device=shlex.quote(args.device),
        runs=int(args.runs),
        queue_depth=int(args.queue_depth),
        runtime_num_threads=int(args.runtime_num_threads),
        output_jsonl=shlex.quote(output_jsonl),
        serial_arg=serial_arg,
    )
    path = package_dir / script_name
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def ssh_target(args):
    if args.board:
        return args.board
    host = board_host(args)
    if not host:
        raise RuntimeError("--board or --host is required for native modes")
    return "root@{}".format(host)


def ssh_options(args):
    options = []
    for option in args.ssh_option:
        options.extend(["-o", option[2:]] if option.startswith("-o") else ["-o", option])
    return options


def upload_and_run_native(args, output_dir, package_dir, serial):
    mode = "native_serial" if serial else "native_pipeline"
    result_name = "native_serial_result" if serial else "native_pipeline_result"
    result_jsonl = result_name + ".jsonl"
    timeout_s = int(args.serial_timeout_s if serial else args.pipeline_timeout_s)
    remote_dir = args.remote_dir.rstrip("/")
    target = ssh_target(args)
    opts = ssh_options(args)
    tar_path = Path(output_dir) / "{}_package.tar.gz".format(result_name)
    with tarfile.open(str(tar_path), "w:gz") as tar:
        for item in sorted(package_dir.iterdir()):
            tar.add(str(item), arcname=item.name)
    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_cmd(["ssh"] + opts + [target, "rm -rf {d} && mkdir -p {d}".format(d=shlex.quote(remote_dir))])
    run_cmd(["scp"] + opts + [str(tar_path), "{}:{}/package.tar.gz".format(target, remote_dir)])
    run_cmd(["ssh"] + opts + [target, "cd {d} && tar xzf package.tar.gz".format(d=shlex.quote(remote_dir))])
    script = "./run_native_serial.sh" if serial else "./run_native_pipeline.sh"
    run_cmd(
        ["ssh"] + opts + [target, "cd {d} && {s}".format(d=shlex.quote(remote_dir), s=script)],
        timeout=timeout_s,
        stdout_path=logs_dir / "{}.log".format(result_name),
    )
    local_jsonl = Path(output_dir) / result_jsonl
    run_cmd(
        ["scp"] + opts + ["{}:{}/{}".format(target, remote_dir, result_jsonl), str(local_jsonl)],
        timeout=int(args.fetch_timeout_s),
    )
    run_cmd(["ssh"] + opts + [target, "rm -rf {}".format(shlex.quote(remote_dir))])
    rows = []
    with local_jsonl.open(encoding="utf-8") as inp:
        for line in inp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("{} produced no result rows".format(mode))
    latencies = [float(row.get("total_latency_ms", 0.0) or 0.0) for row in rows]
    stage0 = [float(row.get("stage0_ms", 0.0) or 0.0) for row in rows]
    mean_ms = float(np.mean(latencies))
    completion_key = "stage0_end_ms"
    if rows and rows[0].get("stage_count"):
        completion_key = "stage{}_end_ms".format(int(rows[0].get("stage_count")) - 1)
    completion_times = [float(row.get(completion_key, 0.0) or 0.0) for row in rows]
    if (not serial) and len(completion_times) > 1 and completion_times[-1] > completion_times[0]:
        throughput_fps = (len(completion_times) - 1) * 1000.0 / (
            completion_times[-1] - completion_times[0]
        )
    else:
        throughput_fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    result = {
        "mode": mode,
        "status": "ok",
        "stage_split": "single_stage_all_vta",
        "native_runner_output_mode": "raw",
        "runs": int(args.runs),
        "queue_depth": int(args.queue_depth),
        "mean_ms": mean_ms,
        "std_ms": float(np.std(latencies)),
        "fps": throughput_fps,
        "throughput_fps": throughput_fps,
        "mean_total_latency_ms": mean_ms,
        "stage0_mean_ms": float(np.mean(stage0)),
        "raw_outputs": rows[-1].get("raw_outputs", []),
        "result_jsonl": result_jsonl,
        "log": "logs/{}.log".format(result_name),
    }
    write_json(Path(output_dir) / "{}.json".format(result_name), result)
    return result


def run_native_mode(args, output_dir, serial):
    package_dir, manifest = build_native_package(args, output_dir)
    result = upload_and_run_native(args, output_dir, package_dir, serial)
    result["package_manifest"] = manifest
    return result


def unsupported_native_result(mode, output_dir):
    result = {
        "mode": mode,
        "status": "not_implemented",
        "passes_correctness_gate": False,
        "reason": (
            "YOLOv3-tiny native stage pipeline needs a detection graph stage splitter. "
            "The existing ResNet18 stage splitter cannot be reused directly because it "
            "depends on ResNet unit metadata and MXNet block boundaries."
        ),
        "next_step": (
            "Implement YOLO coarse units and stage input/output schemas, then connect "
            "them to the existing native runner/package path."
        ),
    }
    filename = "native_serial_result.json" if mode == "native_serial" else "native_pipeline_result.json"
    write_json(Path(output_dir) / filename, result)
    return result


def write_correctness_report(output_dir, results):
    baseline = results.get("rpc_baseline")
    report = {
        "status": "no_comparisons",
        "raw_output_gate": {},
        "detection_gate": None,
        "notes": [],
    }
    baseline_path = Path(output_dir) / "rpc_all_vta_baseline.json"
    if baseline is None or baseline.get("status") not in {"ok", "compile_only"}:
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        else:
            report["notes"].append("RPC baseline is missing or failed.")
    for mode in ["native_serial", "native_pipeline"]:
        item = results.get(mode)
        if not item:
            continue
        if item.get("status") == "not_implemented":
            report["notes"].append("{} is not implemented yet.".format(mode))
            continue
        if baseline and baseline.get("status") == "ok" and item.get("status") == "ok":
            gate = compare_raw_outputs(baseline, item)
            item["passes_correctness_gate"] = bool(gate.get("passes_raw_output_gate"))
            report["raw_output_gate"][mode] = gate
            report["status"] = (
                "passed" if all(
                    entry.get("passes_raw_output_gate")
                    for entry in report["raw_output_gate"].values()
                )
                else "failed"
            )
    write_json(Path(output_dir) / "correctness_report.json", report)
    return report


def render_readme(args, results, correctness):
    lines = [
        "# YOLOv3-tiny Pipeline Baseline",
        "",
        "- Generated: {}".format(time.strftime("%Y-%m-%d %H:%M:%S")),
        "- Board: {}".format(args.board or args.host or "n/a"),
        "- Port: {}".format(args.port),
        "- Modes: {}".format(",".join(args.mode)),
        "- Compile only: {}".format(bool(args.compile_only)),
        "",
        "## Results",
        "",
        "| mode | status | fps | mean ms | notes |",
        "|---|---|---:|---:|---|",
    ]
    for mode in ["rpc_baseline", "native_serial", "native_pipeline"]:
        item = results.get(mode)
        if not item:
            continue
        lines.append(
            "| {} | {} | {:.3f} | {:.3f} | {} |".format(
                mode,
                item.get("status", ""),
                float(item.get("fps", 0.0) or 0.0),
                float(item.get("mean_ms", 0.0) or 0.0),
                item.get("reason", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Correctness",
            "",
            "- Status: {}".format(correctness.get("status")),
            "- Notes: {}".format("; ".join(correctness.get("notes", [])) or "n/a"),
            "",
            "Native modes currently validate a single all-VTA stage through the native "
            "runner with raw multi-output summaries. A true YOLO `cpu/vta/cpu` Relay "
            "stage split is still a separate next step.",
            "",
        ]
    )
    return "\n".join(lines)


def write_summary(output_dir, args, results, correctness):
    rows = []
    for mode, item in results.items():
        rows.append(
            {
                "mode": mode,
                "status": item.get("status", ""),
                "fps": item.get("fps", ""),
                "mean_ms": item.get("mean_ms", ""),
                "std_ms": item.get("std_ms", ""),
                "passes_correctness_gate": item.get("passes_correctness_gate", ""),
                "reason": item.get("reason", ""),
            }
        )
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL_NAME,
        "board": args.board,
        "host": board_host(args),
        "port": int(args.port),
        "modes": list(args.mode),
        "results": results,
        "correctness": correctness,
    }
    write_json(Path(output_dir) / "summary.json", summary)
    write_csv(Path(output_dir) / "summary.csv", rows)
    Path(output_dir, "README.md").write_text(
        render_readme(args, results, correctness),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    results = {}
    if args.rpc_baseline_json and "rpc_baseline" not in args.mode:
        baseline = json.loads(Path(args.rpc_baseline_json).read_text(encoding="utf-8"))
        results["rpc_baseline"] = baseline
        write_json(output_dir / "rpc_all_vta_baseline.json", baseline)
    if "rpc_baseline" in args.mode:
        results["rpc_baseline"] = run_rpc_baseline(args, output_dir)
    for mode in ["native_serial", "native_pipeline"]:
        if mode in args.mode:
            results[mode] = run_native_mode(args, output_dir, serial=(mode == "native_serial"))
    correctness = write_correctness_report(output_dir, results)
    for mode in ["native_serial", "native_pipeline"]:
        if mode in results and results[mode].get("status") == "ok":
            filename = "native_serial_result.json" if mode == "native_serial" else "native_pipeline_result.json"
            write_json(output_dir / filename, results[mode])
    write_summary(output_dir, args, results, correctness)


if __name__ == "__main__":
    main()
