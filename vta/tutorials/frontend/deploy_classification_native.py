#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Build and deploy a VTA ResNet classifier without using a VTA RPC server."""

from __future__ import absolute_import, print_function

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

PACK_DICT = {
    "resnet18_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet18_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet50_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet101_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def repo_root():
    return Path(__file__).resolve().parents[3]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a VTA graph and deploy/run it natively on an AXU5EVB board."
    )
    parser.add_argument("--board", default="", help="SSH target, for example root@192.168.1.247")
    parser.add_argument("--remote-dir", default="/mnt/sd/vta_native", help="Board deploy directory")
    parser.add_argument("--runs", type=int, default=20, help="Native runner inference repeats")
    parser.add_argument("--pipeline", action="store_true", help="Enable experimental GraphExecutor pipeline")
    parser.add_argument("--max-inflight", type=int, default=2, help="Pipeline slot count")
    parser.add_argument("--pipeline-window", type=int, default=2, help="Submitted request window")
    parser.add_argument("--model", default="resnet18_v1", choices=sorted(PACK_DICT))
    parser.add_argument("--tune-log", default="", help="Optional AutoTVM tuning log")
    parser.add_argument("--image", default="", help="Input image path; defaults to cached/downloaded cat")
    parser.add_argument("--image-dir", default="", help="Directory of input jpg/jpeg/png images")
    parser.add_argument("--max-images", type=int, default=0, help="Maximum images to package; 0 means all")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=0)
    parser.add_argument("--pack-start-op", default="", help="Override graph_pack start op")
    parser.add_argument("--pack-stop-op", default="", help="Override graph_pack stop op")
    parser.add_argument(
        "--no-hetero",
        action="store_true",
        help="Build a single VTA target graph instead of the default CPU/VTA heterogeneous graph",
    )
    parser.add_argument("--build-dir", default="", help="Host build/package directory")
    parser.add_argument("--keep-build-dir", action="store_true", help="Keep temporary host build dir")
    parser.add_argument("--skip-copy", action="store_true", help="Do not scp package to the board")
    parser.add_argument("--skip-run", action="store_true", help="Do not ssh-run the deployed package")
    parser.add_argument("--package-only", action="store_true", help="Build package but do not scp or ssh")
    parser.add_argument(
        "--fetch-results-dir",
        default="",
        help="Optional local directory to scp native_result.jsonl and profiler JSON back",
    )
    parser.add_argument(
        "--vta-runtime-profile-dir",
        default="",
        help="Profile directory inside the board deploy dir; empty disables profiler dumps",
    )
    parser.add_argument("--vta-runtime-profile-events-limit", type=int, default=200)
    parser.add_argument("--vta-runtime-profile-checkpoint-every", type=int, default=0)
    return parser.parse_args()


def run(cmd, cwd=None):
    print("[CMD]", " ".join(shlex.quote(str(x)) for x in cmd))
    subprocess.check_call([str(x) for x in cmd], cwd=str(cwd) if cwd else None)


def read_text(path):
    with open(path, "r", encoding="utf-8") as inp:
        return inp.read()


def write_text(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        out.write(data)


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as out:
        out.write(data)


def default_image_path():
    from tvm.contrib import download

    cached = repo_root() / "build_axu_aarch64" / "cat.png"
    if cached.exists():
        return str(cached)
    return download.download("https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg", "cat.png")


def preprocess_image_path(args, env, image_path):
    from PIL import Image
    import numpy as np

    height = args.image_height if args.image_height > 0 else args.image_size
    width = args.image_width if args.image_width > 0 else args.image_size

    print("[DATA] preprocess image:", image_path)
    image = Image.open(image_path).resize((width, height))
    image = np.array(image).astype("float32") - np.array([123.0, 117.0, 104.0], dtype="float32")
    image /= np.array([58.395, 57.12, 57.375], dtype="float32")
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    image = np.repeat(image, env.BATCH, axis=0).astype("float32")
    return image, image_path, height, width


def discover_image_paths(args):
    if not args.image_dir:
        return [Path(args.image) if args.image else Path(default_image_path())]
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        raise RuntimeError(f"--image-dir is not a directory: {image_dir}")
    images = sorted(
        [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS],
        key=lambda path: path.name,
    )
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError(f"No jpg/jpeg/png images found in {image_dir}")
    return images


def prepare_inputs(args, env, package_dir):
    image_paths = discover_image_paths(args)
    input_records = []
    first_image = None
    height = args.image_height if args.image_height > 0 else args.image_size
    width = args.image_width if args.image_width > 0 else args.image_size

    if args.image_dir:
        inputs_dir = package_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        input_list_lines = []
        for idx, image_path in enumerate(image_paths):
            image, resolved_path, height, width = preprocess_image_path(args, env, str(image_path))
            if first_image is None:
                first_image = image
            rel_input = f"inputs/input_{idx:06d}.bin"
            write_bytes(package_dir / rel_input, image.tobytes(order="C"))
            input_list_lines.append(rel_input)
            input_records.append(
                {
                    "index": idx,
                    "source_image": str(resolved_path),
                    "input_file": rel_input,
                }
            )
        write_text(package_dir / "inputs.txt", "\n".join(input_list_lines) + "\n")
        return first_image, input_records, height, width

    image, resolved_path, height, width = preprocess_image_path(args, env, str(image_paths[0]))
    write_bytes(package_dir / "input.bin", image.tobytes(order="C"))
    input_records.append({"index": 0, "source_image": str(resolved_path), "input_file": "input.bin"})
    return image, input_records, height, width


def build_vta_graph(args, package_dir):
    from mxnet.gluon.model_zoo import vision

    import tvm
    from tvm import autotvm, relay
    from tvm.contrib import cc

    import vta
    from vta.top import graph_pack

    env = vta.get_env()
    target = env.target
    target_with_host = tvm.target.Target(target, host=env.target_host)
    height = args.image_height if args.image_height > 0 else args.image_size
    width = args.image_width if args.image_width > 0 else args.image_size
    pack_start = args.pack_start_op or PACK_DICT[args.model][0]
    pack_stop = args.pack_stop_op or PACK_DICT[args.model][1]

    print("[BUILD] env.TARGET =", env.TARGET)
    print("[BUILD] model      =", args.model)
    print("[BUILD] shape      =", (env.BATCH, 3, height, width))
    print("[BUILD] hetero     =", not args.no_hetero)

    build_ctx = autotvm.tophub.context(target)
    if args.tune_log:
        if os.path.exists(args.tune_log):
            print("[BUILD] apply_history_best:", args.tune_log)
            build_ctx = autotvm.apply_history_best(args.tune_log)
        else:
            print("[BUILD] tune log not found, using tophub:", args.tune_log)

    with build_ctx:
        shape_dict = {"data": (env.BATCH, 3, height, width)}
        print("[BUILD] loading Gluon model:", args.model)
        gluon_model = vision.get_model(args.model, pretrained=True)
        print("[BUILD] relay.frontend.from_mxnet ...")
        mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

        print("[BUILD] quantization for VTA ...")
        with tvm.transform.PassContext(opt_level=3):
            with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
                mod = relay.quantize.quantize(mod, params=params)

        print("[BUILD] graph_pack ...")
        relay_prog = graph_pack(
            mod["main"],
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name=pack_start,
            stop_name=pack_stop,
            device_annot=not args.no_hetero,
            annot_start_name="nn.conv2d",
            annot_end_name="annotation.stop_fusion",
        )

        if args.no_hetero:
            build_target = target_with_host
        else:
            build_target = {
                "cpu": tvm.target.Target(env.target_vta_cpu, host=env.target_host),
                "ext_dev": target_with_host,
            }

        print("[BUILD] relay.build ...")
        with vta.build_config(
            opt_level=3,
            disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
        ):
            graph, lib, lowered_params = relay.build(
                relay_prog,
                target=build_target,
                params=params,
            )

    sysroot = os.environ.get("SDKTARGETSYSROOT")
    if not sysroot:
        raise RuntimeError("SDKTARGETSYSROOT is not set; source the AXU5EVB SDK environment first")

    graphlib_path = package_dir / "graphlib.so"
    fcompile = cc.cross_compiler("aarch64-xilinx-linux-g++")
    print("[BUILD] export graphlib.so ->", graphlib_path)
    lib.export_library(
        str(graphlib_path),
        fcompile=fcompile,
        options=[
            f"--sysroot={sysroot}",
            f"-Wl,-rpath-link,{sysroot}/lib",
            f"-Wl,-rpath-link,{sysroot}/usr/lib",
            f"-L{sysroot}/lib",
            f"-L{sysroot}/usr/lib",
        ],
    )

    write_text(package_dir / "graph.json", graph)
    write_bytes(package_dir / "params.params", tvm.runtime.save_param_dict(lowered_params))
    return env


def compile_runner(package_dir):
    root = repo_root()
    sysroot = os.environ.get("SDKTARGETSYSROOT")
    if not sysroot:
        raise RuntimeError("SDKTARGETSYSROOT is not set; source the AXU5EVB SDK environment first")
    cxx = shutil.which("aarch64-xilinx-linux-g++")
    if not cxx:
        raise RuntimeError("aarch64-xilinx-linux-g++ not found in PATH")

    source = root / "vta" / "apps" / "native_deploy" / "vta_native_runner.cc"
    output = package_dir / "vta_native_runner"
    cmd = [
        cxx,
        "-std=c++17",
        "-O2",
        f"--sysroot={sysroot}",
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
        f"-Wl,-rpath-link,{sysroot}/lib",
        f"-Wl,-rpath-link,{sysroot}/usr/lib",
        "-ltvm_runtime",
        "-lvta",
        "-ldl",
        "-pthread",
    ]
    run(cmd)
    return output


def copy_runtime_libs(package_dir):
    root = repo_root()
    for name in ["libtvm_runtime.so", "libvta.so"]:
        src = root / "build_axu_aarch64" / name
        if not src.exists():
            raise RuntimeError(f"Missing {src}; build AXU5EVB runtime first")
        shutil.copy2(src, package_dir / name)


def write_run_script(args, package_dir):
    input_args = "--input-list inputs.txt" if args.image_dir else "--input input.bin"
    pipeline_args = ""
    if args.pipeline:
        pipeline_args = (
            " \\\n  --pipeline"
            + " \\\n  --max-inflight "
            + str(int(args.max_inflight))
            + " \\\n  --pipeline-window "
            + str(int(args.pipeline_window))
        )
    profile_args = ""
    if args.vta_runtime_profile_dir:
        profile_args += (
            " \\\n  --vta-runtime-profile-dir "
            + shlex.quote(args.vta_runtime_profile_dir)
            + " \\\n  --vta-runtime-profile-events-limit "
            + str(int(args.vta_runtime_profile_events_limit))
            + " \\\n  --vta-runtime-profile-checkpoint-every "
            + str(int(args.vta_runtime_profile_checkpoint_every))
        )
    script = f"""#!/bin/sh
set -eu
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
export LD_PRELOAD="$PWD/libtvm_runtime.so:$PWD/libvta.so"
: "${{AXU5EVB_DRIVER_POST_START_SLEEP_NS:=1000}}"
: "${{AXU5EVB_DRIVER_POLL_SLEEP_NS:=1000}}"
export AXU5EVB_DRIVER_POST_START_SLEEP_NS
export AXU5EVB_DRIVER_POLL_SLEEP_NS

exec ./vta_native_runner \\
  --graph graph.json \\
  --lib graphlib.so \\
  --params params.params \\
  {input_args} \\
  --input-name data \\
  --runs {int(args.runs)} \\
  --output-jsonl native_result.jsonl{pipeline_args}{profile_args}
"""
    path = package_dir / "run_native.sh"
    write_text(path, script)
    path.chmod(0o755)


def write_manifest(args, package_dir, env, image, input_records, height, width):
    manifest = {
        "model": args.model,
        "target": env.TARGET,
        "input_name": "data",
        "input_file": input_records[0]["input_file"],
        "input_list": "inputs.txt" if args.image_dir else "",
        "input_count": len(input_records),
        "input_shape": list(image.shape),
        "input_dtype": str(image.dtype),
        "source_image": input_records[0]["source_image"],
        "inputs": input_records,
        "image_height": int(height),
        "image_width": int(width),
        "runs": int(args.runs),
        "pipeline": bool(args.pipeline),
        "max_inflight": int(args.max_inflight),
        "pipeline_window": int(args.pipeline_window),
        "runtime_libs": ["libtvm_runtime.so", "libvta.so"],
        "runner": "vta_native_runner",
    }
    write_text(package_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def package_tar(package_dir):
    tar_path = package_dir.parent / (package_dir.name + ".tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for item in sorted(package_dir.iterdir()):
            tar.add(str(item), arcname=item.name)
    return tar_path


def deploy_and_run(args, package_dir, tar_path):
    if not args.board:
        raise RuntimeError("--board USER@HOST is required unless --package-only is set")
    remote_dir = args.remote_dir.rstrip("/")
    remote_tar = f"/tmp/{tar_path.name}"

    if not args.skip_copy:
        run(["scp", str(tar_path), f"{args.board}:{remote_tar}"])
        remote_cmd = (
            f"rm -rf {shlex.quote(remote_dir)} && "
            f"mkdir -p {shlex.quote(remote_dir)} && "
            f"tar -xzf {shlex.quote(remote_tar)} -C {shlex.quote(remote_dir)}"
        )
        run(["ssh", args.board, remote_cmd])

    if not args.skip_run:
        run(["ssh", args.board, f"cd {shlex.quote(remote_dir)} && ./run_native.sh"])

    if args.fetch_results_dir:
        local_dir = Path(args.fetch_results_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        run(["scp", "-r", f"{args.board}:{remote_dir}/native_result.jsonl", str(local_dir)])
        if args.vta_runtime_profile_dir:
            run(["scp", "-r", f"{args.board}:{remote_dir}/{args.vta_runtime_profile_dir}", str(local_dir)])


def main():
    args = parse_args()
    if args.runs <= 0:
        raise RuntimeError("--runs must be positive")
    if args.max_images < 0:
        raise RuntimeError("--max-images must be non-negative")
    if args.image and args.image_dir:
        raise RuntimeError("--image and --image-dir are mutually exclusive")

    if args.build_dir:
        work_dir = Path(args.build_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="vta_native_"))
        cleanup = not args.keep_build_dir

    package_dir = work_dir / "package"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    try:
        env = build_vta_graph(args, package_dir)
        image, input_records, height, width = prepare_inputs(args, env, package_dir)
        write_manifest(args, package_dir, env, image, input_records, height, width)
        compile_runner(package_dir)
        copy_runtime_libs(package_dir)
        write_run_script(args, package_dir)
        tar_path = package_tar(package_dir)

        print("[PACKAGE]", package_dir)
        print("[PACKAGE]", tar_path)
        for required in [
            "vta_native_runner",
            "graphlib.so",
            "graph.json",
            "params.params",
            "manifest.json",
            "libtvm_runtime.so",
            "libvta.so",
            "run_native.sh",
        ]:
            path = package_dir / required
            if not path.exists():
                raise RuntimeError(f"Package missing {path}")
        if args.image_dir:
            for required in ["inputs", "inputs.txt"]:
                path = package_dir / required
                if not path.exists():
                    raise RuntimeError(f"Package missing {path}")
        elif not (package_dir / "input.bin").exists():
            raise RuntimeError(f"Package missing {package_dir / 'input.bin'}")

        if not args.package_only:
            deploy_and_run(args, package_dir, tar_path)
    finally:
        if cleanup:
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:  # pylint: disable=broad-except
        print("[ERROR]", err, file=sys.stderr)
        sys.exit(1)
