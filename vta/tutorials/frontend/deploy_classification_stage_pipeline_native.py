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
"""Build and deploy a native CPU/VTA/CPU stage pipeline for ResNet18."""

from __future__ import absolute_import, print_function

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from split_resnet18_stages import (  # pylint: disable=wrong-import-position
    AUTO_RESOURCE_AWARE_SCHEME,
    SCHEMES,
    build_resnet18_unit_blocks,
    build_resnet18_unit_metadata,
    lower_stage_to_relay,
    make_stage_block,
    relay_inputs_for_stage,
    resolve_scheme_config,
    stage_input_schema_for_stage,
    stage_output_schema_for_stage,
    validate_scheme,
)
from profile_split_resnet18_stages import (  # pylint: disable=wrong-import-position
    build_cpu_stage,
    build_vta_stage,
    get_func_output_info,
    schema_nbytes,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def repo_root():
    return Path(__file__).resolve().parents[3]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build/deploy a native ResNet18 CPU/VTA/CPU stage pipeline."
    )
    parser.add_argument("--board", default="", help="SSH target, for example root@192.168.1.247")
    parser.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="Extra option passed to both ssh and scp, for example -oHostKeyAlgorithms=+ssh-rsa",
    )
    parser.add_argument(
        "--remote-dir",
        default="/mnt/sd/vta_stage_pipeline",
        help="Board deploy directory",
    )
    parser.add_argument(
        "--deploy-mode",
        default="sync",
        choices=["sync", "tar"],
        help="sync uploads only changed files; tar uploads/extracts the whole package",
    )
    parser.add_argument(
        "--no-ssh-control-master",
        action="store_true",
        help="Disable SSH ControlMaster connection reuse",
    )
    parser.add_argument(
        "--ssh-control-path",
        default="",
        help="SSH ControlPath; default is /tmp/vta_stage_pipeline_mux_%%r_%%h_%%p",
    )
    parser.add_argument("--model", default="resnet18_v1", choices=["resnet18_v1"])
    parser.add_argument(
        "--scheme",
        default="three_stage_e",
        choices=sorted(list(SCHEMES.keys()) + [AUTO_RESOURCE_AWARE_SCHEME]),
    )
    parser.add_argument(
        "--resource-aware-candidate-name",
        default="",
        help="When --scheme auto_resource_aware, force a specific enumerated candidate window",
    )
    parser.add_argument(
        "--resource-aware-top-k",
        type=int,
        default=10,
        help="When --scheme auto_resource_aware, print this many static-ranked candidates",
    )
    parser.add_argument("--image", default="", help="Input image path; defaults to cached cat image")
    parser.add_argument("--image-dir", default="", help="Directory of input jpg/jpeg/png images")
    parser.add_argument("--max-images", type=int, default=0, help="Maximum images to package")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--runs", type=int, default=20, help="Total frame requests")
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--runtime-num-threads", type=int, default=4)
    parser.add_argument("--stage0-runtime-num-threads", type=int, default=0)
    parser.add_argument("--stage1-runtime-num-threads", type=int, default=0)
    parser.add_argument("--stage2-runtime-num-threads", type=int, default=0)
    parser.add_argument("--serial", action="store_true", help="Run stages serially for correctness")
    parser.add_argument(
        "--run-serial-before-pipeline",
        action="store_true",
        help="Run board-side serial script before pipeline script",
    )
    parser.add_argument(
        "--compare-serial-pipeline",
        action="store_true",
        help="Compare fetched serial and pipeline JSONL top1 results",
    )
    parser.add_argument(
        "--rpc-baseline-result",
        default="",
        help="Optional RPC all_vta baseline JSONL/CSV with input_index and top1 columns",
    )
    parser.add_argument("--serial-output-jsonl", default="stage_serial_result.jsonl")
    parser.add_argument("--pipeline-output-jsonl", default="native_result.jsonl")
    parser.add_argument("--build-dir", default="", help="Host build/package directory")
    parser.add_argument("--keep-build-dir", action="store_true", help="Keep temporary host build dir")
    parser.add_argument("--package-only", action="store_true", help="Build package only")
    parser.add_argument("--skip-copy", action="store_true", help="Do not scp package to board")
    parser.add_argument("--skip-run", action="store_true", help="Do not ssh-run deployed package")
    parser.add_argument(
        "--fetch-results-dir",
        default="",
        help="Optional local directory to scp native_result.jsonl and profiler JSON back",
    )
    parser.add_argument(
        "--vta-runtime-profile-dir",
        default="",
        help="Profile directory inside board deploy dir; empty disables profiler dumps",
    )
    parser.add_argument("--vta-runtime-profile-events-limit", type=int, default=200)
    parser.add_argument("--vta-runtime-profile-checkpoint-every", type=int, default=0)
    return parser.parse_args()


def run(cmd, cwd=None):
    print("[CMD]", " ".join(shlex.quote(str(x)) for x in cmd))
    subprocess.check_call([str(x) for x in cmd], cwd=str(cwd) if cwd else None)


def run_capture(cmd, cwd=None, check=True):
    print("[CMD]", " ".join(shlex.quote(str(x)) for x in cmd))
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc


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

    print("[DATA] preprocess image:", image_path)
    image = Image.open(image_path).resize((int(args.image_size), int(args.image_size)))
    image = image.convert("RGB")
    image = np.array(image).astype("float32") - np.array([123.0, 117.0, 104.0], dtype="float32")
    image /= np.array([58.395, 57.12, 57.375], dtype="float32")
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    image = np.repeat(image, env.BATCH, axis=0).astype("float32")
    return image, image_path


def discover_image_paths(args):
    if not args.image_dir:
        return [Path(args.image) if args.image else Path(default_image_path())]
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        raise RuntimeError("--image-dir is not a directory: {}".format(image_dir))
    images = sorted(
        [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS],
        key=lambda path: path.name,
    )
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError("No jpg/jpeg/png images found in {}".format(image_dir))
    return images


def prepare_inputs(args, env, package_dir):
    image_paths = discover_image_paths(args)
    input_records = []
    first_image = None

    if args.image_dir:
        (package_dir / "inputs").mkdir(parents=True, exist_ok=True)
        input_list_lines = []
        for idx, image_path in enumerate(image_paths):
            image, resolved_path = preprocess_image_path(args, env, str(image_path))
            if first_image is None:
                first_image = image
            rel_input = "inputs/input_{:06d}.bin".format(idx)
            write_bytes(package_dir / rel_input, image.tobytes(order="C"))
            input_list_lines.append(rel_input)
            input_records.append(
                {
                    "index": int(idx),
                    "source_image": str(resolved_path),
                    "input_file": rel_input,
                }
            )
        write_text(package_dir / "inputs.txt", "\n".join(input_list_lines) + "\n")
        return first_image, input_records

    image, resolved_path = preprocess_image_path(args, env, str(image_paths[0]))
    write_bytes(package_dir / "input.bin", image.tobytes(order="C"))
    input_records.append({"index": 0, "source_image": str(resolved_path), "input_file": "input.bin"})
    return image, input_records


def export_stage_lib(lib, out_path):
    from tvm.contrib import cc

    sysroot = os.environ.get("SDKTARGETSYSROOT")
    if not sysroot:
        raise RuntimeError("SDKTARGETSYSROOT is not set; source the AXU5EVB SDK environment first")
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


def build_stage_modules(args, package_dir):
    from mxnet.gluon.model_zoo import vision

    import tvm
    import vta

    env = vta.get_env()
    cpu_target = tvm.target.Target(env.target_vta_cpu, host=env.target_host)

    print("[BUILD] env.TARGET =", env.TARGET)
    print("[BUILD] model      =", args.model)
    print("[BUILD] scheme     =", args.scheme)
    print("[BUILD] image_size =", args.image_size)

    full_model = vision.get_model(args.model, pretrained=True)
    feature_blocks = list(full_model.features)
    output_block = full_model.output
    unit_blocks = build_resnet18_unit_blocks(feature_blocks, output_block)
    unit_metadata = build_resnet18_unit_metadata(unit_blocks, env.BATCH, args.image_size)
    scheme_cfg, selected, _ = resolve_scheme_config(
        args.scheme,
        feature_blocks,
        output_block,
        env.BATCH,
        args.image_size,
        print_resource_aware=(args.scheme == AUTO_RESOURCE_AWARE_SCHEME),
        selected_candidate_name=args.resource_aware_candidate_name or None,
        candidate_top_k=args.resource_aware_top_k,
    )
    resolved_scheme_name = selected["scheme_name"] if selected is not None else args.scheme
    validate_scheme(feature_blocks, scheme_cfg, unit_blocks)
    if [stage["device"] for stage in scheme_cfg] != ["cpu", "vta", "cpu"]:
        raise RuntimeError(
            "stage pipeline requires cpu/vta/cpu scheme, got {}".format(
                [stage["device"] for stage in scheme_cfg]
            )
        )
    if len(scheme_cfg) != 3:
        raise RuntimeError("stage pipeline requires exactly 3 stages")

    stage_records = []
    for idx, stage in enumerate(scheme_cfg):
        stage_name = stage["name"]
        stage_dir = package_dir / "stages" / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_inputs = relay_inputs_for_stage(stage, unit_metadata)
        stage_block = make_stage_block(feature_blocks, output_block, stage, unit_blocks=unit_blocks)
        mod, params = lower_stage_to_relay(stage_block, stage_inputs)
        output_schema = get_func_output_info(mod["main"]) or stage_output_schema_for_stage(
            stage, unit_metadata
        )
        input_schema = stage_input_schema_for_stage(stage, unit_metadata)

        print("\n========== {} ==========".format(stage_name))
        print("[STAGE] device       =", stage["device"])
        print("[STAGE] input_names  =", [name for name, _ in stage_inputs])
        print("[STAGE] input_schema =", input_schema)
        print("[STAGE] output_schema=", output_schema)

        if stage["device"] == "cpu":
            graph, lib, lowered_params = build_cpu_stage(stage_name, mod, params, cpu_target)
        else:
            graph, lib, lowered_params = build_vta_stage(
                stage_name,
                mod["main"],
                params,
                env,
                use_graph_pack=(resolved_scheme_name == "all_vta"),
            )

        graph_path = stage_dir / "graph.json"
        lib_path = stage_dir / "graphlib.so"
        params_path = stage_dir / "params.params"
        write_text(graph_path, graph)
        write_bytes(params_path, tvm.runtime.save_param_dict(lowered_params))
        print("[BUILD] export {} -> {}".format(stage_name, lib_path))
        export_stage_lib(lib, lib_path)

        stage_records.append(
            {
                "index": int(idx),
                "name": stage_name,
                "device": stage["device"],
                "unit_names": list(stage.get("unit_names", [])),
                "input_names": [name for name, _ in stage_inputs],
                "input_schema": input_schema,
                "output_schema": output_schema,
                "output_bytes": schema_nbytes(output_schema),
                "graph": str(graph_path.relative_to(package_dir)),
                "lib": str(lib_path.relative_to(package_dir)),
                "params": str(params_path.relative_to(package_dir)),
            }
        )

    return env, resolved_scheme_name, stage_records


def compile_runner(package_dir):
    root = repo_root()
    sysroot = os.environ.get("SDKTARGETSYSROOT")
    if not sysroot:
        raise RuntimeError("SDKTARGETSYSROOT is not set; source the AXU5EVB SDK environment first")
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
    run(cmd)
    return output


def copy_runtime_libs(package_dir):
    root = repo_root()
    for name in ["libtvm_runtime.so", "libvta.so"]:
        src = root / "build_axu_aarch64" / name
        if not src.exists():
            raise RuntimeError("Missing {}; build AXU5EVB runtime first".format(src))
        shutil.copy2(src, package_dir / name)


def _stage_cli(stage_record, prefix):
    return (
        "  --{p}-graph {graph} \\\n"
        "  --{p}-lib {lib} \\\n"
        "  --{p}-params {params} \\\n"
        "  --{p}-input-names {input_names}"
    ).format(
        p=prefix,
        graph=shlex.quote(stage_record["graph"]),
        lib=shlex.quote(stage_record["lib"]),
        params=shlex.quote(stage_record["params"]),
        input_names=shlex.quote(",".join(stage_record["input_names"])),
    )


def resolve_stage_runtime_threads(args, serial):
    if serial:
        return {
            "stage0": int(args.stage0_runtime_num_threads or args.runtime_num_threads),
            "stage1": int(args.stage1_runtime_num_threads or 1),
            "stage2": int(args.stage2_runtime_num_threads or args.runtime_num_threads),
        }
    return {
        "stage0": int(args.stage0_runtime_num_threads or 3),
        "stage1": int(args.stage1_runtime_num_threads or 1),
        "stage2": int(args.stage2_runtime_num_threads or 1),
    }


def write_run_script(args, package_dir, stage_records, serial, output_jsonl, script_name):
    input_args = "--input-list inputs.txt" if args.image_dir else "--input input.bin"
    serial_arg = " \\\n  --serial" if serial else ""
    stage_threads = resolve_stage_runtime_threads(args, serial)
    tvm_num_threads = max(
        int(args.runtime_num_threads),
        stage_threads["stage0"],
        stage_threads["stage1"],
        stage_threads["stage2"],
    )
    profile_args = ""
    if args.vta_runtime_profile_dir:
        profile_dir = args.vta_runtime_profile_dir.rstrip("/") + ("/serial" if serial else "/pipeline")
        profile_args = (
            " \\\n  --vta-runtime-profile-dir "
            + shlex.quote(profile_dir)
            + " \\\n  --vta-runtime-profile-events-limit "
            + str(int(args.vta_runtime_profile_events_limit))
            + " \\\n  --vta-runtime-profile-checkpoint-every "
            + str(int(args.vta_runtime_profile_checkpoint_every))
        )

    script = """#!/bin/sh
set -eu
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
export LD_PRELOAD="$PWD/libtvm_runtime.so:$PWD/libvta.so"
export TVM_NUM_THREADS={tvm_num_threads}
: "${{TVM_THREAD_POOL_SPIN_COUNT:=0}}"
: "${{AXU5EVB_DRIVER_POST_START_SLEEP_NS:=1000}}"
: "${{AXU5EVB_DRIVER_POLL_SLEEP_NS:=1000}}"
export TVM_THREAD_POOL_SPIN_COUNT
export AXU5EVB_DRIVER_POST_START_SLEEP_NS
export AXU5EVB_DRIVER_POLL_SLEEP_NS

exec ./vta_stage_pipeline_runner \\
{stage0} \\
{stage1} \\
{stage2} \\
  {input_args} \\
  --runs {runs} \\
  --queue-depth {queue_depth} \\
  --runtime-num-threads {runtime_num_threads} \\
  --stage0-runtime-num-threads {stage0_threads} \\
  --stage1-runtime-num-threads {stage1_threads} \\
  --stage2-runtime-num-threads {stage2_threads} \\
  --output-jsonl {output_jsonl}{serial_arg}{profile_args}
""".format(
        stage0=_stage_cli(stage_records[0], "stage0"),
        stage1=_stage_cli(stage_records[1], "stage1"),
        stage2=_stage_cli(stage_records[2], "stage2"),
        input_args=input_args,
        runs=int(args.runs),
        queue_depth=int(args.queue_depth),
        runtime_num_threads=int(args.runtime_num_threads),
        stage0_threads=stage_threads["stage0"],
        stage1_threads=stage_threads["stage1"],
        stage2_threads=stage_threads["stage2"],
        tvm_num_threads=tvm_num_threads,
        output_jsonl=shlex.quote(output_jsonl),
        serial_arg=serial_arg,
        profile_args=profile_args,
    )
    path = package_dir / script_name
    write_text(path, script)
    path.chmod(0o755)


def write_manifest(args, package_dir, env, resolved_scheme_name, stage_records, image, input_records):
    manifest = {
        "kind": "resnet18_native_stage_pipeline",
        "model": args.model,
        "target": env.TARGET,
        "scheme": resolved_scheme_name,
        "requested_scheme": args.scheme,
        "resource_aware_candidate_name": args.resource_aware_candidate_name,
        "image_size": int(args.image_size),
        "input_shape": list(image.shape),
        "input_dtype": str(image.dtype),
        "input_file": input_records[0]["input_file"],
        "input_list": "inputs.txt" if args.image_dir else "",
        "input_count": len(input_records),
        "inputs": input_records,
        "runs": int(args.runs),
        "queue_depth": int(args.queue_depth),
        "runtime_num_threads": int(args.runtime_num_threads),
        "serial_stage_runtime_threads": resolve_stage_runtime_threads(args, serial=True),
        "pipeline_stage_runtime_threads": resolve_stage_runtime_threads(args, serial=False),
        "serial": bool(args.serial),
        "run_serial_before_pipeline": bool(args.run_serial_before_pipeline),
        "serial_output_jsonl": args.serial_output_jsonl,
        "pipeline_output_jsonl": args.pipeline_output_jsonl,
        "rpc_baseline_result": args.rpc_baseline_result,
        "stages": stage_records,
        "runner": "vta_stage_pipeline_runner",
        "runtime_libs": ["libtvm_runtime.so", "libvta.so"],
        "baseline": "RPC all_vta",
    }
    write_text(package_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def package_tar(package_dir):
    tar_path = package_dir.parent / (package_dir.name + ".tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for item in sorted(package_dir.iterdir()):
            tar.add(str(item), arcname=item.name)
    return tar_path


def build_ssh_options(args):
    ssh_options = []
    for option in args.ssh_option:
        ssh_options.extend(["-o", option[2:]] if option.startswith("-o") else ["-o", option])
    if not args.no_ssh_control_master:
        control_path = args.ssh_control_path or "/tmp/vta_stage_pipeline_mux_%r_%h_%p"
        ssh_options.extend(
            [
                "-o",
                "ControlMaster=auto",
                "-o",
                "ControlPersist=10m",
                "-o",
                "ControlPath={}".format(control_path),
            ]
        )
    return ssh_options


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_deploy_manifest(package_dir):
    entries = {}
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(package_dir).as_posix()
        if rel == ".deploy_manifest.json":
            continue
        entries[rel] = {
            "sha256": file_sha256(path),
            "size": int(path.stat().st_size),
        }
    return {"version": 1, "files": entries}


def read_remote_deploy_manifest(args, ssh_options, remote_dir):
    cmd = "cat {}/.deploy_manifest.json 2>/dev/null || true".format(shlex.quote(remote_dir))
    proc = run_capture(["ssh"] + ssh_options + [args.board, cmd], check=False)
    data = proc.stdout.strip()
    if not data:
        return {"version": 1, "files": {}}
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {"version": 1, "files": {}}


def sync_changed_files(args, package_dir, ssh_options, remote_dir):
    local_manifest = build_deploy_manifest(package_dir)
    remote_manifest = read_remote_deploy_manifest(args, ssh_options, remote_dir)
    remote_files = remote_manifest.get("files", {})
    changed = [
        rel
        for rel, meta in local_manifest["files"].items()
        if remote_files.get(rel, {}) != meta
    ]

    run(["ssh"] + ssh_options + [args.board, "mkdir -p {}".format(shlex.quote(remote_dir))])
    if not changed:
        print("[SYNC] remote package is up to date; no files uploaded")
    else:
        dirs = sorted({str(Path(rel).parent).replace("\\", "/") for rel in changed})
        dirs = [d for d in dirs if d and d != "."]
        if dirs:
            mkdir_cmd = "mkdir -p " + " ".join(
                shlex.quote(remote_dir + "/" + rel_dir) for rel_dir in dirs
            )
            run(["ssh"] + ssh_options + [args.board, mkdir_cmd])
        total_bytes = sum(local_manifest["files"][rel]["size"] for rel in changed)
        print("[SYNC] uploading {} changed file(s), {} bytes".format(len(changed), total_bytes))
        for rel in changed:
            local_path = package_dir / rel
            remote_path = "{}:{}/{}".format(args.board, remote_dir, rel)
            run(["scp"] + ssh_options + [str(local_path), remote_path])

    manifest_path = package_dir / ".deploy_manifest.json"
    write_text(manifest_path, json.dumps(local_manifest, indent=2, sort_keys=True) + "\n")
    run(
        ["scp"]
        + ssh_options
        + [str(manifest_path), "{}:{}/.deploy_manifest.json".format(args.board, remote_dir)]
    )


def deploy_and_run(args, package_dir, tar_path):
    if not args.board:
        raise RuntimeError("--board USER@HOST is required unless --package-only is set")
    remote_dir = args.remote_dir.rstrip("/")
    remote_tar = "/tmp/{}".format(tar_path.name)
    ssh_options = build_ssh_options(args)

    if not args.skip_copy:
        if args.deploy_mode == "sync":
            sync_changed_files(args, package_dir, ssh_options, remote_dir)
        else:
            run(["scp"] + ssh_options + [str(tar_path), "{}:{}".format(args.board, remote_tar)])
            remote_cmd = (
                "rm -rf {remote_dir} && mkdir -p {remote_dir} && "
                "tar -xzf {remote_tar} -C {remote_dir}"
            ).format(remote_dir=shlex.quote(remote_dir), remote_tar=shlex.quote(remote_tar))
            run(["ssh"] + ssh_options + [args.board, remote_cmd])

    if not args.skip_run:
        if args.run_serial_before_pipeline:
            run(
                ["ssh"]
                + ssh_options
                + [args.board, "cd {} && ./run_stage_serial.sh".format(shlex.quote(remote_dir))]
            )
            run(
                ["ssh"]
                + ssh_options
                + [args.board, "cd {} && ./run_stage_pipeline.sh".format(shlex.quote(remote_dir))]
            )
        elif args.serial:
            run(
                ["ssh"]
                + ssh_options
                + [args.board, "cd {} && ./run_stage_serial.sh".format(shlex.quote(remote_dir))]
            )
        else:
            run(
                ["ssh"]
                + ssh_options
                + [args.board, "cd {} && ./run_stage_pipeline.sh".format(shlex.quote(remote_dir))]
            )

    if args.fetch_results_dir:
        local_dir = Path(args.fetch_results_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        if args.run_serial_before_pipeline or args.serial:
            run(
                ["scp"]
                + ssh_options
                + [
                    "-r",
                    "{}:{}/{}".format(args.board, remote_dir, args.serial_output_jsonl),
                    str(local_dir),
                ]
            )
        if args.run_serial_before_pipeline or not args.serial:
            run(
                ["scp"]
                + ssh_options
                + [
                    "-r",
                    "{}:{}/{}".format(args.board, remote_dir, args.pipeline_output_jsonl),
                    str(local_dir),
                ]
            )
        run(
            ["scp"]
            + ssh_options
            + ["-r", "{}:{}/manifest.json".format(args.board, remote_dir), str(local_dir)]
        )
        if args.vta_runtime_profile_dir:
            run(
                ["scp"]
                + ssh_options
                + [
                    "-r",
                    "{}:{}/{}".format(args.board, remote_dir, args.vta_runtime_profile_dir),
                    str(local_dir),
                ]
            )
        if args.compare_serial_pipeline:
            compare_jsonl_top1(
                local_dir / args.serial_output_jsonl,
                local_dir / args.pipeline_output_jsonl,
            )
        if args.rpc_baseline_result:
            native_name = args.serial_output_jsonl if (args.run_serial_before_pipeline or args.serial) else args.pipeline_output_jsonl
            compare_rpc_baseline_top1(Path(args.rpc_baseline_result), local_dir / native_name)


def _read_jsonl_by_frame(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as inp:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[int(row["frame_id"])] = row
    return rows


def compare_jsonl_top1(serial_path, pipeline_path):
    serial_rows = _read_jsonl_by_frame(serial_path)
    pipeline_rows = _read_jsonl_by_frame(pipeline_path)
    if set(serial_rows) != set(pipeline_rows):
        raise RuntimeError(
            "Serial/pipeline frame_id mismatch: serial={} pipeline={}".format(
                sorted(serial_rows), sorted(pipeline_rows)
            )
        )
    mismatches = []
    for frame_id in sorted(serial_rows):
        serial_row = serial_rows[frame_id]
        pipeline_row = pipeline_rows[frame_id]
        for key in ["input_index", "input_file", "top1"]:
            if serial_row.get(key) != pipeline_row.get(key):
                mismatches.append(
                    {
                        "frame_id": frame_id,
                        "key": key,
                        "serial": serial_row.get(key),
                        "pipeline": pipeline_row.get(key),
                    }
                )
    if mismatches:
        raise RuntimeError(
            "Serial/pipeline top1 comparison failed: {}".format(
                json.dumps(mismatches[:10], sort_keys=True)
            )
        )
    print("[COMPARE] serial vs pipeline top1 matched for {} frame(s)".format(len(serial_rows)))


def _read_baseline_top1_by_input(path):
    path = Path(path)
    rows = {}
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8") as inp:
            reader = csv.DictReader(inp)
            for row_idx, row in enumerate(reader):
                if "top1" not in row:
                    continue
                input_index = int(row.get("input_index", row_idx))
                rows[input_index] = int(row["top1"])
        return rows

    with open(path, "r", encoding="utf-8") as inp:
        for row_idx, line in enumerate(inp):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "top1" not in row:
                continue
            input_index = int(row.get("input_index", row_idx))
            rows[input_index] = int(row["top1"])
    return rows


def compare_rpc_baseline_top1(baseline_path, native_path):
    baseline = _read_baseline_top1_by_input(baseline_path)
    native_rows = _read_jsonl_by_frame(native_path)
    if not baseline:
        raise RuntimeError("No top1 rows found in RPC baseline {}".format(baseline_path))
    mismatches = []
    for frame_id, native_row in sorted(native_rows.items()):
        input_index = int(native_row["input_index"])
        if input_index not in baseline:
            mismatches.append(
                {
                    "frame_id": frame_id,
                    "input_index": input_index,
                    "error": "missing baseline input_index",
                }
            )
            continue
        if int(native_row["top1"]) != int(baseline[input_index]):
            mismatches.append(
                {
                    "frame_id": frame_id,
                    "input_index": input_index,
                    "baseline_top1": int(baseline[input_index]),
                    "native_top1": int(native_row["top1"]),
                }
            )
    if mismatches:
        raise RuntimeError(
            "RPC baseline/native top1 comparison failed: {}".format(
                json.dumps(mismatches[:10], sort_keys=True)
            )
        )
    print("[COMPARE] RPC baseline vs native top1 matched for {} frame(s)".format(len(native_rows)))


def check_package(package_dir, image_dir_used):
    required = [
        "vta_stage_pipeline_runner",
        "manifest.json",
        "libtvm_runtime.so",
        "libvta.so",
        "run_stage_serial.sh",
        "run_stage_pipeline.sh",
        "stages/stage0_cpu/graph.json",
        "stages/stage0_cpu/graphlib.so",
        "stages/stage0_cpu/params.params",
        "stages/stage1_vta/graph.json",
        "stages/stage1_vta/graphlib.so",
        "stages/stage1_vta/params.params",
        "stages/stage2_cpu/graph.json",
        "stages/stage2_cpu/graphlib.so",
        "stages/stage2_cpu/params.params",
    ]
    for rel in required:
        path = package_dir / rel
        if not path.exists():
            raise RuntimeError("Package missing {}".format(path))
    if image_dir_used:
        for rel in ["inputs", "inputs.txt"]:
            path = package_dir / rel
            if not path.exists():
                raise RuntimeError("Package missing {}".format(path))
    elif not (package_dir / "input.bin").exists():
        raise RuntimeError("Package missing {}".format(package_dir / "input.bin"))


def main():
    args = parse_args()
    if args.model != "resnet18_v1":
        raise RuntimeError("Only resnet18_v1 is supported")
    if args.runs <= 0:
        raise RuntimeError("--runs must be positive")
    if args.queue_depth <= 0:
        raise RuntimeError("--queue-depth must be positive")
    if args.max_images < 0:
        raise RuntimeError("--max-images must be non-negative")
    if args.image and args.image_dir:
        raise RuntimeError("--image and --image-dir are mutually exclusive")
    if args.runtime_num_threads < 0:
        raise RuntimeError("--runtime-num-threads must be non-negative")
    if (
        args.stage0_runtime_num_threads < 0
        or args.stage1_runtime_num_threads < 0
        or args.stage2_runtime_num_threads < 0
    ):
        raise RuntimeError("--stage*-runtime-num-threads must be non-negative")
    if args.compare_serial_pipeline and not args.fetch_results_dir:
        raise RuntimeError("--compare-serial-pipeline requires --fetch-results-dir")
    if args.compare_serial_pipeline and not args.run_serial_before_pipeline:
        raise RuntimeError("--compare-serial-pipeline requires --run-serial-before-pipeline")
    if args.rpc_baseline_result and not args.fetch_results_dir:
        raise RuntimeError("--rpc-baseline-result requires --fetch-results-dir")

    if args.build_dir:
        work_dir = Path(args.build_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="vta_stage_pipeline_"))
        cleanup = not args.keep_build_dir

    package_dir = work_dir / "package"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    try:
        env, resolved_scheme_name, stage_records = build_stage_modules(args, package_dir)
        image, input_records = prepare_inputs(args, env, package_dir)
        compile_runner(package_dir)
        copy_runtime_libs(package_dir)
        write_run_script(
            args,
            package_dir,
            stage_records,
            serial=True,
            output_jsonl=args.serial_output_jsonl,
            script_name="run_stage_serial.sh",
        )
        write_run_script(
            args,
            package_dir,
            stage_records,
            serial=False,
            output_jsonl=args.pipeline_output_jsonl,
            script_name="run_stage_pipeline.sh",
        )
        write_manifest(args, package_dir, env, resolved_scheme_name, stage_records, image, input_records)
        check_package(package_dir, bool(args.image_dir))
        tar_path = package_tar(package_dir)

        print("[PACKAGE]", package_dir)
        print("[PACKAGE]", tar_path)
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
