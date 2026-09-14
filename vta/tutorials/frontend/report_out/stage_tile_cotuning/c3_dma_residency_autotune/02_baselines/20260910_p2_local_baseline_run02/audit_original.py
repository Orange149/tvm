"""Local-only post-refactor audit of the frozen VTA original schedule."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import tvm
from tvm import autotvm, rpc
from tvm.contrib import cc
import vta

from tune_resnet18_vta import register_vta_conv2d_template
from vta_autotvm_measure import reference_data


SEEDS = (0, 20250901, 20260910)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_tasks(selected_path, env):
    register_vta_conv2d_template()
    selected = json.loads(Path(selected_path).read_text(encoding="utf-8"))
    rows = []
    for index, item in enumerate(selected):
        task = autotvm.task.create(
            item["workload"][0],
            args=item["workload"][1:],
            target=env.target,
            target_host=env.target_host,
        )
        config = task.config_space.get(item["incumbent_index"])
        rows.append(("W{:02d}".format(index), item, task, config))
    return rows


def instantiate(task, config):
    with task.target:
        schedule, tensors = task.instantiate(config)
    if not config.valid():
        raise RuntimeError("Invalid config {}: {}".format(config.index, config.errors))
    return schedule, tensors


def build(task, config, env):
    schedule, tensors = instantiate(task, config)
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        return vta.build(
            schedule,
            tensors,
            target=tvm.target.Target(env.target, host=env.target_host),
            name="main",
        )


def run_fsim(rows, env):
    if env.TARGET != "sim":
        raise RuntimeError("fsim mode requires a TARGET=sim VTA_HW_PATH")
    from vta.testing import simulator

    if not simulator.enabled():
        raise RuntimeError("VTA FSim is not enabled")
    output = []
    with tempfile.TemporaryDirectory(prefix="c3_p2_fsim_") as temporary:
        for workload_id, item, task, config in rows:
            module = build(task, config, env)
            filename = workload_id + ".o"
            path = Path(temporary) / filename
            module.save(str(path))
            remote = rpc.LocalSession()
            remote.upload(str(path))
            function = remote.load_module(filename)
            device = remote.ext_dev(0)
            passed = []
            for seed in SEEDS:
                data, weight, expected = reference_data(task, seed=seed)
                actual = tvm.nd.empty(expected.shape, "int8", device=device)
                simulator.clear_stats()
                function(
                    tvm.nd.array(data, device),
                    tvm.nd.array(weight, device),
                    actual,
                )
                observed = actual.numpy()
                if not np.array_equal(observed, expected):
                    raise AssertionError(
                        "{} seed {} mismatch_count={}".format(
                            workload_id, seed, int(np.count_nonzero(observed != expected))
                        )
                    )
                passed.append(seed)
            record = {
                "workload_id": workload_id,
                "config_index": item["incumbent_index"],
                "correct_seeds": passed,
            }
            output.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    return output


def run_cross(rows, env):
    if env.TARGET != "axu5evb" or str(env.target_host) != "llvm -keys=arm_cpu,cpu -mtriple=aarch64-linux-gnu":
        raise RuntimeError("cross mode requires the frozen AXU5EVB configuration")
    sysroot = os.environ["SDKTARGETSYSROOT"]
    options = [
        "--sysroot=" + sysroot,
        "-Wl,-rpath-link," + sysroot + "/lib",
        "-Wl,-rpath-link," + sysroot + "/usr/lib",
        "-L" + sysroot + "/lib",
        "-L" + sysroot + "/usr/lib",
    ]
    output = []
    with tempfile.TemporaryDirectory(prefix="c3_p2_cross_") as temporary:
        for workload_id, item, task, config in rows:
            module = build(task, config, env)
            path = Path(temporary) / (workload_id + ".so")
            module.export_library(
                str(path),
                fcompile=cc.cross_compiler("aarch64-xilinx-linux-g++"),
                options=options,
            )
            payload = path.read_bytes()
            record = {
                "workload_id": workload_id,
                "config_index": item["incumbent_index"],
                "binary_size_bytes": len(payload),
                "binary_sha256": sha256_bytes(payload),
            }
            output.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    return output


def run_tir(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output = []
    for workload_id, item, task, config in rows:
        schedule, tensors = instantiate(task, config)
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            module = tvm.lower(schedule, tensors, name="main")
        script = module.script()
        ir_json = tvm.ir.save_json(module).encode()
        path = output_dir / (workload_id + ".tir")
        path.write_text(script, encoding="utf-8")
        record = {
            "workload_id": workload_id,
            "config_index": item["incumbent_index"],
            "tir_script_sha256": sha256_bytes(script.encode()),
            "tir_ir_json_sha256": sha256_bytes(ir_json),
        }
        output.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("fsim", "cross", "tir"))
    parser.add_argument("--selected-tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tir-dir")
    args = parser.parse_args()
    env = vta.get_env()
    rows = load_tasks(args.selected_tasks, env)
    if args.mode == "fsim":
        result = run_fsim(rows, env)
    elif args.mode == "cross":
        result = run_cross(rows, env)
    else:
        if not args.tir_dir:
            parser.error("tir mode requires --tir-dir")
        result = run_tir(rows, Path(args.tir_dir))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
