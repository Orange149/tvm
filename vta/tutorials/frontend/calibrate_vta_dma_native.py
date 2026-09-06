#!/usr/bin/env python3
"""Build and run native VTA load/store identification microkernels.

This tool never uses RPC.  It cross-compiles static VTA packed functions, invokes
them through a small native C++ runner, and records one runtime-profiler snapshot
per repeated measurement.  The first matrix is intentionally diagnostic: it
checks whether load/store byte coefficients are identifiable before formal RAMPS
calibration is allowed to proceed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import statistics
import subprocess
import tarfile
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import nnls
import tvm
from tvm import te, topi
from tvm.contrib import cc

import vta


DEFAULT_OUTPUT = (
    "vta/tutorials/frontend/report_out/resource_aware_maxplus/"
    "stage4_calibration_smoke/native_vta_dma_identification"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="build,measure,analyze")
    parser.add_argument("--board", default="root@192.168.1.240")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--remote-dir", default="/var/volatile/ramps_vta_dma_identification")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--inner-repeats", type=int, default=100)
    parser.add_argument("--driver-post-start-sleep-ns", type=int, default=1000)
    parser.add_argument("--driver-poll-sleep-ns", type=int, default=1000)
    parser.add_argument("--case-order-seed", type=int, default=4101)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--ssh-option", action="append", default=[])
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run(command: List[Any], **kwargs) -> None:
    printable = " ".join(shlex.quote(str(item)) for item in command)
    print("[RUN]", printable, flush=True)
    subprocess.run([str(item) for item in command], check=True, **kwargs)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def protocol_hash(value: Dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def shuffled_cases(matrix: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    ordered = list(matrix)
    random.Random(seed).shuffle(ordered)
    return ordered


def _case(
    *,
    case_id: str,
    family: str,
    access_kind: str,
    input_y: int,
    input_x: int,
    output_y: int,
    output_x: int,
    input_count: int = 1,
    output_count: int = 1,
    pad: int = 0,
    pair_id: str = "",
    pair_role: str = "",
    input_fill: str = "pattern",
    logical_input_y: int | None = None,
    logical_input_x: int | None = None,
    alu_repeats: int = 1,
    batch: int,
    block_out: int,
) -> Dict[str, Any]:
    measurement_kind = (
        "vta_compute_slope" if family == "compute_sweep" else
        "matched_access_pair" if pair_id else
        "dma_component_identification"
    )
    return {
        "case_id": case_id,
        "family": family,
        "measurement_kind": measurement_kind,
        "includes": ["device_load", "alu_add", "device_store"],
        "excludes": ["host_set_input", "host_get_output", "host_submit_sync"],
        "eligible_for_direct_physical_model": False,
        "access_kind": access_kind,
        "target_access_kind": access_kind if not pair_id else pair_id.split("_")[0],
        "pair_id": pair_id,
        "pair_role": pair_role,
        "input_fill": input_fill,
        "alu_opcode": "add",
        "alu_fixed_sequence": ["identity_shift", "add"],
        "alu_repeats": alu_repeats,
        "y": output_y,
        "x": output_x,
        "input_y": input_y,
        "input_x": input_x,
        "logical_input_y": logical_input_y or input_y,
        "logical_input_x": logical_input_x or input_x,
        "output_y": output_y,
        "output_x": output_x,
        "pad_top": pad,
        "pad_left": pad,
        "input_count": input_count,
        "output_count": output_count,
        "output_bits": 8,
        "batch": batch,
        "block_out": block_out,
        "elements": output_y * output_x * batch * block_out,
    }


def cases(batch: int | None = None, block_out: int | None = None) -> List[Dict[str, Any]]:
    if batch is None or block_out is None:
        env = vta.get_env()
        batch = env.BATCH if batch is None else batch
        block_out = env.BLOCK_OUT if block_out is None else block_out
    batch = int(batch)
    block_out = int(block_out)
    result: List[Dict[str, Any]] = []
    for y, x in [(2, 16), (8, 16), (16, 16), (16, 32)]:
        for input_count in (1, 2):
            for output_count in (1, 2):
                case_id = "acc{}_{}x{}_out{}".format(input_count, y, x, output_count)
                result.append(_case(
                    case_id=case_id, family="contiguous_load_store",
                    access_kind="contiguous", input_y=y, input_x=x,
                    output_y=y, output_x=x, input_count=input_count,
                    output_count=output_count, batch=batch, block_out=block_out,
                ))
    for input_y, input_x, output_y, output_x in [(8, 20, 8, 16), (16, 24, 16, 16)]:
        for output_count in (1, 2):
            pair_id = "strided_{}x{}_stride{}_out{}".format(
                output_y, output_x, input_x, output_count)
            common = dict(
                family="strided_matched", output_y=output_y, output_x=output_x,
                output_count=output_count, pair_id=pair_id, batch=batch,
                block_out=block_out,
            )
            result.append(_case(
                case_id=pair_id + "_control", access_kind="contiguous",
                input_y=output_y, input_x=output_x, pair_role="control", **common,
            ))
            result.append(_case(
                case_id=pair_id + "_treatment", access_kind="strided",
                input_y=input_y, input_x=input_x, pair_role="treatment", **common,
            ))
    for input_y, input_x, pad in [(8, 16, 1), (16, 16, 2)]:
        output_y = input_y + 2 * pad
        output_x = input_x + 2 * pad
        for output_count in (1, 2):
            pair_id = "padded_{}x{}_p{}_out{}".format(input_y, input_x, pad, output_count)
            common = dict(
                family="padded_matched", output_y=output_y, output_x=output_x,
                output_count=output_count, pad=pad, pair_id=pair_id,
                logical_input_y=input_y, logical_input_x=input_x,
                batch=batch, block_out=block_out,
            )
            result.append(_case(
                case_id=pair_id + "_control", access_kind="contiguous",
                input_y=output_y, input_x=output_x, pair_role="control",
                input_fill="pre_padded", **common,
            ))
            result.append(_case(
                case_id=pair_id + "_treatment", access_kind="padded",
                input_y=input_y, input_x=input_x, pair_role="treatment", **common,
            ))
    for repeats in (1, 2, 4, 8):
        result.append(_case(
            case_id="compute_alu_repeat{}_16x16".format(repeats),
            family="compute_sweep", access_kind="contiguous",
            input_y=16, input_x=16, output_y=16, output_x=16,
            alu_repeats=repeats, batch=batch, block_out=block_out,
        ))
    return result


def make_kernel(case: Dict[str, Any]):
    env = vta.get_env()
    input_shape = (case["input_y"], case["input_x"], env.BATCH, env.BLOCK_OUT)
    output_shape = (case["output_y"], case["output_x"], env.BATCH, env.BLOCK_OUT)
    inputs = []
    input_buffers = []
    for index in range(case["input_count"]):
        tensor = te.placeholder(input_shape, name="input{}".format(index), dtype=env.acc_dtype)
        def make_buffer(source, name):
            if case["access_kind"] == "padded":
                pad = case["pad_top"]
                return topi.nn.pad(
                    source, (pad, pad, 0, 0), (pad, pad, 0, 0), name=name
                )
            return te.compute(
                output_shape,
                lambda *indices: source(*indices),
                name=name,
            )

        buffer = make_buffer(tensor, "input{}_buf".format(index))
        inputs.append(tensor)
        input_buffers.append(buffer)

    if len(input_buffers) == 1:
        acc = te.compute(
            output_shape,
            lambda *indices: input_buffers[0](*indices) + input_buffers[0](*indices),
            name="acc0",
        )
    else:
        acc = te.compute(
            output_shape,
            lambda *indices: input_buffers[0](*indices) + input_buffers[1](*indices),
            name="acc0",
        )
    alu_buffers = [acc]
    for repeat in range(1, int(case["alu_repeats"])):
        previous = alu_buffers[-1]
        alu_buffers.append(te.compute(
            output_shape,
            lambda *indices: previous(*indices) + 1,
            name="acc{}".format(repeat),
        ))
    acc = alu_buffers[-1]
    def make_result(name):
        return te.compute(
            output_shape,
            lambda *indices: acc(*indices).astype(env.inp_dtype),
            name=name,
        )

    results = [make_result("result{}".format(index)) for index in range(case["output_count"])]
    schedule = te.create_schedule([result.op for result in results])
    for buffer in input_buffers:
        schedule[buffer].set_scope(env.acc_scope)
        schedule[buffer].pragma(buffer.op.axis[0], env.dma_copy)
    for buffer in alu_buffers:
        schedule[buffer].set_scope(env.acc_scope)
        schedule[buffer].pragma(buffer.op.axis[0], env.alu)
    for result in results:
        schedule[result].pragma(result.op.axis[0], env.dma_copy)
    return schedule, inputs + results


def _rewrite_self_multiply_as_add():
    """Keep x+x as an ADD after TE simplifies it to x*2."""

    builtin_uop_push = tvm.ir.Op.get("tir.vta.uop_push")

    @tvm.tir.transform.prim_func_pass(opt_level=0)
    def rewrite(func, _, __):
        def post_order(node):
            if not isinstance(node, tvm.tir.Call) or not node.op.same_as(builtin_uop_push):
                return node
            args = list(node.args)
            opcode = int(args[5].value) if isinstance(args[5], tvm.tir.IntImm) else -1
            use_imm = int(args[6].value) if isinstance(args[6], tvm.tir.IntImm) else -1
            immediate = int(args[7].value) if isinstance(args[7], tvm.tir.IntImm) else 0
            if opcode == 4 and use_imm == 1 and immediate == 2:
                args[3] = args[2]
                args[5] = tvm.tir.IntImm("int32", 2)
                args[6] = tvm.tir.IntImm("int32", 0)
                args[7] = tvm.tir.IntImm("int32", 0)
                return tvm.tir.call_intrin("int32", builtin_uop_push, *args)
            return node

        body = tvm.tir.stmt_functor.ir_transform(
            func.body, None, post_order, ["tir.Call"]
        )
        return func.with_body(body)

    return rewrite


def microbench_build_config():
    base = vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"})
    lower_passes = [
        (int(item[0]), item[1]) for item in base.config["tir.add_lower_pass"]
    ]
    inject_index = next(
        index for index, (_, pass_object) in enumerate(lower_passes)
        if "InjectALUIntrin" in str(pass_object)
    )
    lower_passes.insert(inject_index + 1, (2, _rewrite_self_multiply_as_add()))
    return tvm.transform.PassContext(
        config={"tir.add_lower_pass": lower_passes},
        disabled_pass={"tir.CommonSubexprElimTIR"},
    )


def lower_kernel(case: Dict[str, Any]):
    schedule, kernel_args = make_kernel(case)
    with microbench_build_config():
        return tvm.lower(schedule, kernel_args, simple_mode=True)


def build_kernel(case: Dict[str, Any], output: Path) -> None:
    env = vta.get_env()
    schedule, kernel_args = make_kernel(case)
    with microbench_build_config():
        module = vta.build(
            schedule,
            kernel_args,
            tvm.target.Target("ext_dev", host=env.target_host),
            name=case["case_id"],
        )
    sysroot = os.environ.get("SDKTARGETSYSROOT")
    if not sysroot:
        raise RuntimeError("SDKTARGETSYSROOT is not set")
    module.export_library(
        str(output),
        fcompile=cc.cross_compiler("aarch64-xilinx-linux-g++"),
        options=[
            "--sysroot={}".format(sysroot),
            "-Wl,-rpath-link,{}/lib".format(sysroot),
            "-Wl,-rpath-link,{}/usr/lib".format(sysroot),
            "-L{}/lib".format(sysroot),
            "-L{}/usr/lib".format(sysroot),
        ],
    )


def compile_runner(package_dir: Path) -> None:
    root = repo_root()
    sysroot = os.environ.get("SDKTARGETSYSROOT")
    compiler = shutil.which("aarch64-xilinx-linux-g++")
    if not sysroot or not compiler:
        raise RuntimeError("AXU5EVB cross-compilation environment is not active")
    source = root / "vta/apps/native_deploy/vta_dma_microbench_runner.cc"
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "--sysroot={}".format(sysroot),
        "-I", root / "include",
        "-I", root / "3rdparty/dlpack/include",
        "-I", root / "3rdparty/dmlc-core/include",
        "-I", root / "3rdparty/vta-hw/include",
        "-I", root / "vta/include",
        source,
        "-o", package_dir / "vta_dma_microbench_runner",
        "-L", root / "build_axu_aarch64",
        "-Wl,-rpath-link,{}/lib".format(sysroot),
        "-Wl,-rpath-link,{}/usr/lib".format(sysroot),
        "-ltvm_runtime",
        "-lvta",
        "-ldl",
        "-pthread",
    ]
    run(command)


def build(output_dir: Path, selected_case_ids: List[str] | None = None) -> None:
    package_dir = output_dir / "package"
    modules_dir = package_dir / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    env = vta.get_env()
    matrix = cases(env.BATCH, env.BLOCK_OUT)
    selected = set(selected_case_ids or [])
    if selected:
        matrix = [case for case in matrix if case["case_id"] in selected]
        missing = selected - {case["case_id"] for case in matrix}
        if missing:
            raise ValueError("unknown DMA case ids: {}".format(sorted(missing)))
    expected_modules = {case["case_id"] + ".so" for case in matrix}
    for existing in modules_dir.glob("*.so"):
        if existing.name not in expected_modules:
            existing.unlink()
    for index, case in enumerate(matrix, 1):
        print("[BUILD {}/{}] {}".format(index, len(matrix), case["case_id"]), flush=True)
        build_kernel(case, modules_dir / (case["case_id"] + ".so"))
    compile_runner(package_dir)
    root = repo_root()
    for library in ("libtvm_runtime.so", "libvta.so"):
        shutil.copy2(root / "build_axu_aarch64" / library, package_dir / library)
    write_json(
        package_dir / "manifest.json",
        {
            "kind": "ramps_native_vta_dma_identification",
            "cases": matrix,
            "measurement_path": "native_static_packed_function",
            "rpc_used_for_performance": False,
            "physical_store_dtype": "int8",
            "vta_geometry": {"batch": int(env.BATCH), "block_out": int(env.BLOCK_OUT)},
            "unsupported_design_note": (
                "Declaring an int32 output does not change the physical VTA OUT store; "
                "the identification matrix varies output count instead."
            ),
        },
    )
    with tarfile.open(output_dir / "package.tar.gz", "w:gz") as archive:
        archive.add(package_dir, arcname="package")


def transport_options(args: argparse.Namespace) -> List[str]:
    options = args.ssh_option or [
        "HostKeyAlgorithms=+ssh-rsa",
        "PubkeyAcceptedAlgorithms=+ssh-rsa",
    ]
    result = []
    for option in options:
        result.extend(["-o", option])
    return result


def measure(args: argparse.Namespace, output_dir: Path) -> None:
    package_dir = output_dir / "package"
    manifest = json.loads((package_dir / "manifest.json").read_text())
    ssh = ["ssh"] + transport_options(args) + [args.board]
    scp = ["scp"] + transport_options(args)
    results_dir = output_dir / "board_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "case_status.jsonl"
    status_path.write_text("", encoding="utf-8")
    ordered_cases = shuffled_cases(manifest["cases"], args.case_order_seed)
    protocol = {
        "board": args.board,
        "warmup": args.warmup,
        "runs": args.runs,
        "inner_repeats": args.inner_repeats,
        "driver_post_start_sleep_ns": args.driver_post_start_sleep_ns,
        "driver_poll_sleep_ns": args.driver_poll_sleep_ns,
        "case_order_seed": args.case_order_seed,
        "case_order": [case["case_id"] for case in ordered_cases],
        "rpc_used_for_performance": False,
    }
    protocol["protocol_sha256"] = protocol_hash(protocol)
    write_json(output_dir / "measurement_protocol.json", protocol)

    remote_archive = args.remote_dir + ".tar.gz"
    try:
        run(ssh + ["mkdir -p {}".format(shlex.quote(str(Path(args.remote_dir).parent)))], timeout=60)
        run(
            scp + [str(output_dir / "package.tar.gz"), args.board + ":" + remote_archive],
            timeout=600,
        )
        setup = (
            "rm -rf {d} && mkdir -p {d} && tar -xzf {a} -C {d} && "
            "mv {d}/package/* {d}/ && rmdir {d}/package && mkdir -p {d}/results"
        ).format(d=shlex.quote(args.remote_dir), a=shlex.quote(remote_archive))
        run(ssh + [setup], timeout=60)
        for index, case in enumerate(ordered_cases, 1):
            case_id = case["case_id"]
            print("[MEASURE {}/{}] {}".format(index, len(ordered_cases), case_id), flush=True)
            command = (
                "cd {d} && LD_LIBRARY_PATH=. "
                "AXU5EVB_DRIVER_POST_START_SLEEP_NS={post_sleep} "
                "AXU5EVB_DRIVER_POLL_SLEEP_NS={poll_sleep} ./vta_dma_microbench_runner "
                "--lib modules/{cid}.so --function {cid} --case-id {cid} "
                "--access-kind {kind} --y {y} --x {x} "
                "--input-y {input_y} --input-x {input_x} "
                "--logical-input-y {logical_y} --logical-input-x {logical_x} "
                "--output-y {output_y} --output-x {output_x} "
                "--pad-top {pad_top} --pad-left {pad_left} "
                "--batch {batch} --block-out {block_out} "
                "--input-fill {input_fill} --alu-repeats {alu_repeats} "
                "--input-count {inputs} --output-count {outputs} "
                "--output-bits {bits} "
                "--warmup {warmup} --runs {runs} --inner-repeats {inner} "
                "--output-jsonl results/{cid}.jsonl"
            ).format(
                d=shlex.quote(args.remote_dir), cid=shlex.quote(case_id),
                kind=case["access_kind"], y=case["y"], x=case["x"],
                input_y=case["input_y"], input_x=case["input_x"],
                logical_y=case["logical_input_y"], logical_x=case["logical_input_x"],
                output_y=case["output_y"], output_x=case["output_x"],
                pad_top=case["pad_top"], pad_left=case["pad_left"],
                batch=case["batch"], block_out=case["block_out"],
                input_fill=case["input_fill"], alu_repeats=case["alu_repeats"],
                inputs=case["input_count"], outputs=case["output_count"],
                bits=case["output_bits"], warmup=args.warmup, runs=args.runs,
                inner=args.inner_repeats, post_sleep=args.driver_post_start_sleep_ns,
                poll_sleep=args.driver_poll_sleep_ns,
            )
            status: Dict[str, Any] = {"case_id": case_id, "order_index": index}
            try:
                run(ssh + [command], timeout=300)
                status["status"] = "ok"
            except subprocess.TimeoutExpired as error:
                status.update(status="failed", failure_type="timeout", error_summary=str(error))
            except subprocess.CalledProcessError as error:
                status.update(
                    status="failed", failure_type="case_run_failed",
                    error_summary=str(error),
                )
            if status.get("status") == "failed":
                try:
                    subprocess.run(
                        ssh + ["/bin/busybox killall vta_dma_microbench_runner || true"],
                        check=False, timeout=60,
                    )
                except subprocess.TimeoutExpired:
                    status["runner_cleanup_timed_out"] = True
            try:
                fetched = subprocess.run(scp + [
                    args.board + ":" + args.remote_dir + "/results/" + case_id + ".jsonl",
                    str(results_dir / (case_id + ".jsonl")),
                ], check=False, timeout=120)
                status["result_fetched"] = fetched.returncode == 0
                if status.get("status") == "ok" and fetched.returncode != 0:
                    status.update(
                        status="failed", failure_type="result_fetch_failed",
                        error_summary="runner completed but result scp failed",
                    )
            except subprocess.TimeoutExpired as error:
                status["result_fetched"] = False
                if status.get("status") == "ok":
                    status.update(
                        status="failed", failure_type="timeout", error_summary=str(error)
                    )
            append_jsonl(status_path, status)
    finally:
        try:
            subprocess.run(
                ssh + ["rm -rf {} {}".format(
                    shlex.quote(args.remote_dir), shlex.quote(remote_archive))],
                check=False, timeout=60,
            )
        except subprocess.TimeoutExpired:
            print("[WARN] remote cleanup timed out", flush=True)


def analyze(output_dir: Path) -> None:
    manifest = json.loads((output_dir / "package/manifest.json").read_text())
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for case in manifest["cases"]:
        result_path = output_dir / "board_results" / (case["case_id"] + ".jsonl")
        if not result_path.exists():
            failures.append({
                "case_id": case["case_id"], "failure_type": "profile_missing",
                "error_summary": "board result JSONL is missing",
            })
            continue
        samples = [
            json.loads(line)
            for line in result_path.read_text().splitlines()
            if line.strip()
        ]
        if not samples or not all(sample["correct"] for sample in samples):
            failures.append({
                "case_id": case["case_id"], "failure_type": "correctness_failed",
                "error_summary": "no samples or at least one sample failed correctness",
            })
            continue
        inner = int(samples[0]["inner_repeats"])
        keys = [
            "device_run_wait_us", "driver_submit_mmio_us", "load_buffer_2d_bytes",
            "store_buffer_2d_bytes", "load_buffer_2d_calls", "store_buffer_2d_calls",
            "load_buffer_2d_small_calls", "load_buffer_2d_strided_calls",
            "load_buffer_2d_padded_calls",
        ]
        row = dict(case)
        row["sample_total_wall_ms"] = statistics.median(
            sample["wall_us"] for sample in samples
        ) / 1000.0
        row["wall_us"] = statistics.median(sample["wall_us"] / inner for sample in samples)
        for key in keys:
            row[key] = statistics.median(float(sample["profile"].get(key, 0.0)) / inner for sample in samples)
        rows.append(row)

    contiguous_rows = [row for row in rows if row["family"] == "contiguous_load_store"]
    feature_names = [
        "load_buffer_2d_bytes", "store_buffer_2d_bytes",
        "load_buffer_2d_calls", "store_buffer_2d_calls", "intercept",
    ]
    rank = 0
    condition_number = float("inf")
    r_squared = float("nan")
    mae_us = float("nan")
    coefficients = np.zeros(len(feature_names), dtype="float64")
    if len(contiguous_rows) >= len(feature_names):
        matrix = np.asarray(
            [[row[name] for name in feature_names[:-1]] + [1.0] for row in contiguous_rows],
            dtype="float64",
        )
        scales = np.maximum(np.max(np.abs(matrix), axis=0), 1.0)
        rank = int(np.linalg.matrix_rank(matrix / scales))
        condition_number = float(np.linalg.cond(matrix / scales))
        response = np.asarray(
            [row["device_run_wait_us"] for row in contiguous_rows], dtype="float64"
        )
        scaled_coefficients, _ = nnls(matrix / scales, response)
        coefficients = scaled_coefficients / scales
        prediction = matrix @ coefficients
        residual = response - prediction
        total_variance = float(np.sum(np.square(response - np.mean(response))))
        r_squared = 1.0 - float(np.sum(np.square(residual))) / max(total_variance, 1.0e-12)
        mae_us = float(np.mean(np.abs(residual)))
    coefficients_nonnegative = bool(np.all(coefficients >= -1.0e-15))
    contiguous_identification_ready = bool(
        rank == len(feature_names) and condition_number < 100.0
        and coefficients_nonnegative and np.isfinite(r_squared) and r_squared >= 0.90
        and coefficients[0] > 0.0 and coefficients[1] > 0.0
    )
    fitted_parameters = {name: float(value) for name, value in zip(feature_names, coefficients)}

    holdout_apes = []
    holdout_by_size = []
    for holdout_size in sorted({(row["output_y"], row["output_x"]) for row in contiguous_rows}):
        train = [
            row for row in contiguous_rows
            if (row["output_y"], row["output_x"]) != holdout_size
        ]
        test = [
            row for row in contiguous_rows
            if (row["output_y"], row["output_x"]) == holdout_size
        ]
        if len(train) < len(feature_names) or not test:
            continue
        train_matrix = np.asarray(
            [[row[name] for name in feature_names[:-1]] + [1.0] for row in train],
            dtype="float64",
        )
        train_response = np.asarray(
            [row["device_run_wait_us"] for row in train], dtype="float64"
        )
        train_scales = np.maximum(np.max(np.abs(train_matrix), axis=0), 1.0)
        if np.linalg.matrix_rank(train_matrix / train_scales) < len(feature_names):
            continue
        scaled_holdout_coefficients, _ = nnls(train_matrix / train_scales, train_response)
        holdout_coefficients = scaled_holdout_coefficients / train_scales
        size_apes = []
        for row in test:
            vector = np.asarray(
                [row[name] for name in feature_names[:-1]] + [1.0], dtype="float64"
            )
            predicted = float(vector @ holdout_coefficients)
            ape = abs(predicted - row["device_run_wait_us"]) / max(
                row["device_run_wait_us"], 1.0e-12
            ) * 100.0
            size_apes.append(ape)
            holdout_apes.append(ape)
        holdout_by_size.append({
            "output_y": holdout_size[0], "output_x": holdout_size[1],
            "median_ape_percent": float(statistics.median(size_apes)),
            "max_ape_percent": float(max(size_apes)),
        })
    holdout_median_ape = float(statistics.median(holdout_apes)) if holdout_apes else None
    holdout_p95_ape = float(np.percentile(holdout_apes, 95)) if holdout_apes else None
    for row in rows:
        predicted_contiguous_us = (
            fitted_parameters["intercept"]
            + fitted_parameters["load_buffer_2d_bytes"] * row["load_buffer_2d_bytes"]
            + fitted_parameters["store_buffer_2d_bytes"] * row["store_buffer_2d_bytes"]
            + fitted_parameters["load_buffer_2d_calls"] * row["load_buffer_2d_calls"]
            + fitted_parameters["store_buffer_2d_calls"] * row["store_buffer_2d_calls"]
        )
        row["predicted_contiguous_us"] = float(predicted_contiguous_us)
        row["contiguous_model_residual_us"] = float(row["device_run_wait_us"] - predicted_contiguous_us)

    matched_pairs = []
    rows_by_pair: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        if row.get("pair_id"):
            rows_by_pair.setdefault(row["pair_id"], {})[row["pair_role"]] = row
    for pair_id, roles in sorted(rows_by_pair.items()):
        if set(roles) != {"control", "treatment"}:
            continue
        control, treatment = roles["control"], roles["treatment"]
        matched_pairs.append({
            "pair_id": pair_id,
            "target_access_kind": treatment["target_access_kind"],
            "control_case_id": control["case_id"],
            "treatment_case_id": treatment["case_id"],
            "observed_delta_us": treatment["device_run_wait_us"] - control["device_run_wait_us"],
            "byte_adjusted_access_delta_us": (
                treatment["contiguous_model_residual_us"] - control["contiguous_model_residual_us"]
            ),
            "same_alu_opcode": control["alu_opcode"] == treatment["alu_opcode"],
            "same_alu_repeats": control["alu_repeats"] == treatment["alu_repeats"],
            "same_output_shape": (
                control["output_y"], control["output_x"], control["batch"], control["block_out"]
            ) == (
                treatment["output_y"], treatment["output_x"], treatment["batch"],
                treatment["block_out"]
            ),
            "same_store_work": control["output_count"] == treatment["output_count"],
            "same_profile_store_work": (
                control["store_buffer_2d_bytes"] == treatment["store_buffer_2d_bytes"]
                and control["store_buffer_2d_calls"] == treatment["store_buffer_2d_calls"]
            ),
        })

    compute_rows = sorted(
        (row for row in rows if row["family"] == "compute_sweep"),
        key=lambda row: row["alu_repeats"],
    )
    compute_slope_us = None
    compute_fit_r_squared = None
    if len(compute_rows) >= 3:
        compute_x = np.asarray([row["alu_repeats"] for row in compute_rows], dtype="float64")
        compute_y = np.asarray([row["device_run_wait_us"] for row in compute_rows], dtype="float64")
        slope, intercept = np.polyfit(compute_x, compute_y, 1)
        predicted = slope * compute_x + intercept
        variance = float(np.sum(np.square(compute_y - np.mean(compute_y))))
        compute_slope_us = float(slope)
        compute_fit_r_squared = 1.0 - float(np.sum(np.square(compute_y - predicted))) / max(variance, 1e-12)
    compute_dma_fixed = bool(
        compute_rows
        and len({
            (
                row["load_buffer_2d_bytes"], row["store_buffer_2d_bytes"],
                row["load_buffer_2d_calls"], row["store_buffer_2d_calls"],
            )
            for row in compute_rows
        }) == 1
    )

    protocol_path = output_dir / "measurement_protocol.json"
    protocol_sha256_valid = False
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        expected_hash = protocol.pop("protocol_sha256", "")
        protocol_sha256_valid = expected_hash == protocol_hash(protocol)
    sample_total_wall_values = [row["sample_total_wall_ms"] for row in rows]
    sample_window_valid = bool(
        sample_total_wall_values
        and min(sample_total_wall_values) >= 20.0
        and max(sample_total_wall_values) <= 100.0
    )

    access_kinds = sorted({row["access_kind"] for row in rows})
    strided_observed = any(row["load_buffer_2d_strided_calls"] > 0.0 for row in rows)
    padded_observed = any(row["load_buffer_2d_padded_calls"] > 0.0 for row in rows)
    matched_design_ready = bool(
        contiguous_identification_ready
        and strided_observed
        and padded_observed
        and len(matched_pairs) == 8
        and all(
            pair["same_alu_opcode"] and pair["same_alu_repeats"]
            and pair["same_output_shape"] and pair["same_store_work"]
            and pair["same_profile_store_work"]
            for pair in matched_pairs
        )
        and compute_slope_us is not None and compute_slope_us >= 0.0
        and compute_dma_fixed
    )
    qualification_ready = bool(
        not failures and matched_design_ready and protocol_sha256_valid
        and sample_window_valid
        and holdout_median_ape is not None and holdout_median_ape <= 10.0
        and holdout_p95_ape is not None and holdout_p95_ape <= 20.0
    )
    summary = {
        "schema_version": 2,
        "measurement_path": "native_static_packed_function",
        "rpc_used_for_performance": False,
        "case_count": len(manifest["cases"]),
        "successful_case_count": len(rows),
        "failed_case_count": len(failures),
        "all_correct": not failures,
        "design_columns": feature_names,
        "design_rank": rank,
        "full_rank_required": len(feature_names),
        "design_condition_number": condition_number,
        "fit_r_squared": r_squared,
        "fit_mae_us": mae_us,
        "fitted_parameters": fitted_parameters,
        "fitted_load_bandwidth_GBps": 1.0 / coefficients[0] / 1000.0 if coefficients[0] > 0 else None,
        "fitted_store_bandwidth_GBps": 1.0 / coefficients[1] / 1000.0 if coefficients[1] > 0 else None,
        "coefficients_nonnegative": coefficients_nonnegative,
        "contiguous_load_store_identified": contiguous_identification_ready,
        "access_pattern_coverage": access_kinds,
        "strided_load_observed": strided_observed,
        "padded_load_observed": padded_observed,
        "matched_pair_count": len(matched_pairs),
        "matched_pairs": matched_pairs,
        "compute_slope_us_per_alu_repeat": compute_slope_us,
        "compute_fit_r_squared": compute_fit_r_squared,
        "compute_dma_fixed": compute_dma_fixed,
        "leave_one_size_out": {
            "median_ape_percent": holdout_median_ape,
            "p95_ape_percent": holdout_p95_ape,
            "by_size": holdout_by_size,
        },
        "sample_total_wall_ms": {
            "min": min(sample_total_wall_values) if sample_total_wall_values else None,
            "median": statistics.median(sample_total_wall_values) if sample_total_wall_values else None,
            "max": max(sample_total_wall_values) if sample_total_wall_values else None,
            "required_range": [20.0, 100.0],
            "valid": sample_window_valid,
        },
        "protocol_sha256_valid": protocol_sha256_valid,
        "stage4a_r_design_ready": matched_design_ready,
        "stage4a_q_qualification_passed": qualification_ready,
        "formal_calibration_ready": False,
        "blocking_reason": (
            "Stage 4a-Q passed; Stage 4b still requires a separately frozen three-session protocol."
            if qualification_ready else
            "Stage 4a-Q did not pass every correctness, identifiability, holdout, timing, and protocol gate."
        ),
        "failures": failures,
        "rows": rows,
    }
    write_json(output_dir / "dma_identification_summary.json", summary)
    with (output_dir / "dma_identification_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        if rows:
            fieldnames = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    report = [
        "# Native VTA DMA Identification Smoke", "",
        "- Execution: native static packed functions; RPC performance: `false`.",
        "- Cases/success/failure: `{}/{}/{}`.".format(len(manifest["cases"]), len(rows), len(failures)),
        "- Design rank: `{}/{}`.".format(rank, len(feature_names)),
        "- Condition number: `{:.3f}`.".format(condition_number),
        "- Linear diagnostic fit: `R2={:.3f}`, `MAE={:.3f} us`.".format(r_squared, mae_us),
        "- Contiguous load/store identified: `{}`.".format(contiguous_identification_ready),
        "- Stage 4a-R matched design ready: `{}`; matched pairs: `{}`.".format(
            matched_design_ready, len(matched_pairs)
        ),
        "- Compute slope: `{}` us/additional ALU repeat.".format(compute_slope_us),
        "- Compute DMA fixed: `{}`.".format(compute_dma_fixed),
        "- Leave-one-size-out median/P95 APE: `{}` / `{}` percent.".format(
            holdout_median_ape, holdout_p95_ape
        ),
        "- Sample total wall range valid: `{}`; observed `{}` ms.".format(
            sample_window_valid,
            [min(sample_total_wall_values), max(sample_total_wall_values)]
            if sample_total_wall_values else None,
        ),
        "- Protocol SHA256 valid: `{}`.".format(protocol_sha256_valid),
        "- Stage 4a-Q qualification passed: `{}`.".format(qualification_ready),
        "- Formal calibration ready: `{}`.".format(summary["formal_calibration_ready"]), "",
        "| case | access | load B | store B | load calls | store calls | device us | wall us |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            "| `{case_id}` | {access_kind} | {load_buffer_2d_bytes:.0f} | {store_buffer_2d_bytes:.0f} | "
            "{load_buffer_2d_calls:.0f} | {store_buffer_2d_calls:.0f} | "
            "{device_run_wait_us:.3f} | {wall_us:.3f} |".format(**row)
        )
    report.extend(["", "## Matched pairs", ""])
    for pair in matched_pairs:
        report.append(
            "- `{pair_id}`: observed `{observed_delta_us:.3f} us`, byte-adjusted "
            "`{byte_adjusted_access_delta_us:.3f} us`.".format(**pair)
        )
    report.extend(["", "## Gate", "", summary["blocking_reason"]])
    (output_dir / "DMA_IDENTIFICATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    modes = {item.strip() for item in args.mode.split(",") if item.strip()}
    if not modes or not modes.issubset({"build", "measure", "analyze"}):
        raise ValueError("--mode must contain build, measure, and/or analyze")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "build" in modes:
        build(output_dir, args.case_id)
    if "measure" in modes:
        measure(args, output_dir)
    if "analyze" in modes:
        analyze(output_dir)
    print("[RAMPS NATIVE VTA DMA]", output_dir)


if __name__ == "__main__":
    main()
