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

"""AXU5EVB-friendly VTA conv2d benchmark with latency decomposition"""

import json
import os
import time
import csv
from contextlib import nullcontext

import numpy as np
from collections import namedtuple

import tvm
from tvm import te, relay, autotvm, topi, rpc
import tvm.topi.testing
from tvm.contrib import utils, cc

import vta
from vta import program_fpga, reconfig_runtime
from vta.testing import simulator
from tvm.autotvm.task import ApplyConfig


Workload = namedtuple(
    "Conv2DWorkload",
    [
        "batch",
        "height",
        "width",
        "in_filter",
        "out_filter",
        "hkernel",
        "wkernel",
        "hpad",
        "wpad",
        "hstride",
        "wstride",
    ],
)

env = vta.get_env()

default_wkls = [
    ("probe.k1.28.ic16.oc16.s1", Workload(env.BATCH, 28, 28, 16, 16, 1, 1, 0, 0, 1, 1)),
    ("probe.k3.28.ic16.oc16.s1", Workload(env.BATCH, 28, 28, 16, 16, 3, 3, 1, 1, 1, 1)),
    ("probe.k3.28.ic16.oc16.s2", Workload(env.BATCH, 28, 28, 16, 16, 3, 3, 1, 1, 2, 2)),
    ("probe.k3.28.ic32.oc32.s1", Workload(env.BATCH, 28, 28, 32, 32, 3, 3, 1, 1, 1, 1)),
    ("resnet-18.C2", Workload(env.BATCH, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1)),
    ("resnet-18.C3", Workload(env.BATCH, 56, 56, 64, 128, 3, 3, 1, 1, 2, 2)),
    ("resnet-18.C4", Workload(env.BATCH, 56, 56, 64, 128, 1, 1, 0, 0, 2, 2)),
    ("resnet-18.C5", Workload(env.BATCH, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1)),
    ("resnet-18.C6", Workload(env.BATCH, 28, 28, 128, 256, 3, 3, 1, 1, 2, 2)),
    ("resnet-18.C7", Workload(env.BATCH, 28, 28, 128, 256, 1, 1, 0, 0, 2, 2)),
    ("resnet-18.P3", Workload(env.BATCH, 14, 14, 256, 512, 1, 1, 0, 0, 2, 2)),
]

# Favor shapes that avoid padding-heavy innermost tiles and keep channels aligned
# with VTA blocks so direct-RPC sweeps can focus on legal schedule space.
vta_friendly_wkls = [
    ("friendly.k1.56.ic64.oc64.s1", Workload(env.BATCH, 56, 56, 64, 64, 1, 1, 0, 0, 1, 1)),
    ("friendly.k3.56.ic64.oc64.s1.valid", Workload(env.BATCH, 56, 56, 64, 64, 3, 3, 0, 0, 1, 1)),
    ("friendly.k1.28.ic128.oc128.s1", Workload(env.BATCH, 28, 28, 128, 128, 1, 1, 0, 0, 1, 1)),
    ("friendly.k3.28.ic128.oc128.s1.valid", Workload(env.BATCH, 28, 28, 128, 128, 3, 3, 0, 0, 1, 1)),
    ("friendly.k1.14.ic256.oc256.s1", Workload(env.BATCH, 14, 14, 256, 256, 1, 1, 0, 0, 1, 1)),
]


def get_workloads():
    workload_set = os.environ.get("VTA_WORKLOAD_SET", "default").strip().lower()
    if workload_set == "friendly":
        return vta_friendly_wkls
    return default_wkls


@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
    x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return x


def register_vta_conv2d_template():
    from tvm.autotvm.task import TaskExtractEnv

    TaskExtractEnv()

    @autotvm.template("conv2d_packed.vta")
    def _topi_nn_conv2d(*args, **kwargs):
        assert not kwargs, "kwargs are not supported in this template"
        data, kernel, strides, padding, dilation, layout, out_dtype = args

        with tvm.target.vta():
            res = vta.top.conv2d_packed(data, kernel, strides, padding, dilation, layout, out_dtype)
            res = topi.right_shift(res, env.WGT_WIDTH)
            res = my_clip(res, 0, (1 << env.OUT_WIDTH) - 1)
            res = topi.cast(res, env.out_dtype)

        if tvm.target.Target.current().device_name == "vta":
            sch = vta.top.schedule_conv2d_packed([res])
        else:
            sch = te.create_schedule([res.op])
        return sch, [data, kernel, res]


def build_conv2d_module(env, remote, s, data, kernel, bias, res, target):
    """
    仿照你 GEMM 的 _build_and_load():
    在 PC 上交叉编译成 .so，再 upload 到板子，再 load_module。
    """
    if "vta" in target.keys:
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            mod = vta.build(
                s,
                [data, kernel, bias, res],
                target=tvm.target.Target(target, host=env.target_host),
                name="conv2d",
            )
    else:
        mod = tvm.build(
            s,
            [data, kernel, bias, res],
            target=tvm.target.Target(target, host=env.target_host),
            name="conv2d",
        )

    tmp = utils.tempdir()
    so_path = tmp.relpath("conv2d.so")

    if env.TARGET in ["sim", "tsim"]:
        mod.export_library(so_path)
    else:
        sysroot = os.environ["SDKTARGETSYSROOT"]
        fcompile = cc.cross_compiler("aarch64-xilinx-linux-g++")
        mod.export_library(
            so_path,
            fcompile=fcompile,
            options=[
                f"--sysroot={sysroot}",
                f"-Wl,-rpath-link,{sysroot}/lib",
                f"-Wl,-rpath-link,{sysroot}/usr/lib",
                f"-L{sysroot}/lib",
                f"-L{sysroot}/usr/lib",
            ],
        )

    remote.upload(so_path)
    return remote.load_module("conv2d.so")


def build_conv2d_debug_module(env, remote, s, data, kernel, bias, res, target, debug_flag):
    """Build a VTA module with runtime debug flags enabled."""
    with vta.build_config(debug_flag=debug_flag, disabled_pass={"tir.CommonSubexprElimTIR"}):
        mod = vta.build(
            s,
            [data, kernel, bias, res],
            target=tvm.target.Target(target, host=env.target_host),
            name="conv2d_debug",
        )

    tmp = utils.tempdir()
    so_path = tmp.relpath("conv2d_debug.so")

    if env.TARGET in ["sim", "tsim"]:
        mod.export_library(so_path)
    else:
        sysroot = os.environ["SDKTARGETSYSROOT"]
        fcompile = cc.cross_compiler("aarch64-xilinx-linux-g++")
        mod.export_library(
            so_path,
            fcompile=fcompile,
            options=[
                f"--sysroot={sysroot}",
                f"-Wl,-rpath-link,{sysroot}/lib",
                f"-Wl,-rpath-link,{sysroot}/usr/lib",
                f"-L{sysroot}/lib",
                f"-L{sysroot}/usr/lib",
            ],
        )

    remote.upload(so_path)
    return remote.load_module("conv2d_debug.so")


def run_conv2d_decomposed(
    env,
    remote,
    wl,
    target,
    csv_writer,
    records,
    check_correctness=False,
    print_ir=False,
    number=20,
    warmup=2,
    config=None,
):
    assert wl.hpad == wl.wpad

    # 只建议先测 VTA，CPU 版本可以后面再加
    if "arm_cpu" in target.keys:
        data_pack = False
        layout = "NCHW"
        conv2d_fcompute = topi.arm_cpu.conv2d_nchw_spatial_pack
        conv2d_fschedule = topi.arm_cpu.schedule_conv2d_nchw_spatial_pack
        device_name = "CPU"
    elif "vta" in target.keys:
        data_pack = True
        layout = "NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN)
        conv2d_fcompute = vta.top.conv2d_packed
        conv2d_fschedule = vta.top.schedule_conv2d_packed
        device_name = "VTA"
    else:
        raise RuntimeError("Unsupported target")

    # 原始 shape
    a_shape = (wl.batch, wl.in_filter, wl.height, wl.width)
    w_shape = (wl.out_filter, wl.in_filter, wl.hkernel, wl.wkernel)
    b_shape = (wl.batch, wl.out_filter, 1, 1)

    # packed shape
    if data_pack:
        data_shape = (
            wl.batch // env.BATCH,
            wl.in_filter // env.BLOCK_IN,
            wl.height,
            wl.width,
            env.BATCH,
            env.BLOCK_IN,
        )
        kernel_shape = (
            wl.out_filter // env.BLOCK_OUT,
            wl.in_filter // env.BLOCK_IN,
            wl.hkernel,
            wl.wkernel,
            env.BLOCK_OUT,
            env.BLOCK_IN,
        )
        bias_shape = (
            wl.batch // env.BATCH,
            wl.out_filter // env.BLOCK_OUT,
            1,
            1,
            env.BATCH,
            env.BLOCK_OUT,
        )
    else:
        data_shape = a_shape
        kernel_shape = w_shape
        bias_shape = b_shape

    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    kernel = te.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
    bias = te.placeholder(bias_shape, name="bias", dtype=env.acc_dtype)

    padding = relay.nn.get_pad_tuple2d((wl.hpad, wl.wpad))

    cfg_ctx = ApplyConfig(config) if config is not None else nullcontext()
    with cfg_ctx:
        with target:
            if data_pack:
                res = conv2d_fcompute(
                    data,
                    kernel,
                    (wl.hstride, wl.wstride),
                    padding,
                    (1, 1),
                    layout,
                    env.acc_dtype,
                )
            else:
                res = conv2d_fcompute(
                    data,
                    kernel,
                    (wl.hstride, wl.wstride),
                    padding,
                    (1, 1),
                    env.acc_dtype,
                )

            res = topi.right_shift(res, 8)
            res = topi.add(res, bias)
            res = my_clip(res, 0, (1 << env.OUT_WIDTH) - 1)
            res = topi.cast(res, env.out_dtype)

            s = conv2d_fschedule([res])
            if print_ir:
                print(vta.lower(s, [data, kernel, bias, res], simple_mode=True))

    # 输出 feature map 尺寸
    fout_height = (wl.height + 2 * wl.hpad - wl.hkernel) // wl.hstride + 1
    fout_width = (wl.width + 2 * wl.wpad - wl.wkernel) // wl.wstride + 1

    num_ops = (
        2
        * wl.batch
        * fout_height
        * fout_width
        * wl.hkernel
        * wl.wkernel
        * wl.out_filter
        * wl.in_filter
    )

    def get_input_data():
        a_min, a_max = -(1 << (env.INP_WIDTH - 1)), (1 << (env.INP_WIDTH - 1))
        w_min, w_max = -(1 << (env.WGT_WIDTH - 1)), (1 << (env.WGT_WIDTH - 1))
        b_min = -(1 << (env.INP_WIDTH + env.WGT_WIDTH - 2))
        b_max = (1 << (env.INP_WIDTH + env.WGT_WIDTH - 2))

        a_np = np.random.randint(a_min, a_max, size=a_shape).astype(env.inp_dtype)
        w_np = np.random.randint(w_min, w_max, size=w_shape).astype(env.wgt_dtype)
        b_np = np.random.randint(b_min, b_max, size=b_shape).astype(env.acc_dtype)
        return a_np, w_np, b_np

    def get_ref_data(a_np, w_np):
        r_np = tvm.topi.testing.conv2d_nchw_python(
            a_np.astype(env.acc_dtype),
            w_np.astype(env.acc_dtype),
            (wl.hstride, wl.wstride),
            wl.hpad,
        ).astype(env.acc_dtype)
        return r_np

    def pack_inputs(a_np, w_np, b_np):
        if not data_pack:
            return a_np, w_np, b_np, 0.0, 0.0, 0.0

        t_pack_in0 = time.perf_counter()
        data_np = a_np.reshape(
            wl.batch // env.BATCH,
            env.BATCH,
            wl.in_filter // env.BLOCK_IN,
            env.BLOCK_IN,
            wl.height,
            wl.width,
        ).transpose((0, 2, 4, 5, 1, 3))
        t_pack_in1 = time.perf_counter()

        t_pack_w0 = time.perf_counter()
        kernel_np = w_np.reshape(
            wl.out_filter // env.BLOCK_OUT,
            env.BLOCK_OUT,
            wl.in_filter // env.BLOCK_IN,
            env.BLOCK_IN,
            wl.hkernel,
            wl.wkernel,
        ).transpose((0, 2, 4, 5, 1, 3))
        t_pack_w1 = time.perf_counter()

        t_pack_b0 = time.perf_counter()
        bias_np = b_np.reshape(
            wl.batch // env.BATCH,
            wl.out_filter // env.BLOCK_OUT,
            1,
            1,
            env.BATCH,
            env.BLOCK_OUT,
        )
        t_pack_b1 = time.perf_counter()
        return (
            data_np,
            kernel_np,
            bias_np,
            t_pack_in1 - t_pack_in0,
            t_pack_w1 - t_pack_w0,
            t_pack_b1 - t_pack_b0,
        )

    # 1) build + load
    mod = build_conv2d_module(env, remote, s, data, kernel, bias, res, target)
    f = mod["conv2d"]

    # 2) host 生成和 pack
    t_gen0 = time.perf_counter()
    a_np, w_np, b_np = get_input_data()
    t_gen1 = time.perf_counter()
    (
        data_np,
        kernel_np,
        bias_np,
        T_pack_input,
        T_pack_weight,
        T_pack_bias,
    ) = pack_inputs(a_np, w_np, b_np)
    res_ref = None
    T_ref = 0.0
    if check_correctness:
        t_ref0 = time.perf_counter()
        res_ref = get_ref_data(a_np, w_np)
        t_ref1 = time.perf_counter()
        T_ref = t_ref1 - t_ref0

    # Materialize packed views before H2D so implicit host-side copies are not
    # folded into tvm.nd.array timing.
    t_mat0 = time.perf_counter()
    data_host = np.ascontiguousarray(data_np)
    kernel_host = np.ascontiguousarray(kernel_np)
    bias_host = np.ascontiguousarray(bias_np)
    res_host = np.ascontiguousarray(np.zeros(topi.utils.get_const_tuple(res.shape), dtype=res.dtype))
    t_mat1 = time.perf_counter()

    # 3) H2D
    dev = remote.ext_dev(0) if "vta" in target.keys else remote.cpu(0)
    t2 = time.perf_counter()
    data_arr = tvm.nd.array(data_host, dev)
    kernel_arr = tvm.nd.array(kernel_host, dev)
    bias_arr = tvm.nd.array(bias_host, dev)
    res_arr = tvm.nd.array(res_host, dev)
    dev.sync()
    t3 = time.perf_counter()

    # 4) warmup
    for _ in range(warmup):
        f(data_arr, kernel_arr, bias_arr, res_arr)
    dev.sync()

    # 5) host submit + device sync 分离
    submit_times = []
    sync_times = []

    if env.TARGET in ["sim", "tsim"]:
        simulator.clear_stats()

    for _ in range(number):
        t4 = time.perf_counter()
        f(data_arr, kernel_arr, bias_arr, res_arr)
        t5 = time.perf_counter()
        dev.sync()
        t6 = time.perf_counter()
        submit_times.append(t5 - t4)
        sync_times.append(t6 - t5)

    stats = {}
    if env.TARGET in ["sim", "tsim"]:
        stats = simulator.stats()

    # 5b) steady-state kernel benchmark with device-resident tensors.
    # This still includes board-side runtime/driver overhead, but excludes
    # Python packing and host<->device copies performed above.
    ftimer = mod.time_evaluator("conv2d", dev, number=1, repeat=number)
    timer_res = ftimer(data_arr, kernel_arr, bias_arr, res_arr)
    steady_times = list(timer_res.results)

    # 6) D2H
    t7 = time.perf_counter()
    res_out = res_arr.numpy()
    t8 = time.perf_counter()

    # 7) correctness
    correct = True
    if check_correctness:
        res_orig = res_out
        if data_pack:
            res_orig = res_orig.transpose((0, 4, 1, 5, 2, 3)).reshape(
                wl.batch, wl.out_filter, fout_height, fout_width
            )
            bias_orig = bias_np.transpose((0, 4, 1, 5, 2, 3)).reshape(
                wl.batch, wl.out_filter, 1, 1
            )
        else:
            bias_orig = bias_np

        ref = res_ref >> env.WGT_WIDTH
        ref += bias_orig
        ref = np.clip(ref, 0, (1 << env.OUT_WIDTH) - 1)
        ref = ref.astype(env.out_dtype)

        correct = np.allclose(res_orig, ref, atol=0, rtol=0)

    T_input_gen = t_gen1 - t_gen0
    T_pack = T_pack_input + T_pack_weight + T_pack_bias
    T_materialize = t_mat1 - t_mat0
    T_h2d = t3 - t2
    T_submit = float(np.mean(submit_times))
    T_sync = float(np.mean(sync_times))
    T_kernel_steady = float(np.mean(steady_times))
    T_d2h = t8 - t7
    total = T_input_gen + T_ref + T_pack + T_materialize + T_h2d + T_submit + T_sync + T_d2h

    gops_e2e_run = (num_ops / (T_submit + T_sync)) / 1e9
    gops_kernel_steady = (num_ops / T_kernel_steady) / 1e9
    copy_bytes = data_host.nbytes + kernel_host.nbytes + bias_host.nbytes + res_host.nbytes
    copy_gbps = (copy_bytes * 8 / (T_h2d + 1e-12)) / 1e9

    row = {
        "device": device_name,
        "workload": f"{wl.height}x{wl.width}_ic{wl.in_filter}_oc{wl.out_filter}_k{wl.hkernel}s{wl.hstride}",
        "batch": wl.batch,
        "height": wl.height,
        "width": wl.width,
        "in_filter": wl.in_filter,
        "out_filter": wl.out_filter,
        "hkernel": wl.hkernel,
        "wkernel": wl.wkernel,
        "hstride": wl.hstride,
        "wstride": wl.wstride,
        "hpad": wl.hpad,
        "wpad": wl.wpad,
        "T_input_gen_s": T_input_gen,
        "T_ref_s": T_ref,
        "T_pack_input_s": T_pack_input,
        "T_pack_weight_s": T_pack_weight,
        "T_pack_bias_s": T_pack_bias,
        "T_pack_s": T_pack,
        "T_materialize_s": T_materialize,
        "T_h2d_s": T_h2d,
        "T_submit_s": T_submit,
        "T_sync_s": T_sync,
        "T_kernel_steady_s": T_kernel_steady,
        "T_d2h_s": T_d2h,
        "total_s": total,
        "prep_pct": 100.0 * (T_input_gen + T_ref + T_pack + T_materialize) / total if total else 0.0,
        "pack_pct": 100.0 * T_pack / total if total else 0.0,
        "copy_pct": 100.0 * (T_h2d + T_d2h) / total if total else 0.0,
        "gops_submit_sync": gops_e2e_run,
        "gops_kernel_steady": gops_kernel_steady,
        "h2d_eff_gbps": copy_gbps,
        "ok": bool(correct),
    }

    print(
        "{dev:<4} {wl:<28} "
        "gen={gen:>8.4f}ms  ref={ref:>8.4f}ms  pack={pack:>8.4f}ms  "
        "mat={mat:>8.4f}ms  h2d={h2d:>8.4f}ms  submit={sub:>8.4f}ms  sync={syn:>8.4f}ms  "
        "kernel={ker:>8.4f}ms  d2h={d2h:>8.4f}ms  total={tot:>8.4f}ms  "
        "prep%={prep:>5.1f}  pack%={pp:>5.1f}  copy%={cp:>5.1f}  ok={ok}".format(
            dev=device_name,
            wl=row["workload"],
            gen=T_input_gen * 1e3,
            ref=T_ref * 1e3,
            pack=T_pack * 1e3,
            mat=T_materialize * 1e3,
            h2d=T_h2d * 1e3,
            sub=T_submit * 1e3,
            syn=T_sync * 1e3,
            ker=T_kernel_steady * 1e3,
            d2h=T_d2h * 1e3,
            tot=total * 1e3,
            prep=row["prep_pct"],
            pp=row["pack_pct"],
            cp=row["copy_pct"],
            ok=row["ok"],
        )
    )

    csv_writer.writerow(row)
    records.append(row)

    return correct, stats, row


def run_device(device="vta", number=20, warmup=2, check_correctness=False):
    def _run(env, remote):
        if device == "vta":
            target = env.target
            if env.TARGET not in ["sim", "tsim", "intelfocl"]:
                assert tvm.runtime.enabled("rpc")
                # 你已经有自己的 bitstream / runtime 时，这两句可按需保留或注释
                # program_fpga(remote, bitstream=None)
                # reconfig_runtime(remote)
        elif device == "arm_cpu":
            target = env.target_vta_cpu
        else:
            raise RuntimeError("Unsupported device")

        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = f"vta_conv2d_decomp_{device}_{ts}.csv"
        print(f"\n[CSV] Writing records to: {csv_path}\n")

        fieldnames = [
            "device",
            "workload",
            "batch",
            "height",
            "width",
            "in_filter",
            "out_filter",
            "hkernel",
            "wkernel",
            "hstride",
            "wstride",
            "hpad",
            "wpad",
            "T_input_gen_s",
            "T_ref_s",
            "T_pack_input_s",
            "T_pack_weight_s",
            "T_pack_bias_s",
            "T_pack_s",
            "T_materialize_s",
            "T_h2d_s",
            "T_submit_s",
            "T_sync_s",
            "T_kernel_steady_s",
            "T_d2h_s",
            "total_s",
            "prep_pct",
            "pack_pct",
            "copy_pct",
            "gops_submit_sync",
            "gops_kernel_steady",
            "h2d_eff_gbps",
            "ok",
        ]

        records = []
        tuning_log = os.environ.get("TVM_TUNING_LOG", "").strip()
        if tuning_log:
            print(f"[AutoTVM] Using tuning log: {tuning_log}")
            autotvm_ctx = autotvm.tophub.context(target, extra_files=[tuning_log])
        else:
            autotvm_ctx = autotvm.tophub.context(target)

        with open(csv_path, "w", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()

            with autotvm_ctx:
                for name, wl in get_workloads():
                    print(f"\n=== CONV2D decomposition {name} target={device} ===")
                    run_conv2d_decomposed(
                        env,
                        remote,
                        wl,
                        target,
                        writer,
                        records,
                        check_correctness=check_correctness,
                        print_ir=False,
                        number=number,
                        warmup=warmup,
                    )

        print("\n=== Summary (top by total latency) ===")
        records_sorted = sorted(records, key=lambda r: r["total_s"], reverse=True)
        for r in records_sorted[:10]:
            print(
                "{dev:<4} {wl:<28} total={tot:>8.4f}ms kernel={ker:>8.4f}ms prep%={prep:>5.1f} pack%={pp:>5.1f} copy%={cp:>5.1f} gops={g:>6.2f}".format(
                    dev=r["device"],
                    wl=r["workload"],
                    tot=r["total_s"] * 1e3,
                    ker=r["T_kernel_steady_s"] * 1e3,
                    prep=r["prep_pct"],
                    pp=r["pack_pct"],
                    cp=r["copy_pct"],
                    g=r["gops_kernel_steady"],
                )
            )

        print("\nDone.\n")

    remote = rpc.connect("192.168.1.107", 9090)
    _run(env, remote)


def sweep_vta_configs(number=5, warmup=1):
    """Direct-RPC sweep of selected AutoTVM config indices for one VTA workload."""
    register_vta_conv2d_template()
    remote = rpc.connect("192.168.1.107", 9090)
    target = env.target

    selected_wl = os.environ.get("VTA_SWEEP_WORKLOAD", "resnet-18.C2").strip()
    wl_entry = next(((name, wl) for name, wl in get_workloads() if name == selected_wl), None)
    if wl_entry is None:
        raise RuntimeError(f"Unknown VTA_SWEEP_WORKLOAD={selected_wl}")
    name, wl = wl_entry

    layout = "NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN)
    data_shape = (
        wl.batch // env.BATCH,
        wl.in_filter // env.BLOCK_IN,
        wl.height,
        wl.width,
        env.BATCH,
        env.BLOCK_IN,
    )
    kernel_shape = (
        wl.out_filter // env.BLOCK_OUT,
        wl.in_filter // env.BLOCK_IN,
        wl.hkernel,
        wl.wkernel,
        env.BLOCK_OUT,
        env.BLOCK_IN,
    )
    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    kernel = te.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
    task_args = (
        data,
        kernel,
        (wl.hstride, wl.wstride),
        relay.nn.get_pad_tuple2d((wl.hpad, wl.wpad)),
        (1, 1),
        layout,
        env.acc_dtype,
    )
    task = autotvm.task.create("conv2d_packed.vta", args=task_args, target=target, target_host=env.target_host)

    print(f"\n=== VTA config sweep: {name} ===")
    print(f"Config space size: {len(task.config_space)}")

    _, _, cpu_row = run_conv2d_decomposed(
        env,
        remote,
        wl,
        env.target_vta_cpu,
        csv_writer=_NullWriter(),
        records=[],
        check_correctness=False,
        print_ir=False,
        number=number,
        warmup=warmup,
    )
    print(
        "CPU baseline: device_exec={ker:>8.4f}ms  e2e_total={tot:>8.4f}ms".format(
            ker=cpu_row["T_kernel_steady_s"] * 1e3,
            tot=cpu_row["total_s"] * 1e3,
        )
    )

    knob_mode = os.environ.get("VTA_SWEEP_KNOBS", "0") == "1"
    selected_cfgs = []

    if knob_mode:
        max_configs = int(os.environ.get("VTA_SWEEP_MAX_CONFIGS", "32"))
        tile_h_vals = [int(x) for x in os.environ.get("VTA_SWEEP_TILE_H", "56").split(",") if x.strip()]
        tile_w_vals = [int(x) for x in os.environ.get("VTA_SWEEP_TILE_W", "1,2,4,7,8,14,28,56").split(",") if x.strip()]
        tile_ci_vals = [int(x) for x in os.environ.get("VTA_SWEEP_TILE_CI", "1").split(",") if x.strip()]
        tile_co_vals = [int(x) for x in os.environ.get("VTA_SWEEP_TILE_CO", "1").split(",") if x.strip()]
        oc_thread_vals = [int(x) for x in os.environ.get("VTA_SWEEP_OC_THREAD", "1,2").split(",") if x.strip()]
        h_thread_vals = [int(x) for x in os.environ.get("VTA_SWEEP_H_THREAD", "1").split(",") if x.strip()]

        for idx in range(len(task.config_space)):
            cfg = task.config_space.get(idx)
            splits = cfg.to_json_dict()["entity"]
            split_map = {name: value for name, _, value in splits}
            if (
                split_map["tile_h"][-1] in tile_h_vals
                and split_map["tile_w"][-1] in tile_w_vals
                and split_map["tile_ci"][-1] in tile_ci_vals
                and split_map["tile_co"][-1] in tile_co_vals
                and split_map["oc_nthread"] in oc_thread_vals
                and split_map["h_nthread"] in h_thread_vals
            ):
                selected_cfgs.append((idx, cfg))
                if len(selected_cfgs) >= max_configs:
                    break
        print(
            "Selected knob filters: tile_h={th} tile_w={tw} tile_ci={tci} tile_co={tco} oc_nthread={oc} h_nthread={ht} max_configs={mx}".format(
                th=tile_h_vals,
                tw=tile_w_vals,
                tci=tile_ci_vals,
                tco=tile_co_vals,
                oc=oc_thread_vals,
                ht=h_thread_vals,
                mx=max_configs,
            )
        )
    else:
        raw_indices = os.environ.get("VTA_SWEEP_CONFIGS", "0,1,2,3,4,5,6,7").strip()
        config_indices = [int(x) for x in raw_indices.split(",") if x.strip()]
        print(f"Selected config indices: {config_indices}")
        for idx in config_indices:
            if idx < 0 or idx >= len(task.config_space):
                print(f"[skip] config #{idx}: out of range")
                continue
            selected_cfgs.append((idx, task.config_space.get(idx)))

    results = []
    for idx, cfg in selected_cfgs:
        print(f"\n--- config #{idx} ---")
        print(cfg)
        try:
            ok, _, row = run_conv2d_decomposed(
                env,
                remote,
                wl,
                target,
                csv_writer=_NullWriter(),
                records=[],
                check_correctness=False,
                print_ir=False,
                number=number,
                warmup=warmup,
                config=cfg,
            )
            results.append((idx, cfg, ok, row))
            print(
                "compare: vta_device_exec/cpu_device_exec = {ratio:>6.2f}x".format(
                    ratio=row["T_kernel_steady_s"] / cpu_row["T_kernel_steady_s"]
                )
            )
        except Exception as err:  # pylint: disable=broad-except
            print(f"[fail] config #{idx}: {err}")
            results.append((idx, cfg, False, None))


class _NullWriter:
    def writerow(self, row):
        return None


def debug_vta_insn_dump():
    """Dump VTA UOP/insn streams for two representative workloads."""
    debug_workloads = [
        ("probe.k3.28.ic16.oc16.s1", Workload(env.BATCH, 28, 28, 16, 16, 3, 3, 1, 1, 1, 1)),
        ("resnet-18.C3", Workload(env.BATCH, 56, 56, 64, 128, 3, 3, 1, 1, 2, 2)),
    ]
    selected = os.environ.get("VTA_DEBUG_WORKLOAD", "").strip()
    if selected:
        debug_workloads = [(name, wl) for name, wl in debug_workloads if name == selected]
        if not debug_workloads:
            raise RuntimeError(f"Unknown VTA_DEBUG_WORKLOAD={selected}")

    remote = rpc.connect("192.168.1.107", 9090)
    target = env.target
    debug_flag = env.DEBUG_DUMP_INSN
    if os.environ.get("VTA_DEBUG_UOP", "0") == "1":
        debug_flag |= env.DEBUG_DUMP_UOP

    for name, wl in debug_workloads:
        print(f"\n=== VTA debug dump {name} ===")
        layout = "NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN)
        data_shape = (
            wl.batch // env.BATCH,
            wl.in_filter // env.BLOCK_IN,
            wl.height,
            wl.width,
            env.BATCH,
            env.BLOCK_IN,
        )
        kernel_shape = (
            wl.out_filter // env.BLOCK_OUT,
            wl.in_filter // env.BLOCK_IN,
            wl.hkernel,
            wl.wkernel,
            env.BLOCK_OUT,
            env.BLOCK_IN,
        )
        bias_shape = (
            wl.batch // env.BATCH,
            wl.out_filter // env.BLOCK_OUT,
            1,
            1,
            env.BATCH,
            env.BLOCK_OUT,
        )

        data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
        kernel = te.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
        bias = te.placeholder(bias_shape, name="bias", dtype=env.acc_dtype)
        padding = relay.nn.get_pad_tuple2d((wl.hpad, wl.wpad))

        with target:
            res = vta.top.conv2d_packed(
                data,
                kernel,
                (wl.hstride, wl.wstride),
                padding,
                (1, 1),
                layout,
                env.acc_dtype,
            )
            res = topi.right_shift(res, 8)
            res = topi.add(res, bias)
            res = my_clip(res, 0, (1 << env.OUT_WIDTH) - 1)
            res = topi.cast(res, env.out_dtype)
            s = vta.top.schedule_conv2d_packed([res])

        mod = build_conv2d_debug_module(env, remote, s, data, kernel, bias, res, target, debug_flag)
        f = mod["conv2d_debug"]
        dev = remote.ext_dev(0)

        a_min, a_max = -(1 << (env.INP_WIDTH - 1)), (1 << (env.INP_WIDTH - 1))
        w_min, w_max = -(1 << (env.WGT_WIDTH - 1)), (1 << (env.WGT_WIDTH - 1))
        b_min = -(1 << (env.INP_WIDTH + env.WGT_WIDTH - 2))
        b_max = (1 << (env.INP_WIDTH + env.WGT_WIDTH - 2))

        a_np = np.random.randint(a_min, a_max, size=(wl.batch, wl.in_filter, wl.height, wl.width)).astype(
            env.inp_dtype
        )
        w_np = np.random.randint(
            w_min, w_max, size=(wl.out_filter, wl.in_filter, wl.hkernel, wl.wkernel)
        ).astype(env.wgt_dtype)
        b_np = np.random.randint(b_min, b_max, size=(wl.batch, wl.out_filter, 1, 1)).astype(
            env.acc_dtype
        )

        data_np = np.ascontiguousarray(
            a_np.reshape(
                wl.batch // env.BATCH,
                env.BATCH,
                wl.in_filter // env.BLOCK_IN,
                env.BLOCK_IN,
                wl.height,
                wl.width,
            ).transpose((0, 2, 4, 5, 1, 3))
        )
        kernel_np = np.ascontiguousarray(
            w_np.reshape(
                wl.out_filter // env.BLOCK_OUT,
                env.BLOCK_OUT,
                wl.in_filter // env.BLOCK_IN,
                env.BLOCK_IN,
                wl.hkernel,
                wl.wkernel,
            ).transpose((0, 2, 4, 5, 1, 3))
        )
        bias_np = np.ascontiguousarray(
            b_np.reshape(
                wl.batch // env.BATCH,
                wl.out_filter // env.BLOCK_OUT,
                1,
                1,
                env.BATCH,
                env.BLOCK_OUT,
            )
        )
        out_np = np.ascontiguousarray(np.zeros(topi.utils.get_const_tuple(res.shape), dtype=res.dtype))

        data_arr = tvm.nd.array(data_np, dev)
        kernel_arr = tvm.nd.array(kernel_np, dev)
        bias_arr = tvm.nd.array(bias_np, dev)
        out_arr = tvm.nd.array(out_np, dev)

        print("Running one debug invocation; board-side runtime should dump instruction streams.")
        if debug_flag & env.DEBUG_DUMP_UOP:
            print("UOP dumping is enabled.")
        f(data_arr, kernel_arr, bias_arr, out_arr)
        dev.sync()


def benchmark_rpc_noop(number=50):
    """Measure host<->remote call overhead with a minimal remotely loaded module."""
    remote = rpc.connect("192.168.1.107", 9090)
    sysroot = os.environ["SDKTARGETSYSROOT"]

    ib = tvm.tir.ir_builder.create()
    ib.emit(tvm.tir.Evaluate(tvm.tir.const(0, "int32")))
    noop = tvm.tir.PrimFunc([], ib.get())
    noop = noop.with_attr("global_symbol", "rpc_noop")
    noop = noop.with_attr("tir.noalias", True)
    ir_mod = tvm.IRModule({"rpc_noop": noop})
    mod = tvm.build(ir_mod, target=env.target_host)

    tmp = utils.tempdir()
    so_path = tmp.relpath("rpc_noop.so")
    fcompile = cc.cross_compiler("aarch64-xilinx-linux-g++")
    mod.export_library(
        so_path,
        fcompile=fcompile,
        options=[
            f"--sysroot={sysroot}",
            f"-Wl,-rpath-link,{sysroot}/lib",
            f"-Wl,-rpath-link,{sysroot}/usr/lib",
            f"-L{sysroot}/lib",
            f"-L{sysroot}/usr/lib",
        ],
    )
    remote.upload(so_path)
    remote_mod = remote.load_module("rpc_noop.so")
    f = remote_mod["rpc_noop"]

    # Host-observed round-trip cost.
    host_times = []
    for _ in range(number):
        t0 = time.perf_counter()
        f()
        t1 = time.perf_counter()
        host_times.append(t1 - t0)

    # Remote-side time_evaluator cost. This excludes local Python timing noise,
    # but still measures remote packed-func invocation overhead.
    dev = remote.cpu(0)
    ftimer = remote_mod.time_evaluator("rpc_noop", dev, number=1, repeat=number)
    remote_times = list(ftimer().results)

    print("\n=== RPC noop benchmark ===")
    print(
        "host_roundtrip={host:>8.4f}ms  remote_eval={remote_t:>8.4f}ms  samples={n}".format(
            host=float(np.mean(host_times)) * 1e3,
            remote_t=float(np.mean(remote_times)) * 1e3,
            n=number,
        )
    )
    print(
        "Interpretation: host_roundtrip includes local Python + RPC request/response; "
        "remote_eval is the remote packed-func invocation cost."
    )


if __name__ == "__main__":
    if os.environ.get("RPC_NOOP_BENCH", "0") == "1":
        benchmark_rpc_noop(number=int(os.environ.get("RPC_NOOP_REPEAT", "50")))
    elif os.environ.get("VTA_SWEEP", "0") == "1":
        sweep_vta_configs(
            number=int(os.environ.get("VTA_SWEEP_NUMBER", "5")),
            warmup=int(os.environ.get("VTA_SWEEP_WARMUP", "1")),
        )
    elif os.environ.get("VTA_DEBUG_DUMP", "0") == "1":
        debug_vta_insn_dump()
    else:
        run_device(device="vta", number=5, warmup=1, check_correctness=False)
        run_device(device="arm_cpu", number=5, warmup=1, check_correctness=False)
