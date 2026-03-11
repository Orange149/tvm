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

import numpy as np
from collections import namedtuple

import tvm
from tvm import te, relay, autotvm, topi, rpc
import tvm.topi.testing
from tvm.contrib import utils, cc

import vta
from vta import program_fpga, reconfig_runtime
from vta.testing import simulator


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

# 先只保留一个 workload，确认能在板子上跑通后再扩
resnet_wkls = [
    ("resnet-18.C2", Workload(env.BATCH, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1)),
    # 之后你可以逐步加：
    # ("resnet-18.C3", Workload(env.BATCH, 56, 56, 64, 128, 3, 3, 1, 1, 2, 2)),
    # ("resnet-18.C4", Workload(env.BATCH, 56, 56, 64, 128, 1, 1, 0, 0, 2, 2)),
]


@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
    x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return x


def build_conv2d_module(env, remote, s, data, kernel, bias, res, target):
    """
    仿照你 GEMM 的 _build_and_load():
    在 PC 上交叉编译成 .so，再 upload 到板子，再 load_module。
    """
    sysroot = os.environ["SDKTARGETSYSROOT"]

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
    return remote.load_module("conv2d.so")["conv2d"]


def run_conv2d_decomposed(
    env,
    remote,
    wl,
    target,
    csv_writer,
    records,
    check_correctness=True,
    print_ir=False,
    number=20,
    warmup=2,
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

    def get_ref_data():
        a_min, a_max = -(1 << (env.INP_WIDTH - 1)), (1 << (env.INP_WIDTH - 1))
        w_min, w_max = -(1 << (env.WGT_WIDTH - 1)), (1 << (env.WGT_WIDTH - 1))
        b_min = -(1 << (env.INP_WIDTH + env.WGT_WIDTH - 2))
        b_max = (1 << (env.INP_WIDTH + env.WGT_WIDTH - 2))

        a_np = np.random.randint(a_min, a_max, size=a_shape).astype(env.inp_dtype)
        w_np = np.random.randint(w_min, w_max, size=w_shape).astype(env.wgt_dtype)
        b_np = np.random.randint(b_min, b_max, size=b_shape).astype(env.acc_dtype)

        r_np = tvm.topi.testing.conv2d_nchw_python(
            a_np.astype(env.acc_dtype),
            w_np.astype(env.acc_dtype),
            (wl.hstride, wl.wstride),
            wl.hpad,
        ).astype(env.acc_dtype)
        return a_np, w_np, b_np, r_np

    def pack_inputs(a_np, w_np, b_np):
        if not data_pack:
            return a_np, w_np, b_np

        data_np = a_np.reshape(
            wl.batch // env.BATCH,
            env.BATCH,
            wl.in_filter // env.BLOCK_IN,
            env.BLOCK_IN,
            wl.height,
            wl.width,
        ).transpose((0, 2, 4, 5, 1, 3))

        kernel_np = w_np.reshape(
            wl.out_filter // env.BLOCK_OUT,
            env.BLOCK_OUT,
            wl.in_filter // env.BLOCK_IN,
            env.BLOCK_IN,
            wl.hkernel,
            wl.wkernel,
        ).transpose((0, 2, 4, 5, 1, 3))

        bias_np = b_np.reshape(
            wl.batch // env.BATCH,
            wl.out_filter // env.BLOCK_OUT,
            1,
            1,
            env.BATCH,
            env.BLOCK_OUT,
        )
        return data_np, kernel_np, bias_np

    # 1) build + load
    f = build_conv2d_module(env, remote, s, data, kernel, bias, res, target)

    # 2) host 生成和 pack
    t0 = time.perf_counter()
    a_np, w_np, b_np, res_ref = get_ref_data()
    data_np, kernel_np, bias_np = pack_inputs(a_np, w_np, b_np)
    t1 = time.perf_counter()

    # 3) H2D
    dev = remote.ext_dev(0) if "vta" in target.keys else remote.cpu(0)
    res_np = np.zeros(topi.utils.get_const_tuple(res.shape), dtype=res.dtype)

    t2 = time.perf_counter()
    data_arr = tvm.nd.array(data_np, dev)
    kernel_arr = tvm.nd.array(kernel_np, dev)
    bias_arr = tvm.nd.array(bias_np, dev)
    res_arr = tvm.nd.array(res_np, dev)
    dev.sync()
    t3 = time.perf_counter()

    # 4) warmup
    for _ in range(warmup):
        f(data_arr, kernel_arr, bias_arr, res_arr)
    dev.sync()

    # 5) run + sync 分离
    run_times = []
    sync_times = []

    if env.TARGET in ["sim", "tsim"]:
        simulator.clear_stats()

    for _ in range(number):
        t4 = time.perf_counter()
        f(data_arr, kernel_arr, bias_arr, res_arr)
        t5 = time.perf_counter()
        dev.sync()
        t6 = time.perf_counter()
        run_times.append(t5 - t4)
        sync_times.append(t6 - t5)

    stats = {}
    if env.TARGET in ["sim", "tsim"]:
        stats = simulator.stats()

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

    T_pack = t1 - t0
    T_h2d = t3 - t2
    T_run = float(np.mean(run_times))
    T_sync = float(np.mean(sync_times))
    T_d2h = t8 - t7
    total = T_pack + T_h2d + T_run + T_sync + T_d2h

    gops = (num_ops / (T_run + T_sync)) / 1e9
    copy_bytes = data_np.nbytes + kernel_np.nbytes + bias_np.nbytes + res_np.nbytes
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
        "T_pack_s": T_pack,
        "T_h2d_s": T_h2d,
        "T_run_s": T_run,
        "T_sync_s": T_sync,
        "T_d2h_s": T_d2h,
        "total_s": total,
        "pack_pct": 100.0 * T_pack / total if total else 0.0,
        "copy_pct": 100.0 * (T_h2d + T_d2h) / total if total else 0.0,
        "gops": gops,
        "h2d_eff_gbps": copy_gbps,
        "ok": bool(correct),
    }

    print(
        "{dev:<4} {wl:<28} "
        "pack={pack:>8.4f}ms  h2d={h2d:>8.4f}ms  run={run:>8.4f}ms  "
        "sync={syn:>8.4f}ms  d2h={d2h:>8.4f}ms  total={tot:>8.4f}ms  "
        "pack%={pp:>5.1f}  copy%={cp:>5.1f}  ok={ok}".format(
            dev=device_name,
            wl=row["workload"],
            pack=T_pack * 1e3,
            h2d=T_h2d * 1e3,
            run=T_run * 1e3,
            syn=T_sync * 1e3,
            d2h=T_d2h * 1e3,
            tot=total * 1e3,
            pp=row["pack_pct"],
            cp=row["copy_pct"],
            ok=row["ok"],
        )
    )

    csv_writer.writerow(row)
    records.append(row)

    return correct, stats


def test_conv2d(device="vta"):
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
            "T_pack_s",
            "T_h2d_s",
            "T_run_s",
            "T_sync_s",
            "T_d2h_s",
            "total_s",
            "pack_pct",
            "copy_pct",
            "gops",
            "h2d_eff_gbps",
            "ok",
        ]

        records = []
        with open(csv_path, "w", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()

            with autotvm.tophub.context(target):
                for name, wl in resnet_wkls:
                    print(f"\n=== CONV2D decomposition {name} target={device} ===")
                    run_conv2d_decomposed(
                        env,
                        remote,
                        wl,
                        target,
                        writer,
                        records,
                        check_correctness=True,
                        print_ir=False,
                        number=20,
                        warmup=2,
                    )

        print("\n=== Summary (top by total latency) ===")
        records_sorted = sorted(records, key=lambda r: r["total_s"], reverse=True)
        for r in records_sorted[:10]:
            print(
                "{dev:<4} {wl:<28} total={tot:>8.4f}ms pack%={pp:>5.1f} copy%={cp:>5.1f} gops={g:>6.2f}".format(
                    dev=r["device"],
                    wl=r["workload"],
                    tot=r["total_s"] * 1e3,
                    pp=r["pack_pct"],
                    cp=r["copy_pct"],
                    g=r["gops"],
                )
            )

        print("\nDone.\n")

    remote = rpc.connect("192.168.1.248", 9090)
    _run(env, remote)


if __name__ == "__main__":
    # 建议先只测 VTA
    test_conv2d(device="vta")
    # 稳定后再打开 CPU
    # test_conv2d(device="arm_cpu")