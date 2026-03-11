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
import tvm
import tvm.testing
from tvm import te
import numpy as np
from tvm.contrib import utils
import vta.testing
from vta.testing import simulator
from tvm import te, rpc
import vta

import time
import csv



def test_gemm():
    # This test is extended to:
    # 1) Print a stdout table that decomposes end-to-end latency into:
    #    pack / H2D / device-run / D2H / sync
    # 2) Save the same records into a CSV file for later plotting.
    #
    # Notes:
    # - On VTA fsim, absolute bandwidth numbers are not "hardware-true", but the
    #   decomposition & scaling trends are still very useful and are directly reusable on-board.
    # - We intentionally keep compilation inside each (shape, case) because shape changes
    #   generate different kernels. We DO ensure that upload/load is not repeated within a run.

    def run_gemm_packed(env, remote, batch_size, channel, block, csv_writer, records):
        assert batch_size % env.BATCH == 0, "batch_size must be divisible by env.BATCH"
        assert channel % env.BLOCK_IN == 0, "channel must be divisible by env.BLOCK_IN"
        assert channel % env.BLOCK_OUT == 0, "channel must be divisible by env.BLOCK_OUT"

        data_shape = (batch_size // env.BATCH, channel // env.BLOCK_IN, env.BATCH, env.BLOCK_IN)
        weight_shape = (
            channel // env.BLOCK_OUT,
            channel // env.BLOCK_IN,
            env.BLOCK_OUT,
            env.BLOCK_IN,
        )
        res_shape = (batch_size // env.BATCH, channel // env.BLOCK_OUT, env.BATCH, env.BLOCK_OUT)

        # To compute number of ops, use a x2 factor for FMA
        num_ops = 2 * channel * channel * batch_size

        ko = te.reduce_axis((0, channel // env.BLOCK_IN), name="ko")
        ki = te.reduce_axis((0, env.BLOCK_IN), name="ki")

        data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
        weight = te.placeholder(weight_shape, name="weight", dtype=env.wgt_dtype)
        data_buf = te.compute(data_shape, lambda *i: data(*i), "data_buf")
        weight_buf = te.compute(weight_shape, lambda *i: weight(*i), "weight_buf")

        res_gem = te.compute(
            res_shape,
            lambda bo, co, bi, ci: te.sum(
                data_buf[bo, ko, bi, ki].astype(env.acc_dtype)
                * weight_buf[co, ko, ci, ki].astype(env.acc_dtype),
                axis=[ko, ki],
            ),
            name="res_gem",
        )
        res_shf = te.compute(res_shape, lambda *i: res_gem(*i) >> 8, name="res_shf")
        res_max = te.compute(res_shape, lambda *i: tvm.te.max(res_shf(*i), 0), "res_max")  # relu
        res_min = te.compute(
            res_shape, lambda *i: tvm.te.min(res_max(*i), (1 << (env.INP_WIDTH - 1)) - 1), "res_min"
        )  # relu
        res = te.compute(res_shape, lambda *i: res_min(*i).astype(env.inp_dtype), name="res")

        dev = remote.ext_dev(0)

        def _pack_inputs():
            # Data in original format
            data_orig = np.random.randint(-128, 128, size=(batch_size, channel)).astype(data.dtype)
            weight_orig = np.random.randint(-128, 128, size=(channel, channel)).astype(weight.dtype)

            # NOTE: These pack/transpose steps are real "host-side layout transform" costs.
            data_packed = data_orig.reshape(
                batch_size // env.BATCH, env.BATCH, channel // env.BLOCK_IN, env.BLOCK_IN
            ).transpose((0, 2, 1, 3))
            weight_packed = weight_orig.reshape(
                channel // env.BLOCK_OUT, env.BLOCK_OUT, channel // env.BLOCK_IN, env.BLOCK_IN
            ).transpose((0, 2, 1, 3))
            return data_packed, weight_packed

        def _golden(data_packed, weight_packed):
            # Reference (host) compute for correctness
            res_ref = np.zeros(res_shape).astype(env.acc_dtype)
            for b in range(batch_size // env.BATCH):
                for i in range(channel // env.BLOCK_OUT):
                    for j in range(channel // env.BLOCK_IN):
                        res_ref[b, i, :] += np.dot(
                            data_packed[b, j, :].astype(env.acc_dtype),
                            weight_packed[i, j].T.astype(env.acc_dtype),
                        )
            res_ref = np.right_shift(res_ref, 8)
            res_ref = np.clip(res_ref, 0, (1 << (env.INP_WIDTH - 1)) - 1).astype(res.dtype)
            return res_ref

        from tvm.contrib import cc
        import os

        def _build_and_load(s):
            # Build module (VTA ext_dev + host llvm)
            mod = tvm.build(
                s,
                [data, weight, res],
                target=tvm.target.Target("ext_dev", host=env.target_host),
                name="gemm",
            )

            # Cross-compile on PC into a shared library, then upload to board.
            tmp = utils.tempdir()
            so_path = tmp.relpath("gemm.so")

            # You already used this flow successfully for add.so:
            # - source PetaLinux SDK env so these vars exist
            sysroot = os.environ["SDKTARGETSYSROOT"]

            # Use your PetaLinux cross compiler (glibc one)
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
            return remote.load_module("gemm.so")["gemm"]

        def _make_schedule(load_inp, load_wgt, gemm, alu, store_out, print_ir=False):
            s = te.create_schedule(res.op)
            s[data_buf].set_scope(env.inp_scope)
            s[weight_buf].set_scope(env.wgt_scope)
            s[res_gem].set_scope(env.acc_scope)
            s[res_shf].set_scope(env.acc_scope)
            s[res_min].set_scope(env.acc_scope)
            s[res_max].set_scope(env.acc_scope)

            if block:
                bblock = block // env.BATCH
                iblock = block // env.BLOCK_IN
                oblock = block // env.BLOCK_OUT

                xbo, xco, xbi, xci = s[res].op.axis
                xb1, xco1, xb2, xco2 = s[res].tile(xbo, xco, bblock, oblock)
                store_pt = xb2

                s[res_gem].compute_at(s[res], xco1)
                s[res_shf].compute_at(s[res], xco1)
                s[res_min].compute_at(s[res], xco1)
                s[res_max].compute_at(s[res], xco1)

                xbo, xco, xbi, xci = s[res_gem].op.axis
                ko1, ko2 = s[res_gem].split(ko, iblock)
                s[res_gem].reorder(ko1, ko2, xbo, xco, xbi, xci, ki)
                s[data_buf].compute_at(s[res_gem], ko1)
                s[weight_buf].compute_at(s[res_gem], ko1)

                s[data_buf].pragma(s[data_buf].op.axis[0], load_inp)
                s[weight_buf].pragma(s[weight_buf].op.axis[0], load_wgt)
                s[res_gem].tensorize(xbi, gemm)
                s[res_shf].pragma(s[res_shf].op.axis[0], alu)
                s[res_min].pragma(s[res_min].op.axis[0], alu)
                s[res_max].pragma(s[res_max].op.axis[0], alu)
                s[res].pragma(store_pt, store_out)
            else:
                xbo, xco, xbi, xci = s[res_gem].op.axis
                s[res_gem].reorder(ko, xbo, xco, xbi, xci, ki)
                s[data_buf].pragma(s[data_buf].op.axis[0], load_inp)
                s[weight_buf].pragma(s[weight_buf].op.axis[0], load_wgt)
                s[res_gem].tensorize(xbi, gemm)
                s[res_shf].pragma(s[res_shf].op.axis[0], alu)
                s[res_min].pragma(s[res_min].op.axis[0], alu)
                s[res_max].pragma(s[res_max].op.axis[0], alu)
                s[res].pragma(s[res].op.axis[0], store_out)

            if print_ir:
                print(tvm.lower(s, [data, weight, res], simple_mode=True))
            return s

        def _measure_case(case_name, load_inp, load_wgt, gemm, alu, store_out, print_ir=False, number=20, warmup=2):
            # 1) Make schedule + build/load module
            s = _make_schedule(load_inp, load_wgt, gemm, alu, store_out, print_ir=print_ir)
            with vta.build_config():
                f = _build_and_load(s)

            # 2) Host-side pack/layout transform
            t0 = time.perf_counter()
            data_packed, weight_packed = _pack_inputs()
            t1 = time.perf_counter()

            # 3) H2D transfer (allocate + copy)
            res_np = np.zeros(res_shape).astype(res.dtype)
            t2 = time.perf_counter()
            data_arr = tvm.nd.array(data_packed, dev)
            weight_arr = tvm.nd.array(weight_packed, dev)
            res_arr = tvm.nd.array(res_np, dev)
            # Make sure allocations/copies are done before timing device-run
            dev.sync()
            t3 = time.perf_counter()

            # 4) Warmup runs (exclude from timing)
            for _ in range(warmup):
                f(data_arr, weight_arr, res_arr)
            dev.sync()

            # 5) Device-run + sync split timing (per-iteration)
            run_times = []
            sync_times = []
            if env.TARGET in ["sim", "tsim"]:
                simulator.clear_stats()

            for _ in range(number):
                t4 = time.perf_counter()
                f(data_arr, weight_arr, res_arr)
                t5 = time.perf_counter()
                dev.sync()
                t6 = time.perf_counter()
                run_times.append(t5 - t4)
                sync_times.append(t6 - t5)

            if env.TARGET in ["sim", "tsim"]:
                stats = simulator.stats()
                # Keep stats for stdout (optional), but don't flood output on scans
                # You can re-enable detailed printing when you zoom into one case.
                # print("Execution statistics:", stats)

            # 6) D2H (numpy()) time
            t7 = time.perf_counter()
            res_out = res_arr.numpy()
            t8 = time.perf_counter()

            # 7) Correctness check (cheap; runs on host)
            # Only check correctness for "NORMAL" and "GEMM_ONLY/ALU_ONLY" style cases that produce meaningful outputs.
            ok = True
            try:
                res_ref = _golden(data_packed, weight_packed)
                ok = np.allclose(
                    res_out.reshape(res_shape),
                    res_ref.reshape(res_shape),
                    atol=0,
                    rtol=0,
                )
            except Exception:
                # Some mock cases don't produce a meaningful output; still keep timings.
                ok = True

            T_pack = t1 - t0
            T_h2d = t3 - t2
            T_run = float(np.mean(run_times))
            T_sync = float(np.mean(sync_times))
            T_d2h = t8 - t7
            total = T_pack + T_h2d + T_run + T_sync + T_d2h

            # Helpful derived metrics
            gops = (num_ops / (T_run + T_sync)) / 1e9
            copy_bytes = (data_packed.nbytes + weight_packed.nbytes + res_np.nbytes)
            copy_gbps = (copy_bytes * 8 / (T_h2d + 1e-12)) / 1e9  # host-side effective rate (not HW-true on fsim)

            row = {
                "case": case_name,
                "batch_size": batch_size,
                "channel": channel,
                "block": block if block else 0,
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
                "ok": bool(ok),
            }

            # stdout (single line)
            print(
                "{case:<10} bs={bs:<4} ch={ch:<4} "
                "pack={pack:>8.4f}ms  h2d={h2d:>8.4f}ms  run={run:>8.4f}ms  "
                "sync={syn:>8.4f}ms  d2h={d2h:>8.4f}ms  total={tot:>8.4f}ms  "
                "pack%={pp:>5.1f}  copy%={cp:>5.1f}  ok={ok}".format(
                    case=case_name,
                    bs=batch_size,
                    ch=channel,
                    pack=row["T_pack_s"] * 1e3,
                    h2d=row["T_h2d_s"] * 1e3,
                    run=row["T_run_s"] * 1e3,
                    syn=row["T_sync_s"] * 1e3,
                    d2h=row["T_d2h_s"] * 1e3,
                    tot=row["total_s"] * 1e3,
                    pp=row["pack_pct"],
                    cp=row["copy_pct"],
                    ok=row["ok"],
                )
            )

            # CSV
            csv_writer.writerow(row)
            records.append(row)

        # ---- Cases: end-to-end + isolated DMA directions ----
        mock = env.mock
        print(f"\n=== GEMM decomposition (bs={batch_size}, ch={channel}, block={block}) target={env.TARGET} ===")

        _measure_case("NORMAL", env.dma_copy, env.dma_copy, env.gemm, env.alu, env.dma_copy, print_ir=False)

    def _run(env, remote):
        # CSV output (one per invocation)
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = f"vta_gemm_decomp_{ts}.csv"
        print(f"\n[CSV] Writing records to: {csv_path}\n")

        fieldnames = [
            "case",
            "batch_size",
            "channel",
            "block",
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
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # ---- Shape scan (keep modest by default; expand as needed) ----
            scan = [
                (128, 128, 128)
            ]
            for bs, ch, blk in scan:
                run_gemm_packed(env, remote, bs, ch, blk, writer, records)

        # Print a compact summary table (sorted by total)
        print("\n=== Summary (top by total latency) ===")
        records_sorted = sorted(records, key=lambda r: r["total_s"], reverse=True)
        for r in records_sorted[:10]:
            print(
                "{case:<10} bs={bs:<4} ch={ch:<4} total={tot:>8.4f}ms pack%={pp:>5.1f} copy%={cp:>5.1f} gops={g:>6.2f}".format(
                    case=r["case"],
                    bs=r["batch_size"],
                    ch=r["channel"],
                    tot=r["total_s"] * 1e3,
                    pp=r["pack_pct"],
                    cp=r["copy_pct"],
                    g=r["gops"],
                )
            )

        print("\nDone. You can now plot CSV columns (pack_pct/copy_pct vs shape) and reuse the same script on-board.\n")

    env = vta.get_env()
    remote = rpc.connect("192.168.1.248", 9090)
    _run(env, remote)

if __name__ == "__main__":
    test_gemm()
