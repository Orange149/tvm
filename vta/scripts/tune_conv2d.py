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

"""Tuning a single conv2d operator"""

from collections import namedtuple
import logging
import os

import tvm
from tvm import te
from tvm import autotvm
from tvm import topi
from tvm import rpc
import vta
import vta.testing

env = vta.get_env()

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

resnet_wkls = [
    # Workloads of resnet18 on imagenet
    # ('resnet-18.C1',  Workload(env.BATCH, 224, 224, 3,   64,  7, 7, 3, 3, 2, 2)),
    ("resnet-18.C2", Workload(env.BATCH, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1)),
    ("resnet-18.C3", Workload(env.BATCH, 56, 56, 64, 128, 3, 3, 1, 1, 2, 2)),
    ("resnet-18.C4", Workload(env.BATCH, 56, 56, 64, 128, 1, 1, 0, 0, 2, 2)),
    ("resnet-18.C5", Workload(env.BATCH, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1)),
    ("resnet-18.C6", Workload(env.BATCH, 28, 28, 128, 256, 3, 3, 1, 1, 2, 2)),
    ("resnet-18.C7", Workload(env.BATCH, 28, 28, 128, 256, 1, 1, 0, 0, 2, 2)),
    ("resnet-18.C8", Workload(env.BATCH, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1)),
    ("resnet-18.C9", Workload(env.BATCH, 14, 14, 256, 512, 3, 3, 1, 1, 2, 2)),
    ("resnet-18.C10", Workload(env.BATCH, 14, 14, 256, 512, 1, 1, 0, 0, 2, 2)),
    ("resnet-18.C11", Workload(env.BATCH, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1)),
]


@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
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


def create_conv2d_task_args(N, CI, H, W, CO, KH, KW, strides, padding, dilation):
    data_shape = (N // env.BATCH, CI // env.BLOCK_IN, H, W, env.BATCH, env.BLOCK_IN)
    kernel_shape = (CO // env.BLOCK_OUT, CI // env.BLOCK_IN, KH, KW, env.BLOCK_OUT, env.BLOCK_IN)

    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    kernel = te.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
    layout = "NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN)
    return (data, kernel, strides, padding, dilation, layout, env.acc_dtype)


if __name__ == "__main__":

    # Logging config (for printing tuning log to the screen)
    logging.basicConfig()
    # logging.getLogger('autotvm').setLevel(logging.DEBUG)

    # Tuning log files
    log_file = "%s.conv2d.log" % (env.TARGET)
    # create tmp log file
    tmp_log_file = log_file + ".tmp"
    if os.path.exists(log_file):
        os.remove(log_file)

    # Get tracker info from env
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = os.environ.get("TVM_TRACKER_PORT", None)
    if not tracker_host or not tracker_port:
        print("Set your AutoTVM tracker node host and port variables to run the autotuner")
        exit()

    register_vta_conv2d_template()

    if env.TARGET != "sim":
        print("Request remote device...")
        remote = autotvm.measure.request_remote(env.TARGET, tracker_host, int(tracker_port), timeout=10000)
        if os.environ.get("VTA_TUNE_RECONFIG", "0") == "1":
            print("Reconfiguring VTA runtime...")
            vta.reconfig_runtime(remote)
        if os.environ.get("VTA_TUNE_PROGRAM_FPGA", "0") == "1":
            print("Programming FPGA bitstream...")
            vta.program_fpga(remote, bitstream=None)
    else:
        remote = rpc.LocalSession()

    selected_wl = os.environ.get("VTA_TUNE_WORKLOAD", "").strip()
    if selected_wl:
        workloads = [(name, wl) for name, wl in resnet_wkls if name == selected_wl]
        if not workloads:
            raise RuntimeError(f"Unknown VTA_TUNE_WORKLOAD={selected_wl}")
    else:
        workloads = resnet_wkls

    measure_number = int(os.environ.get("VTA_TUNE_MEASURE_NUMBER", "1"))
    measure_timeout = int(os.environ.get("VTA_TUNE_TIMEOUT", "10"))
    max_trials = int(os.environ.get("VTA_TUNE_TRIALS", "16"))

    for idx, (wl_name, wl) in enumerate(workloads):
        prefix = "[Task %2d/%2d] " % (idx, len(workloads))

        # Read in workload parameters
        N = wl.batch
        CI = wl.in_filter
        H = wl.height
        W = wl.width
        CO = wl.out_filter
        KH = wl.hkernel
        KW = wl.wkernel
        strides = (wl.hstride, wl.wstride)
        padding = (wl.hpad, wl.wpad)
        dilation = (1, 1)

        # Create task
        task_args = create_conv2d_task_args(N, CI, H, W, CO, KH, KW, strides, padding, dilation)
        task = autotvm.task.create(
            "conv2d_packed.vta",
            args=task_args,
            target=env.target,
            target_host=env.target_host,
        )
        print(task.config_space)

        # Tune
        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.RPCRunner(
                env.TARGET,
                host=tracker_host,
                port=int(tracker_port),
                number=measure_number,
                timeout=measure_timeout,
                # check_correctness=True, # TODO: re-enable when check_correctness works again.
            ),
        )

        # Run Tuner
        tuner = autotvm.tuner.RandomTuner(task)
        n_trial = min(len(task.config_space), max_trials)
        tuner.tune(
            n_trial=n_trial,
            early_stopping=None,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # Pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_file)
    os.remove(tmp_log_file)
