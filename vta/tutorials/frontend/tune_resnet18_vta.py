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
"""Tune packed conv2d tasks for VTA ResNet on AXU5EVB."""

from __future__ import absolute_import, print_function

import argparse
import os
import traceback

from mxnet.gluon.model_zoo import vision

import tvm
from tvm import te, relay, autotvm, topi
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

import vta
from vta.top import graph_pack


PACK_DICT = {
    "resnet18_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet18_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet50_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet101_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18_v1", choices=sorted(PACK_DICT.keys()))
    parser.add_argument("--tracker-host", default="127.0.0.1")
    parser.add_argument("--tracker-port", type=int, default=9190)
    parser.add_argument("--rpc-key", default="axu5evb")
    parser.add_argument("--log-file", default="vta.resnet18_v1.axu5evb.tuning.log")
    parser.add_argument("--tuner", default="random", choices=["xgb", "ga", "random", "gridsearch"])
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--early-stopping", type=int, default=0)
    parser.add_argument("--measure-number", type=int, default=3)
    parser.add_argument("--measure-timeout", type=int, default=120)
    return parser.parse_args()


def compile_network(env, model):
    dtype_dict = {"data": "float32"}
    shape_dict = {"data": (env.BATCH, 3, 224, 224)}

    gluon_model = vision.get_model(model, pretrained=True)
    mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params=params)

    start_name, stop_name = PACK_DICT[model]
    relay_prog = graph_pack(
        mod["main"],
        env.BATCH,
        env.BLOCK_OUT,
        env.WGT_WIDTH,
        start_name=start_name,
        stop_name=stop_name,
    )
    return relay_prog, params


def register_vta_conv2d_template():
    from tvm.autotvm.task import TaskExtractEnv

    @tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
    def my_clip(x, a_min, a_max):
        const_min = tvm.tir.const(a_min, x.dtype)
        const_max = tvm.tir.const(a_max, x.dtype)
        x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
        x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
        return x

    TaskExtractEnv()

    @autotvm.template("conv2d_packed.vta")
    def _topi_nn_conv2d(*args, **kwargs):
        assert not kwargs, "kwargs are not supported in this template"
        a, w, strides, padding, dilation, layout, out_dtype = args

        # Newer Relay may pass 4D padding (top, left, bottom, right).
        # VTA packed conv2d template expects symmetric 2D (h, w).
        if isinstance(padding, (tuple, list)) and len(padding) == 4:
            pt, pl, pb, pr = padding
            if pt == pb and pl == pr:
                padding = (pt, pl)
            else:
                raise ValueError("Asymmetric padding is not supported for conv2d_packed.vta")

        norm_args = (a, w, strides, padding, dilation, layout, out_dtype)

        with tvm.target.vta():
            res = vta.top.conv2d_packed(*norm_args)
            res = topi.right_shift(res, 8)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == "vta":
            sch = vta.top.schedule_conv2d_packed([res])
        else:
            sch = te.create_schedule([res.op])
        return sch, [a, w, res]


def create_tuner(task, tuner_name):
    if tuner_name == "xgb":
        try:
            return XGBTuner(task, loss_type="reg")
        except ImportError:
            print("[TUNE] xgboost is not installed, fallback to random tuner")
            return RandomTuner(task)
    if tuner_name == "ga":
        return GATuner(task, pop_size=50)
    if tuner_name == "gridsearch":
        return GridSearchTuner(task)
    return RandomTuner(task)


def tune_tasks(tasks, args, measure_option):
    tmp_log = args.log_file + ".tmp"
    if os.path.exists(tmp_log):
        os.remove(tmp_log)

    for i, task in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner = create_tuner(task, args.tuner)
        if os.path.isfile(tmp_log):
            tuner.load_history(autotvm.record.load_from_file(tmp_log))

        n_trial = min(args.trials, len(task.config_space))
        tuner.tune(
            n_trial=n_trial,
            early_stopping=(args.early_stopping or None),
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log),
            ],
        )

    autotvm.record.pick_best(tmp_log, args.log_file)
    os.remove(tmp_log)


def filter_buildable_tasks(tasks):
    """Keep only tasks that can be lowered with at least one config."""
    buildable = []
    for idx, task in enumerate(tasks):
        try:
            cfg = task.config_space.get(0)
            with tvm.target.Target(task.target):
                sch, tensor_args = task.instantiate(cfg)
                _ = vta.build(sch, tensor_args, target=task.target, target_host=task.target_host)
            buildable.append(task)
            print("[TUNE] task #%d precheck: buildable" % idx)
        except Exception as err:  # pylint: disable=broad-except
            print("[TUNE] task #%d precheck: skip (%s)" % (idx, str(err).splitlines()[-1]))
            # Keep traceback short but available for debugging.
            tb = traceback.format_exc().splitlines()
            if tb:
                print("[TUNE] task #%d precheck traceback tail: %s" % (idx, tb[-1]))
    return buildable


def main():
    args = parse_args()
    env = vta.get_env()
    target = env.target

    print("========== Tuning Configuration ==========")
    print("env.TARGET      =", env.TARGET)
    print("model           =", args.model)
    print("target          =", target)
    print("tracker_host    =", args.tracker_host)
    print("tracker_port    =", args.tracker_port)
    print("rpc_key         =", args.rpc_key)
    print("tuner           =", args.tuner)
    print("trials/task     =", args.trials)
    print("early_stopping  =", args.early_stopping if args.early_stopping else "<none>")
    print("log_file        =", args.log_file)
    print("==========================================")

    register_vta_conv2d_template()
    relay_prog, params = compile_network(env, args.model)

    print("[TUNE] extracting packed conv2d tasks ...")
    mod = tvm.IRModule.from_expr(relay_prog)
    tasks = autotvm.task.extract_from_program(
        mod,
        params=params,
        ops=(relay.op.get("nn.conv2d"),),
        target=target,
        target_host=env.target_host,
    )
    tasks = list(filter(lambda t: len(t.args[0][1]) > 4 and "conv" in t.name, tasks))
    print("[TUNE] extracted %d tasks" % len(tasks))
    if not tasks:
        raise RuntimeError("No VTA packed conv2d tasks extracted")
    tasks = filter_buildable_tasks(tasks)
    print("[TUNE] buildable tasks for tuning: %d" % len(tasks))
    if not tasks:
        raise RuntimeError("No buildable VTA conv2d tasks; tuning log would be empty")

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.RPCRunner(
            args.rpc_key,
            host=args.tracker_host,
            port=args.tracker_port,
            number=args.measure_number,
            timeout=args.measure_timeout,
        ),
    )

    tune_tasks(tasks, args, measure_option)
    print("[TUNE] done, best records saved to:", args.log_file)


if __name__ == "__main__":
    main()
