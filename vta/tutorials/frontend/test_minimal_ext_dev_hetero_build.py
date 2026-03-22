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
"""
Minimal heterogeneous Relay build reproducer for VTA ext_dev.

Instead of hand-writing packed operators, this script constructs a tiny NCHW
network, quantizes it, then runs ``graph_pack(..., device_annot=True)`` so the
input to ``relay.build`` follows the same path as the working AXU5EVB baseline.
"""

from __future__ import absolute_import, print_function

import argparse

import numpy as np

import tvm
from tvm import relay

import vta
from vta.top import graph_pack


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-relay",
        action="store_true",
        help="Dump the annotated Relay program before build",
    )
    return parser.parse_args()


def make_small_float_module():
    # Keep channels aligned to VTA block size so graph_pack can legally pack.
    data = relay.var("data", shape=(1, 16, 16, 16), dtype="float32")
    weight0 = relay.var("weight0", shape=(16, 16, 3, 3), dtype="float32")
    weight1 = relay.var("weight1", shape=(16, 16, 3, 3), dtype="float32")
    bias = relay.var("bias", shape=(16, 1, 1), dtype="float32")
    dense_w = relay.var("dense_w", shape=(8, 16), dtype="float32")
    dense_b = relay.var("dense_b", shape=(8,), dtype="float32")

    # First conv stays outside the packed VTA region, matching the official
    # skip_conv_layers=[0] quantization setup.
    x = relay.nn.conv2d(data, weight0, padding=(1, 1), channels=16, kernel_size=(3, 3))
    x = relay.nn.relu(x)
    x = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2))

    # Second conv is the actual packed/VTA candidate.
    x = relay.nn.conv2d(x, weight1, padding=(1, 1), channels=16, kernel_size=(3, 3))
    x = relay.add(x, bias)
    x = relay.nn.relu(x)
    x = relay.nn.global_avg_pool2d(x)
    x = relay.nn.batch_flatten(x)
    x = relay.nn.dense(x, dense_w, units=8)
    x = relay.add(x, dense_b)

    func = relay.Function([data, weight0, weight1, bias, dense_w, dense_b], x)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    params = {
        "weight0": tvm.nd.array(np.random.randint(-2, 3, size=(16, 16, 3, 3)).astype("float32")),
        "weight1": tvm.nd.array(np.random.randint(-2, 3, size=(16, 16, 3, 3)).astype("float32")),
        "bias": tvm.nd.array(np.random.randint(-2, 3, size=(16, 1, 1)).astype("float32")),
        "dense_w": tvm.nd.array(np.random.randint(-2, 3, size=(8, 16)).astype("float32")),
        "dense_b": tvm.nd.array(np.random.randint(-2, 3, size=(8,)).astype("float32")),
    }
    return mod, params


def build_minimal_module():
    env = vta.get_env()
    mod, params = make_small_float_module()

    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params=params)

    relay_prog = graph_pack(
        mod["main"],
        env.BATCH,
        env.BLOCK_OUT,
        env.WGT_WIDTH,
        start_name="nn.max_pool2d",
        stop_name="nn.global_avg_pool2d",
        device_annot=True,
        annot_start_name="nn.conv2d",
        annot_end_name="annotation.stop_fusion",
    )
    relay_prog = relay.transform.InferType()(tvm.IRModule.from_expr(relay_prog))["main"]

    build_target = {
        "cpu": env.target_vta_cpu,
        "ext_dev": env.target,
    }
    return env, relay_prog, params, build_target


def main():
    args = parse_args()
    env, relay_prog, params, build_target = build_minimal_module()

    print("========== Configuration ==========")
    print("env.TARGET =", env.TARGET)
    print("env.target =", env.target)
    print("env.host   =", env.target_host)
    print("build_target =", build_target)
    print("===================================")

    if args.print_relay:
        print("\n[RELAY]")
        print(relay_prog.astext(show_meta_data=False))

    print("[BUILD] relay.build ...")
    with vta.build_config(
        opt_level=3,
        disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
    ):
        graph, lib, lowered_params = relay.build(
            relay_prog,
            target=build_target,
            target_host=env.target_host,
            params=params,
        )

    print("[BUILD] success")
    print("[GRAPH] length =", len(graph))
    print("[LIB] type =", type(lib))
    print("[PARAMS] keys =", sorted(lowered_params.keys()))


if __name__ == "__main__":
    main()
