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
"""Unit tests for VTA graph_pack boundary bridge mode."""

import tvm
from tvm import relay
from tvm.relay import op, transform

import vta
from vta.top import graphpack as vta_graphpack


def _infer(expr):
    return transform.InferType()(tvm.IRModule.from_expr(expr))["main"]


def _bridge_pack(func):
    env = vta.get_env()
    return vta_graphpack.graph_pack(
        _infer(func),
        env.BATCH,
        env.BLOCK_OUT,
        env.WGT_WIDTH,
        start_name=None,
        stop_name=None,
        boundary_bridge=True,
    )


def _conv(data, channels):
    weight = relay.var("weight", shape=(channels, channels, 3, 3), dtype="int8")
    conv = relay.nn.conv2d(
        data,
        weight,
        padding=(1, 1),
        channels=channels,
        kernel_size=(3, 3),
        out_dtype="int32",
    )
    return weight, conv


def test_boundary_bridge_tuple_output():
    env = vta.get_env()
    channels = env.BLOCK_OUT
    bitpack_start = op.op.get("annotation.bitpack_start")
    bitpack_end = op.op.get("annotation.bitpack_end")
    data = relay.var("data", shape=(env.BATCH, channels, 8, 8), dtype="int8")
    packed_data = relay.Call(bitpack_start, [data])
    weight, conv = _conv(packed_data, channels)
    out = relay.Tuple(
        [
            relay.Call(bitpack_end, [conv]),
            relay.Call(bitpack_end, [packed_data]),
        ]
    )
    func = relay.Function([data, weight], out)

    packed = _bridge_pack(func)
    packed = _infer(packed)

    assert isinstance(packed.ret_type, tvm.ir.type.TupleType)
    assert [tuple(int(dim) for dim in field.shape) for field in packed.ret_type.fields] == [
        (env.BATCH, channels, 8, 8),
        (env.BATCH, channels, 8, 8),
    ]


def test_boundary_bridge_auto_unpack_for_plain_consumer():
    env = vta.get_env()
    channels = env.BLOCK_OUT
    bitpack_start = op.op.get("annotation.bitpack_start")
    data = relay.var("data", shape=(env.BATCH, channels, 8, 8), dtype="int8")
    packed_data = relay.Call(bitpack_start, [data])
    weight, conv = _conv(packed_data, channels)
    out = relay.nn.global_avg_pool2d(conv)
    func = relay.Function([data, weight], out)

    packed = _infer(_bridge_pack(func))

    assert tuple(int(dim) for dim in packed.ret_type.shape) == (env.BATCH, channels, 1, 1)


def test_boundary_bridge_auto_pack_for_conv_input():
    env = vta.get_env()
    channels = env.BLOCK_OUT
    bitpack_end = op.op.get("annotation.bitpack_end")
    data = relay.var("data", shape=(env.BATCH, channels, 8, 8), dtype="int8")
    weight, conv = _conv(data, channels)
    out = relay.Call(bitpack_end, [conv])
    func = relay.Function([data, weight], out)

    packed = _infer(_bridge_pack(func))

    assert tuple(int(dim) for dim in packed.ret_type.shape) == (env.BATCH, channels, 8, 8)


if __name__ == "__main__":
    tvm.testing.main()
