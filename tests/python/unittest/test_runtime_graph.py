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
import tempfile
import tvm
import tvm.testing
from tvm import te, runtime
import numpy as np
import json
import pytest
from tvm import rpc
from tvm import relay
from tvm.contrib import utils, graph_executor


@tvm.testing.requires_llvm
def test_graph_simple():
    n = 4
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    s = te.create_schedule(B.op)

    node0 = {"op": "null", "name": "x", "inputs": []}
    node1 = {
        "op": "tvm_op",
        "name": "add",
        "inputs": [[0, 0, 0]],
        "attrs": {"func_name": "myadd", "flatten_data": "1", "num_inputs": "1", "num_outputs": "1"},
    }
    nodes = [node0, node1]
    arg_nodes = [0]
    node_row_ptr = [0, 1, 2]
    outputs = [[1, 0, 0]]
    shape = (4,)
    attrs = {
        "shape": ["list_shape", [shape, shape]],
        "dltype": ["list_str", ["float32", "float32"]],
        "storage_id": ["list_int", [0, 1]],
    }
    graph = {
        "nodes": nodes,
        "arg_nodes": arg_nodes,
        "node_row_ptr": node_row_ptr,
        "heads": outputs,
        "attrs": attrs,
    }
    graph = json.dumps(graph)

    def check_verify():
        mlib = tvm.build(s, [A, B], "llvm", name="myadd")
        mod = graph_executor.create(graph, mlib, tvm.cpu(0))
        a = np.random.uniform(size=(n,)).astype(A.dtype)
        mod.run(x=a)
        out = mod.get_output(0, tvm.nd.empty((n,)))
        np.testing.assert_equal(out.numpy(), a + 1)

    def check_remote(server):
        mlib = tvm.build(s, [A, B], "llvm", name="myadd")
        remote = rpc.connect(server.host, server.port)
        temp = utils.tempdir()
        dev = remote.cpu(0)
        path_dso = temp.relpath("dev_lib.so")
        mlib.export_library(path_dso)
        remote.upload(path_dso)
        mlib = remote.load_module("dev_lib.so")
        mod = graph_executor.create(graph, mlib, remote.cpu(0))
        a = np.random.uniform(size=(n,)).astype(A.dtype)
        mod.run(x=tvm.nd.array(a, dev))
        out = tvm.nd.empty((n,), device=dev)
        out = mod.get_output(0, out)
        np.testing.assert_equal(out.numpy(), a + 1)

    def check_sharing():
        x = relay.var("x", shape=(1, 10))
        y = relay.var("y", shape=(1, 10))
        z = relay.add(x, y)
        func = relay.Function([x, y], z)

        x_in = np.ones((1, 10)).astype("float32")
        params = {"x": x_in}
        graph, lib, params = relay.build(func, target="llvm", params=params)

        mod_shared = graph_executor.create(graph, lib, tvm.cpu(0))
        mod_shared.load_params(runtime.save_param_dict(params))
        num_mods = 10
        mods = [graph_executor.create(graph, lib, tvm.cpu(0)) for _ in range(num_mods)]

        for mod in mods:
            mod.share_params(mod_shared, runtime.save_param_dict(params))

        a = np.random.uniform(size=(1, 10)).astype("float32")
        for mod in mods:
            mod.run(y=a)
            out = mod.get_output(0, tvm.nd.empty((1, 10)))
            np.testing.assert_equal(out.numpy(), x_in + a)

        # Explicitly delete the shared module and verify correctness.
        del mod_shared
        for mod in mods:
            mod.run(y=a)
            out = mod.get_output(0, tvm.nd.empty((1, 10)))
            np.testing.assert_equal(out.numpy(), x_in + a)
            del mod

    check_verify()
    check_remote(rpc.Server("127.0.0.1"))
    check_sharing()


def test_load_unexpected_params():
    # Test whether graph_executor.load_params works if parameters
    # are provided that are not an expected input.
    mod = tvm.IRModule()
    params = {}
    x = relay.var("x", shape=(1, 10))
    y = relay.var("y", shape=(1, 10))
    z = relay.add(x, y)
    mod["main"] = relay.Function([x, y], z)

    graph_module = relay.build(mod, target="llvm", params=params)
    rt_mod = tvm.contrib.graph_executor.create(
        graph_module.get_graph_json(), graph_module.get_lib(), tvm.cpu(0)
    )

    new_params = graph_module.get_params()
    new_params.update({"y_unknown": np.ones((1,)).astype("float32")})
    rt_mod.load_params(runtime.save_param_dict(new_params))


@tvm.testing.requires_llvm
def test_graph_executor_pipeline_cpu():
    dev = tvm.cpu(0)
    x = relay.var("x", shape=(1, 4), dtype="float32")
    bias = relay.var("bias", shape=(1, 4), dtype="float32")
    mod = tvm.IRModule.from_expr(relay.Function([x, bias], relay.add(x, bias)))
    bias_data = np.array([[10.0, 20.0, 30.0, 40.0]], dtype="float32")

    graph_module = relay.build(mod, target="llvm", params={"bias": bias_data})
    rt_mod = graph_executor.create(graph_module.get_graph_json(), graph_module.get_lib(), dev)
    rt_mod.load_params(runtime.save_param_dict(graph_module.get_params()))

    pipeline_init = rt_mod["pipeline_init"]
    pipeline_submit = rt_mod["pipeline_submit"]
    pipeline_try_submit = rt_mod["pipeline_try_submit"]
    pipeline_wait = rt_mod["pipeline_wait"]
    pipeline_poll = rt_mod["pipeline_poll"]
    pipeline_close = rt_mod["pipeline_close"]
    pipeline_stats = rt_mod["pipeline_stats"]

    pipeline_init(2)
    x0 = tvm.nd.array(np.array([[1.0, 2.0, 3.0, 4.0]], dtype="float32"), dev)
    x1 = tvm.nd.array(np.array([[5.0, 6.0, 7.0, 8.0]], dtype="float32"), dev)
    req0 = pipeline_submit("x", x0)
    req1 = pipeline_submit("x", x1)
    assert pipeline_poll(req0) in ("queued", "running", "done")
    assert pipeline_poll(req1) in ("queued", "running", "done")

    out0 = pipeline_wait(req0, dev.device_type, dev.device_id)[0]
    out1 = pipeline_wait(req1, dev.device_type, dev.device_id)[0]
    np.testing.assert_allclose(out0.numpy(), x0.numpy() + bias_data)
    np.testing.assert_allclose(out1.numpy(), x1.numpy() + bias_data)

    stats = json.loads(pipeline_stats())
    assert stats["submitted"] == 2
    assert stats["completed"] == 2
    assert stats["errors"] == 0

    pipeline_init(1)
    req2 = pipeline_try_submit("x", x0)
    assert req2 >= 0
    assert pipeline_try_submit("x", x1) == -1
    out2 = pipeline_wait(req2, dev.device_type, dev.device_id)[0]
    np.testing.assert_allclose(out2.numpy(), x0.numpy() + bias_data)

    pipeline_close()
    with pytest.raises(tvm.error.TVMError):
        pipeline_submit("x", x0)


def test_save_load_file():
    p = np.random.randn(10)
    params = {"x": p}

    with tempfile.NamedTemporaryFile() as fp:
        tvm.runtime.save_param_dict_to_file(params, fp.name)
        params_loaded = tvm.runtime.load_param_dict_from_file(fp.name)

        assert "x" in params_loaded
        np.testing.assert_equal(p, params_loaded["x"].numpy())


if __name__ == "__main__":
    tvm.testing.main()
