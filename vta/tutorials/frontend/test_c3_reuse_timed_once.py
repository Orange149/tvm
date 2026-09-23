"""Native/RPC timing helper regression: one call, and no hidden warmup."""
from pathlib import Path
import subprocess

import numpy as np
import tvm
from tvm import rpc
from tvm.script import tir as T


def test_one_call_through_rpc(tmp_path):
    root = Path(__file__).resolve().parents[3]
    source = Path(__file__).with_name("c3_reuse_timed_once.cc")
    helper = tmp_path / "timed_once.so"
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", str(source),
                    "-I" + str(root / "include"), "-I" + str(root / "3rdparty/dlpack/include"),
                    "-I" + str(root / "3rdparty/dmlc-core/include"),
                    "-L" + str(root / "build"), "-ltvm", "-Wl,-rpath," + str(root / "build"),
                    "-o", str(helper)], check=True, capture_output=True)
    # In-place increment exposes any extra invocation as output 2, not 1.
    @T.prim_func
    def main(a: T.Buffer((1,), "int32"), b: T.Buffer((1,), "int32"),
             out: T.Buffer((1,), "int32")):
        T.func_attr({"global_symbol": "main"})
        out[0] = out[0] + 1

    module = tvm.build(main, target="llvm", name="main")
    binary = tmp_path / "increment.so"
    module.export_library(str(binary))
    server = rpc.Server("127.0.0.1", port=19180, port_end=19200, silent=True)
    try:
        remote = rpc.connect("127.0.0.1", server.port, session_timeout=30)
        remote.upload(str(helper))
        helper_module = remote.load_module(helper.name)
        remote.upload(str(binary))
        operator = remote.load_module(binary.name)
        arrays = [tvm.nd.array(np.zeros(1, dtype="int32"), remote.cpu()) for _ in range(3)]
        elapsed = remote.get_function("c3.reuse.timed_once")(operator, *arrays)
        assert elapsed > 0
        np.testing.assert_array_equal(arrays[2].numpy(), [1])
        elapsed = remote.get_function("c3.reuse.timed_once")(operator, *arrays)
        assert elapsed > 0
        np.testing.assert_array_equal(arrays[2].numpy(), [2])
    finally:
        server.terminate()
