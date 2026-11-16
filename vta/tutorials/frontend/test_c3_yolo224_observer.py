"""Native observer forwarding tests; no VTA/FPGA runtime is loaded."""
from pathlib import Path
import subprocess

import pytest


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    import tvm
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    binary = tmp_path_factory.mktemp("c3_observer") / "observer.so"
    cmd = ["g++", "-std=c++17", "-shared", "-fPIC", "-w",
           str(here / "c3_yolo224_graph_observer.cc"),
           str(here / "test_c3_yolo224_observer_fixture.cc"),
           "-I" + str(root / "include"), "-I" + str(root / "3rdparty/dlpack/include"),
           "-I" + str(root / "3rdparty/dmlc-core/include"),
           "-L" + str(root / "build"), "-ltvm",
           "-Wl,-rpath," + str(root / "build"), "-o", str(binary)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    handle = tvm.runtime.load_module(str(binary))
    return tvm, handle


def test_exact_forwarding_repeated_symbols_and_module_lifetime(native):
    import gc
    import json
    tvm, retained_helper = native
    original = tvm.get_global_func("c3.incremental.test_fixture")()
    proxy = tvm.get_global_func("c3.incremental.observe_module")(original, "19 a\n29 b\n57 b")
    proxy["c3_observer_begin"]()
    assert proxy["a"](3) == 20
    assert proxy["b"](8) == 25
    assert proxy["b"](9) == 26
    events = json.loads(proxy["c3_observer_end"]())
    assert [r["node"] for r in events] == [19, 29, 57]
    assert all(r["host_call_ms"] >= 0 for r in events)
    assert original["calls"]() == 3
    del original
    gc.collect()
    assert proxy["a"](4) == 21
    assert proxy["calls"]() == 4


def test_observer_rejects_missing_wrong_order_and_hidden_calls(native):
    tvm, retained_helper = native
    make = tvm.get_global_func("c3.incremental.observe_module")
    factory = tvm.get_global_func("c3.incremental.test_fixture")
    proxy = make(factory(), "19 a\n42 b")
    proxy["c3_observer_begin"]()
    with pytest.raises(tvm.error.TVMError):
        proxy["b"](1)
    with pytest.raises(tvm.error.TVMError):
        proxy["c3_observer_end"]()
    proxy = make(factory(), "19 a")
    proxy["c3_observer_begin"]()
    proxy["a"](1)
    with pytest.raises(tvm.error.TVMError):
        proxy["a"](1)


def test_observer_rejects_invalid_mapping(native):
    tvm, retained_helper = native
    make = tvm.get_global_func("c3.incremental.observe_module")
    original = tvm.get_global_func("c3.incremental.test_fixture")()
    for spec in ("", "19 a\n19 a", "42 b\n19 a", "19 __copy", "19 missing"):
        with pytest.raises(tvm.error.TVMError):
            make(original, spec)
